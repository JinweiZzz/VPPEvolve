"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import (
    safe_numeric_average,
    get_fitness_score,
    format_feature_coordinates,
)

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(custom_template_dir=config.template_dir)

        # Initialize the random number generator
        random.seed()

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        # Only log once to reduce duplication
        if not hasattr(logger, "_prompt_sampler_logged"):
            logger.info("Initialized prompt sampler")
            logger._prompt_sampler_logged = True

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],  # Add inspirations parameter
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            evolution_round: Current evolution round
            diff_based_evolution: Whether to use diff-based evolution (True) or full rewrites (False)
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on evolution mode (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        else:
            # Default behavior: diff-based vs full rewrite
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs, feature_dimensions
        )

        # Format evolution history (returns content and has_inspiration_comparison flag)
        evolution_history_result = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language, feature_dimensions, evolution_round
        )
        if isinstance(evolution_history_result, tuple):
            evolution_history, has_inspiration_comparison = evolution_history_result
        else:
            evolution_history = evolution_history_result
            has_inspiration_comparison = False

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Calculate fitness and feature coordinates for the new template format
        feature_dimensions = feature_dimensions or []
        fitness_score = get_fitness_score(program_metrics, feature_dimensions)
        feature_coords = format_feature_coordinates(program_metrics, feature_dimensions)

        # Scenario temporal/spatial context for evolution guidance (if evaluator provides it)
        scenario_context = self._get_scenario_context(program_metrics)

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            fitness_score=f"{fitness_score:.4f}",
            feature_coords=feature_coords,
            feature_dimensions=", ".join(feature_dimensions) if feature_dimensions else "None",
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            scenario_context=scenario_context,
            **kwargs,
        )

        out = {
            "system": system_message,
            "user": user_message,
        }
        # Whether enable_inspiration_comparison produced non-empty result (for logging/analysis)
        out["has_inspiration_comparison"] = has_inspiration_comparison
        return out

    def _get_scenario_context(self, program_metrics: Optional[Dict[str, Any]]) -> str:
        """Get scenario temporal/spatial context for LLM evolution guidance from program metrics.
        When program_metrics contains memory.reflection (from evaluator reflection), append it."""
        if not program_metrics:
            return "No scenario context available."
        parts = []
        if isinstance(program_metrics.get("scenario_context_text"), str):
            parts.append(program_metrics["scenario_context_text"])
        else:
            summary = program_metrics.get("program_summary")
            if isinstance(summary, dict) and summary:
                import json
                try:
                    parts.append("Scenario summary (program_summary):\n" + json.dumps(summary, ensure_ascii=False, indent=2))
                except Exception:
                    parts.append(str(summary))
            else:
                parts.append("No scenario context available.")
        # Inject reasoning reflection from memory (generated by evaluator from node/global profile)
        memory = program_metrics.get("memory")
        if isinstance(memory, dict) and memory.get("reflection"):
            parts.append("\n## Previous reflection (for evolution guidance)\n" + memory["reflection"])
        return "\n".join(parts).strip()

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts = []
        skip_keys = {"program_summary", "scenario_context_text", "memory"}  # Shown in scenario_context / reflection block
        for name, value in metrics.items():
            if name in skip_keys:
                continue
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Identify improvement areas with proper fitness/feature separation"""

        improvement_areas = []
        feature_dimensions = feature_dimensions or []

        # Calculate fitness (excluding feature dimensions)
        current_fitness = get_fitness_score(metrics, feature_dimensions)

        # Track fitness changes (not individual metrics)
        if previous_programs:
            prev_metrics = previous_programs[-1].get("metrics", {})
            prev_fitness = get_fitness_score(prev_metrics, feature_dimensions)

            if current_fitness > prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_improved", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif current_fitness < prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_declined", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif abs(current_fitness - prev_fitness) < 1e-6:  # Essentially unchanged
                msg = self.template_manager.get_fragment("fitness_stable", current=current_fitness)
                improvement_areas.append(msg)

        # Note feature exploration (not good/bad, just informational)
        if feature_dimensions:
            feature_coords = format_feature_coordinates(metrics, feature_dimensions)
            if feature_coords != "No feature coordinates":
                msg = self.template_manager.get_fragment(
                    "exploring_region", features=feature_coords
                )
                improvement_areas.append(msg)

        # Code length check (configurable threshold)
        threshold = (
            self.config.suggest_simplification_after_chars or self.config.code_length_threshold
        )
        if threshold and len(current_program) > threshold:
            msg = self.template_manager.get_fragment("code_too_long", threshold=threshold)
            improvement_areas.append(msg)

        # Default guidance if nothing specific
        if not improvement_areas:
            improvement_areas.append(self.template_manager.get_fragment("no_specific_guidance"))

        return "\n".join(f"- {area}" for area in improvement_areas)

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
        evolution_round: int = 0,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("metadata", {}).get("changes", "Unknown changes")

            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("metadata", {}).get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Use the full program code
            program_code = program.get("code", "")

            # Calculate fitness score (prefers combined_score, excludes feature dimensions)
            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Use the full program code
                    program_code = program.get("code", "")

                    # Calculate fitness score (prefers combined_score, excludes feature dimensions)
                    score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Format inspirations section (returns content and whether non-empty inspiration comparison was generated)
        inspirations_section_str, has_inspiration_comparison = self._format_inspirations_section(
            inspirations, language, feature_dimensions, evolution_round
        )

        # Combine into full history (has_inspiration_comparison is used by build_prompt)
        full_history = history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )
        return full_history, has_inspiration_comparison

    def _format_inspirations_section(
        self,
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
        evolution_round: int = 0,
    ) -> str:
        """
        Format the inspirations section for the prompt

        Args:
            inspirations: List of inspiration programs
            language: Programming language
            feature_dimensions: Optional feature dimensions for scoring
            evolution_round: Current evolution round (iteration number)

        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return "", False

        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")

        inspiration_programs_str = ""

        for i, program in enumerate(inspirations):
            # Use the full program code
            program_code = program.get("code", "")

            # Calculate fitness score (prefers combined_score, excludes feature dimensions)
            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

            # Determine program type based on metadata and score
            program_type = self._determine_program_type(program, feature_dimensions or [])

            # Extract unique features (emphasizing diversity rather than just performance)
            unique_features = self._extract_unique_features(program)

            # Borrowable (from parent reflection): triggered by enable_inspiration_borrowable_from_reflection, injected in process_parallel/iteration
            borrowable_summary = program.get("borrowable_summary") or ""
            borrowable_from_reflection = (
                f"Borrowable (from parent reflection): {borrowable_summary}" if borrowable_summary else ""
            )

            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_code,
                    unique_features=unique_features,
                    borrowable_from_reflection=borrowable_from_reflection,
                )
                + "\n\n"
            )

        # Generate comparison summary if enabled and iteration >= min_round
        full_inspirations_str = inspiration_programs_str.strip()
        has_inspiration_comparison = False
        min_round = getattr(self.config, "inspiration_comparison_min_round", 5)
        if not self.config.enable_inspiration_comparison:
            pass  # No log for disabled config
        elif evolution_round < min_round:
            if evolution_round == 0 or evolution_round == min_round - 1:
                logger.info(
                    f"Inspiration comparison: will start from iteration {min_round} "
                    f"(current: {evolution_round})"
                )
        else:
            comparison_summary = self._generate_inspiration_comparison(inspirations)
            if comparison_summary and comparison_summary.strip():
                full_inspirations_str += "\n\n" + comparison_summary
                has_inspiration_comparison = True
                logger.info(
                    f"Inspiration comparison generated (iteration {evolution_round}): "
                    f"{len(comparison_summary)} chars"
                )
            else:
                n_with_attr = sum(
                    1 for p in inspirations
                    if p.get("metadata", {}).get("term_attribution", {}).get("new")
                )
                logger.info(
                    f"Inspiration comparison skipped (iteration {evolution_round}): "
                    f"need >=2 programs with term_attribution, got {n_with_attr}/{len(inspirations)}"
                )

        return inspirations_section_template.format(
            inspiration_programs=full_inspirations_str
        ), has_inspiration_comparison

    def _determine_program_type(
        self, program: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
    ) -> str:
        """
        Determine the type/category of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"

        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"

    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing unique aspects of the program
        """
        features = []

        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if (
                isinstance(changes, str)
                and self.config.include_changes_under_chars
                and len(changes) < self.config.include_changes_under_chars
            ):
                features.append(f"Modification: {changes}")

        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")

        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if (
                self.config.concise_implementation_max_lines
                and len(code.split("\n")) <= self.config.concise_implementation_max_lines
            ):
                features.append("Concise implementation")
            elif (
                self.config.comprehensive_implementation_min_lines
                and len(code.split("\n")) >= self.config.comprehensive_implementation_min_lines
            ):
                features.append("Comprehensive implementation")

        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")

        # Use num_top_programs as limit for features (similar to how we limit programs)
        feature_limit = self.config.num_top_programs
        return ", ".join(features[:feature_limit])

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _generate_inspiration_comparison(self, inspirations: List[Dict[str, Any]]) -> str:
        """
        Generate comparison summary of inspiration programs, analyzing strengths/weaknesses
        in revenue, degradation, deviation and per-device scheduling.

        Args:
            inspirations: List of inspiration programs

        Returns:
            Formatted comparison summary string
        """
        if not inspirations or len(inspirations) < 2:
            return ""

        # Collect term_attribution from all programs
        programs_with_attribution = []
        for i, program in enumerate(inspirations):
            term_attribution = program.get("metadata", {}).get("term_attribution")
            if term_attribution and "new" in term_attribution:
                programs_with_attribution.append({
                    "index": i + 1,
                    "program": program,
                    "attribution": term_attribution["new"]
                })
        
        if len(programs_with_attribution) < 2:
            return ""
        
        summary_lines = []
        summary_lines.append("## Inspiration Programs Comparison Summary")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("This section compares the device attribution across all inspiration programs,")
        summary_lines.append("highlighting strengths and weaknesses in revenue, degradation, and deviation metrics.")
        summary_lines.append("")
        
        device_types = ["pv", "wind", "storage", "vehicle", "AC", "wash"]
        metrics = ["revenue", "degradation", "deviation"]
        
        # Compare each metric
        for metric in metrics:
            summary_lines.append(f"### {metric.upper()} Analysis")
            summary_lines.append("")
            
            # Collect values for all programs and device types for this metric
            metric_data = {}
            for device_type in device_types:
                metric_data[device_type] = []
                for prog_info in programs_with_attribution:
                    if metric in prog_info["attribution"] and device_type in prog_info["attribution"][metric]:
                        value = prog_info["attribution"][metric][device_type]
                        metric_data[device_type].append({
                            "program_index": prog_info["index"],
                            "value": value
                        })
            
            # For each device type, find best and worst performing programs
            for device_type in device_types:
                if not metric_data[device_type]:
                    continue

                # Sort to find best and worst
                sorted_data = sorted(metric_data[device_type], key=lambda x: x["value"], reverse=(metric == "revenue"))
                # revenue: higher is better; degradation and deviation: lower is better

                best_prog = sorted_data[0]
                worst_prog = sorted_data[-1]

                # Compute average
                avg_value = sum(d["value"] for d in metric_data[device_type]) / len(metric_data[device_type])

                # Generate analysis text
                if metric == "revenue":
                    # revenue: higher is better
                    if best_prog["value"] > avg_value * 1.1:  # >10% above average
                        summary_lines.append(
                            f"- **Program {best_prog['program_index']}** excels in {device_type} {metric} "
                            f"({best_prog['value']:.2f}, avg: {avg_value:.2f})"
                        )
                    if worst_prog["value"] < avg_value * 0.9:  # >10% below average
                        summary_lines.append(
                            f"- **Program {worst_prog['program_index']}** underperforms in {device_type} {metric} "
                            f"({worst_prog['value']:.2f}, avg: {avg_value:.2f})"
                        )
                else:
                    # degradation and deviation: lower is better
                    if best_prog["value"] < avg_value * 0.9:  # >10% below average
                        summary_lines.append(
                            f"- **Program {best_prog['program_index']}** minimizes {device_type} {metric} "
                            f"({best_prog['value']:.2f}, avg: {avg_value:.2f})"
                        )
                    if worst_prog["value"] > avg_value * 1.1:  # >10% above average
                        summary_lines.append(
                            f"- **Program {worst_prog['program_index']}** has high {device_type} {metric} "
                            f"({worst_prog['value']:.2f}, avg: {avg_value:.2f})"
                        )
            
            summary_lines.append("")
        
        # Generate overall strengths/weaknesses summary
        summary_lines.append("### Overall Strengths and Weaknesses by Program")
        summary_lines.append("")
        
        for prog_info in programs_with_attribution:
            program_index = prog_info["index"]
            attribution = prog_info["attribution"]
            
            strengths = []
            weaknesses = []
            
            # Analyze performance per metric and device type
            for metric in metrics:
                if metric not in attribution:
                    continue
                
                for device_type in device_types:
                    if device_type not in attribution[metric]:
                        continue
                    
                    value = attribution[metric][device_type]
                    
                    # Average value of other programs for this device type and metric
                    other_values = []
                    for other_prog in programs_with_attribution:
                        if other_prog["index"] != program_index:
                            if (metric in other_prog["attribution"] and 
                                device_type in other_prog["attribution"][metric]):
                                other_values.append(other_prog["attribution"][metric][device_type])
                    
                    if not other_values:
                        continue
                    
                    avg_other = sum(other_values) / len(other_values)
                    
                    if metric == "revenue":
                        # revenue: higher is better
                        if value > avg_other * 1.1:
                            strengths.append(f"{device_type} {metric} ({value:.2f} vs avg {avg_other:.2f})")
                        elif value < avg_other * 0.9:
                            weaknesses.append(f"{device_type} {metric} ({value:.2f} vs avg {avg_other:.2f})")
                    else:
                        # degradation and deviation: lower is better
                        if value < avg_other * 0.9:
                            strengths.append(f"{device_type} {metric} ({value:.2f} vs avg {avg_other:.2f})")
                        elif value > avg_other * 1.1:
                            weaknesses.append(f"{device_type} {metric} ({value:.2f} vs avg {avg_other:.2f})")
            
            if strengths or weaknesses:
                summary_lines.append(f"**Program {program_index}:**")
                if strengths:
                    summary_lines.append(f"  - Strengths: {', '.join(strengths)}")
                if weaknesses:
                    summary_lines.append(f"  - Weaknesses: {', '.join(weaknesses)}")
                summary_lines.append("")
        
        summary_lines.append("=" * 80)
        
        return "\n".join(summary_lines)

    def _format_attribution_summary(self, term_attribution: Dict[str, Any]) -> str:
        """
        Format term_attribution as a short summary.

        Args:
            term_attribution: Result from attribution_comparison

        Returns:
            Formatted summary string
        """
        if not term_attribution:
            return ""

        summary_parts = []

        # Extract main device contribution differences per metric
        for metric in ["revenue", "degradation", "deviation"]:
            if metric not in term_attribution.get("new", {}):
                continue
            
            new_metric = term_attribution["new"][metric]
            best_metric = term_attribution["best"][metric]
            
            # Find device type with largest contribution difference
            max_diff_device = None
            max_diff_value = 0.0
            
            for device_type in ["pv", "wind", "storage", "vehicle", "AC", "wash"]:
                if device_type in new_metric and device_type in best_metric:
                    diff = abs(new_metric[device_type] - best_metric[device_type])
                    if diff > max_diff_value:
                        max_diff_value = diff
                        max_diff_device = device_type
            
            if max_diff_device:
                new_val = new_metric[max_diff_device]
                best_val = best_metric[max_diff_device]
                summary_parts.append(
                    f"{metric}:{max_diff_device}({new_val:.1f} vs {best_val:.1f})"
                )
        
        return "Attribution: " + ", ".join(summary_parts) if summary_parts else ""

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
