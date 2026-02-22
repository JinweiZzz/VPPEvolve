import asyncio
import os
from openevolve import OpenEvolve
from openevolve.config import load_config
import time
import logging

# Program summaries are written here after each run (same as openevolve output_dir)
import evaluator_joint_raw as evaluator_joint_raw_module


async def main():
    # Load config (raw config; config_joint_raw.yaml has no fixed alpha_score requirements)
    config = load_config("config_joint_raw.yaml")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"openevolve_output/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 设置摘要输出目录：每个 program 评估完成后会在此目录下生成 program_summary_<ts>.json
    summary_dir = os.path.join(output_dir, "program_summaries")
    evaluator_joint_raw_module.SUMMARY_OUTPUT_DIR = summary_dir
    
    # Initialize OpenEvolve with the first initial program
    openevolve = OpenEvolve(
        initial_program_path="program_seed_joint_raw.py",
        evaluation_file="evaluator_joint_raw.py",
        config=config,
        output_dir=output_dir,  # Use default output directory
    )
    
    # Log config.yaml content to the log file
    logger = logging.getLogger(__name__)
    config_path = "config_joint_raw.yaml"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config_content = f.read()
            logger.info("=" * 80)
            logger.info("Configuration file (config_joint_raw.yaml) content:")
            logger.info("=" * 80)
            # Log config content line by line for better readability
            for line in config_content.split('\n'):
                logger.info(f"  {line}")
            logger.info("=" * 80)
        except Exception as e:
            logger.warning(f"Failed to read {config_path} for logging: {e}")
    else:
        logger.warning(f"Config file not found: {config_path}")
    
    # Add multiple initial programs if needed
    # Method 1: Specify in code
    additional_initial_programs = []  # e.g., ["program_seed_2.py", "program_seed_3.py"]
    
    # Method 2: Load from config file (if specified)
    # You can add this to config.yaml:
    # initial_programs:
    #   - "program_seed_2.py"
    #   - "program_seed_3.py"
    if hasattr(config, 'initial_programs'):
        if isinstance(config.initial_programs, list):
            additional_initial_programs.extend(config.initial_programs)
        elif isinstance(config.initial_programs, str):
            # If it's a single string, convert to list
            additional_initial_programs.append(config.initial_programs)
    
    # Add additional initial programs to database before evolution starts
    if additional_initial_programs:
        from openevolve.database import Program
        import uuid
        
        print(f"\n{'='*60}")
        print(f"Adding {len(additional_initial_programs)} additional initial programs...")
        print(f"{'='*60}")
        
        for prog_path in additional_initial_programs:
            if os.path.exists(prog_path):
                # Load program code
                with open(prog_path, "r") as f:
                    program_code = f.read()
                
                # Evaluate the program
                program_id = str(uuid.uuid4())
                print(f"Evaluating {prog_path}...")
                initial_metrics = await openevolve.evaluator.evaluate_program(
                    program_code, program_id
                )
                
                # Extract term_attribution (device attribution) from evaluation result
                term_attribution = initial_metrics.pop("term_attribution", None)
                initial_metadata = {}
                if term_attribution is not None:
                    initial_metadata["term_attribution"] = term_attribution
                
                # Create Program object
                initial_program = Program(
                    id=program_id,
                    code=program_code,
                    language=openevolve.config.language,
                    metrics=initial_metrics,
                    iteration_found=0,
                    metadata=initial_metadata,
                )
                
                # Add to database
                openevolve.database.add(initial_program)
                print(f"  ✓ Added {prog_path} to database (ID: {program_id[:8]}...)")
                print(f"    Metrics: {initial_metrics}")
            else:
                print(f"  ✗ Warning: {prog_path} not found, skipping")
        
        print(f"{'='*60}\n")
    
    # Run evolution
    best_program = await openevolve.run(
        iterations=config.max_iterations,
        target_score=None,
        checkpoint_path=None,
    )
    
    # Get the checkpoint path
    checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            os.path.join(checkpoint_dir, d)
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        if checkpoints:
            latest_checkpoint = sorted(
                checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
            )[-1]
    
    print(f"\nEvolution complete!")
    print(f"Best program metrics (on predicted scenarios):")
    for name, value in best_program.metrics.items():
        # Handle mixed types: format numbers as floats, others as strings
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    # Test best program on real scenarios
    print(f"\n{'='*60}")
    print(f"Evaluating best program on REAL scenarios...")
    print(f"{'='*60}")
    
    from evaluator_joint_raw import evaluate_real
    best_program_path = os.path.join(openevolve.output_dir, "best", "best_program.py")
    if os.path.exists(best_program_path):
        real_results = evaluate_real(best_program_path)
        print(f"\nReal scenario evaluation complete!")
        print(f"Best program path: {best_program_path}")
    else:
        print(f"Warning: Best program file not found at {best_program_path}")
    
    if latest_checkpoint:
        print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
        print(f"To resume, use: --checkpoint {latest_checkpoint}")


if __name__ == "__main__":
    asyncio.run(main())
