import importlib.util
import math
import yaml
import os
import json
import logging
from collections import defaultdict
import numpy as np
from simulator_joint_stub import run_joint_episode
from data_stub import SCENARIOS, SCENARIOS_REAL

# Logger
logger = logging.getLogger(__name__)

# Config file used by run_joint_raw (distinct from run_joint.py's config.yaml)
CONFIG_PATH_RAW = "config_joint_raw.yaml"

# Load lambda coefficients from config
def load_lambda_coefficients():
    """Load lambda coefficients from config_joint_raw.yaml (top-level, not under evaluator section)."""
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_PATH_RAW)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Read lambda from top level (EvaluatorConfig does not support custom fields)
        lambda_rev = config.get('lambda_rev', 1e-9)
        lambda_dev = config.get('lambda_dev', 1e-7)
        lambda_deg = config.get('lambda_deg', 1e-8)
        lambda_risk = config.get('lambda_risk', 1e-2)
        
        return lambda_rev, lambda_dev, lambda_deg, lambda_risk
    except Exception as e:
        logger.warning(f"Failed to load lambda coefficients from {CONFIG_PATH_RAW}: {e}")
        logger.warning("Using default values")
        return 1e-9, 1e-7, 1e-8, 1e-2

# Load lambda coefficients
LAMBDA_REV, LAMBDA_DEV, LAMBDA_DEG, LAMBDA_RISK = load_lambda_coefficients()

# Lambda tuning: combined_score in single-digit range; LAMBDA_DEV increased to emphasize deviation.
# Typical: revenue ~3-4e9, deviation ~2-9e7, deg ~80-100, risk ~60-100. Values loaded from config.

BAD_SCORE = -1e18


def build_program_summary(episode_outputs):
    """
    Build summary from multi-scenario episode outputs: per-node temporal/spatial + global analysis + coupling.
    episode_outputs: list of dict, each element is return value of run_joint_episode.
    Returns:
      node: { node_id: { temporal: {...}, spatial: {pv:[cap,net], ...} } }
      global_analysis: { temporal: {...}, spatial: {...} }
      coupling: homogeneous vs complementary
        homogeneous: similar roles/patterns, unified strategy
        complementary: pairs that can balance each other
    """
    if not episode_outputs:
        return {"node": {}, "global_analysis": {}, "coupling": {}}
    first = episode_outputs[0]
    nodes = list(first.get("node_prices", {}).keys())
    total_elec = np.array(first["total_elec"], dtype=float)
    total_bid = np.array(first["total_bid"], dtype=float)
    T = len(total_elec)
    deviation_per_hour = total_elec - total_bid

    n_high = max(1, T // 3)
    n_low = max(1, T // 3)
    n_peak = max(1, T // 3)
    n_min = max(1, T // 3)

    # Per node: temporal + spatial [capacity, net_volume]
    node_summary = {}
    for n in nodes:
        price_list = first["node_prices"].get(n, [0.0] * T)
        price_arr = np.array(price_list, dtype=float)
        high_ix = np.argsort(-price_arr)[:n_high]
        low_ix = np.argsort(price_arr)[:n_low]
        peak_ix = np.argsort(-deviation_per_hour)[:n_peak]
        min_ix = np.argsort(deviation_per_hour)[:n_min]

        cap = first["node_device_capacity"].get(n, {})
        net = {k: 0.0 for k in ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]}
        for out in episode_outputs:
            ndn = out.get("node_device_net", {}).get(n, {})
            for k in net:
                net[k] += ndn.get(k, 0.0)

        node_summary[str(n)] = {
            "temporal": {
                "high_price_hours": high_ix.tolist(),
                "low_price_hours": low_ix.tolist(),
                "deviation_peak_hours": peak_ix.tolist(),
                "deviation_minimum_hours": min_ix.tolist(),
            },
            "spatial": {
                "pv": [float(cap.get("pv", 0.0)), float(net.get("pv", 0.0))],
                "wind": [float(cap.get("wind", 0.0)), float(net.get("wind", 0.0))],
                "storage": [float(cap.get("storage", 0.0)), float(net.get("storage", 0.0))],
                "vehicle": [float(cap.get("vehicle", 0.0)), float(net.get("vehicle", 0.0))],
                "AC": [float(cap.get("AC", 0.0)), float(net.get("AC", 0.0))],
                "washing_machine": [float(cap.get("washing_machine", 0.0)), float(net.get("washing_machine", 0.0))],
            },
        }

    # Global temporal/spatial: system-level temporal (first scenario total_elec/total_bid/avg price) + spatial (device capacity and net across nodes)
    avg_price = np.zeros(T)
    for n in nodes:
        avg_price += np.array(first["node_prices"].get(n, [0.0] * T), dtype=float)
    if nodes:
        avg_price /= len(nodes)
    global_high = np.argsort(-avg_price)[:n_high].tolist()
    global_low = np.argsort(avg_price)[:n_low].tolist()
    global_peak = np.argsort(-deviation_per_hour)[:n_peak].tolist()
    global_min = np.argsort(deviation_per_hour)[:n_min].tolist()

    global_cap = {k: 0.0 for k in ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]}
    global_net = {k: 0.0 for k in global_cap}
    for n in nodes:
        c = first["node_device_capacity"].get(n, {})
        for k in global_cap:
            global_cap[k] += float(c.get(k, 0.0))
    for out in episode_outputs:
        for n in nodes:
            ndn = out.get("node_device_net", {}).get(n, {})
            for k in global_net:
                global_net[k] += ndn.get(k, 0.0)
    global_spatial = {
        k: [global_cap[k], global_net[k]] for k in global_cap
    }

    global_analysis = {
        "temporal": {
            "high_price_hours": global_high,
            "low_price_hours": global_low,
            "deviation_peak_hours": global_peak,
            "deviation_minimum_hours": global_min,
        },
        "spatial": global_spatial,
    }

    # Coupling: homogeneous (similar, unified strategy) vs complementary (can balance)
    # Homogeneous - temporal: same role within group (high/low price, deviation peak/valley)
    homogeneous_temporal_groups = {
        "high_price_hours": global_high,
        "low_price_hours": global_low,
        "deviation_peak_hours": global_peak,
        "deviation_minimum_hours": global_min,
    }
    # Complementary - temporal: two groups can balance (high↔low price arbitrage, peak↔valley tracking)
    complementary_temporal_pairs = [
        {"roles": ["high_price_hours", "low_price_hours"], "hours": [global_high, global_low]},
        {"roles": ["deviation_peak_hours", "deviation_minimum_hours"], "hours": [global_peak, global_min]},
    ]
    # Homogeneous - spatial: similar capacity/net pattern across nodes (same dominant device and net direction)
    device_keys = ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]
    node_dominant_device = {}
    node_total_net_sign = {}
    for n in nodes:
        s = node_summary[n]["spatial"]
        caps = [s.get(k, [0, 0])[0] for k in device_keys]
        nets = [s.get(k, [0, 0])[1] for k in device_keys]
        if caps:
            node_dominant_device[n] = device_keys[np.argmax(caps)]
        else:
            node_dominant_device[n] = "pv"
        node_total_net_sign[n] = 1 if sum(nets) >= 0 else -1
    # Group by (dominant device, net sign) -> same group is homogeneous
    homogeneous_groups_key = defaultdict(list)
    for n in nodes:
        key = (node_dominant_device[n], node_total_net_sign[n])
        homogeneous_groups_key[key].append(n)
    homogeneous_spatial_groups = [sorted(g) for g in homogeneous_groups_key.values() if g]
    # Complementary - spatial: node pairs with opposite net direction (export/import balance) or different dominant devices
    complementary_spatial_pairs = []
    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            ni, nj = node_list[i], node_list[j]
            if node_total_net_sign[ni] != node_total_net_sign[nj]:
                complementary_spatial_pairs.append([ni, nj])
            elif node_dominant_device[ni] != node_dominant_device[nj]:
                # Different dominant resources can be complementary (e.g. PV node and storage node)
                complementary_spatial_pairs.append([ni, nj])

    coupling = {
        "homogeneous": {
            "description": "Homogeneous: similar roles/patterns within group, unified strategy",
            "temporal": {
                "description": "Homogeneous temporal groups (same role within group)",
                "groups": homogeneous_temporal_groups,
            },
            "spatial": {
                "description": "Homogeneous node groups (same dominant device and net direction)",
                "node_groups": homogeneous_spatial_groups,
            },
        },
        "complementary": {
            "description": "Complementary: pairs/groups can balance (arbitrage, tracking, or resource coordination)",
            "temporal": {
                "description": "Complementary temporal pairs (high↔low price, deviation peak↔valley)",
                "pairs": complementary_temporal_pairs,
            },
            "spatial": {
                "description": "Complementary node pairs (opposite net direction or different dominant resource)",
                "node_pairs": complementary_spatial_pairs,
            },
        },
    }

    return {
        "node": node_summary,
        "global_analysis": global_analysis,
        "coupling": coupling,
    }


def build_device_attribution(episode_outputs, tot_rev, tot_dev, tot_deg, tot_risk):
    """
    Compute per-device attribution of revenue, deviation, degradation from multi-scenario episode outputs.
    Used for prompt inspiration comparison so LLM can compare program scheduling.

    episode_outputs: list of dict, each is return of run_joint_episode (with node_device_net).
    Returns: term_attribution: {"new": {metric: {device_type: value}}} for metric in revenue/degradation/deviation, device_type in pv, wind, storage, vehicle, AC, wash.
    """
    device_keys = ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]
    # Aggregate total net per device (across nodes and scenarios)
    device_net = {k: 0.0 for k in device_keys}
    for out in episode_outputs:
        ndn_all = out.get("node_device_net", {})
        for nid, ndn in ndn_all.items():
            for k in device_keys:
                device_net[k] += float(ndn.get(k, 0.0))
    
    # Map washing_machine -> wash (consistent with sampler device_types)
    device_net["wash"] = device_net.pop("washing_machine", 0.0)
    device_keys = ["pv", "wind", "storage", "vehicle", "AC", "wash"]

    # Allocate metrics by |net| proportion (simplified attribution: larger contribution gets more)
    total_abs = sum(abs(device_net[k]) for k in device_keys)
    if total_abs < 1e-12:
        # Uniform when no contribution
        n_dev = len(device_keys)
        rev_attrib = {k: tot_rev / n_dev for k in device_keys}
        dev_attrib = {k: tot_dev / n_dev for k in device_keys}
        deg_attrib = {k: tot_deg / n_dev for k in device_keys}
    else:
        rev_attrib = {k: tot_rev * abs(device_net[k]) / total_abs for k in device_keys}
        dev_attrib = {k: tot_dev * abs(device_net[k]) / total_abs for k in device_keys}
        deg_attrib = {k: tot_deg * abs(device_net[k]) / total_abs for k in device_keys}
    
    return {
        "new": {
            "revenue": rev_attrib,
            "degradation": deg_attrib,
            "deviation": dev_attrib,
        }
    }


def _is_profile_reflection_enabled():
    """Read from config_joint_raw.yaml whether to call LLM for reasoning reflection from profile."""
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_PATH_RAW)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config.get("reflection", {}).get("enable_profile_reflection", True)  # Default True
    except Exception:
        return False


def format_profile_for_reflection(program_summary):
    """
    Format node + global_analysis + coupling into short text for LLM reasoning reflection.
    Injected into LLM prompt; reflection is stored in memory.
    """
    if not program_summary:
        return ""
    node = program_summary.get("node", {})
    global_analysis = program_summary.get("global_analysis", {})
    coupling = program_summary.get("coupling", {})
    lines = ["## Node & Global Profile (for reflection)", ""]
    # Global
    gt = global_analysis.get("temporal", {})
    gs = global_analysis.get("spatial", {})
    lines.append("### Global")
    lines.append(f"Temporal: high_price_hours={gt.get('high_price_hours', [])}, low_price_hours={gt.get('low_price_hours', [])}, deviation_peak={gt.get('deviation_peak_hours', [])}, deviation_min={gt.get('deviation_minimum_hours', [])}.")
    lines.append("Spatial (capacity, net): " + ", ".join(f"{k}={gs.get(k, [0,0])}" for k in ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]))
    lines.append("")
    # Per-node summary
    for nid, data in node.items():
        t = data.get("temporal", {})
        s = data.get("spatial", {})
        lines.append(f"### Node {nid}")
        lines.append(f"  Temporal: high={t.get('high_price_hours', [])}, low={t.get('low_price_hours', [])}, dev_peak={t.get('deviation_peak_hours', [])}, dev_min={t.get('deviation_minimum_hours', [])}.")
        lines.append("  Spatial: " + ", ".join(f"{k}={s.get(k, [0,0])}" for k in ["pv", "wind", "storage", "vehicle", "AC", "washing_machine"]))
    lines.append("")
    # Coupling: homogeneous vs complementary
    hom = coupling.get("homogeneous", {})
    comp = coupling.get("complementary", {})
    lines.append("### Coupling")
    lines.append("Homogeneous (temporal groups): " + json.dumps(hom.get("temporal", {}).get("groups", {}), ensure_ascii=False))
    lines.append("Homogeneous (spatial node_groups): " + json.dumps(hom.get("spatial", {}).get("node_groups", []), ensure_ascii=False))
    lines.append("Complementary (temporal pairs): " + json.dumps(comp.get("temporal", {}).get("pairs", []), ensure_ascii=False))
    lines.append("Complementary (spatial node_pairs): " + json.dumps(comp.get("spatial", {}).get("node_pairs", []), ensure_ascii=False))
    return "\n".join(lines)


def generate_reflection_via_llm(profile_text, metrics):
    """
    Call LLM to generate reasoning reflection from node and global profile; store in memory.
    LLM config read from config_joint_raw.yaml llm section. Returns 1–3 paragraphs; empty on failure or no LLM.
    """
    if not profile_text or not profile_text.strip():
        return ""
    config_path = os.path.join(os.path.dirname(__file__), CONFIG_PATH_RAW)
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.debug(f"Reflection LLM: config not found {e}")
        return ""
    llm_cfg = config.get("llm", {})
    api_base = llm_cfg.get("api_base") or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    models = llm_cfg.get("models", [])
    model_name = models[0].get("name", "gpt-4o-mini") if models else "gpt-4o-mini"
    if not api_key:
        logger.debug("Reflection LLM: no api_key, skip")
        return ""
    # Azure OpenAI requires api-version in URL, else 404
    is_azure = "openai.azure.com" in (api_base or "")
    # Reasoning models (o1/o3/o4) use max_completion_tokens, not max_tokens/temperature
    _reasoning_prefixes = ("o1-", "o1", "o3-", "o3", "o4-", "o4", "gpt-5-", "gpt-5", "gpt-oss-120b", "gpt-oss-20b")
    model_lower = (model_name or "").lower()
    is_reasoning_model = any(model_lower.startswith(p) for p in _reasoning_prefixes)
    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        score_info = f"combined_score={metrics.get('combined_score', 0):.4f}, revenue={metrics.get('revenue', 0):.4f}, deviation={metrics.get('deviation', 0):.4f}."
        user_content = (
            "Below is the **temporal/spatial node and global profile** for a VPP multi-node scheduling evaluation "
            "(including per-node temporal/spatial and coupling). "
            "Write a **reasoning reflection** (1–3 paragraphs):\n"
            "1. Note strengths/weaknesses of the current strategy in time and space (high/low price utilization, deviation peak/valley tracking).\n"
            "2. Explain coupling with deviation and price (homogeneous segments/nodes → unified strategy; complementary → balance).\n"
            "3. Suggest improvement directions (alpha_score or device_allocation).\n"
            "Be concise and actionable. Answer in English.\n\n"
            f"Current metrics: {score_info}\n\n"
            "---\n\n"
            f"{profile_text}"
        )
        messages = [
            {"role": "system", "content": "You are a VPP scheduling and evolution analyst. Write a short reasoning reflection from the temporal/spatial profile for later code evolution."},
            {"role": "user", "content": user_content},
        ]
        if is_reasoning_model:
            create_kw = {"model": model_name, "messages": messages, "max_completion_tokens": 1024}
        else:
            create_kw = {"model": model_name, "messages": messages, "max_tokens": 1024, "temperature": 0.3}
        if is_azure:
            create_kw["extra_query"] = {"api-version": "2025-01-01-preview"}
        resp = client.chat.completions.create(**create_kw)
        text = (resp.choices[0].message.content or "").strip()
        return text if text else ""
    except Exception as e:
        logger.warning(f"Reflection LLM failed: {e}")
        return ""


def load_program(path):
    spec = importlib.util.spec_from_file_location("prog", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def extract_parameters(prog):
    """Extract evolvable parameter values from program if present."""
    params = {}
    # Try common parameter names
    param_names = ['W_P', 'W_PV', 'W_WIND', 'W_FLEX', 'TEMP', 'W_DEG', 'W_TRACK', 'W_RISK']
    
    for name in param_names:
        if hasattr(prog, name):
            params[name] = getattr(prog, name)
        else:
            params[name] = None
    
    # Also try attributes starting with uppercase (likely parameters)
    for attr_name in dir(prog):
        if attr_name.startswith('W_') or attr_name.startswith('TEMP') or attr_name.startswith('ALPHA'):
            if not attr_name.startswith('_') and not callable(getattr(prog, attr_name, None)):
                try:
                    params[attr_name] = getattr(prog, attr_name)
                except:
                    pass
    
    return params

def print_parameters(params, program_path=None):
    """Print parameter values."""
    if program_path:
        logger.info(f"\n[Parameters] Program: {program_path.split('/')[-1]}")
    else:
        logger.info(f"\n[Parameters]")

    logger.info("  " + "=" * 60)

    # Print all found parameters
    found_params = {k: v for k, v in params.items() if v is not None}
    if found_params:
        logger.info("  Found parameters:")
        for name, val in sorted(found_params.items()):
            if isinstance(val, (int, float)):
                logger.info(f"    {name:15s} = {val:.6f}")
            else:
                logger.info(f"    {name:15s} = {val}")
    else:
        logger.info("  No extractable parameters found (code may use other means)")
    
    logger.info("  " + "=" * 60)


def evaluate(program_path):
    """
    For evolution only: uses predicted scenarios.
    """
    prog = load_program(program_path)
    params = extract_parameters(prog)

    tot_rev = tot_dev = tot_deg = tot_risk = 0.0
    episode_outputs = []

    for sc in SCENARIOS:
        out = run_joint_episode(prog, sc)

        # Guard against NaN / inf
        for k in ["revenue", "deviation", "deg", "risk"]:
            if k not in out or not math.isfinite(out[k]):
                raise ValueError(f"Invalid {k}")

        tot_rev  += out["revenue"]
        tot_dev  += out["deviation"]
        tot_deg  += out["deg"]
        tot_risk += out["risk"]
        episode_outputs.append(out)

    # Compute combined_score (lambda tuned for single-digit range)
    combined_score = (
        float(LAMBDA_REV) * tot_rev
        - float(LAMBDA_DEV)  * tot_dev
        - float(LAMBDA_DEG)  * tot_deg
        - float(LAMBDA_RISK) * tot_risk
    )

    # Per-node temporal/spatial + global analysis + coupling
    program_summary = build_program_summary(episode_outputs)

    # Per-device attribution for revenue, deviation, degradation (for prompt inspiration comparison)
    term_attribution = build_device_attribution(episode_outputs, tot_rev, tot_dev, tot_deg, tot_risk)

    result = {
        "combined_score": combined_score,
        "revenue": tot_rev,
        "deviation": tot_dev,
        "deg": tot_deg,
        "risk": tot_risk,
        "program_summary": program_summary,
        "term_attribution": term_attribution,
    }

    # Optional: LLM generates reasoning reflection from node + global profile, stored in memory (config.reflection.enable_profile_reflection)
    if _is_profile_reflection_enabled():
        profile_text = format_profile_for_reflection(program_summary)
        reflection = generate_reflection_via_llm(profile_text, result)
        has_reflection = bool(reflection and reflection.strip())
        result["memory"] = {
            "reflection": reflection,
            "profile_snapshot": profile_text[:3000] if profile_text else "",  # Truncate for storage
            "has_reflection": has_reflection,
        }
    else:
        result["memory"] = {"reflection": "", "profile_snapshot": "", "has_reflection": False}

    # Print metrics to console
    print(f"\n[Evaluation result]")
    print(f"  Revenue: {tot_rev:.3f}")
    print(f"  Deviation: {tot_dev:.3f}")
    print(f"  Degradation: {tot_deg:.3f}")
    print(f"  Risk: {tot_risk:.3f}")
    print(f"  Combined Score: {combined_score:.3f}")
    print()
    
    logger.info(f"  Revenue: {tot_rev:.3f}")
    logger.info(f"  Deviation: {tot_dev:.3f}")
    logger.info(f"  Degradation: {tot_deg:.3f}")
    logger.info(f"  Risk: {tot_risk:.3f}")
    logger.info(f"  Combined Score: {combined_score:.3f}")

    logger.info(f'Evaluation Result: {result}')

    # Record parameters (for MAP-Elites / logging)
    for k, v in params.items():
        result[f"param_{k}"] = v

    # Strong penalty if no parameters defined
    if len(params) == 0:
        result["combined_score"] -= 1e6

    return result


def evaluate_real(program_path):
    """
    Final test on real scenarios (similar to execute_actions in run_genetic.ipynb).
    Returns performance metrics on real scenarios.
    """
    prog = load_program(program_path)

    # Extract and print parameters
    params = extract_parameters(prog)
    print_parameters(params, program_path)
    logger.info("\n[Real scenario test] Evaluating with real PV/Wind data...")

    tot_rev_real = tot_dev_real = tot_deg_real = tot_risk_real = 0.0
    episode_outputs_real = []

    # Run on real scenarios
    for sc_real in SCENARIOS_REAL:
        out = run_joint_episode(prog, sc_real)
        tot_rev_real += out["revenue"]
        tot_dev_real += out["deviation"]
        tot_deg_real += out["deg"]
        tot_risk_real += out["risk"]
        episode_outputs_real.append(out)

    # Compute combined_score (lambda tuned for single-digit range)
    combined_score_real = float(LAMBDA_REV) * tot_rev_real - float(LAMBDA_DEV) * tot_dev_real - float(LAMBDA_DEG) * tot_deg_real - float(LAMBDA_RISK) * tot_risk_real

    # Temporal/spatial summary on real scenarios (node + global_analysis + coupling)
    program_summary_real = build_program_summary(episode_outputs_real)

    # Print real scenario test results to console
    print(f"\n[Real scenario test result]")
    print(f"  Revenue: {tot_rev_real:.3f}")
    print(f"  Deviation: {tot_dev_real:.3f}")
    print(f"  Degradation: {tot_deg_real:.3f}")
    print(f"  Risk: {tot_risk_real:.3f}")
    print(f"  Combined Score: {combined_score_real:.3f}")
    print()
    
    logger.info(f"\n[Real scenario test result]")
    logger.info(f"  Revenue: {tot_rev_real:.3f}")
    logger.info(f"  Deviation: {tot_dev_real:.3f}")
    logger.info(f"  Degradation: {tot_deg_real:.3f}")
    logger.info(f"  Risk: {tot_risk_real:.3f}")
    logger.info(f"  Combined Score: {combined_score_real:.3f}")

    # 返回真实场景的结果
    result = {
        "combined_score": combined_score_real,  # 注意：这里返回真实场景的分数
        "revenue": tot_rev_real,
        "deviation": tot_dev_real,
        "deg": tot_deg_real,
        "risk": tot_risk_real,
        "real_revenue": tot_rev_real,
        "real_deviation": tot_dev_real,
        "real_deg": tot_deg_real,
        "real_risk": tot_risk_real,
        "program_summary": program_summary_real,
    }
    
    # Add parameter values to result
    for key, value in params.items():
        if value is not None:
            result[f"param_{key}"] = value
    
    return result
