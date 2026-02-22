import math
import numpy as np
import yaml
import os
import inspect

# Project root: parent of VPPEvolve_released; device configs live at <project_root>/config/device/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEVICE_CONFIG_BASE = os.path.join(_PROJECT_ROOT, "config", "device")

temperature = 1.0

# ========== Normalization helpers ==========
def normalize_value(value, min_val, max_val, range_val):
    """Normalize value to [0, 1]."""
    if range_val == 0:
        return 0.5  # Mid value when range is zero
    normalized = (value - min_val) / range_val
    return np.clip(normalized, 0.0, 1.0)

def normalize_power(value, power_max):
    """Normalize power-related values to [0, 1] using unified power_max (pv, wind, bq, wash, etc.)."""
    if power_max == 0:
        return 0.0
    normalized = value / power_max
    return np.clip(normalized, 0.0, 1.0)

def get_normalization_params():
    """Load normalization params from data_stub."""
    try:
        from data_stub import NORMALIZATION_PARAMS
        return NORMALIZATION_PARAMS
    except ImportError:
        return {
            "price": {"min": 0.0, "max": 1.0, "range": 1.0},
            "delta_p": {"min": -100.0, "max": 100.0, "range": 200.0},
            "track_error": {"min": 0.0, "max": 1000.0, "range": 1000.0},
            "power_max": 10000.0,
        }

# Device cache for loading device configs
_DEVICE_CACHE = {}

def load_device_yaml(device_type, device_id, base_dir):
    """Load device configuration from YAML file (consistent with run_genetic.ipynb)"""
    key = (device_type, int(device_id))
    if key in _DEVICE_CACHE:
        return _DEVICE_CACHE[key]
    path = f"{base_dir}/{device_type}/simulator_{int(device_id)}.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    dev = data[device_type]
    _DEVICE_CACHE[key] = dev
    return dev

def ensure_2d_curve(curve, T, n_devices):
    """Ensure curve is 2D array (consistent with run_genetic.ipynb)"""
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 1:
        if n_devices == 0:
            return np.zeros((0, T))
        if n_devices == 1:
            return arr[None, :]
        return np.tile(arr[None, :] / n_devices, (n_devices, 1))
    return arr

# add temperature to softmax
def softmax(scores, temperature):
    m = max(scores.values())
    exps = {k: math.exp(v - m) / temperature for k, v in scores.items()}
    s = sum(exps.values())
    return {k: exps[k] / s for k in exps}

def dispatch_node_devices(program, node_id, t, bid_nt, node_ctx):
    """
    Simple dispatch using device_allocation. Revenue uses absolute PV/Wind values.
    """
    devices = node_ctx["devices"]
    price = node_ctx["price"][t]  # Absolute price value
    pv_abs = node_ctx.get("pv", [0.0] * 24)[t] if isinstance(node_ctx.get("pv"), list) else 0.0  # Absolute PV generation
    wind_abs = node_ctx.get("wind", [0.0] * 24)[t] if isinstance(node_ctx.get("wind"), list) else 0.0  # Absolute Wind generation

    # state (stub)
    soc = node_ctx["soc"]
    track_error = node_ctx["track_error"]

    # Get normalization params
    norm_params = get_normalization_params()

    # Normalize inputs for device_allocation
    price_norm = normalize_value(
        price,
        norm_params["price"]["min"],
        norm_params["price"]["max"],
        norm_params["price"]["range"]
    )
    # Normalize all power-related values using unified power_max
    power_max = norm_params.get("power_max", 10000.0)
    pv_norm = normalize_power(pv_abs, power_max)
    wind_norm = normalize_power(wind_abs, power_max)
    bid_quantity_norm = normalize_power(bid_nt, power_max)

    # Compute flex values (load actual params from device config; devices list has no capacity/power_max, use device_dict)
    device_dict = node_ctx.get("device_dict", {})

    flex_storage_val = 0.0
    flex_vehicle_val = 0.0
    flex_AC_val = 0.0
    flex_wash_count = 0  # int: unit count

    # Storage: load capacity from device config
    storage_ids = device_dict.get("storage_id", [])
    for storage_id in storage_ids:
        try:
            dev_config = load_device_yaml("storage", storage_id, BASE_DIR)
            capacity = float(dev_config.get("capacity", 0.0))
            flex_storage_val += capacity
        except:
            pass
    
    # Vehicle: load capacity from device config
    vehicle_ids = device_dict.get("vehicle_id", [])
    for vehicle_id in vehicle_ids:
        try:
            dev_config = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
            capacity = float(dev_config.get("capacity", 0.0))
            flex_vehicle_val += capacity
        except:
            pass
    
    # AC: load power_max from device config
    ac_ids = device_dict.get("AC_id", [])
    for ac_id in ac_ids:
        try:
            dev_config = load_device_yaml("AC", ac_id, BASE_DIR)
            power_max_ac = float(dev_config.get("power_max", 0.0))
            flex_AC_val += power_max_ac
        except:
            pass
    
    # Wash: use device count directly
    wash_ids = device_dict.get("wash_id", [])
    flex_wash_count = len(wash_ids)  # int: total units

    # Normalize flex values (using power_max)
    flex_storage_norm = normalize_power(flex_storage_val, power_max)
    flex_vehicle_norm = normalize_power(flex_vehicle_val, power_max)
    flex_AC_norm = normalize_power(flex_AC_val, power_max)

    # Get current values from node_ctx (0 if missing). current_* in kW except current_wash (int, units on)
    current_storage_val = node_ctx.get("current_storage", 0.0)
    current_vehicle_val = node_ctx.get("current_vehicle", 0.0)
    current_AC_val = node_ctx.get("current_AC", 0.0)
    current_wash_count = node_ctx.get("current_wash", 0)  # int: units on

    # Normalize current values (using power_max)
    current_storage_norm = normalize_power(current_storage_val, power_max)
    current_vehicle_norm = normalize_power(current_vehicle_val, power_max)
    current_AC_norm = normalize_power(current_AC_val, power_max)
    
    # Call device_allocation for device ratios. Check signature for backward compat (bid_quantity vs bq).
    try:
        sig = inspect.signature(program.device_allocation)
        accepts_bq = 'bq' in sig.parameters
        accepts_bid_quantity = 'bid_quantity' in sig.parameters
    except:
        accepts_bq = True
        accepts_bid_quantity = False

    # Build call kwargs
    call_kwargs = {
        "p": price_norm,
        "pv": pv_norm,
        "wind": wind_norm,
        "flex_storage": flex_storage_norm,
        "current_storage": current_storage_norm,
        "flex_vehicle": flex_vehicle_norm,
        "current_vehicle": current_vehicle_norm,
        "flex_AC": flex_AC_norm,
        "current_AC": current_AC_norm,
        "flex_wash": flex_wash_count,  # int: total units
        "current_wash": current_wash_count,  # int: units on
        "t": t,
        "n": node_id,
        "ctx": {"T": 24, "power_max": power_max}
    }

    # Prefer bq, else bid_quantity (backward compat)
    if accepts_bq:
        call_kwargs["bq"] = bid_quantity_norm
    elif accepts_bid_quantity:
        call_kwargs["bid_quantity"] = bid_quantity_norm
    
    alloc = program.device_allocation(**call_kwargs)

    # Compute actual power from allocation ratios
    actual = 0.0
    deg_cost = 0.0
    risk_cost = 0.0
    storage_power_used = 0.0
    vehicle_power_used = 0.0

    # Per-device dispatch (for logging)
    device_dispatch_info = {
        "storage": {},  # {device_id: power_kWh}
        "vehicle": {},  # {device_id: power_kWh}
        "AC": {},       # {device_id: power_kWh}
        "wash": {}      # {device_id: power_kWh}
    }
    
    # PV: use grid connection ratio from device_allocation
    pv_ratio = alloc.get("pv_ratio", 1.0)
    pv_ratio = np.clip(pv_ratio, 0.0, 1.0)
    actual += pv_ratio * pv_abs

    # Wind: use grid connection ratio
    wind_ratio = alloc.get("wind_ratio", 1.0)
    wind_ratio = np.clip(wind_ratio, 0.0, 1.0)
    actual += wind_ratio * wind_abs

    # Storage: use charge/discharge ratio [-1, 1] (positive=discharge, negative=charge)
    storage_ratio = alloc.get("storage_ratio", 0.0)
    storage_ratio = np.clip(storage_ratio, -1.0, 1.0)
    storage_ids = device_dict.get("storage_id", [])
    current_storage = node_ctx.get("current_storage", 0.0)

    # Greedy: prefer devices with lower degradation cost
    # 1. Collect device info
    storage_devices = []
    total_capacity = 0.0
    for storage_id in storage_ids:
        try:
            dev_config = load_device_yaml("storage", storage_id, BASE_DIR)
            max_power = float(dev_config.get("max_power", 0.0))
            capacity = float(dev_config.get("capacity", 0.0))
            # Degradation cost proxy: max_power^2 (deg_cost = 0.01 * power^2); can read degradation_cost from config
            degradation_coeff = dev_config.get("degradation_cost", 0.01)
            degradation_cost_proxy = degradation_coeff * (max_power ** 2)

            storage_devices.append({
                "id": storage_id,
                "max_power": max_power,
                "capacity": capacity,
                "degradation_cost": degradation_cost_proxy,
                "dev_config": dev_config
            })
            total_capacity += capacity
        except:
            pass
    
    # 2. Sort by degradation cost (low first)
    storage_devices.sort(key=lambda x: x["degradation_cost"])

    # 3. Total power needed
    total_max_power = sum(d["max_power"] for d in storage_devices)
    total_power_needed = storage_ratio * total_max_power

    # 4. Greedy: fill low-cost devices first
    power_remaining = abs(total_power_needed)
    power_direction = 1.0 if total_power_needed >= 0 else -1.0  # positive=discharge, negative=charge

    soc_gain = 0.01
    for dev in devices:
        if dev.get("type") == "storage":
            soc_gain = dev.get("soc_gain", 0.01)
            break

    for device in storage_devices:
        if power_remaining <= 1e-6:
            device_power = 0.0
        else:
            capacity_ratio = device["capacity"] / total_capacity if total_capacity > 0 else 1.0 / len(storage_devices)
            device_energy = current_storage * capacity_ratio

            if power_direction > 0:  # discharge
                max_available_power = min(device["max_power"], device_energy)
            else:  # charge
                remaining_capacity = device["capacity"] - device_energy
                max_available_power = min(device["max_power"], remaining_capacity)

            device_power = min(power_remaining, max_available_power) * power_direction
            power_remaining -= abs(device_power)

        storage_power = device_power
        actual += storage_power
        storage_power_used += storage_power

        if abs(storage_power) > 0:
            soc_change = -storage_power * soc_gain
            soc = np.clip(soc + soc_change, 0.0, 1.0)
        
        deg_cost += 0.01 * (storage_power ** 2)

    risk_cost += max(0.0, soc - 0.95) ** 2 + max(0.0, 0.05 - soc) ** 2  # once per node

    # Vehicle: use charge/discharge ratio
    vehicle_ratio = alloc.get("vehicle_ratio", 0.0)
    vehicle_ratio = np.clip(vehicle_ratio, -1.0, 1.0)
    vehicle_ids = device_dict.get("vehicle_id", [])
    current_vehicle = node_ctx.get("current_vehicle", 0.0)

    # Greedy: prefer lower degradation cost
    # 1. Collect device info
    vehicle_devices = []
    total_capacity = 0.0
    for vehicle_id in vehicle_ids:
        try:
            dev_config = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
            max_power = float(dev_config.get("max_power", 0.0))
            capacity = float(dev_config.get("capacity", 0.0))
            degradation_coeff = dev_config.get("degradation_cost", 0.01)
            degradation_cost_proxy = degradation_coeff * (max_power ** 2)

            vehicle_devices.append({
                "id": vehicle_id,
                "max_power": max_power,
                "capacity": capacity,
                "degradation_cost": degradation_cost_proxy,
                "dev_config": dev_config
            })
            total_capacity += capacity
        except:
            pass
    
    # 2. Sort by degradation cost (low first)
    vehicle_devices.sort(key=lambda x: x["degradation_cost"])

    # 3. Total power needed
    total_max_power = sum(d["max_power"] for d in vehicle_devices)
    total_power_needed = vehicle_ratio * total_max_power

    # 4. Greedy: fill low-cost devices first
    power_remaining = abs(total_power_needed)
    power_direction = 1.0 if total_power_needed >= 0 else -1.0

    soc_gain = 0.01
    for dev in devices:
        if dev.get("type") == "vehicle":
            soc_gain = dev.get("soc_gain", 0.01)
            break

    for device in vehicle_devices:
        if power_remaining <= 1e-6:
            device_power = 0.0
        else:
            capacity_ratio = device["capacity"] / total_capacity if total_capacity > 0 else 1.0 / len(vehicle_devices)
            device_energy = current_vehicle * capacity_ratio

            if power_direction > 0:  # discharge
                max_available_power = min(device["max_power"], device_energy)
            else:  # charge
                remaining_capacity = device["capacity"] - device_energy
                max_available_power = min(device["max_power"], remaining_capacity)

            device_power = min(power_remaining, max_available_power) * power_direction
            power_remaining -= abs(device_power)

        vehicle_power = device_power
        actual += vehicle_power
        vehicle_power_used += vehicle_power

        if abs(vehicle_power) > 0:
            soc_change = -vehicle_power * soc_gain
            soc = np.clip(soc + soc_change, 0.0, 1.0)
        
        deg_cost += 0.01 * (vehicle_power ** 2)

    risk_cost += max(0.0, soc - 0.95) ** 2 + max(0.0, 0.05 - soc) ** 2  # once per node

    # AC: use usage ratio [0, 1]. Degradation: 0.01 * (P_ac_total)^2 per node to avoid deg blow-up
    ac_ratio = alloc.get("ac_ratio", 0.0)
    ac_ratio = np.clip(ac_ratio, 0.0, 1.0)
    ac_ids = device_dict.get("AC_id", [])
    ac_power_total = 0.0
    for ac_id in ac_ids:
        try:
            dev_config = load_device_yaml("AC", ac_id, BASE_DIR)
            power_max = float(dev_config.get("power_max", 0.0))
            ac_power = ac_ratio * power_max
            actual -= ac_power  # consumption
            ac_power_total += ac_power
        except:
            pass
    if ac_power_total != 0.0:
        deg_cost += 0.01 * (ac_power_total ** 2)
    
    # Wash: use on count
    wash_on_number = alloc.get("wash_on_number", 0)
    wash_ids = device_dict.get("wash_id", [])
    wash_on_number = max(0, min(int(wash_on_number), len(wash_ids)))
    wash_power_total = 0.0
    for idx, wash_id in enumerate(wash_ids):
        if idx < wash_on_number:
            try:
                dev_config = load_device_yaml("wash", wash_id, DEVICE_CONFIG_BASE)
                wash_power = float(dev_config.get("rate_power", 0.0))
                actual -= wash_power  # consumption
                wash_power_total += wash_power
            except:
                pass

    # Per-device net for this step (pv/wind=grid injection +; storage/vehicle=discharge +/charge -; AC/wash=consumption +)
    device_net = {
        "pv": pv_ratio * pv_abs,
        "wind": wind_ratio * wind_abs,
        "storage": storage_power_used,
        "vehicle": vehicle_power_used,
        "AC": ac_power_total,
        "washing_machine": wash_power_total,
    }
    
    # Device operation penalty (aligned with VPP_dispatching). AC/storage/vehicle/wash penalties handled in run_joint_episode.
    device_penalty = 0.0

    # Revenue: price * elec (penalty computed separately)
    elec = actual  # This will be accumulated across nodes (absolute value)
    revenue = price * elec  # Economic revenue (aligned with VPP_dispatching)
    penalty = device_penalty  # Device operation penalty (will be accumulated in run_joint_episode)
    
    # Update device usage in node_ctx (storage_power_used, vehicle_power_used already accumulated above)
    ac_power_used = 0.0
    wash_count_used = 0

    # Extract from alloc
    ac_ratio = alloc.get("ac_ratio", 0.0)
    wash_on_number = alloc.get("wash_on_number", 0)
    
    ac_ids = device_dict.get("AC_id", [])
    for ac_id in ac_ids:
        try:
            dev_config = load_device_yaml("AC", ac_id, BASE_DIR)
            power_max = float(dev_config.get("power_max", 0.0))
            ac_power_used += ac_ratio * power_max
        except:
            pass

    wash_count_used = int(wash_on_number)

    # Update node_ctx device state. Storage: charge increases, discharge decreases (storage_power_used < 0 = charge).
    current_storage = node_ctx.get("current_storage", 0.0)
    new_storage = current_storage - storage_power_used  # charge: storage_power_used < 0 => increase
    # Clamp to [0, flex_storage]
    flex_storage_val = 0.0
    for storage_id in storage_ids:
        try:
            dev_config = load_device_yaml("storage", storage_id, BASE_DIR)
            flex_storage_val += float(dev_config.get("capacity", 0.0))
        except:
            pass
    node_ctx["current_storage"] = max(0.0, min(flex_storage_val, new_storage))

    # Vehicle: same as storage
    current_vehicle = node_ctx.get("current_vehicle", 0.0)
    new_vehicle = current_vehicle - vehicle_power_used
    flex_vehicle_val = 0.0
    for vehicle_id in vehicle_ids:
        try:
            dev_config = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
            flex_vehicle_val += float(dev_config.get("capacity", 0.0))
        except:
            pass
    node_ctx["current_vehicle"] = max(0.0, min(flex_vehicle_val, new_vehicle))

    node_ctx["current_AC"] = ac_power_used
    node_ctx["current_wash"] = wash_count_used

    node_ctx["soc"] = soc
    node_ctx["track_error"] = abs(actual - bid_nt)

    # Return: elec, revenue, penalty_device, deg, risk, device_net (per-device net this step)
    # Note: revenue = price * elec (economic revenue, penalty calculated separately)
    return elec, revenue, penalty, deg_cost, risk_cost, device_net

def _get_node_device_capacity(scenario):
    """Compute per-node per-device capacity from scenario (for post-run stats)."""
    nodes = scenario["nodes"]
    T = scenario.get("T", 24)
    pv = scenario.get("pv", {})
    wind = scenario.get("wind", {})
    device_dict_all = scenario.get("device_dict", {})
    BASE_DIR = DEVICE_CONFIG_BASE
    out = {}
    for n in nodes:
        cap = {"pv": 0.0, "wind": 0.0, "storage": 0.0, "vehicle": 0.0, "AC": 0.0, "washing_machine": 0.0}
        if n in pv and pv[n]:
            cap["pv"] = float(np.max(pv[n])) if hasattr(pv[n], "__len__") else float(pv[n])
        if n in wind and wind[n]:
            cap["wind"] = float(np.max(wind[n])) if hasattr(wind[n], "__len__") else float(wind[n])
        dev = device_dict_all.get(n, {})
        for storage_id in dev.get("storage_id", []):
            try:
                d = load_device_yaml("storage", storage_id, BASE_DIR)
                cap["storage"] += float(d.get("capacity", 0.0))
            except Exception:
                pass
        for vehicle_id in dev.get("vehicle_id", []):
            try:
                d = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
                cap["vehicle"] += float(d.get("capacity", 0.0))
            except Exception:
                pass
        for ac_id in dev.get("AC_id", []):
            try:
                d = load_device_yaml("AC", ac_id, BASE_DIR)
                cap["AC"] += float(d.get("power_max", 0.0))
            except Exception:
                pass
        for wash_id in dev.get("wash_id", []):
            try:
                d = load_device_yaml("wash", wash_id, BASE_DIR)
                cap["washing_machine"] += float(d.get("rate_power", 0.0))
            except Exception:
                pass
        out[n] = cap
    return out


def run_joint_episode(program, scenario):
    """
    Run a joint episode consistent with run_genetic.ipynb:
    - Revenue: price * elec - penalty (no bid limit on revenue)
    - Deviation: L1 norm of (total_elec - total_bid) over 24 hours
    """
    nodes = scenario["nodes"]
    T = scenario["T"]
    total_bid = scenario["total_bid"]  # list length T
    price = scenario["price"]          # dict node->list[T]
    pv = scenario["pv"]                 # dict node->list[T]
    wind = scenario["wind"]            # dict node->list[T]
    flex_storage = scenario["flex_storage"]  # dict node->float
    flex_vehicle = scenario["flex_vehicle"]  # dict node->float
    node_device_capacity = _get_node_device_capacity(scenario)
    flex_AC = scenario.get("flex_AC", {})
    flex_wash = scenario.get("flex_wash", {})

    if not flex_AC or not flex_wash:
        for n in nodes:
            if n not in flex_AC or n not in flex_wash:
                node_devices = scenario.get("device_dict", {}).get(n, {})
                ac_flex = 0.0
                ac_ids = node_devices.get("AC_id", [])
                for ac_id in ac_ids:
                    try:
                        ac_dev = load_device_yaml("AC", ac_id, DEVICE_CONFIG_BASE)
                        ac_flex += float(ac_dev.get("power_max", 0.0))
                    except:
                        pass
                if n not in flex_AC:
                    flex_AC[n] = ac_flex

                wash_flex = 0.0
                wash_ids = node_devices.get("wash_id", [])
                for wash_id in wash_ids:
                    try:
                        wash_dev = load_device_yaml("wash", wash_id, DEVICE_CONFIG_BASE)
                        wash_flex += float(wash_dev.get("rate_power", 0.0))
                    except:
                        pass
                if n not in flex_wash:
                    flex_wash[n] = wash_flex

    # Init node contexts (pv/wind in scenario are absolute; normalized when calling alpha_score)
    node_ctx = {}
    for n in nodes:
        node_ctx[n] = {
            "price": price[n],
            "pv": pv[n],
            "wind": wind[n],
            "soc": scenario["init_soc"].get(n, 0.5),
            "track_error": 0.0,
            "devices": scenario["devices"][n],
            "current_storage": 0.0,
            "current_vehicle": 0.0,
            "current_AC": 0.0,
            "current_wash": 0,
            "device_dict": scenario.get("device_dict", {}).get(n, {}),
        }

    # Aligned with VPP_dispatching: accumulate revenue and penalty separately
    total_revenue = 0.0  # Economic revenue = price * elec_clip (after bidding threshold clipping)
    total_penalty_device = 0.0  # Device operation penalty
    total_penalty_bid = 0.0  # Bidding deviation penalty
    total_elec = np.zeros(T, dtype=float)  # Total elec across all nodes for each time step
    total_deg = 0.0
    total_risk = 0.0

    # Per-node per-device 24h net accumulation (for post-run stats)
    node_device_net = {n: {"pv": 0.0, "wind": 0.0, "storage": 0.0, "vehicle": 0.0, "AC": 0.0, "washing_machine": 0.0} for n in nodes}
    
    # Load evaluation configuration
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        eval_config = config.get("evaluation", {})
        bidding_penalty = eval_config.get("bidding_penalty", 1.0)
        bidding_ratio = eval_config.get("bidding_ratio", 0.2)
        device_penalty_effi = eval_config.get("device_penalty_effi", 1.0)
        normalize_ratio = eval_config.get("normalize_ratio", 1.0)
    except:
        # Default values if config not found
        bidding_penalty = 1.0
        bidding_ratio = 0.2
        device_penalty_effi = 1.0
        normalize_ratio = 1.0

    norm_params = get_normalization_params()

    for t in range(T):
        scores = {}
        for i,n in enumerate(nodes):
            p_norm = normalize_value(
                price[n][t],
                norm_params["price"]["min"],
                norm_params["price"]["max"],
                norm_params["price"]["range"]
            )
            power_max = norm_params.get("power_max", 10000.0)
            pv_norm = normalize_power(pv[n][t], power_max)
            wind_norm = normalize_power(wind[n][t], power_max)
            flex_storage_norm = normalize_power(flex_storage[n], power_max)
            flex_vehicle_norm = normalize_power(flex_vehicle[n], power_max)
            flex_AC_norm = normalize_power(flex_AC.get(n, 0.0), power_max)
            flex_wash_norm = normalize_power(flex_wash.get(n, 0.0), power_max)

            scores[n] = program.alpha_score(
                p=p_norm,
                pv=pv_norm,
                wind=wind_norm,
                flex_storage=flex_storage_norm,
                current_storage=0.0,
                flex_vehicle=flex_vehicle_norm,
                current_vehicle=0.0,
                flex_AC=flex_AC_norm,
                current_AC=0.0,
                flex_wash=flex_wash_norm,
                current_wash=0.0,
                t=t, n=n,
                ctx={"T": T}
            )
            
            # if i == 0 and t == 0:
            #     print(f"score: {scores[n]}, price: {price[n][t]}, pv: {pv[n][t]}, wind: {wind[n][t]}, flex: {flex[n]}")
            # safety: avoid non-positive / nan
            if not (scores[n] > 0.0) or scores[n] != scores[n]:
                scores[n] = 1e-6

        alpha = softmax(scores, temperature)
        bids = {n: alpha[n] * total_bid[t] for n in nodes}

        # Step2: per node dispatch using device_utility
        for n in nodes:
            # Get elec, revenue, device penalty, and device_net for this node at this time
            elec_nt, revenue_nt, penalty_device_nt, deg_nt, risk_nt, device_net_nt = dispatch_node_devices(
                program, n, t, bids[n], node_ctx[n]
            )
            
            # Accumulate: aligned with VPP_dispatching
            total_revenue += revenue_nt  # price * elec (economic revenue)
            total_penalty_device += penalty_device_nt  # Device operation penalty
            total_elec[t] += elec_nt  # Accumulate elec across nodes for each time step
            total_deg += deg_nt
            total_risk += risk_nt
            for key in node_device_net[n]:
                node_device_net[n][key] += device_net_nt.get(key, 0.0)
        
        # Calculate bidding deviation penalty for this time step (aligned with VPP_dispatching)
        elec_quan = total_elec[t]  # Total electricity at time t
        bid = total_bid[t]  # Bidding value at time t
        
        # Calculate threshold (aligned with VPP_dispatching logic)
        if abs(bid) < 50:
            thresh = 10
        else:
            thresh = abs(bidding_ratio * bid)
        
        # Calculate elec_clip and penalty_bid (aligned with VPP_dispatching)
        if elec_quan - bid > thresh:
            # Exceeded upper threshold
            penalty_bid_t = (elec_quan - bid - thresh) * np.mean([price[n][t] for n in nodes]) / normalize_ratio * bidding_penalty
            elec_clip = bid + thresh
        elif elec_quan - bid < -thresh:
            # Below lower threshold
            penalty_bid_t = (bid - elec_quan - thresh) * np.mean([price[n][t] for n in nodes]) / normalize_ratio * (1 + bidding_penalty)
            elec_clip = bid - thresh
        else:
            # Within threshold
            penalty_bid_t = 0.0
            elec_clip = elec_quan
        
        # Update revenue with clipped electricity (aligned with VPP_dispatching)
        # Note: We need to recalculate revenue with elec_clip instead of elec_quan
        # But since we already calculated revenue per node, we need to adjust
        # For simplicity, we'll calculate revenue adjustment here
        revenue_adjustment = (elec_clip - elec_quan) * np.mean([price[n][t] for n in nodes])
        total_revenue += revenue_adjustment
        total_penalty_bid += penalty_bid_t
        
        # Calculate device operation penalty at end of day (aligned with VPP_dispatching)
        if t == T - 1:  # Last time step of the day
            for n in nodes:
                node_ctx_n = node_ctx[n]
                
                # Storage penalty: if SoC < init_SoC at end of day
                init_soc = scenario["init_soc"].get(n, 0.5)
                current_soc = node_ctx_n.get("soc", init_soc)
                if current_soc < init_soc:
                    storage_penalty = 0.1 * (init_soc - current_soc)
                    total_penalty_device += storage_penalty
                
                # Vehicle penalty: if SoC < capacity_demand at departure time
                # Note: This requires tracking vehicle departure times, simplified here
                # For each vehicle, check if departure time reached
                vehicle_ids = node_ctx_n.get("device_dict", {}).get("vehicle_id", [])
                base_dir = DEVICE_CONFIG_BASE
                for vehicle_id in vehicle_ids:
                    try:
                        dev_config = load_device_yaml("vehicle", vehicle_id, base_dir)
                        capacity_demand = float(dev_config.get("capacity_demand", 80.0))
                        capacity_max = float(dev_config.get("capacity", 100.0))
                        # Simplified: check if vehicle SoC < capacity_demand at end of day
                        # In reality, this should be checked at departure time
                        vehicle_soc = node_ctx_n.get("current_vehicle", 0.0) / capacity_max if capacity_max > 0 else 0.0
                        if vehicle_soc < (capacity_demand / capacity_max):
                            vehicle_penalty = 0.2 * (capacity_demand / capacity_max - vehicle_soc) * capacity_max
                            total_penalty_device += vehicle_penalty
                    except:
                        pass
                
                # Wash penalty: if work_state < t_dur at end of day
                wash_ids = node_ctx_n.get("device_dict", {}).get("wash_id", [])
                for wash_id in wash_ids:
                    try:
                        dev_config = load_device_yaml("wash", wash_id, base_dir)
                        t_dur = int(dev_config.get("t_dur", 2))
                        # Simplified: check if wash work_state < t_dur at end of day
                        # In reality, this should be tracked per device
                        wash_work_state = node_ctx_n.get("current_wash", 0)
                        if wash_work_state < t_dur:
                            wash_penalty = 5.0 * (t_dur - wash_work_state)
                            total_penalty_device += wash_penalty
                    except:
                        pass

    # Calculate total penalty (device + bidding deviation)
    total_penalty = device_penalty_effi * total_penalty_device + total_penalty_bid
    
    # Calculate final revenue (aligned with VPP_dispatching: revenue = price * elec_clip - penalty)
    final_revenue = total_revenue - total_penalty
    
    # Deviation: L1 norm of (total_elec - total_bid) over 24 hours
    # This represents the overall deviation from bidding values
    deviation = float(np.abs(total_elec - np.array(total_bid, dtype=float)).sum())

    return {
        "revenue": final_revenue,  # Economic revenue after penalty (aligned with VPP_dispatching)
        "penalty": total_penalty,  # Total penalty (device + bidding deviation)
        "penalty_device": total_penalty_device,  # Device operation penalty
        "penalty_bid": total_penalty_bid,  # Bidding deviation penalty
        "deviation": deviation,   # L1 norm over 24 hours
        "deg": total_deg,
        "risk": total_risk,
        # For post-run summary stats
        "total_elec": total_elec.tolist(),
        "total_bid": list(total_bid),
        "node_prices": {n: list(price[n]) for n in nodes},
        "node_device_capacity": node_device_capacity,
        "node_device_net": node_device_net,
    }
