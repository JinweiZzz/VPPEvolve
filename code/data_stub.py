import pandas as pd
import pickle
import numpy as np
import yaml
import os
import random

# ========== Data path configuration ==========
# Project root: parent of VPPEvolve_released (when run from VPPEvolve_released, BASE_DIR = "../" is cwd-relative)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEVICE_CONFIG_BASE = os.path.join(PROJECT_ROOT, "config", "device")

BASE_DIR = "../"  # Relative to cwd; must run from VPPEvolve_released so this resolves to project root
DATA_DIR = os.path.join(BASE_DIR, "data/Shanxi")  # Data directory
DEVICE_SET_CONFIG = os.path.join(BASE_DIR, "config/device_set/shanxi_15nodes.yaml")

# ========== Load real data ==========
print("Loading data...")

# 1. Load electricity price data
sampled_node_prices = pd.read_csv(os.path.join(DATA_DIR, 'price_data.csv'))
node_set = sampled_node_prices['node_name'].unique()
print(f"Loaded price data for {len(node_set)} nodes")

# 2. Load PV/Wind prediction and real data
with open(os.path.join(BASE_DIR, 'data/Shanxi/station2pv_pred_list.pkl'), 'rb') as f:
    station2pv_pred_list = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data/Shanxi/station2pv_real_list.pkl'), 'rb') as f:
    station2pv_real_list = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data/Shanxi/station2wind_pred_list.pkl'), 'rb') as f:
    station2wind_pred_list = pickle.load(f)
with open(os.path.join(BASE_DIR, 'data/Shanxi/station2wind_real_list.pkl'), 'rb') as f:
    station2wind_real_list = pickle.load(f)

# 3. Load bid data
with open(os.path.join(DATA_DIR, 'bid_data.pkl'), 'rb') as f:
    bid_data = pickle.load(f)

# 4. Load device configuration
with open(DEVICE_SET_CONFIG, 'r') as yaml_file:
    device_set_config = yaml.safe_load(yaml_file)
node_device_mapping = device_set_config.get('node_device_mapping', {})

print("Data loading complete!")

# ========== Data preprocessing ==========
def preprocess_price_data(node_prices_df, T=24):
    """Convert 15-minute granularity price data to 24-hour data"""
    price_list = node_prices_df['price_value'].tolist()
    # Average every 4 values (15 min -> 1 hour)
    price_list = [np.mean(price_list[i*4:(i+1)*4]) for i in range(len(price_list)//4)]
    return price_list[:T]

def preprocess_ren_data(ren_list, T=24):
    """Convert 15-minute granularity renewable data to 24-hour data"""
    # Average every 4 values
    ren_list = [np.mean(ren_list[i*4:(i+1)*4]) for i in range(len(ren_list)//4)]
    return ren_list[:T]

def aggregate_bid_data(bid_data, day=0):
    """Aggregate bid data for all nodes"""
    bid_sum = None
    for node, bid in bid_data.items():
        if day < len(bid):
            day_bid = np.array(bid[day])
            if bid_sum is None:
                bid_sum = day_bid
            else:
                bid_sum += day_bid
    return bid_sum.tolist() if bid_sum is not None else [0.0] * 24

def make_devices_from_config(node_name, node_device_mapping):
    """
    Generate simplified device list from node device configuration.
    Device format: {"type": "storage/vehicle/wash/AC", "candidates": [...], "soc_gain": ...}
    Note: PV and Wind are not devices; they use node total power data directly.
    """
    devices = []

    if node_name not in node_device_mapping:
        # If no config, use default devices (excluding PV and Wind)
        devices = [
            {"type": "storage", "candidates": [-1.0, 0.0, 1.0], "soc_gain": 0.02},
            {"type": "vehicle", "candidates": [-1.0, 0.0, 1.0], "soc_gain": 0.015},
        ]
        return devices
    
    node_devices = node_device_mapping[node_name]
    
    # Generate devices by count (excluding PV and Wind)
    storage_count = len(node_devices.get('storage_id', []))
    vehicle_count = len(node_devices.get('vehicle_id', []))
    wash_count = len(node_devices.get('wash_id', []))
    ac_count = len(node_devices.get('AC_id', []))

    # Storage devices
    for i in range(storage_count):
        devices.append({
            "type": "storage",
            "candidates": [-1.0, -0.5, 0.0, 0.5, 1.0],  # charge/discharge
            "soc_gain": 0.02,
        })

    # Vehicle devices
    for i in range(vehicle_count):
        devices.append({
            "type": "vehicle",
            "candidates": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "soc_gain": 0.015,
        })

    # Wash devices
    for i in range(wash_count):
        devices.append({
            "type": "wash",
            "candidates": [0.0, 1.0],  # on/off
        })

    # AC devices
    for i in range(ac_count):
        devices.append({
            "type": "AC",
            "candidates": [0.0, 0.3, 0.6, 1.0],  # power ratio
        })
    
    return devices

def calculate_flexibility(node_name, node_device_mapping, base_dir=None):
    """
    Compute three flexibility values from node device configuration.

    Flexibility definitions:
    1. flex_storage: total dispatchable storage capacity
    2. flex_vehicle: total dispatchable EV chargeable capacity
    3. flex_load: total dispatchable AC+wash power (AC power_max + wash rate_power)

    Returns:
        tuple: (flex_storage, flex_vehicle, flex_load) - absolute values, not normalized
    """
    if base_dir is None:
        base_dir = DEVICE_CONFIG_BASE
    if node_name not in node_device_mapping:
        return (0.0, 0.0, 0.0)

    node_devices = node_device_mapping[node_name]
    flex_storage = 0.0
    flex_vehicle = 0.0
    flex_load = 0.0

    # 1. Total storage capacity
    storage_ids = node_devices.get('storage_id', [])
    for storage_id in storage_ids:
        try:
            import yaml
            device_path = f"{base_dir}/storage/simulator_{int(storage_id)}.yaml"
            with open(device_path, 'r') as f:
                device_data = yaml.safe_load(f)
            storage_capacity = float(device_data.get('storage', {}).get('capacity', 0.0))
            flex_storage += storage_capacity
        except Exception as e:
            pass
    
    # 2. Total EV chargeable capacity
    vehicle_ids = node_devices.get('vehicle_id', [])
    for vehicle_id in vehicle_ids:
        try:
            import yaml
            device_path = f"{base_dir}/vehicle/simulator_{int(vehicle_id)}.yaml"
            with open(device_path, 'r') as f:
                device_data = yaml.safe_load(f)
            vehicle_capacity = float(device_data.get('vehicle', {}).get('capacity', 0.0))
            flex_vehicle += vehicle_capacity
        except Exception as e:
            pass
    
    # 3. Total AC+wash power
    ac_ids = node_devices.get('AC_id', [])
    for ac_id in ac_ids:
        try:
            import yaml
            device_path = f"{base_dir}/AC/simulator_{int(ac_id)}.yaml"
            with open(device_path, 'r') as f:
                device_data = yaml.safe_load(f)
            ac_power_max = float(device_data.get('AC', {}).get('power_max', 0.0))
            flex_load += ac_power_max
        except Exception as e:
            pass
    
    wash_ids = node_devices.get('wash_id', [])
    for wash_id in wash_ids:
        try:
            import yaml
            device_path = f"{base_dir}/wash/simulator_{int(wash_id)}.yaml"
            with open(device_path, 'r') as f:
                device_data = yaml.safe_load(f)
            wash_rate_power = float(device_data.get('wash', {}).get('rate_power', 0.0))
            flex_load += wash_rate_power
        except Exception as e:
            pass
    
    return (flex_storage, flex_vehicle, flex_load)

# ========== Normalization parameters ==========
# Compute normalization parameters (based on all data)
def compute_normalization_params(node_price_data, node_pv_data, node_wind_data, node_device_mapping=None, selected_nodes=None, bid_quantities=None):
    """Compute normalization parameters"""
    # Price normalization parameters
    all_prices = []
    for node, prices in node_price_data.items():
        all_prices.extend(prices)
    price_min = min(all_prices) if all_prices else 0.0
    price_max = max(all_prices) if all_prices else 1.0
    
    # Collect all power-related values (for unified max)
    # Includes: PV, Wind, bid_quantity, wash power, etc.
    all_power_values = []

    # 1. Collect PV power (all nodes, all time steps)
    for node, pv_list in node_pv_data.items():
        all_power_values.extend(pv_list)

    # 2. Collect Wind power (all nodes, all time steps)
    for node, wind_list in node_wind_data.items():
        all_power_values.extend(wind_list)

    # 3. Collect bid_quantity values
    if bid_quantities:
        all_power_values.extend(bid_quantities)

    # delta_p normalization (assume power change in [-100, 100], adjust as needed)
    delta_p_min = -100.0
    delta_p_max = 100.0

    # track_error normalization (assume range [0, 1000], adjust as needed)
    track_error_min = 0.0
    track_error_max = 1000.0

    # Collect all flexibility values (flex_storage, flex_vehicle, flex_load, flex_AC, flex_wash)
    all_flex_storage = []
    all_flex_vehicle = []
    all_flex_load = []
    all_flex_AC = []
    all_flex_wash = []
    
    if node_device_mapping is not None and selected_nodes is not None:
        for node in selected_nodes:
            flex_storage, flex_vehicle, flex_load = calculate_flexibility(node, node_device_mapping)
            all_flex_storage.append(flex_storage)
            all_flex_vehicle.append(flex_vehicle)
            all_flex_load.append(flex_load)
            
            # Compute flex_AC and flex_wash
            node_devices = node_device_mapping.get(node, {})
            ac_flex = 0.0
            ac_ids = node_devices.get('AC_id', [])
            for ac_id in ac_ids:
                try:
                    import yaml
                    device_path = os.path.join(DEVICE_CONFIG_BASE, "AC", f"simulator_{int(ac_id)}.yaml")
                    with open(device_path, 'r') as f:
                        device_data = yaml.safe_load(f)
                    ac_power_max = float(device_data.get('AC', {}).get('power_max', 0.0))
                    ac_flex += ac_power_max
                except:
                    pass
            all_flex_AC.append(ac_flex)
            
            wash_flex = 0.0
            wash_ids = node_devices.get('wash_id', [])
            for wash_id in wash_ids:
                try:
                    import yaml
                    device_path = os.path.join(DEVICE_CONFIG_BASE, "wash", f"simulator_{int(wash_id)}.yaml")
                    with open(device_path, 'r') as f:
                        device_data = yaml.safe_load(f)
                    wash_rate_power = float(device_data.get('wash', {}).get('rate_power', 0.0))
                    wash_flex += wash_rate_power
                except:
                    pass
            all_flex_wash.append(wash_flex)
    
    # 4. Collect wash power (total wash device power per node)
    # wash power = sum of wash device rate_power
    for node in (selected_nodes if selected_nodes else []):
        if node_device_mapping and node in node_device_mapping:
            node_devices = node_device_mapping.get(node, {})
            wash_ids = node_devices.get('wash_id', [])
            wash_total_power = 0.0
            for wash_id in wash_ids:
                try:
                    import yaml
                    device_path = os.path.join(DEVICE_CONFIG_BASE, "wash", f"simulator_{int(wash_id)}.yaml")
                    with open(device_path, 'r') as f:
                        device_data = yaml.safe_load(f)
                    wash_rate_power = float(device_data.get('wash', {}).get('rate_power', 0.0))
                    wash_total_power += wash_rate_power
                except:
                    pass
            if wash_total_power > 0:
                all_power_values.append(wash_total_power)
    
    # 5. Collect other power-related values (flex_storage, flex_vehicle, flex_AC, etc.)
    all_quantity_flex_current_values = []
    all_quantity_flex_current_values.extend(all_flex_storage)
    all_quantity_flex_current_values.extend(all_flex_vehicle)
    all_quantity_flex_current_values.extend(all_flex_load)
    all_quantity_flex_current_values.extend(all_flex_AC)
    all_quantity_flex_current_values.extend(all_flex_wash)

    # Merge all power-related values
    all_power_values.extend(all_quantity_flex_current_values)

    # Unified power max (max over all nodes and time steps)
    if all_power_values:
        power_max = max(all_power_values)
        if power_max < 1.0:
            power_max = 10000.0  # default
    else:
        power_max = 10000.0  # default

    result = {
        "price": {"min": price_min, "max": price_max, "range": price_max - price_min if price_max > price_min else 1.0},
        "delta_p": {"min": delta_p_min, "max": delta_p_max, "range": delta_p_max - delta_p_min},
        "track_error": {"min": track_error_min, "max": track_error_max, "range": track_error_max - track_error_min},
        # Unified power_max for normalizing all power-related values (pv, wind, bid_quantity, flex_*, etc.)
        "power_max": power_max,
    }

    # Keep legacy flexibility params (compatibility; normalization uses unified power_max)
    if all_flex_storage:
        result["flex_storage"] = {"min": 0.0, "max": power_max, "range": power_max}
    else:
        result["flex_storage"] = {"min": 0.0, "max": power_max, "range": power_max}
    
    if all_flex_vehicle:
        result["flex_vehicle"] = {"min": 0.0, "max": power_max, "range": power_max}
    else:
        result["flex_vehicle"] = {"min": 0.0, "max": power_max, "range": power_max}
    
    if all_flex_load:
        result["flex_load"] = {"min": 0.0, "max": power_max, "range": power_max}
    else:
        result["flex_load"] = {"min": 0.0, "max": power_max, "range": power_max}
    
    # Add flex_AC and flex_wash
    result["flex_AC"] = {"min": 0.0, "max": power_max, "range": power_max}
    result["flex_wash"] = {"min": 0.0, "max": power_max, "range": power_max}

    # Add bid_quantity
    result["bid_quantity"] = {"min": 0.0, "max": power_max, "range": power_max}
    
    return result

# ========== Generate scenarios ==========
# SCENARIOS: predicted scenarios for evolution
# SCENARIOS_REAL: real scenarios for final testing
SCENARIOS = []
SCENARIOS_REAL = []
T = 24

# Select nodes to use (all or a subset)
# For testing, use first N nodes or all
selected_nodes = list(node_set)[:15]  # Use all 15 nodes, or subset e.g. [:5]

print(f"Generating scenarios with {len(selected_nodes)} nodes...")

# Prepare data per node (prediction and real)
node_price_data = {}
node_pv_pred_data = {}
node_wind_pred_data = {}
node_pv_real_data = {}
node_wind_real_data = {}

for node in selected_nodes:
    # Price data
    node_prices_df = sampled_node_prices[sampled_node_prices['node_name'] == node]
    if len(node_prices_df) > 0:
        price_list = preprocess_price_data(node_prices_df, T)
        node_price_data[node] = price_list
    
    # PV and Wind prediction data
    if node in station2pv_pred_list:
        pv_list = preprocess_ren_data(station2pv_pred_list[node], T)
        node_pv_pred_data[node] = pv_list
    else:
        node_pv_pred_data[node] = [0.0] * T
    
    if node in station2wind_pred_list:
        wind_list = preprocess_ren_data(station2wind_pred_list[node], T)
        node_wind_pred_data[node] = wind_list
    else:
        node_wind_pred_data[node] = [0.0] * T
    
    # PV and Wind real data
    if node in station2pv_real_list:
        pv_list = preprocess_ren_data(station2pv_real_list[node], T)
        node_pv_real_data[node] = pv_list
    else:
        node_pv_real_data[node] = [0.0] * T
    
    if node in station2wind_real_list:
        wind_list = preprocess_ren_data(station2wind_real_list[node], T)
        node_wind_real_data[node] = wind_list
    else:
        node_wind_real_data[node] = [0.0] * T

# Collect all bid_quantity values (from all days in bid_data)
all_bid_quantities = []
for day in range(7):  # Assume at most 7 days
    if day < len(bid_data.get(list(bid_data.keys())[0], [])) if bid_data else 0:
        total_bid = aggregate_bid_data(bid_data, day)
        all_bid_quantities.extend(total_bid)

# Compute normalization params (after all data ready, including flexibility)
NORMALIZATION_PARAMS = compute_normalization_params(
    node_price_data, node_pv_pred_data, node_wind_pred_data,
    node_device_mapping=node_device_mapping,
    selected_nodes=selected_nodes,
    bid_quantities=all_bid_quantities
)

# Generate scenarios: first 4 days for evolution training, next 3 for validation
available_days = min(7, len(bid_data.get(list(bid_data.keys())[0], [])) if bid_data else 0)
train_days = 4  # First 4 days for training
test_days = 3   # Next 3 days for validation

# ========== Generate training scenarios (first 4 days, predicted data) ==========
for day in range(train_days):
    if day >= available_days:
        break

    # Total bid for this day
    total_bid = aggregate_bid_data(bid_data, day)

    # Prepare data per node (prediction scenario for evolution)
    price_dict_pred = {}
    pv_dict_pred = {}
    wind_dict_pred = {}
    flex_storage_dict = {}
    flex_vehicle_dict = {}
    flex_load_dict = {}
    init_soc_dict = {}
    devices_dict = {}
    
    for node in selected_nodes:
        price_dict_pred[node] = node_price_data[node].copy()
        pv_dict_pred[node] = node_pv_pred_data[node].copy()
        wind_dict_pred[node] = node_wind_pred_data[node].copy()
        
        # Three flexibilities (stored separately)
        flex_storage, flex_vehicle, flex_load = calculate_flexibility(node, node_device_mapping)
        flex_storage_dict[node] = flex_storage
        flex_vehicle_dict[node] = flex_vehicle
        flex_load_dict[node] = flex_load

        # Initial SOC (random)
        init_soc_dict[node] = random.uniform(0.3, 0.7)

        # Device configuration
        devices_dict[node] = make_devices_from_config(node, node_device_mapping)

    # Prediction scenario (for evolution training)
    SCENARIOS.append({
        "nodes": selected_nodes,
        "T": T,
        "total_bid": total_bid,
        "price": price_dict_pred,
        "pv": pv_dict_pred,         # pv is total node PV prediction
        "wind": wind_dict_pred,
        "flex_storage": flex_storage_dict,
        "flex_vehicle": flex_vehicle_dict,
        "flex_load": flex_load_dict,
        "init_soc": init_soc_dict,
        "devices": devices_dict,
        "device_dict": node_device_mapping,
        "day": day,  # Which day
    })

# ========== Generate test scenarios (last 3 days, real data) ==========
for day in range(train_days, train_days + test_days):
    if day >= available_days:
        break

    # Total bid for this day
    total_bid = aggregate_bid_data(bid_data, day)

    # Prepare data per node (real scenario for final testing)
    price_dict_real = {}
    pv_dict_real = {}
    wind_dict_real = {}
    flex_dict = {}
    init_soc_dict = {}
    devices_dict = {}
    
    for node in selected_nodes:
        price_dict_real[node] = node_price_data[node].copy()  # Same price
        pv_dict_real[node] = node_pv_real_data[node].copy()
        wind_dict_real[node] = node_wind_real_data[node].copy()

        # Three flexibilities (stored separately)
        flex_storage, flex_vehicle, flex_load = calculate_flexibility(node, node_device_mapping)
        flex_storage_dict[node] = flex_storage
        flex_vehicle_dict[node] = flex_vehicle
        flex_load_dict[node] = flex_load

        # Initial SOC (random)
        init_soc_dict[node] = random.uniform(0.3, 0.7)

        # Device configuration
        devices_dict[node] = make_devices_from_config(node, node_device_mapping)

    # Real scenario (for final testing)
    SCENARIOS_REAL.append({
        "nodes": selected_nodes,
        "T": T,
        "total_bid": total_bid,
        "price": price_dict_real,
        "pv": pv_dict_real,
        "wind": wind_dict_real,
        "flex_storage": flex_storage_dict,
        "flex_vehicle": flex_vehicle_dict,
        "flex_load": flex_load_dict,
        "init_soc": init_soc_dict,
        "devices": devices_dict,
        "device_dict": node_device_mapping, # Pass full device mapping for simulator
        "day": day,  # Which day
    })

# Compute normalization params (after all data ready; flexibility needs node_device_mapping)
NORMALIZATION_PARAMS = compute_normalization_params(
    node_price_data, node_pv_pred_data, node_wind_pred_data,
    node_device_mapping=node_device_mapping,
    selected_nodes=selected_nodes
)

print(f"Generated {len(SCENARIOS)} training scenarios (first {train_days} days, predicted data, for evolution)")
print(f"Generated {len(SCENARIOS_REAL)} test scenarios (last {test_days} days, real data, for validation)")
print(f"Each scenario has {len(selected_nodes)} nodes, {T} hours per node")
print(f"\nNormalization parameters:")
print(f"  Price: [{NORMALIZATION_PARAMS['price']['min']:.2f}, {NORMALIZATION_PARAMS['price']['max']:.2f}] (range: {NORMALIZATION_PARAMS['price']['range']:.2f})")
print(f"  Power max: {NORMALIZATION_PARAMS.get('power_max', 10000.0):.2f} (for PV, Wind, bid_quantity, wash, etc.)")
print(f"  Flex_Storage: [{NORMALIZATION_PARAMS['flex_storage']['min']:.2f}, {NORMALIZATION_PARAMS['flex_storage']['max']:.2f}] (range: {NORMALIZATION_PARAMS['flex_storage']['range']:.2f})")
print(f"  Flex_Vehicle: [{NORMALIZATION_PARAMS['flex_vehicle']['min']:.2f}, {NORMALIZATION_PARAMS['flex_vehicle']['max']:.2f}] (range: {NORMALIZATION_PARAMS['flex_vehicle']['range']:.2f})")
print(f"  Flex_Load: [{NORMALIZATION_PARAMS['flex_load']['min']:.2f}, {NORMALIZATION_PARAMS['flex_load']['max']:.2f}] (range: {NORMALIZATION_PARAMS['flex_load']['range']:.2f})")
print(f"  Delta_p: [{NORMALIZATION_PARAMS['delta_p']['min']:.2f}, {NORMALIZATION_PARAMS['delta_p']['max']:.2f}] (range: {NORMALIZATION_PARAMS['delta_p']['range']:.2f})")
print(f"  Track_error: [{NORMALIZATION_PARAMS['track_error']['min']:.2f}, {NORMALIZATION_PARAMS['track_error']['max']:.2f}] (range: {NORMALIZATION_PARAMS['track_error']['range']:.2f})")
