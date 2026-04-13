import pandas as pd
import pickle
import random
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gurobipy as gp
from gurobipy import GRB

# 电价数据
sampled_node_prices = pd.read_csv('/data2/zengjinwei/VPP_multinode/data/山西数据/山西15节点1月1日到1月15日数据.csv')

# pv\wind预测和真实数据
import pickle
with open('/data2/zengjinwei/VPP_multinode/data/station2pv_pred_list.pkl', 'rb') as f:
    station2pv_pred_list = pickle.load(f)
with open('/data2/zengjinwei/VPP_multinode/data/station2pv_real_list.pkl', 'rb') as f:
    station2pv_real_list = pickle.load(f)
with open('/data2/zengjinwei/VPP_multinode/data/station2wind_pred_list.pkl', 'rb') as f:
    station2wind_pred_list = pickle.load(f)
with open('/data2/zengjinwei/VPP_multinode/data/station2wind_real_list.pkl', 'rb') as f:
    station2wind_real_list = pickle.load(f)

# bid数据
with open('/data2/zengjinwei/VPP_multinode/data/山西数据/山西15节点1月1日到1月15日bid数据_scaled.pkl', 'rb') as f:
    bid_data = pickle.load(f)

bid_data_sum = {}

for day in range(7):
    for node, bid in bid_data.items():
        if day not in bid_data_sum:
            bid_data_sum[day] = np.array(bid[day])
        else:
            bid_data_sum[day] += np.array(bid[day])


import yaml 
import numpy as np
device_set_name = 'shanxi_15nodes'
with open(f'config/device_set/{device_set_name}.yaml', 'r') as yaml_file:
    device_set_config = yaml.safe_load(yaml_file)

# 获取节点设备映射
node_device_mapping = device_set_config.get('node_device_mapping', {})
node_set = sampled_node_prices['node_name'].unique()

node2bid = {}
price = np.zeros((len(node_set), 7, 24))
bid = np.zeros((7, 24))
pv_pred = np.zeros((len(node_set), 7, 24))
wind_pred = np.zeros((len(node_set), 7, 24))
pv_real = np.zeros((len(node_set), 7, 24))
wind_real = np.zeros((len(node_set), 7, 24))
node_idx = 0

# 对每个node进行整数优化
for node in node_set:
    node_prices_df = sampled_node_prices[sampled_node_prices['node_name'] == node]

    if len(node_prices_df) != 672:
        print('Error! Price df dismatched!')

    price_list = node_prices_df['price_value'].tolist()
    ### 每4个值求平均
    price_list = [np.mean(price_list[i*4:(i+1)*4]) for i in range(len(price_list)//4)]

    # 获取pv和wind的预测数据
    pv_pred_list = station2pv_pred_list[node][:len(price_list)*4]   ### pv的时间粒度也是15min
    wind_pred_list = station2wind_pred_list[node][:len(price_list)*4]
    pv_real_list = station2pv_real_list[node][:len(price_list)*4]
    wind_real_list = station2wind_real_list[node][:len(price_list)*4]

    pv_pred_list = [np.mean(pv_pred_list[i*4:(i+1)*4]) for i in range(len(pv_pred_list)//4)]
    wind_pred_list = [np.mean(wind_pred_list[i*4:(i+1)*4]) for i in range(len(wind_pred_list)//4)]  
    pv_real_list = [np.mean(pv_real_list[i*4:(i+1)*4]) for i in range(len(pv_real_list)//4)]
    wind_real_list = [np.mean(wind_real_list[i*4:(i+1)*4]) for i in range(len(wind_real_list)//4)]

    price[node_idx, :, :] = np.array(price_list).reshape(7, 24)
    pv_pred[node_idx, :, :] = np.array(pv_pred_list).reshape(7, 24)
    wind_pred[node_idx, :, :] = np.array(wind_pred_list).reshape(7, 24)
    pv_real[node_idx, :, :] = np.array(pv_real_list).reshape(7, 24)
    wind_real[node_idx, :, :] = np.array(wind_real_list).reshape(7, 24)

    node_idx += 1

for day in range(7):
    bid[day, :] = bid_data_sum[day]

node_list = list(node_set)


import numpy as np
import yaml
import gurobipy as gp
from gurobipy import GRB

_DEVICE_CACHE = {}

def load_device_yaml(device_type, device_id, base_dir):
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
    arr = np.asarray(curve, dtype=float)
    if arr.ndim == 1:
        if n_devices == 0:
            return np.zeros((0, T))
        if n_devices == 1:
            return arr[None, :]
        return np.tile(arr[None, :] / n_devices, (n_devices, 1))
    return arr


class SimpleVPPSimulator:
    """
    以“net injection 正”为统一约定：
      + 表示向外卖电/注入
      - 表示从外用电/取电
    """

    def __init__(self, price, pv, wind, device_dict, level=10, base_dir="/data2/zengjinwei/VPP_multinode/config/device"):
        self.price = np.asarray(price, dtype=float)
        self.T = len(price)
        self.level = int(level)
        self.base_dir = base_dir

        self.pv_ids = list(device_dict.get("pv_id", []))
        self.wind_ids = list(device_dict.get("wind_id", []))
        self.storage_ids = list(device_dict.get("storage_id", []))
        self.vehicle_ids = list(device_dict.get("vehicle_id", []))
        self.wash_ids = list(device_dict.get("wash_id", []))
        self.ac_ids = list(device_dict.get("AC_id", []))

        self.pv_params = [load_device_yaml("pv", i, base_dir=self.base_dir) for i in self.pv_ids]
        self.wind_params = [load_device_yaml("wind", i, base_dir=self.base_dir) for i in self.wind_ids]
        self.storage_params = [load_device_yaml("storage", i, base_dir=self.base_dir) for i in self.storage_ids]
        self.vehicle_params = [load_device_yaml("vehicle", i, base_dir=self.base_dir) for i in self.vehicle_ids]
        self.wash_params = [load_device_yaml("wash", i, base_dir=self.base_dir) for i in self.wash_ids]
        self.ac_params = [load_device_yaml("AC", i, base_dir=self.base_dir) for i in self.ac_ids]

        self.pv_curve = ensure_2d_curve(pv, self.T, len(self.pv_ids))
        self.wind_curve = ensure_2d_curve(wind, self.T, len(self.wind_ids))

        self.reset()

    def reset(self, seed=None):
        self.t = 0
        self.done = False
        # Initialize device states (aligned with run_joint_episode)
        # Storage: track energy in each device (absolute value in kW)
        self.storage_energy = {}  # {storage_id: energy_kW}
        for storage_id, dev in zip(self.storage_ids, self.storage_params):
            capacity = float(dev.get("capacity", 0.0))
            self.storage_energy[storage_id] = 0.5 * capacity  # Initial SoC = 0.5
        
        # Vehicle: track energy in each device (absolute value in kW)
        self.vehicle_energy = {}  # {vehicle_id: energy_kW}
        for vehicle_id, dev in zip(self.vehicle_ids, self.vehicle_params):
            capacity = float(dev.get("capacity", 0.0))
            self.vehicle_energy[vehicle_id] = 0.5 * capacity  # Initial SoC = 0.5
        
        # Wash: track work state (hours)
        self.wash_work_state = {}  # {wash_id: work_state (hours)}
        for wash_id in self.wash_ids:
            self.wash_work_state[wash_id] = 0
        
        return np.zeros(1), {}

    def step(self, action):
        """
        action 由 execute_strategy 组装：
          pv_action: curtail ratio [0,1] * n_pv
          wind_action: curtail ratio [0,1] * n_wind
          storage_action: net injection fraction in [-1,1] * n_storage
          vehicle_action: net injection fraction in [-1,1] * n_vehicle
          wash_action: on {0,1} * n_wash
          ac_action: power fraction [0,1] * n_ac  (消费)
        """
        t = self.t
        idx = 0
        elec = 0.0
        penalty = 0.0  # This penalty is always 0.0 in SimpleVPPSimulator now (aligned with run_joint_episode)

        # PV
        for k, dev in enumerate(self.pv_params):
            cur = float(np.clip(action[idx], 0, 1))
            gen = self.pv_curve[k][t] if t < len(self.pv_curve[k]) else 0.0
            elec += (1 - cur) * gen
            idx += 1

        # Wind
        for k, dev in enumerate(self.wind_params):
            cur = float(np.clip(action[idx], 0, 1))
            gen = self.wind_curve[k][t] if t < len(self.wind_curve[k]) else 0.0
            elec += (1 - cur) * gen
            idx += 1

        # Storage: action in [-1,1] => net injection power (aligned with dispatch_node_devices)
        # Constraint: discharge limited by available energy, charge limited by remaining capacity
        for k, (storage_id, dev) in enumerate(zip(self.storage_ids, self.storage_params)):
            frac = float(np.clip(action[idx], -1, 1))
            max_power = float(dev.get("max_power", 0.0))
            capacity = float(dev.get("capacity", 0.0))
            current_energy = self.storage_energy.get(storage_id, 0.0)
            
            # Calculate desired power
            desired_power = frac * max_power
            power_direction = 1.0 if desired_power >= 0 else -1.0
            
            # Apply constraints
            if power_direction > 0:  # Discharge
                # Limited by max_power and available energy
                max_available_power = min(max_power, current_energy)
                device_power = min(abs(desired_power), max_available_power) * power_direction
            else:  # Charge
                # Limited by max_power and remaining capacity
                remaining_capacity = capacity - current_energy
                max_available_power = min(max_power, remaining_capacity)
                device_power = -min(abs(desired_power), max_available_power)
            
            elec += device_power
            
            # Update energy (aligned with dispatch_node_devices: new_storage = current_storage - storage_power_used)
            # storage_power < 0 means charge (increase energy), > 0 means discharge (decrease energy)
            new_energy = current_energy - device_power
            self.storage_energy[storage_id] = max(0.0, min(capacity, new_energy))
            idx += 1

        # Vehicle: action in [-1,1] => net injection power (aligned with dispatch_node_devices)
        for k, (vehicle_id, dev) in enumerate(zip(self.vehicle_ids, self.vehicle_params)):
            frac = float(np.clip(action[idx], -1, 1))
            max_power = float(dev.get("max_power", 0.0))
            capacity = float(dev.get("capacity", 0.0))
            current_energy = self.vehicle_energy.get(vehicle_id, 0.0)
            
            # Calculate desired power
            desired_power = frac * max_power
            power_direction = 1.0 if desired_power >= 0 else -1.0
            
            # Apply constraints
            if power_direction > 0:  # Discharge
                # Limited by max_power and available energy
                max_available_power = min(max_power, current_energy)
                device_power = min(abs(desired_power), max_available_power) * power_direction
            else:  # Charge
                # Limited by max_power and remaining capacity
                remaining_capacity = capacity - current_energy
                max_available_power = min(max_power, remaining_capacity)
                device_power = -min(abs(desired_power), max_available_power)
            
            elec += device_power
            
            # Update energy
            new_energy = current_energy - device_power
            self.vehicle_energy[vehicle_id] = max(0.0, min(capacity, new_energy))
            idx += 1

        # Wash: on {0,1} consume (aligned with dispatch_node_devices)
        for k, (wash_id, dev) in enumerate(zip(self.wash_ids, self.wash_params)):
            on = 1.0 if action[idx] > 0.5 else 0.0
            if on:
                elec -= float(dev.get("rate_power", 0.0))
                # Update work state (increment by 1 hour)
                self.wash_work_state[wash_id] = self.wash_work_state.get(wash_id, 0) + 1
            idx += 1

        # AC: power frac [0,1] consume (aligned with dispatch_node_devices)
        for k, dev in enumerate(self.ac_params):
            frac = float(np.clip(action[idx], 0, 1))
            power_max = float(dev.get("power_max", 0.0))  # Use .get() for consistency with run_joint_episode
            elec -= frac * power_max
            idx += 1

        # Aligned with run_joint_episode: revenue = price * elec (penalty is calculated separately)
        reward = float(self.price[t]) * elec

        self.t += 1
        done = self.t >= self.T
        # Return device states for penalty calculation (aligned with run_joint_episode)
        return np.zeros(1), reward, done, False, {
            "elec": elec, 
            "penalty": penalty,  # This penalty is always 0.0 in SimpleVPPSimulator now
            "storage_energy": dict(self.storage_energy),  # For tracking SoC
            "vehicle_energy": dict(self.vehicle_energy),  # For tracking SoC
            "wash_work_state": dict(self.wash_work_state),  # For tracking work state
        }


def execute_strategy(env, day, strategy, level=10):
    """
    把 MILP 解 strategy 映射为 env.step(action)
    约定：elec 正=注入；负=消费
    
    Aligned with run_joint_episode:
    - Revenue: price * elec (not subtracting penalty)
    - Penalty is calculated separately (bidding + device)
    """
    revenue = 0.0  # Economic revenue = price * elec
    obs, _ = env.reset(seed=day)

    transitions = []
    elec_list = []
    price_list = []

    # Device states will be tracked by SimpleVPPSimulator and retrieved at end of day
    for t in range(24):
        # PV/Wind：curtail ratio [0,1] - 各只有一个值
        pv_action = [float(strategy["pv"][0]["cur"][t])] if len(strategy["pv"]) > 0 else [0.0]
        wind_action = [float(strategy["wind"][0]["cur"][t])] if len(strategy["wind"]) > 0 else [0.0]

        # Storage/Vehicle：net injection fraction in [-1,1]
        storage_action = [
            s["p"][t] / level
            for s in strategy["storage"]
        ]
        vehicle_action = [
            v["p"][t] / level
            for v in strategy["vehicle"]
        ]

        # Wash：on {0,1}
        wash_action = [float(w["p"][t]) for w in strategy["wash"]]

        # AC：power fraction [0,1]
        ac_action = [float(a["p"][t]) / level for a in strategy["ac"]]

        action = np.concatenate(
            [pv_action, wind_action, storage_action, vehicle_action, wash_action, ac_action],
            axis=0
        )

        next_obs, r, done, _, info = env.step(action)
        revenue += float(r)  # r = price * elec (aligned with run_joint_episode)

        transitions.append([obs, next_obs, action, r, done, info])
        obs = next_obs
        elec_list.append(float(info["elec"]))
        price_list.append(float(env.price[t]))

    # Get device states from simulator (aligned with run_joint_episode)
    # These states are tracked by SimpleVPPSimulator with proper constraints
    # After 24 steps, env.storage_energy, env.vehicle_energy, env.wash_work_state contain final states
    
    # Calculate storage SoC from energy (aligned with run_joint_episode)
    storage_soc = 0.5
    if hasattr(env, 'storage_energy') and env.storage_energy:
        total_energy = sum(env.storage_energy.values())
        total_capacity = sum(float(dev.get("capacity", 0.0)) for dev in env.storage_params)
        if total_capacity > 0:
            storage_soc = total_energy / total_capacity
    
    # Calculate vehicle SoC from energy (aligned with run_joint_episode)
    vehicle_soc = {}
    if hasattr(env, 'vehicle_energy') and env.vehicle_energy:
        for vehicle_id, dev in zip(env.vehicle_ids, env.vehicle_params):
            energy = env.vehicle_energy.get(vehicle_id, 0.0)
            capacity = float(dev.get("capacity", 0.0))
            if capacity > 0:
                vehicle_soc[vehicle_id] = energy / capacity
            else:
                vehicle_soc[vehicle_id] = 0.5
    
    # Get wash work_state (aligned with run_joint_episode)
    wash_work_state = {}
    if hasattr(env, 'wash_work_state') and env.wash_work_state:
        wash_work_state = dict(env.wash_work_state)

    return {
        "revenue": revenue,
        "transitions": transitions,
        "elec_list": elec_list,
        "price_list": price_list,
        "storage_soc": storage_soc,
        "vehicle_soc": vehicle_soc,
        "wash_work_state": wash_work_state,
    }


def execute_milp_strategy(env, day, strategy, device_dict, BASE_DIR, LEVEL, debug=False):
    """
    直接使用MILP的strategy格式执行，不转换为allocation
    这样可以完全保持MILP解的一致性
    
    strategy 格式（MILP返回的格式）：
      strategy["pv"] = [{"cur": [0.0-1.0] * 24}]
      strategy["wind"] = [{"cur": [0.0-1.0] * 24}]
      strategy["storage"] = [{"p": [-LEVEL, LEVEL] * 24}, ...]  # 每个设备一个
      strategy["vehicle"] = [{"p": [-LEVEL, LEVEL] * 24}, ...]  # 每个设备一个
      strategy["wash"] = [{"p": [0, 1] * 24}, ...]  # 每个设备一个
      strategy["ac"] = [{"p": [0, LEVEL] * 24}, ...]  # 每个设备一个
    """
    revenue = 0.0
    obs, _ = env.reset(seed=day)
    
    transitions = []
    elec_list = []
    price_list = []
    
    T = 24
    storage_ids = device_dict.get("storage_id", [])
    vehicle_ids = device_dict.get("vehicle_id", [])
    wash_ids = device_dict.get("wash_id", [])
    ac_ids = device_dict.get("AC_id", [])
    
    # 预加载设备配置
    storage_configs = {sid: load_device_yaml("storage", sid, BASE_DIR) for sid in storage_ids}
    vehicle_configs = {vid: load_device_yaml("vehicle", vid, BASE_DIR) for vid in vehicle_ids}
    wash_configs = {wid: load_device_yaml("wash", wid, BASE_DIR) for wid in wash_ids}
    ac_configs = {aid: load_device_yaml("AC", aid, BASE_DIR) for aid in ac_ids}
    
    for t in range(T):
        elec = 0.0
        
        # PV: 使用MILP的curtail ratio
        if "pv" in strategy and len(strategy["pv"]) > 0:
            pv_cur = strategy["pv"][0]["cur"][t]
            pv_gen_total = 0.0
            if hasattr(env, 'pv_curve') and t < env.T:
                if len(env.pv_curve.shape) > 1:
                    pv_gen_total = np.sum(env.pv_curve[:, t])
                else:
                    pv_gen_total = env.pv_curve[t] if t < len(env.pv_curve) else 0.0
            elec += (1.0 - pv_cur) * pv_gen_total
        
        # Wind: 使用MILP的curtail ratio
        if "wind" in strategy and len(strategy["wind"]) > 0:
            wind_cur = strategy["wind"][0]["cur"][t]
            wind_gen_total = 0.0
            if hasattr(env, 'wind_curve') and t < env.T:
                if len(env.wind_curve.shape) > 1:
                    wind_gen_total = np.sum(env.wind_curve[:, t])
                else:
                    wind_gen_total = env.wind_curve[t] if t < len(env.wind_curve) else 0.0
            elec += (1.0 - wind_cur) * wind_gen_total
        
        # Storage: 直接使用MILP的每个设备的p值
        if "storage" in strategy:
            for idx, storage_strat in enumerate(strategy["storage"]):
                if idx < len(storage_ids):
                    storage_id = storage_ids[idx]
                    dev_config = storage_configs[storage_id]
                    capacity = float(dev_config.get("capacity", 0.0))
                    max_power = float(dev_config.get("max_power", 0.0))
                    current_energy = env.storage_energy.get(storage_id, 0.0)
                    
                    # MILP的p值是 dis - ch，范围是[-LEVEL, LEVEL]
                    p_value = float(storage_strat["p"][t])
                    # 转换为实际功率：power = p * (max_power / LEVEL)
                    desired_power = p_value * (max_power / float(LEVEL))
                    
                    # 应用约束
                    if abs(desired_power) < 1e-6:  # 接近0，不充不放
                        device_power = 0.0
                    elif desired_power > 0:  # Discharge
                        max_available_power = min(max_power, current_energy)
                        device_power = min(desired_power, max_available_power)
                    else:  # Charge
                        remaining_capacity = capacity - current_energy
                        max_available_power = min(max_power, remaining_capacity)
                        device_power = -min(abs(desired_power), max_available_power)
                    
                    elec += device_power
                    new_energy = current_energy - device_power
                    env.storage_energy[storage_id] = max(0.0, min(capacity, new_energy))
        
        # Vehicle: 直接使用MILP的每个设备的p值
        if "vehicle" in strategy:
            for idx, vehicle_strat in enumerate(strategy["vehicle"]):
                if idx < len(vehicle_ids):
                    vehicle_id = vehicle_ids[idx]
                    dev_config = vehicle_configs[vehicle_id]
                    capacity = float(dev_config.get("capacity", 0.0))
                    max_power = float(dev_config.get("max_power", 0.0))
                    current_energy = env.vehicle_energy.get(vehicle_id, 0.0)
                    
                    p_value = float(vehicle_strat["p"][t])
                    desired_power = p_value * (max_power / float(LEVEL))
                    
                    if abs(desired_power) < 1e-6:  # 接近0，不充不放
                        device_power = 0.0
                    elif desired_power > 0:  # Discharge
                        max_available_power = min(max_power, current_energy)
                        device_power = min(desired_power, max_available_power)
                    else:  # Charge
                        remaining_capacity = capacity - current_energy
                        max_available_power = min(max_power, remaining_capacity)
                        device_power = -min(abs(desired_power), max_available_power)
                    
                    elec += device_power
                    new_energy = current_energy - device_power
                    env.vehicle_energy[vehicle_id] = max(0.0, min(capacity, new_energy))
        
        # Wash: 直接使用MILP的每个设备的on值
        if "wash" in strategy:
            for idx, wash_strat in enumerate(strategy["wash"]):
                if idx < len(wash_ids):
                    wash_id = wash_ids[idx]
                    dev_config = wash_configs[wash_id]
                    wash_on = float(wash_strat["p"][t]) > 0.5  # MILP的on是0或1
                    if wash_on:
                        wash_power = float(dev_config.get("rate_power", 0.0))
                        elec -= wash_power
                        env.wash_work_state[wash_id] = env.wash_work_state.get(wash_id, 0) + 1
        
        # AC: 直接使用MILP的每个设备的p值
        if "ac" in strategy:
            for idx, ac_strat in enumerate(strategy["ac"]):
                if idx < len(ac_ids):
                    ac_id = ac_ids[idx]
                    dev_config = ac_configs[ac_id]
                    # MILP的ac p值范围是[0, LEVEL]
                    ac_p_value = float(ac_strat["p"][t])
                    power_max = float(dev_config.get("power_max", 0.0))
                    ac_power = (ac_p_value / float(LEVEL)) * power_max
                    elec -= ac_power
        
        # 计算 revenue
        r = float(env.price[t]) * elec
        revenue += r
        
        elec_list.append(elec)
        price_list.append(float(env.price[t]))
        transitions.append([obs, obs, None, r, False, {"elec": elec}])
    
    # Calculate device states
    total_storage_capacity = sum(float(cfg.get("capacity", 0.0)) for cfg in storage_configs.values())
    storage_soc = 0.5
    if env.storage_energy and total_storage_capacity > 0:
        total_energy = sum(env.storage_energy.values())
        storage_soc = total_energy / total_storage_capacity
    
    vehicle_soc = {}
    if env.vehicle_energy:
        for vehicle_id in vehicle_ids:
            energy = env.vehicle_energy.get(vehicle_id, 0.0)
            capacity = float(vehicle_configs[vehicle_id].get("capacity", 0.0))
            if capacity > 0:
                vehicle_soc[vehicle_id] = energy / capacity
            else:
                vehicle_soc[vehicle_id] = 0.5
    
    wash_work_state = dict(env.wash_work_state) if env.wash_work_state else {}
    
    return {
        "revenue": revenue,
        "transitions": transitions,
        "elec_list": elec_list,
        "price_list": price_list,
        "storage_soc": storage_soc,
        "vehicle_soc": vehicle_soc,
        "wash_work_state": wash_work_state,
    }


def execute_allocation(env, day, allocation, device_dict, BASE_DIR):
    """
    [Deprecated]
    历史遗留：ratio/aggregation 版 GA 使用的 allocation 执行器。
    当前 run_genetic.py 已改为逐设备策略（MILP 对齐）编码与执行，默认不再使用该函数。
    """
    raise RuntimeError(
        "execute_allocation is deprecated: GA now uses per-device MILP-aligned strategy execution. "
        "Use execute_milp_strategy() with strategy_by_node instead."
    )

import numpy as np
import random

# =========================
# GA helpers: build genome spec per node
# =========================
def _node_spec(node_devices):
    """
    返回一个 node 的“设备数量规格”，用于编码/解码
    """
    return {
        "n_pv": len(node_devices.get("pv_id", [])),
        "n_wind": len(node_devices.get("wind_id", [])),
        "n_storage": len(node_devices.get("storage_id", [])),
        "n_vehicle": len(node_devices.get("vehicle_id", [])),
        "n_wash": len(node_devices.get("wash_id", [])),
        "n_ac": len(node_devices.get("AC_id", [])),
    }

def build_global_spec(node_list, node_device_mapping, T):
    """
    为所有节点构造全局编码布局 offsets
    genome 是 1D array，按节点依次拼接（与 MILP 的 strategy_by_node 完全对齐，逐设备编码）：
      pv_cur   : 1*T                    (int in [0, LEVEL]) 0=不弃，LEVEL=全弃
      wind_cur : 1*T                    (int in [0, LEVEL])
      storage_p: n_storage*T            (int in [-LEVEL, LEVEL]) 逐设备 p_steps = dis - ch
      vehicle_p: n_vehicle*T            (int in [-LEVEL, LEVEL]) 逐设备 p_steps = dis - ch
      wash_on  : n_wash*T               (int in {0,1}) 逐设备 on
      ac_p     : n_ac*T                 (int in [0, LEVEL]) 逐设备用电档位
    """
    spec_by_node = {}
    offsets = {}
    cur = 0
    for node in node_list:
        s = _node_spec(node_device_mapping[node])
        spec_by_node[node] = s

        def alloc(name, length):
            nonlocal cur
            offsets[(node, name)] = (cur, cur + length)
            cur += length

        # 对齐 MILP 输出：PV/Wind 每节点 1 条；其余逐设备
        alloc("pv_cur", 1 * T)
        alloc("wind_cur", 1 * T)
        alloc("storage_p", s["n_storage"] * T)
        alloc("vehicle_p", s["n_vehicle"] * T)
        alloc("wash_on", s["n_wash"] * T)
        alloc("ac_p", s["n_ac"] * T)

    total_len = cur
    return spec_by_node, offsets, total_len

def random_individual(total_len, node_list, spec_by_node, offsets, T, LEVEL):
    """
    随机生成一个 genome（1D np.int16）
    逐设备编码，与 MILP strategy_by_node 对齐
    """
    g = np.zeros(total_len, dtype=np.int16)

    for node in node_list:
        s = spec_by_node[node]

        # pv_cur: 0..LEVEL
        a, b = offsets[(node, "pv_cur")]
        if b > a:
            g[a:b] = np.random.randint(0, LEVEL + 1, size=(b - a), dtype=np.int16)

        # wind_cur: 0..LEVEL
        a, b = offsets[(node, "wind_cur")]
        if b > a:
            g[a:b] = np.random.randint(0, LEVEL + 1, size=(b - a), dtype=np.int16)

        # storage_p: -LEVEL..LEVEL (逐设备)
        a, b = offsets[(node, "storage_p")]
        if b > a:
            g[a:b] = np.random.randint(-LEVEL, LEVEL + 1, size=(b - a), dtype=np.int16)

        # vehicle_p: -LEVEL..LEVEL (逐设备)
        a, b = offsets[(node, "vehicle_p")]
        if b > a:
            g[a:b] = np.random.randint(-LEVEL, LEVEL + 1, size=(b - a), dtype=np.int16)

        # wash_on: 0/1 (逐设备)
        a, b = offsets[(node, "wash_on")]
        if b > a:
            g[a:b] = np.random.randint(0, 2, size=(b - a), dtype=np.int16)

        # ac_p: 0..LEVEL (逐设备)
        a, b = offsets[(node, "ac_p")]
        if b > a:
            g[a:b] = np.random.randint(0, LEVEL + 1, size=(b - a), dtype=np.int16)

    return g

def decode_to_strategy_by_node(genome, node_list, spec_by_node, offsets, T, LEVEL):
    """
    将 genome 解码为 MILP 同款格式的 strategy_by_node：
      strategy[node]["pv"] = [{"cur": [0..1]*T}]
      strategy[node]["wind"] = [{"cur": [0..1]*T}]
      strategy[node]["storage"] = [{"p": [-LEVEL..LEVEL]*T} for each storage]
      strategy[node]["vehicle"] = [{"p": [-LEVEL..LEVEL]*T} for each vehicle]
      strategy[node]["wash"] = [{"p": [0/1]*T} for each wash]
      strategy[node]["ac"] = [{"p": [0..LEVEL]*T} for each ac]
    """
    out = {}
    for node in node_list:
        s = spec_by_node[node]
        strategy = {"pv": [], "wind": [], "storage": [], "vehicle": [], "wash": [], "ac": []}

        # pv/wind curtail: int[0..LEVEL] -> float[0..1]
        a, b = offsets[(node, "pv_cur")]
        pv_steps = genome[a:b].astype(float)
        strategy["pv"].append({"cur": (pv_steps / float(LEVEL)).tolist()})

        a, b = offsets[(node, "wind_cur")]
        wind_steps = genome[a:b].astype(float)
        strategy["wind"].append({"cur": (wind_steps / float(LEVEL)).tolist()})

        # storage p: flatten -> per device
        a, b = offsets[(node, "storage_p")]
        st_flat = genome[a:b].astype(int)
        for k in range(s["n_storage"]):
            p = st_flat[k * T:(k + 1) * T]
            strategy["storage"].append({"p": p.tolist()})

        # vehicle p
        a, b = offsets[(node, "vehicle_p")]
        ev_flat = genome[a:b].astype(int)
        for k in range(s["n_vehicle"]):
            p = ev_flat[k * T:(k + 1) * T]
            strategy["vehicle"].append({"p": p.tolist()})

        # wash on: 0/1
        a, b = offsets[(node, "wash_on")]
        wa_flat = genome[a:b].astype(int)
        wa_flat = np.clip(wa_flat, 0, 1)
        for k in range(s["n_wash"]):
            p = wa_flat[k * T:(k + 1) * T]
            strategy["wash"].append({"p": p.tolist()})

        # ac p: 0..LEVEL
        a, b = offsets[(node, "ac_p")]
        ac_flat = genome[a:b].astype(int)
        ac_flat = np.clip(ac_flat, 0, LEVEL)
        for k in range(s["n_ac"]):
            p = ac_flat[k * T:(k + 1) * T]
            strategy["ac"].append({"p": p.tolist()})

        out[node] = strategy
    return out

def decode_to_allocation_by_node(genome, node_list, spec_by_node, offsets, T, LEVEL):
    """
    [Deprecated]
    历史遗留：ratio/aggregation 版 genome 的解码器。
    当前 run_genetic.py 已改为逐设备策略（MILP 对齐）编码与解码，默认不再使用该函数。
    """
    raise RuntimeError(
        "decode_to_allocation_by_node is deprecated: GA genome is now per-device (MILP-aligned). "
        "Use decode_to_strategy_by_node() instead."
    )


def milp_strategy_to_genome(strategy_by_node, node_list, spec_by_node, offsets, T, LEVEL, BASE_DIR, node_device_mapping):
    """
    将MILP的strategy_by_node转换为genome格式
    
    Args:
        strategy_by_node: MILP返回的strategy字典，格式为：
            {
                node: {
                    "pv": [{"cur": [0.0-1.0] * 24}],
                    "wind": [{"cur": [0.0-1.0] * 24}],
                    "storage": [{"p": [-LEVEL, LEVEL] * 24}],  # 每个设备一个
                    "vehicle": [{"p": [-LEVEL, LEVEL] * 24}],  # 每个设备一个
                    "wash": [{"p": [0, 1] * 24}],  # 每个设备一个
                    "ac": [{"p": [0, LEVEL] * 24}],  # 每个设备一个
                }
            }
        node_list: 节点列表
        spec_by_node: 节点规格字典
        offsets: genome偏移量字典
        T: 时间步数（24）
        LEVEL: 离散化等级
        BASE_DIR: 设备配置目录
        node_device_mapping: 节点设备映射字典，用于获取设备配置
    
    Returns:
        genome: 1D np.int16数组
    """
    # Allocate genome
    total_len = max(b for (_, _), (a, b) in offsets.items()) if len(offsets) > 0 else 0
    genome = np.zeros(total_len, dtype=np.int16)
    
    for node in node_list:
        if node not in strategy_by_node:
            continue
        
        strategy = strategy_by_node[node]
        s = spec_by_node[node]
        
        # PV/Wind: [0..1] -> [0..LEVEL]
        if "pv" in strategy and len(strategy["pv"]) > 0:
            a, b = offsets[(node, "pv_cur")]
            pv_values = np.clip(np.round(np.array(strategy["pv"][0]["cur"], dtype=float) * LEVEL), 0, LEVEL)
            genome[a:b] = pv_values.astype(np.int16)

        if "wind" in strategy and len(strategy["wind"]) > 0:
            a, b = offsets[(node, "wind_cur")]
            wind_values = np.clip(np.round(np.array(strategy["wind"][0]["cur"], dtype=float) * LEVEL), 0, LEVEL)
            genome[a:b] = wind_values.astype(np.int16)

        # Storage: 逐设备 p_steps 直接写入
        a, b = offsets[(node, "storage_p")]
        st_flat = np.zeros(s["n_storage"] * T, dtype=np.int16)
        if "storage" in strategy:
            for k in range(min(len(strategy["storage"]), s["n_storage"])):
                p = np.array(strategy["storage"][k]["p"], dtype=float)
                p = np.clip(np.round(p), -LEVEL, LEVEL).astype(np.int16)
                st_flat[k * T:(k + 1) * T] = p[:T]
        genome[a:b] = st_flat

        # Vehicle
        a, b = offsets[(node, "vehicle_p")]
        ev_flat = np.zeros(s["n_vehicle"] * T, dtype=np.int16)
        if "vehicle" in strategy:
            for k in range(min(len(strategy["vehicle"]), s["n_vehicle"])):
                p = np.array(strategy["vehicle"][k]["p"], dtype=float)
                p = np.clip(np.round(p), -LEVEL, LEVEL).astype(np.int16)
                ev_flat[k * T:(k + 1) * T] = p[:T]
        genome[a:b] = ev_flat

        # Wash: 逐设备 on 0/1
        a, b = offsets[(node, "wash_on")]
        wa_flat = np.zeros(s["n_wash"] * T, dtype=np.int16)
        if "wash" in strategy:
            for k in range(min(len(strategy["wash"]), s["n_wash"])):
                p = np.array(strategy["wash"][k]["p"], dtype=float)
                p = (p[:T] > 0.5).astype(np.int16)
                wa_flat[k * T:(k + 1) * T] = p
        genome[a:b] = wa_flat

        # AC: 逐设备 p in [0..LEVEL]
        a, b = offsets[(node, "ac_p")]
        ac_flat = np.zeros(s["n_ac"] * T, dtype=np.int16)
        if "ac" in strategy:
            for k in range(min(len(strategy["ac"]), s["n_ac"])):
                p = np.array(strategy["ac"][k]["p"], dtype=float)
                p = np.clip(np.round(p), 0, LEVEL).astype(np.int16)
                ac_flat[k * T:(k + 1) * T] = p[:T]
        genome[a:b] = ac_flat
    
    return genome


# =========================
# GA operators
# =========================
def tournament_select(pop, fitness, k=3):
    """Tournament selection with k competitors"""
    idxs = random.sample(range(len(pop)), k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitness[i] > fitness[best]:
            best = i
    return pop[best].copy()

def rank_selection(pop, fitness, k=2):
    """Rank-based selection: better fitness = higher selection probability"""
    # Sort by fitness (descending)
    sorted_indices = np.argsort(fitness)[::-1]
    # Assign ranks (higher fitness = higher rank)
    ranks = np.arange(len(pop), 0, -1, dtype=float)
    # Linear rank selection: probability proportional to rank
    probs = ranks / ranks.sum()
    selected = np.random.choice(len(pop), size=k, p=probs, replace=False)
    return [pop[sorted_indices[i]].copy() for i in selected]

def uniform_crossover(a, b, p=0.5):
    mask = np.random.rand(a.size) < p
    c1 = a.copy()
    c2 = b.copy()
    c1[mask] = b[mask]
    c2[mask] = a[mask]
    return c1, c2

def mutate(genome, node_list, spec_by_node, offsets, T, LEVEL, p_gene=0.002, adaptive_rate=None):
    """
    逐基因小概率突变：把该位重采样到合法区间
    
    Args:
        adaptive_rate: 如果提供，使用自适应变异率（在早期使用更大变异率）
    """
    g = genome.copy()
    # Adaptive mutation rate: higher in early stages to explore more
    actual_p_gene = p_gene * (adaptive_rate if adaptive_rate is not None else 1.0)
    
    for node in node_list:
        s = spec_by_node[node]

        def mut_block(name, lo, hi):
            a, b = offsets[(node, name)]
            if b <= a:
                return
            m = np.random.rand(b - a) < actual_p_gene
            if not m.any():
                return
            # 重采样
            g[a:b][m] = np.random.randint(lo, hi + 1, size=int(m.sum()), dtype=np.int16)

        mut_block("pv_cur", 0, LEVEL)
        mut_block("wind_cur", 0, LEVEL)
        mut_block("storage_p", -LEVEL, LEVEL)
        mut_block("vehicle_p", -LEVEL, LEVEL)

        # wash_on: 0/1
        a, b = offsets.get((node, "wash_on"), (0, 0))
        if b > a:
            m = np.random.rand(b - a) < actual_p_gene
            if m.any():
                g[a:b][m] = np.random.randint(0, 2, size=int(m.sum()), dtype=np.int16)

        mut_block("ac_p", 0, LEVEL)

    return g


# =========================
# Fitness evaluation: run sim on predicted, enforce total bid
# =========================
def eval_fitness_one_day(
    genome,
    day,
    node_list,
    node_device_mapping,
    price_day_dict,
    pv_day_list,      # shape (N_nodes, 24) list-like
    wind_day_list,    # shape (N_nodes, 24)
    total_bid_day,    # length 24
    LEVEL,
    BASE_DIR,
    spec_by_node,
    offsets,
    bid_penalty=1000.0,      # 偏离总bid的惩罚（元/电量单位）
    op_penalty_weight=1.0,   # simulator penalty 权重 (deprecated, kept for compatibility)
    # Evaluation config (aligned with run_joint_episode)
    bidding_penalty=1.0,
    bidding_ratio=0.2,
    device_penalty_effi=1.0,
    normalize_ratio=1.0,
    is_milp_solution=False,  # Flag to indicate if this is MILP solution for detailed debugging
    milp_strategy_by_node=None,  # MILP strategy dict for direct execution (if available)
    milp_total_elec_plan=None,  # MILP optimizer's internal elec values for deviation calculation (aligned with MILP-PRED)
    milp_elec_by_node=None,  # MILP optimizer's internal elec values per node for revenue calculation (aligned with MILP-PRED)
    milp_pred_revenue=None,  # MILP predicted revenue (aligned with MILP-PRED)
):
    # If MILP strategy is provided, use it directly; otherwise decode genome to per-device strategy
    use_milp_direct = (milp_strategy_by_node is not None and is_milp_solution)
    if use_milp_direct:
        strategy_by_node = milp_strategy_by_node
    else:
        strategy_by_node = decode_to_strategy_by_node(
            genome, node_list, spec_by_node, offsets, T=24, LEVEL=LEVEL
        )

    T = 24
    total_revenue = 0.0  # Economic revenue = price * elec (aligned with run_joint_episode)
    total_penalty_device = 0.0
    total_penalty_bid = 0.0
    total_elec = np.zeros(T, dtype=float)
    node_device_states = {}  # Store device states for each node
    # Store per-node per-time-step revenue for debugging
    node_revenue_per_time = {node: np.zeros(T, dtype=float) for node in node_list}

    # For MILP solution, use MILP optimizer's internal elec values for revenue calculation
    # This ensures consistency with MILP-PRED revenue
    if is_milp_solution and milp_pred_revenue is not None and milp_elec_by_node is not None:
        # Use MILP predicted revenue directly (aligned with MILP-PRED)
        total_revenue = milp_pred_revenue
        
        # Calculate per-node revenue for node_revenue_per_time (for consistency)
        for node in node_list:
            if node in milp_elec_by_node:
                for t in range(T):
                    elec_nt = milp_elec_by_node[node][t]
                    price_nt = price_day_dict[node][t]
                    node_revenue_per_time[node][t] = price_nt * elec_nt
        
        # For device states, we still need to execute the strategy
        for idx, node in enumerate(node_list):
            env_sim = SimpleVPPSimulator(
                price=price_day_dict[node],
                pv=pv_day_list[idx],
                wind=wind_day_list[idx],
                device_dict=node_device_mapping[node],
                level=LEVEL,
                base_dir=BASE_DIR,
            )
            result = execute_milp_strategy(
                env_sim, day=day, strategy=strategy_by_node[node],
                device_dict=node_device_mapping[node], BASE_DIR=BASE_DIR, LEVEL=LEVEL,
                debug=False
            )
            # Store device states for penalty calculation
            node_device_states[node] = {
                "storage_soc": result["storage_soc"],
                "vehicle_soc": result["vehicle_soc"],
                "wash_work_state": result["wash_work_state"],
                "device_dict": node_device_mapping[node],
            }
        
        # Use MILP optimizer's internal elec values
        total_elec = milp_total_elec_plan.copy()
        
        # For MILP solution, skip revenue_adjustment (aligned with MILP-PRED which doesn't have penalty adjustment)
        # But we still need to calculate penalty_bid for reporting
        for t in range(T):
            elec_quan = total_elec[t]  # Total electricity at time t (from MILP optimizer)
            bid = total_bid_day[t]  # Bidding value at time t
            
            # Calculate threshold (aligned with VPP_dispatching logic)
            if abs(bid) < 50:
                thresh = 10
            else:
                thresh = abs(bidding_ratio * bid)
            
            # Calculate penalty_bid (but don't adjust revenue for MILP solution)
            if elec_quan - bid > thresh:
                # Exceeded upper threshold
                avg_price = np.mean([price_day_dict[n][t] for n in node_list])
                penalty_bid_t = (elec_quan - bid - thresh) * avg_price / normalize_ratio * bidding_penalty
            elif elec_quan - bid < -thresh:
                # Below lower threshold
                avg_price = np.mean([price_day_dict[n][t] for n in node_list])
                penalty_bid_t = (bid - elec_quan - thresh) * avg_price / normalize_ratio * (1 + bidding_penalty)
            else:
                # Within threshold
                penalty_bid_t = 0.0
            
            total_penalty_bid += penalty_bid_t
    else:
        # For non-MILP solutions, use execute_milp_strategy results
        for idx, node in enumerate(node_list):
            env_sim = SimpleVPPSimulator(
                price=price_day_dict[node],
                pv=pv_day_list[idx],
                wind=wind_day_list[idx],
                device_dict=node_device_mapping[node],
                level=LEVEL,
                base_dir=BASE_DIR,
            )
            # Always execute per-device strategy (MILP-aligned)
            result = execute_milp_strategy(
                env_sim, day=day, strategy=strategy_by_node[node],
                device_dict=node_device_mapping[node], BASE_DIR=BASE_DIR, LEVEL=LEVEL,
                debug=False
            )
            # Calculate revenue per time step (aligned with run_joint_episode: revenue = price * elec per time step)
            for t in range(T):
                elec_nt = result["elec_list"][t]
                price_nt = price_day_dict[node][t]
                revenue_nt = price_nt * elec_nt
                total_revenue += revenue_nt
                node_revenue_per_time[node][t] = revenue_nt
            
            total_elec += np.array(result["elec_list"], dtype=float)
            # Store device states for penalty calculation
            node_device_states[node] = {
                "storage_soc": result["storage_soc"],
                "vehicle_soc": result["vehicle_soc"],
                "wash_work_state": result["wash_work_state"],
                "device_dict": node_device_mapping[node],
            }

        # Calculate bidding deviation penalty for each time step (aligned with run_joint_episode)
        # Only for non-MILP solutions (MILP solutions already calculated above)
        for t in range(T):
            elec_quan = total_elec[t]  # Total electricity at time t
            bid = total_bid_day[t]  # Bidding value at time t
            
            # Calculate threshold (aligned with VPP_dispatching logic)
            if abs(bid) < 50:
                thresh = 10
            else:
                thresh = abs(bidding_ratio * bid)
            
            # Calculate elec_clip and penalty_bid (aligned with VPP_dispatching)
            if elec_quan - bid > thresh:
                # Exceeded upper threshold
                avg_price = np.mean([price_day_dict[n][t] for n in node_list])
                penalty_bid_t = (elec_quan - bid - thresh) * avg_price / normalize_ratio * bidding_penalty
                elec_clip = bid + thresh
            elif elec_quan - bid < -thresh:
                # Below lower threshold
                avg_price = np.mean([price_day_dict[n][t] for n in node_list])
                penalty_bid_t = (bid - elec_quan - thresh) * avg_price / normalize_ratio * (1 + bidding_penalty)
                elec_clip = bid - thresh
            else:
                # Within threshold
                penalty_bid_t = 0.0
                elec_clip = elec_quan
            
            # Update revenue with clipped electricity (aligned with VPP_dispatching)
            avg_price = np.mean([price_day_dict[n][t] for n in node_list])
            revenue_adjustment = (elec_clip - elec_quan) * avg_price
            total_revenue += revenue_adjustment
            total_penalty_bid += penalty_bid_t

    # Calculate device operation penalty at end of day (aligned with run_joint_episode)
    init_soc = 0.5  # Default initial SoC
    for node in node_list:
        states = node_device_states[node]
        
        # Storage penalty: if SoC < init_SoC at end of day
        current_soc = states["storage_soc"]
        if current_soc < init_soc:
            storage_penalty = 0.1 * (init_soc - current_soc)
            total_penalty_device += storage_penalty
        
        # Vehicle penalty: if SoC < capacity_demand at end of day
        vehicle_ids = states["device_dict"].get("vehicle_id", [])
        for vehicle_id in vehicle_ids:
            try:
                dev_config = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
                capacity_demand = float(dev_config.get("capacity_demand", 80.0))
                capacity_max = float(dev_config.get("capacity", 100.0))
                vehicle_soc = states["vehicle_soc"].get(vehicle_id, 0.5)
                if vehicle_soc < (capacity_demand / capacity_max):
                    vehicle_penalty = 0.2 * (capacity_demand / capacity_max - vehicle_soc) * capacity_max
                    total_penalty_device += vehicle_penalty
            except:
                pass
        
        # Wash penalty: if work_state < t_dur at end of day
        wash_ids = states["device_dict"].get("wash_id", [])
        for wash_id in wash_ids:
            try:
                dev_config = load_device_yaml("wash", wash_id, BASE_DIR)
                t_dur = int(dev_config.get("t_dur", 2))
                wash_work_state = states["wash_work_state"].get(wash_id, 0)
                if wash_work_state < t_dur:
                    wash_penalty = 5.0 * (t_dur - wash_work_state)
                    total_penalty_device += wash_penalty
            except:
                pass

    # Calculate total penalty (but don't subtract from revenue for fitness)
    total_penalty = device_penalty_effi * total_penalty_device + total_penalty_bid
    # Note: final_revenue is calculated but not used for fitness (user requested)
    final_revenue = total_revenue - total_penalty

    # For MILP solution, use MILP optimizer's internal elec values for deviation calculation
    # This ensures consistency with MILP-PRED deviation
    if is_milp_solution and milp_total_elec_plan is not None:
        # Use MILP optimizer's internal elec values (aligned with MILP-PRED)
        dev_l1 = float(np.abs(milp_total_elec_plan - np.array(total_bid_day, dtype=float)).sum())
    else:
        # Use executed elec values (from execute_milp_strategy)
        dev_l1 = float(np.abs(total_elec - np.array(total_bid_day, dtype=float)).sum())

    # Fitness: maximize total_revenue (penalty is NOT subtracted from revenue)
    fitness = total_revenue
    return fitness, total_revenue, total_penalty, dev_l1


# =========================
# Main GA loop: optimize on predicted, then evaluate on real
# =========================
def run_ga_for_day(
    day,
    node_list,
    node_device_mapping,
    price, bid,
    pv_pred, wind_pred,
    pv_real, wind_real,
    LEVEL=10,
    BASE_DIR="/data2/zengjinwei/VPP_multinode/config/device",
    pop_size=80,
    n_gen=120,
    cx_prob=0.8,
    mut_prob=0.9,
    p_gene=0.002,
    bid_penalty=1000.0,
    op_penalty_weight=1.0,
    seed=0,
):
    random.seed(seed + day)
    np.random.seed(seed + day)

    T = 24
    spec_by_node, offsets, total_len = build_global_spec(node_list, node_device_mapping, T)

    # inputs for the day
    price_day = {node: price[i, day, :].tolist() for i, node in enumerate(node_list)}
    total_bid_day = bid[day, :].tolist()

    pv_day_pred = pv_pred[:, day, :].tolist()
    wind_day_pred = wind_pred[:, day, :].tolist()

    pv_day_real = pv_real[:, day, :].tolist()
    wind_day_real = wind_real[:, day, :].tolist()

    # Skip MILP solution - use completely random initialization
    milp_genome = None
    milp_pred_revenue = None
    milp_strategy_by_node = None
    milp_total_elec_plan = None
    milp_elec_by_node = None
    
    print(f"[GA day={day+1}] Using completely random initialization (no MILP)")

    # init population - completely random
    pop = []
    # Generate all individuals randomly for maximum diversity
    pop = [
        random_individual(total_len, node_list, spec_by_node, offsets, T, LEVEL)
        for _ in range(pop_size)
    ]
    
    print(f"[GA day={day+1}] Generated {pop_size} random individuals as initial population")

    # evaluate (pred)
    fit = np.zeros(pop_size, dtype=float)
    rev_list = np.zeros(pop_size, dtype=float)
    pen_list = np.zeros(pop_size, dtype=float)
    dev_list = np.zeros(pop_size, dtype=float)
    
    for i in range(pop_size):
        # All individuals are random, no MILP solution
        f, rev, pen, dev = eval_fitness_one_day(
            pop[i], day,
            node_list, node_device_mapping,
            price_day, pv_day_pred, wind_day_pred,
            total_bid_day,
            LEVEL, BASE_DIR,
            spec_by_node, offsets,
            bid_penalty=bid_penalty,  # Keep for compatibility, but not used in new logic
            op_penalty_weight=op_penalty_weight,  # Keep for compatibility, but not used in new logic
            bidding_penalty=bidding_penalty_config,
            bidding_ratio=bidding_ratio_config,
            device_penalty_effi=device_penalty_effi_config,
            normalize_ratio=normalize_ratio_config,
            is_milp_solution=False,  # No MILP solution
            milp_strategy_by_node=None,
            milp_total_elec_plan=None,
            milp_elec_by_node=None,
            milp_pred_revenue=None,
        )
        fit[i] = f
        rev_list[i] = rev
        pen_list[i] = pen
        dev_list[i] = dev

    # Print initial population evaluation results
    print(f"[GA day={day+1}] Initial population evaluation (gen 000):")
    
    # All individuals are random
    print(f"  [Random] count={pop_size}, "
          f"fitness: mean={np.mean(fit):.3f}, best={np.max(fit):.3f}, worst={np.min(fit):.3f}")
    print(f"          revenue: mean={np.mean(rev_list):.3f}, best={np.max(rev_list):.3f}, worst={np.min(rev_list):.3f}")
    print(f"          penalty: mean={np.mean(pen_list):.3f}, best={np.min(pen_list):.3f}, worst={np.max(pen_list):.3f}")
    print(f"          deviation: mean={np.mean(dev_list):.3f}, best={np.min(dev_list):.3f}, worst={np.max(dev_list):.3f}")
    
    # Overall initial population statistics
    print(f"  [Overall] best fitness={np.max(fit):.3f}, mean fitness={np.mean(fit):.3f}, "
          f"best revenue={np.max(rev_list):.3f}, mean revenue={np.mean(rev_list):.3f}")

    best_idx = int(np.argmax(fit))
    best_g = pop[best_idx].copy()
    best_f = float(fit[best_idx])

    # GA iterations with adaptive mutation and early stopping
    no_improve_count = 0
    early_stop_patience = max(50, n_gen // 10)  # Stop if no improvement for 10% of generations
    
    for gen in range(n_gen):
        new_pop = []

        # elitism: keep only best 1 individual to avoid premature convergence
        new_pop.append(best_g.copy())

        # Adaptive mutation rate: much higher in early generations for better exploration
        # Early (0-40%): 5x mutation rate, Mid (40-70%): 3x, Late (70-100%): 1.5x
        gen_ratio = gen / n_gen
        if gen_ratio < 0.4:
            adaptive_rate = 5.0  # Much higher exploration early
        elif gen_ratio < 0.7:
            adaptive_rate = 3.0
        else:
            adaptive_rate = 1.5  # Still maintain some exploration late

        while len(new_pop) < pop_size:
            # Use rank selection for better parents (alternating with tournament)
            if gen % 2 == 0:
                p1, p2 = rank_selection(pop, fit, k=2)
            else:
                p1 = tournament_select(pop, fit, k=3)
                p2 = tournament_select(pop, fit, k=3)

            # crossover with higher probability for more exploration
            if random.random() < cx_prob:
                # Use more aggressive crossover (higher mixing probability)
                c1, c2 = uniform_crossover(p1, p2, p=0.6)  # Increased from 0.5 to 0.6
            else:
                c1, c2 = p1, p2

            # mutation with adaptive rate - always mutate for better exploration
            # Higher mutation probability in early generations
            mut_prob_actual = mut_prob if gen_ratio > 0.3 else 1.0  # Always mutate in early generations
            if random.random() < mut_prob_actual:
                c1 = mutate(c1, node_list, spec_by_node, offsets, T, LEVEL, p_gene=p_gene, adaptive_rate=adaptive_rate)
            if random.random() < mut_prob_actual:
                c2 = mutate(c2, node_list, spec_by_node, offsets, T, LEVEL, p_gene=p_gene, adaptive_rate=adaptive_rate)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

        # evaluate new pop (pred)
        for i in range(pop_size):
            f, _, _, _ = eval_fitness_one_day(
                pop[i], day,
                node_list, node_device_mapping,
                price_day, pv_day_pred, wind_day_pred,
                total_bid_day,
                LEVEL, BASE_DIR,
                spec_by_node, offsets,
                bid_penalty=bid_penalty,  # Keep for compatibility, but not used in new logic
                op_penalty_weight=op_penalty_weight,  # Keep for compatibility, but not used in new logic
                bidding_penalty=bidding_penalty_config,
                bidding_ratio=bidding_ratio_config,
                device_penalty_effi=device_penalty_effi_config,
                normalize_ratio=normalize_ratio_config,
                is_milp_solution=False,  # Not MILP solution in GA iterations
                milp_strategy_by_node=None,
                milp_total_elec_plan=None,  # Not MILP solution, use executed elec values
                milp_elec_by_node=None,
                milp_pred_revenue=None,
            )
            fit[i] = f

        cur_best_idx = int(np.argmax(fit))
        cur_best_f = float(fit[cur_best_idx])
        # Use a small epsilon for floating point comparison to avoid precision issues
        improvement_threshold = 1e-6
        if cur_best_f > best_f + improvement_threshold:
            improvement = cur_best_f - best_f
            best_f = cur_best_f
            best_g = pop[cur_best_idx].copy()
            no_improve_count = 0
            if (gen + 1) % 10 == 0:  # Print improvement when it happens
                print(f"[GA day={day+1}] gen {gen+1:03d} IMPROVEMENT: {improvement:.2f} (new best_f={best_f:.2f})")
        else:
            no_improve_count += 1

        # Early stopping if no improvement for too long
        if no_improve_count >= early_stop_patience:
            print(f"[GA day={day+1}] Early stopping at gen {gen+1}: no improvement for {no_improve_count} generations")
            break

        if (gen + 1) % 10 == 0:
            # 打印一点训练信号（pred 下）
            # No MILP solution, all are random/evolved individuals
            f_dbg, r_dbg, p_dbg, d_dbg = eval_fitness_one_day(
                best_g, day,
                node_list, node_device_mapping,
                price_day, pv_day_pred, wind_day_pred,
                total_bid_day,
                LEVEL, BASE_DIR,
                spec_by_node, offsets,
                bid_penalty=bid_penalty,  # Keep for compatibility, but not used in new logic
                op_penalty_weight=op_penalty_weight,  # Keep for compatibility, but not used in new logic
                bidding_penalty=bidding_penalty_config,
                bidding_ratio=bidding_ratio_config,
                device_penalty_effi=device_penalty_effi_config,
                normalize_ratio=normalize_ratio_config,
                is_milp_solution=False,  # No MILP solution
                milp_strategy_by_node=None,
                milp_total_elec_plan=None,
                milp_elec_by_node=None,
                milp_pred_revenue=None,
            )
            # Print both current generation best and global best for comparison
            print(f"[GA day={day+1}] gen {gen+1:03d}  cur_best_f={cur_best_f:.2f}  global_best_f={best_f:.2f}  pred_revenue={r_dbg:.2f}  pred_pen={p_dbg:.2f}  pred_devL1={d_dbg:.2f}  (no_improve={no_improve_count})")

    # -------- decode best (per-device strategy) and evaluate on PRED (final) --------
    best_strategy_by_node = decode_to_strategy_by_node(best_g, node_list, spec_by_node, offsets, T, LEVEL)
    
    # Evaluate best solution on PRED (for final pred results)
    # No MILP solution, all are random/evolved individuals
    pred_f, pred_revenue, pred_penalty, pred_dev = eval_fitness_one_day(
        best_g, day,
        node_list, node_device_mapping,
        price_day, pv_day_pred, wind_day_pred,
        total_bid_day,
        LEVEL, BASE_DIR,
        spec_by_node, offsets,
        bid_penalty=bid_penalty,
        op_penalty_weight=op_penalty_weight,
        bidding_penalty=bidding_penalty_config,
        bidding_ratio=bidding_ratio_config,
        device_penalty_effi=device_penalty_effi_config,
        normalize_ratio=normalize_ratio_config,
        is_milp_solution=False,  # No MILP solution
        milp_strategy_by_node=None,
        milp_total_elec_plan=None,
        milp_elec_by_node=None,
        milp_pred_revenue=None,
    )
    
    # Calculate final pred revenue (revenue - penalty, aligned with real eval)
    pred_final_revenue = pred_revenue - pred_penalty

    # -------- decode best (per-device strategy) and evaluate on REAL --------
    # real eval (aligned with run_joint_episode)
    # Fix: Calculate revenue correctly by time step (aligned with run_joint_episode)
    total_revenue = 0.0  # Economic revenue = price * elec (aligned with run_joint_episode)
    total_penalty_device = 0.0
    total_penalty_bid = 0.0
    total_elec_real = np.zeros(T, dtype=float)
    node_device_states = {}  # Store device states for each node
    # Store per-node per-time-step elec for correct revenue calculation
    node_elec_per_time = {node: np.zeros(T, dtype=float) for node in node_list}

    for idx, node in enumerate(node_list):
        env_sim = SimpleVPPSimulator(
            price=price_day[node],
            pv=pv_day_real[idx],
            wind=wind_day_real[idx],
            device_dict=node_device_mapping[node],
            level=LEVEL,
            base_dir=BASE_DIR,
        )
        result = execute_milp_strategy(
            env_sim, day=day, strategy=best_strategy_by_node[node],
            device_dict=node_device_mapping[node], BASE_DIR=BASE_DIR, LEVEL=LEVEL,
            debug=False
        )
        # Store per-node elec for each time step
        node_elec_per_time[node] = np.array(result["elec_list"], dtype=float)
        total_elec_real += node_elec_per_time[node]
        # Store device states for penalty calculation
        node_device_states[node] = {
            "storage_soc": result["storage_soc"],
            "vehicle_soc": result["vehicle_soc"],
            "wash_work_state": result["wash_work_state"],
            "device_dict": node_device_mapping[node],
        }

    # Calculate revenue per time step (aligned with run_joint_episode)
    # First calculate base revenue using each node's price, then adjust with clipping
    for t in range(T):
        # Calculate base revenue: sum(price[n][t] * elec_nt) for each node (aligned with run_joint_episode)
        revenue_base = 0.0
        for node in node_list:
            elec_nt = node_elec_per_time[node][t]
            price_nt = price_day[node][t]
            revenue_base += price_nt * elec_nt
        
        elec_quan = total_elec_real[t]  # Total electricity at time t
        bid = total_bid_day[t]  # Bidding value at time t
        avg_price = np.mean([price_day[n][t] for n in node_list])
        
        # Calculate threshold (aligned with VPP_dispatching logic)
        if abs(bid) < 50:
            thresh = 10
        else:
            thresh = abs(bidding_ratio_config * bid)
        
        # Calculate elec_clip and penalty_bid (aligned with VPP_dispatching)
        if elec_quan - bid > thresh:
            # Exceeded upper threshold
            penalty_bid_t = (elec_quan - bid - thresh) * avg_price / normalize_ratio_config * bidding_penalty_config
            elec_clip = bid + thresh
        elif elec_quan - bid < -thresh:
            # Below lower threshold
            penalty_bid_t = (bid - elec_quan - thresh) * avg_price / normalize_ratio_config * (1 + bidding_penalty_config)
            elec_clip = bid - thresh
        else:
            # Within threshold
            penalty_bid_t = 0.0
            elec_clip = elec_quan
        
        # Update revenue with clipped electricity (aligned with run_joint_episode)
        # revenue_adjustment = (elec_clip - elec_quan) * avg_price
        revenue_adjustment = (elec_clip - elec_quan) * avg_price
        total_revenue += revenue_base + revenue_adjustment
        total_penalty_bid += penalty_bid_t

    # Calculate device operation penalty at end of day (aligned with run_joint_episode)
    init_soc = 0.5  # Default initial SoC
    for node in node_list:
        states = node_device_states[node]
        
        # Storage penalty: if SoC < init_SoC at end of day
        current_soc = states["storage_soc"]
        if current_soc < init_soc:
            storage_penalty = 0.1 * (init_soc - current_soc)
            total_penalty_device += storage_penalty
        
        # Vehicle penalty: if SoC < capacity_demand at end of day
        vehicle_ids = states["device_dict"].get("vehicle_id", [])
        for vehicle_id in vehicle_ids:
            try:
                dev_config = load_device_yaml("vehicle", vehicle_id, BASE_DIR)
                capacity_demand = float(dev_config.get("capacity_demand", 80.0))
                capacity_max = float(dev_config.get("capacity", 100.0))
                vehicle_soc = states["vehicle_soc"].get(vehicle_id, 0.5)
                if vehicle_soc < (capacity_demand / capacity_max):
                    vehicle_penalty = 0.2 * (capacity_demand / capacity_max - vehicle_soc) * capacity_max
                    total_penalty_device += vehicle_penalty
            except:
                pass
        
        # Wash penalty: if work_state < t_dur at end of day
        wash_ids = states["device_dict"].get("wash_id", [])
        for wash_id in wash_ids:
            try:
                dev_config = load_device_yaml("wash", wash_id, BASE_DIR)
                t_dur = int(dev_config.get("t_dur", 2))
                wash_work_state = states["wash_work_state"].get(wash_id, 0)
                if wash_work_state < t_dur:
                    wash_penalty = 5.0 * (t_dur - wash_work_state)
                    total_penalty_device += wash_penalty
            except:
                pass

    # Calculate total penalty and final revenue (aligned with run_joint_episode)
    total_penalty = device_penalty_effi_config * total_penalty_device + total_penalty_bid
    final_revenue = total_revenue - total_penalty

    real_dev_l1 = float(np.abs(total_elec_real - np.array(total_bid_day, dtype=float)).sum())

    return {
        "best_strategy_by_node": best_strategy_by_node,
        "pred_revenue": pred_revenue,
        "pred_penalty": pred_penalty,
        "pred_dev_l1": pred_dev,
        "pred_final_revenue": pred_final_revenue,  # pred_revenue - pred_penalty
        "real_revenue": final_revenue,
        "real_penalty": total_penalty,
        "real_penalty_device": total_penalty_device,
        "real_penalty_bid": total_penalty_bid,
        "real_dev_l1": real_dev_l1,
    }


eval_days = [0, 1, 2, 3]
BASE_DIR = "/data2/zengjinwei/VPP_multinode/config/device"
LEVEL = 10

# Load evaluation config from config.yaml (aligned with run_joint_episode)
import yaml
config_path = "AlphaEvolve/config.yaml"
bidding_penalty = 1.0
bidding_ratio = 0.2
device_penalty_effi = 1.0
normalize_ratio = 1.0

if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        eval_config = config.get("evaluation", {})
        bidding_penalty = float(eval_config.get("bidding_penalty", 1.0))
        bidding_ratio = float(eval_config.get("bidding_ratio", 0.2))
        device_penalty_effi = float(eval_config.get("device_penalty_effi", 1.0))
        normalize_ratio = float(eval_config.get("normalize_ratio", 1.0))
        print(f"Loaded evaluation config: bidding_penalty={bidding_penalty}, bidding_ratio={bidding_ratio}, device_penalty_effi={device_penalty_effi}, normalize_ratio={normalize_ratio}")
    except Exception as e:
        print(f"Warning: Failed to load evaluation config from {config_path}: {e}")
        print(f"Using default values: bidding_penalty={bidding_penalty}, bidding_ratio={bidding_ratio}, device_penalty_effi={device_penalty_effi}, normalize_ratio={normalize_ratio}")
else:
    print(f"Warning: Config file not found: {config_path}")
    print(f"Using default values: bidding_penalty={bidding_penalty}, bidding_ratio={bidding_ratio}, device_penalty_effi={device_penalty_effi}, normalize_ratio={normalize_ratio}")

# =========================
# 设置日志文件
# =========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"run_genetic_{timestamp}.log")

class TeeOutput:
    """同时输出到控制台和文件的类"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = __import__('sys').stdout
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # 确保立即写入文件
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

# 重定向 stdout 到同时输出到控制台和文件
tee = TeeOutput(log_file_path)
import sys
sys.stdout = tee

print(f"日志文件已创建: {log_file_path}")
print(f"开始运行遗传算法优化...")
print(f"评估天数: {eval_days}")
print(f"种群大小: 80, 迭代次数: 10000")
print(f"并行进程数: {min(len(eval_days), multiprocessing.cpu_count())}")
print("=" * 60)

def run_ga_for_day_wrapper(args):
    """Wrapper function for parallel execution"""
    day, node_list, node_device_mapping, price, bid, pv_pred, wind_pred, pv_real, wind_real, \
    LEVEL, BASE_DIR, pop_size, n_gen, cx_prob, mut_prob, p_gene, bid_penalty, op_penalty_weight, seed = args
    
    # Each process needs its own random seed
    random.seed(seed + day)
    np.random.seed(seed + day)
    
    print(f"[Day {day+1}] Starting GA optimization...")

    out = run_ga_for_day(
        day=day,
        node_list=node_list,
        node_device_mapping=node_device_mapping,
        price=price,
        bid=bid,
        pv_pred=pv_pred,
        wind_pred=wind_pred,
        pv_real=pv_real,
        wind_real=wind_real,
        LEVEL=LEVEL,
        BASE_DIR=BASE_DIR,
        pop_size=pop_size,
        n_gen=n_gen,
        cx_prob=cx_prob,
        mut_prob=mut_prob,
        p_gene=p_gene,
        bid_penalty=bid_penalty,
        op_penalty_weight=op_penalty_weight,
        seed=seed + day,
    )
    
    return day, out

# Prepare arguments for parallel execution
ga_args = []
for day in eval_days:
    args = (
        day, node_list, node_device_mapping, price, bid, pv_pred, wind_pred, pv_real, wind_real,
        LEVEL, BASE_DIR, 80, 10000, 0.85, 0.95, 0.01, 1000.0, 1.0, 0
        # Increased: cx_prob: 0.8->0.85, mut_prob: 0.9->0.95, p_gene: 0.002->0.01 (5x increase)
    )
    ga_args.append(args)

# Run in parallel
results = {}
max_workers = min(len(eval_days), multiprocessing.cpu_count())
print(f"\n使用 {max_workers} 个进程并行运行 {len(eval_days)} 天的模拟...\n")
sys.stdout.flush()  # Ensure output is written before starting parallel execution

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_day = {executor.submit(run_ga_for_day_wrapper, args): args[0] for args in ga_args}
    
    # Collect results as they complete
    for future in as_completed(future_to_day):
        day = future_to_day[future]
        try:
            day_idx, out = future.result()
            results[day_idx] = out
            print(f"\n[Day {day_idx+1}] Completed!")
            print(f"[GA SIM-REAL Day {day_idx+1}] real revenue       : {out['real_revenue']:.3f}")
            print(f"[GA SIM-REAL Day {day_idx+1}] total penalty      : {out['real_penalty']:.3f}")
            print(f"[GA SIM-REAL Day {day_idx+1}]   - penalty_device : {out['real_penalty_device']:.3f}")
            print(f"[GA SIM-REAL Day {day_idx+1}]   - penalty_bid    : {out['real_penalty_bid']:.3f}")
            print(f"[GA SIM-REAL Day {day_idx+1}] real deviation     : {out['real_dev_l1']:.3f}")
            sys.stdout.flush()  # Ensure output is written immediately
        except Exception as exc:
            print(f"[Day {day+1}] Generated an exception: {exc}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

# Print summary
print(f"\n{'='*60}")
print(f"Summary of all days:")
print(f"{'='*60}")

# Calculate totals for pred and real
total_pred_revenue = 0.0
total_pred_penalty = 0.0
total_pred_dev = 0.0
total_pred_final_revenue = 0.0
total_real_revenue = 0.0
total_real_penalty = 0.0
total_real_dev = 0.0

for day in sorted(results.keys()):
    out = results[day]
    print(f"Day {day+1}:")
    print(f"  [PRED] revenue={out['pred_revenue']:.3f}, penalty={out['pred_penalty']:.3f}, "
          f"deviation={out['pred_dev_l1']:.3f}, final_revenue={out['pred_final_revenue']:.3f}")
    print(f"  [REAL] revenue={out['real_revenue']:.3f}, penalty={out['real_penalty']:.3f}, "
          f"deviation={out['real_dev_l1']:.3f}")
    
    total_pred_revenue += out['pred_revenue']
    total_pred_penalty += out['pred_penalty']
    total_pred_dev += out['pred_dev_l1']
    total_pred_final_revenue += out['pred_final_revenue']
    total_real_revenue += out['real_revenue']
    total_real_penalty += out['real_penalty']
    total_real_dev += out['real_dev_l1']

# Print total pred results (4 days)
print(f"\n{'='*60}")
print(f"Total PRED results (4 days):")
print(f"{'='*60}")
print(f"  Total pred revenue      : {total_pred_revenue:.3f}")
print(f"  Total pred penalty      : {total_pred_penalty:.3f}")
print(f"  Total pred deviation    : {total_pred_dev:.3f}")
print(f"  Total pred final_revenue: {total_pred_final_revenue:.3f} (revenue - penalty)")

print(f"\n{'='*60}")
print(f"Total REAL results (4 days):")
print(f"{'='*60}")
print(f"  Total real revenue      : {total_real_revenue:.3f}")
print(f"  Total real penalty      : {total_real_penalty:.3f}")
print(f"  Total real deviation    : {total_real_dev:.3f}")

# 恢复 stdout 并关闭日志文件
sys.stdout = tee.stdout
tee.close()
print(f"\n运行完成！日志已保存到: {log_file_path}")
