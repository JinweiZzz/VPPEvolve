# run_transformerVPP.py
# DTCE (Decentralized Training Centralized Execution) with Transformer Critic
# 每个节点有一个policy actor，输入该节点对应的信息
# 共享的critic采用transformer整合不同节点的信息并评估value

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
import yaml
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from typing import Dict, List, Tuple

# =========================
# 0. 设备配置（GPU支持）
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print()

# =========================
# 1. 数据读取模块（与run_group_rl.py一致）
# =========================
print("正在加载数据...")

# 电价数据
sampled_node_prices = pd.read_csv('/data2/zengjinwei/VPP_multinode/data/山西数据/山西15节点1月1日到1月15日数据.csv')

# pv\wind预测和真实数据
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

# 设备配置
device_set_name = 'shanxi_15nodes'
with open(f'/data2/zengjinwei/VPP_multinode/config/device_set/{device_set_name}.yaml', 'r') as yaml_file:
    device_set_config = yaml.safe_load(yaml_file)

# 获取节点设备映射
node_device_mapping = device_set_config.get('node_device_mapping', {})
node_set = sampled_node_prices['node_name'].unique()

# 数据预处理
price = np.zeros((len(node_set), 7, 24))
bid = np.zeros((7, 24))
pv_pred = np.zeros((len(node_set), 7, 24))
wind_pred = np.zeros((len(node_set), 7, 24))
pv_real = np.zeros((len(node_set), 7, 24))
wind_real = np.zeros((len(node_set), 7, 24))
node_idx = 0

for node in node_set:
    node_prices_df = sampled_node_prices[sampled_node_prices['node_name'] == node]
    
    if len(node_prices_df) != 672:
        print('Error! Price df dismatched!')
    
    price_list = node_prices_df['price_value'].tolist()
    # 每4个值求平均
    price_list = [np.mean(price_list[i*4:(i+1)*4]) for i in range(len(price_list)//4)]
    
    # 获取pv和wind的预测数据
    pv_pred_list = station2pv_pred_list[node][:len(price_list)*4]
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
NUM_NODES = len(node_list)

print(f"数据加载完成！节点数量: {NUM_NODES}")

# =========================
# 2. 全局参数
# =========================
STATE_DIM = 13  # [p, pv, wind, flex_storage, current_storage, flex_vehicle, current_vehicle,
                #  flex_AC, current_AC, flex_wash, current_wash, t, total_bid]
ACTION_DIM = 6  # [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
LAMBDA = 1e3    # deviation penalty
LR_ACTOR = 1e-5
LR_CRITIC = 1e-5  # 降低Critic学习率，防止训练不稳定
GAMMA = 0.99    # discount factor
TAU = 0.005     # soft update coefficient

# =========================
# 3. 辅助函数：计算设备灵活性（与run_group_rl.py一致）
# =========================
def load_device_yaml(device_type, device_id, base_dir="/data2/zengjinwei/VPP_multinode/config/device"):
    """加载设备YAML配置"""
    path = f"{base_dir}/{device_type}/simulator_{int(device_id)}.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data[device_type]

def get_node_flexibility(node_name, node_device_mapping, base_dir="/data2/zengjinwei/VPP_multinode/config/device"):
    """
    获取节点的所有灵活性值（flex）和最大功率（max_power）
    返回: (flex_storage, flex_vehicle, flex_AC, flex_wash, storage_max_power, vehicle_max_power)
    """
    if node_name not in node_device_mapping:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    node_devices = node_device_mapping[node_name]
    flex_storage = 0.0
    flex_vehicle = 0.0
    flex_AC = 0.0
    flex_wash = 0.0
    storage_max_power = 0.0
    vehicle_max_power = 0.0
    
    # 1. 储能的总储电量和最大功率
    storage_ids = node_devices.get('storage_id', [])
    for storage_id in storage_ids:
        try:
            dev = load_device_yaml("storage", storage_id, base_dir)
            flex_storage += float(dev.get('capacity', 0.0))
            storage_max_power += float(dev.get('max_power', 0.0))
        except:
            pass
    
    # 2. EV的总可充电量和最大功率
    vehicle_ids = node_devices.get('vehicle_id', [])
    for vehicle_id in vehicle_ids:
        try:
            dev = load_device_yaml("vehicle", vehicle_id, base_dir)
            flex_vehicle += float(dev.get('capacity', 0.0))
            vehicle_max_power += float(dev.get('max_power', 0.0))
        except:
            pass
    
    # 3. AC的总用电量
    ac_ids = node_devices.get('AC_id', [])
    for ac_id in ac_ids:
        try:
            dev = load_device_yaml("AC", ac_id, base_dir)
            flex_AC += float(dev.get('power_max', 0.0))
        except:
            pass
    
    # 4. Wash的总可开次数（每台功率约40kW，这里返回总功率）
    wash_ids = node_devices.get('wash_id', [])
    for wash_id in wash_ids:
        try:
            dev = load_device_yaml("wash", wash_id, base_dir)
            rate_power = float(dev.get('rate_power', 0.0))
            flex_wash += rate_power  # 总功率，而不是次数
        except:
            pass
    
    return (flex_storage, flex_vehicle, flex_AC, flex_wash, storage_max_power, vehicle_max_power)

# =========================
# 4. 环境类（与run_group_rl.py一致）
# =========================
class Env:
    def __init__(self, price_data, pv_data, wind_data, node_list, day=0, node_device_mapping=None, base_dir="/data2/zengjinwei/VPP_multinode/config/device"):
        """
        price_data: (N_nodes, 7, 24) 或 (N_nodes, 24)
        pv_data: (N_nodes, 7, 24) 或 (N_nodes, 24)
        wind_data: (N_nodes, 7, 24) 或 (N_nodes, 24)
        node_list: 节点名称列表
        day: 第几天（0-6），初始day
        node_device_mapping: 节点设备映射
        base_dir: 设备配置目录
        """
        self.node_list = node_list
        self.node_device_mapping = node_device_mapping or {}
        self.base_dir = base_dir
        
        # 保存完整的数据（所有天），而不是只保存当天的数据
        if price_data.ndim == 3:
            self.price_data_full = price_data  # (N_nodes, 7, 24)
            self.prices = price_data[:, day, :]  # (N_nodes, 24) - 当前day的数据
        else:
            self.price_data_full = price_data  # (N_nodes, 24)
            self.prices = price_data  # (N_nodes, 24)
        
        if pv_data.ndim == 3:
            self.pv_data_full = pv_data  # (N_nodes, 7, 24)
            self.pv = pv_data[:, day, :]  # (N_nodes, 24) - 当前day的数据
        else:
            self.pv_data_full = pv_data  # (N_nodes, 24)
            self.pv = pv_data  # (N_nodes, 24)
        
        if wind_data.ndim == 3:
            self.wind_data_full = wind_data  # (N_nodes, 7, 24)
            self.wind = wind_data[:, day, :]  # (N_nodes, 24) - 当前day的数据
        else:
            self.wind_data_full = wind_data  # (N_nodes, 24)
            self.wind = wind_data  # (N_nodes, 24)
        
        self.day = day  # 当前day
        
        # 预计算每个节点的灵活性值（flex）
        self.node_flex = {}
        for i, node_name in enumerate(node_list):
            self.node_flex[i] = get_node_flexibility(node_name, self.node_device_mapping, self.base_dir)
        
        # 计算归一化因子（用于状态归一化）
        # 找到所有节点中的最大值，用于归一化
        max_flex_storage = max([self.node_flex[i][0] for i in range(len(node_list))], default=1.0)
        max_flex_vehicle = max([self.node_flex[i][1] for i in range(len(node_list))], default=1.0)
        max_flex_AC = max([self.node_flex[i][2] for i in range(len(node_list))], default=1.0)
        max_flex_wash = max([self.node_flex[i][3] for i in range(len(node_list))], default=1.0)
        max_pv = float(pv_data.max()) if pv_data.size > 0 else 1000.0
        max_wind = float(wind_data.max()) if wind_data.size > 0 else 1000.0
        max_price = float(price_data.max()) if price_data.size > 0 else 1.0
        
        # 存储归一化因子
        self.norm_factors = {
            'flex_storage': max(max_flex_storage, 1.0),
            'flex_vehicle': max(max_flex_vehicle, 1.0),
            'flex_AC': max(max_flex_AC, 1.0),
            'flex_wash': max(max_flex_wash, 1.0),
            'pv': max(max_pv, 1.0),
            'wind': max(max_wind, 1.0),
            'price': max(max_price, 1.0),
            'total_bid': 1e6,  # 假设最大bid为1e6
        }
        
        # 初始化当前状态（current值）
        self.current_state = {}
        self.node_max_power = {}  # 存储每个节点的max_power信息
        for i, node_name in enumerate(node_list):
            flex_storage, flex_vehicle, flex_AC, flex_wash, storage_max_power, vehicle_max_power = self.node_flex[i]
            self.current_state[i] = {
                'storage': flex_storage * 0.5,  # 初始SoC = 0.5
                'vehicle': flex_vehicle * 0.5,   # 初始SoC = 0.5
                'AC': 0.0,                       # AC初始用电量为0
                'wash': 0.0,                     # Wash初始开启次数为0
            }
            self.node_max_power[i] = {
                'storage': storage_max_power,
                'vehicle': vehicle_max_power,
            }

    def get_state(self, node_idx, total_bid, time):
        """
        获取完整的状态向量，包含 program_seed_joint.py 中的所有参数
        
        node_idx: 节点索引（0-based）
        total_bid: 总bid值
        time: 当前时刻（0-23，归一化到[0,1]）
        
        返回状态向量：[p, pv, wind, flex_storage, current_storage, flex_vehicle, current_vehicle,
                     flex_AC, current_AC, flex_wash, current_wash, t, total_bid]
        """
        time_idx = int(np.clip(time, 0, 23))
        
        # 获取原始值
        # 1. p: electricity price (归一化到[0, 1])
        p_raw = float(self.prices[node_idx, time_idx] if time_idx < self.prices.shape[1] else self.prices[node_idx, -1])
        p = p_raw / self.norm_factors['price']
        
        # 2. pv: photovoltaic generation (归一化到[0, 1])
        pv_raw = float(self.pv[node_idx, time_idx] if time_idx < self.pv.shape[1] else self.pv[node_idx, -1])
        pv = max(0.0, pv_raw) / self.norm_factors['pv']
        
        # 3. wind: wind generation (归一化到[0, 1])
        wind_raw = float(self.wind[node_idx, time_idx] if time_idx < self.wind.shape[1] else self.wind[node_idx, -1])
        wind = max(0.0, wind_raw) / self.norm_factors['wind']
        
        # 4-7. flex 和 current 值 (归一化到[0, 1])
        flex_storage, flex_vehicle, flex_AC, flex_wash, _, _ = self.node_flex[node_idx]
        current_storage = self.current_state[node_idx]['storage']
        current_vehicle = self.current_state[node_idx]['vehicle']
        current_AC = self.current_state[node_idx]['AC']
        current_wash = self.current_state[node_idx]['wash']
        
        # 归一化flex和current值
        flex_storage_norm = max(0.0, float(flex_storage)) / self.norm_factors['flex_storage']
        current_storage_norm = max(0.0, float(current_storage)) / self.norm_factors['flex_storage']
        flex_vehicle_norm = max(0.0, float(flex_vehicle)) / self.norm_factors['flex_vehicle']
        current_vehicle_norm = max(0.0, float(current_vehicle)) / self.norm_factors['flex_vehicle']
        flex_AC_norm = max(0.0, float(flex_AC)) / self.norm_factors['flex_AC']
        current_AC_norm = max(0.0, float(current_AC)) / self.norm_factors['flex_AC']
        flex_wash_norm = max(0.0, float(flex_wash)) / self.norm_factors['flex_wash']
        current_wash_norm = max(0.0, float(current_wash)) / self.norm_factors['flex_wash']
        
        # 8. t: time step (归一化到[0,1])
        t_normalized = time / 23.0
        
        # 9. total_bid (归一化到[0, 1])
        total_bid_val = float(total_bid) if not isinstance(total_bid, torch.Tensor) else total_bid.item()
        total_bid_norm = total_bid_val / self.norm_factors['total_bid']
        
        # 构建状态向量（所有值都在合理范围内）
        state = np.array([
            p,                    # 0: price (归一化)
            pv,                   # 1: pv (归一化)
            wind,                 # 2: wind (归一化)
            flex_storage_norm,    # 3: flex_storage (归一化)
            current_storage_norm, # 4: current_storage (归一化)
            flex_vehicle_norm,    # 5: flex_vehicle (归一化)
            current_vehicle_norm, # 6: current_vehicle (归一化)
            flex_AC_norm,         # 7: flex_AC (归一化)
            current_AC_norm,      # 8: current_AC (归一化)
            flex_wash_norm,       # 9: flex_wash (归一化)
            current_wash_norm,    # 10: current_wash (归一化)
            t_normalized,         # 11: t (归一化)
            total_bid_norm,       # 12: total_bid (归一化)
        ], dtype=np.float32)
        
        return torch.tensor(state, dtype=torch.float32).to(device)
    
    def update_current_state(self, node_idx, delta_storage=0.0, delta_vehicle=0.0, delta_AC=0.0, delta_wash=0.0):
        """
        更新节点的当前状态（用于跟踪设备状态变化）
        """
        if node_idx in self.current_state:
            flex_storage, flex_vehicle, _, _, _, _ = self.node_flex[node_idx]
            self.current_state[node_idx]['storage'] = max(0.0, min(
                flex_storage,
                self.current_state[node_idx]['storage'] + delta_storage
            ))
            self.current_state[node_idx]['vehicle'] = max(0.0, min(
                flex_vehicle,
                self.current_state[node_idx]['vehicle'] + delta_vehicle
            ))
            self.current_state[node_idx]['AC'] = max(0.0, self.current_state[node_idx]['AC'] + delta_AC)
            self.current_state[node_idx]['wash'] = max(0.0, self.current_state[node_idx]['wash'] + delta_wash)
    
    def reset_current_state(self):
        """重置所有节点的当前状态到初始值"""
        for i in range(len(self.node_list)):
            flex_storage, flex_vehicle, flex_AC, flex_wash, _, _ = self.node_flex[i]
            self.current_state[i] = {
                'storage': flex_storage * 0.5,
                'vehicle': flex_vehicle * 0.5,
                'AC': 0.0,
                'wash': 0.0,
            }
    
    def set_day(self, day):
        """
        切换环境到指定的day（不重置状态，只切换数据）
        
        Args:
            day: 第几天（0-6）
        """
        self.day = day
        
        # 更新当前day的数据
        if self.price_data_full.ndim == 3:
            self.prices = self.price_data_full[:, day, :]  # (N_nodes, 24)
        else:
            self.prices = self.price_data_full  # (N_nodes, 24)
        
        if self.pv_data_full.ndim == 3:
            self.pv = self.pv_data_full[:, day, :]  # (N_nodes, 24)
        else:
            self.pv = self.pv_data_full  # (N_nodes, 24)
        
        if self.wind_data_full.ndim == 3:
            self.wind = self.wind_data_full[:, day, :]  # (N_nodes, 24)
        else:
            self.wind = self.wind_data_full  # (N_nodes, 24)
    
    def reset(self, day=None):
        """
        重置环境：切换day（如果提供）并重置所有节点的当前状态
        
        Args:
            day: 第几天（0-6），如果提供则切换day，否则保持当前day
        """
        if day is not None:
            self.set_day(day)
        self.reset_current_state()

    def evaluate(self, actions, total_bid, time):
        """
        actions: dict[node_idx] -> action vector (6维，对应 program_seed_joint.py 格式)
        action向量格式: [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
        total_bid: 总bid值（可以是float或tensor）
        time: 当前时刻
        返回: revenue (tensor), deviation (tensor)
        """
        time_idx = int(np.clip(time, 0, 23))
        
        # 确保total_bid是tensor并在正确的device上（与run_group_rl.py完全一致）
        if not isinstance(total_bid, torch.Tensor):
            total_bid = torch.tensor(float(total_bid), dtype=torch.float32, device=device)
        else:
            total_bid = total_bid.to(device)
        
        # 使用累积方式，确保梯度连接
        total_supply = None
        revenue = None

        for node_idx, a in actions.items():
            # 确保action tensor在正确的device上
            if isinstance(a, torch.Tensor):
                a_vals = a.to(device)
            else:
                a_vals = torch.tensor(a, dtype=torch.float32, device=device)
            
            # 解析动作向量（6维）
            # a_vals[0]: pv_ratio [0, 1]
            # a_vals[1]: wind_ratio [0, 1]
            # a_vals[2]: storage_ratio [-1, 1] (正=放电, 负=充电)
            # a_vals[3]: vehicle_ratio [-1, 1] (正=放电, 负=充电)
            # a_vals[4]: ac_ratio [0, 1]
            # a_vals[5]: wash_on_number >=0
            
            # 获取当前时刻的PV和Wind发电量（标量，用于计算）
            pv_val = float(self.pv[node_idx, time_idx] if time_idx < self.pv.shape[1] else 0.0)
            wind_val = float(self.wind[node_idx, time_idx] if time_idx < self.wind.shape[1] else 0.0)
            
            # 获取节点的灵活性值和最大功率（用于计算storage和vehicle功率）
            flex_storage, flex_vehicle, flex_AC, flex_wash, storage_max_power, vehicle_max_power = self.node_flex[node_idx]
            
            # 计算电力供应（保持梯度连接）
            # 1. PV发电并网
            pv_supply = a_vals[0] * pv_val  # pv_ratio * pv_generation
            
            # 2. Wind发电并网
            wind_supply = a_vals[1] * wind_val  # wind_ratio * wind_generation
            
            # 3. Storage充放电
            # storage_ratio * max_power (正=放电, 负=充电)
            storage_supply = a_vals[2] * storage_max_power  # storage_ratio * max_power
            
            # 4. Vehicle充放电
            # vehicle_ratio * max_power (正=放电, 负=充电)
            vehicle_supply = a_vals[3] * vehicle_max_power  # vehicle_ratio * max_power
            
            # 5. AC用电（负值，因为是消费）
            ac_consumption = a_vals[4] * flex_AC  # ac_ratio * flex_AC
            
            # 6. Wash用电（负值，根据开启数量）
            # wash_on_number是浮点数（保持梯度），表示开启的台数
            # flex_wash是总功率（kW），需要知道每台功率来计算台数
            # 简化：假设每台wash功率为40kW（根据program_seed_joint.py注释）
            wash_power_per_unit = 40.0  # 每台wash功率（kW）
            max_wash_units = flex_wash / wash_power_per_unit if wash_power_per_unit > 0 else 0.0
            wash_on_number = torch.clamp(a_vals[5], min=0.0, max=max_wash_units)  # 限制在有效范围内
            wash_consumption = wash_on_number * wash_power_per_unit  # 总用电功率
            
            # 总电力供应 = 发电 - 用电
            supply = pv_supply + wind_supply + storage_supply + vehicle_supply - ac_consumption - wash_consumption
            
            # 计算收益（与run_group_rl.py完全一致）
            price_val = float(self.prices[node_idx, time_idx] if time_idx < self.prices.shape[1] else self.prices[node_idx, -1])
            node_revenue = supply * price_val
            
            # 累积，保持梯度连接
            if total_supply is None:
                total_supply = supply
                revenue = node_revenue
            else:
                total_supply = total_supply + supply
                revenue = revenue + node_revenue

        # 如果没有action，返回零值
        if total_supply is None:
            total_supply = torch.tensor(0.0, dtype=torch.float32, device=device)
            revenue = torch.tensor(0.0, dtype=torch.float32, device=device)

        deviation = torch.abs(total_supply - total_bid)
        return revenue, deviation

# =========================
# 5. Actor网络（共享参数，所有节点共用）
# =========================
class SharedActor(nn.Module):
    """
    共享的Actor网络（所有节点共用同一个网络）
    输入：13维状态向量 [p, pv, wind, flex_storage, current_storage, flex_vehicle, current_vehicle,
                     flex_AC, current_AC, flex_wash, current_wash, t, total_bid]
    输出：6维动作向量 [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
    """
    def __init__(self, state_dim=13, action_dim=6, hidden_dim=128):
        super(SharedActor, self).__init__()
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 分别的输出层，使用不同的激活函数
        # pv_ratio, wind_ratio, ac_ratio: [0, 1] -> Sigmoid
        self.pv_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.wind_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.ac_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
        # storage_ratio, vehicle_ratio: [-1, 1] -> Tanh
        self.storage_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.vehicle_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        
        # wash_on_number: >=0 -> ReLU
        self.wash_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU())

    def forward(self, state):
        """
        输出动作向量，格式与 program_seed_joint.py 一致：
        [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
        """
        features = self.feature_net(state)
        
        pv_ratio = self.pv_head(features)
        wind_ratio = self.wind_head(features)
        storage_ratio = self.storage_head(features)
        vehicle_ratio = self.vehicle_head(features)
        ac_ratio = self.ac_head(features)
        wash_on_number = self.wash_head(features)
        
        # 拼接成6维动作向量
        action = torch.cat([
            pv_ratio,      # [0, 1]
            wind_ratio,    # [0, 1]
            storage_ratio, # [-1, 1]
            vehicle_ratio, # [-1, 1]
            ac_ratio,      # [0, 1]
            wash_on_number # >=0
        ], dim=-1)
        
        return action

# =========================
# 6. Transformer Critic网络（共享）
# =========================
class TransformerCritic(nn.Module):
    """
    共享的Transformer Critic网络
    输入：所有节点的状态和动作对
    输出：全局价值函数 V(s_1, ..., s_n)
    """
    def __init__(self, state_dim=13, action_dim=6, hidden_dim=128, n_heads=8, n_layers=3):
        super(TransformerCritic, self).__init__()
        
        # 输入归一化层（防止输入值过大导致NaN）
        self.state_norm = nn.LayerNorm(state_dim)
        self.action_norm = nn.LayerNorm(action_dim)
        
        # 状态和动作的嵌入层
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 全局价值输出
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        """
        states: (batch_size, n_agents, state_dim) - 所有节点的状态
        actions: (batch_size, n_agents, action_dim) - 所有节点的动作
        返回: (batch_size, 1) - 全局价值
        """
        batch_size, n_agents, _ = states.shape
        
        # 归一化输入（防止值过大导致NaN）
        states_norm = self.state_norm(states)
        actions_norm = self.action_norm(actions)
        
        # 嵌入状态和动作
        state_emb = self.state_embedding(states_norm)  # (batch_size, n_agents, hidden_dim)
        action_emb = self.action_embedding(actions_norm)  # (batch_size, n_agents, hidden_dim)
        
        # 拼接状态和动作嵌入
        agent_emb = state_emb + action_emb  # (batch_size, n_agents, hidden_dim)
        
        # Transformer编码
        encoded = self.transformer(agent_emb)  # (batch_size, n_agents, hidden_dim)
        
        # 全局池化（平均池化）
        global_feature = encoded.mean(dim=1)  # (batch_size, hidden_dim)
        
        # 输出全局价值
        value = self.value_head(global_feature)  # (batch_size, 1)
        
        return value

# =========================
# 7. Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, state_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.n_agents = n_agents
        self.ptr = 0
        self.size = 0

        # store (s, a, r, s', done) for all agents
        self.states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)  # 全局奖励
        self.next_states = np.zeros((capacity, n_agents, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, states, actions, reward, next_states, done):
        self.states[self.ptr] = states
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_states
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            states=torch.FloatTensor(self.states[idx]).to(self.device),
            actions=torch.FloatTensor(self.actions[idx]).to(self.device),
            rewards=torch.FloatTensor(self.rewards[idx]).to(self.device),
            next_states=torch.FloatTensor(self.next_states[idx]).to(self.device),
            done=torch.FloatTensor(self.done[idx]).to(self.device),
        )
        return batch

# =========================
# 8. 初始化网络和优化器
# =========================
# 创建共享的Actor网络（所有节点共用）
shared_actor = SharedActor(STATE_DIM, ACTION_DIM).to(device)
actor_optimizer = optim.Adam(shared_actor.parameters(), lr=LR_ACTOR)

# 创建Critic网络（共享）
critic = TransformerCritic(STATE_DIM, ACTION_DIM).to(device)
critic_target = TransformerCritic(STATE_DIM, ACTION_DIM).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC)

# 创建Replay Buffer
buffer = ReplayBuffer(capacity=100000, n_agents=NUM_NODES, state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)

print(f"初始化了共享 Actor 网络 (在 {device} 上)")
print(f"初始化了共享 Transformer Critic (在 {device} 上)")
print()

# =========================
# 9. 训练循环（DTCE，与run_group_rl.py一致：每个epoch遍历4天）
# =========================
NUM_EPISODES = 100000
BATCH_SIZE = 256
UPDATE_EVERY = 1
WARMUP_STEPS = 1000
TRAIN_DAYS = [0, 1, 2, 3]  # 使用前4天进行训练（pred场景）
VAL_DAYS = [0, 1, 2, 3]    # 验证场景：真实场景的前4天（与run_MILP.py对齐）
TEST_DAYS = [0, 1, 2, 3]   # 测试场景：真实场景的前4天（与run_MILP.py对齐）
EVAL_INTERVAL = 10  # 每10个epoch评估一次
EARLY_STOP_PATIENCE = 300  # 早停耐心值：300个epoch不提升就停止

# 准备训练数据：四天的所有时刻
train_samples = []
for day in TRAIN_DAYS:
    for t in range(24):
        train_samples.append((day, t))

# 创建训练环境（使用pred数据）
env_train = Env(price, pv_pred, wind_pred, node_list, day=0,
                node_device_mapping=node_device_mapping,
                base_dir="/data2/zengjinwei/VPP_multinode/config/device")

# 创建评估环境（使用real数据）
env_eval = Env(price, pv_real, wind_real, node_list, day=0,
               node_device_mapping=node_device_mapping,
               base_dir="/data2/zengjinwei/VPP_multinode/config/device")

print("开始训练...")
print(f"训练轮数: {NUM_EPISODES}")
print(f"批量大小: {BATCH_SIZE}")
print(f"训练数据：{len(TRAIN_DAYS)} 天 × 24 小时 = {len(train_samples)} 个样本（pred场景）")
print(f"验证数据：{len(VAL_DAYS)} 天 × 24 小时（real场景，用于早停）")
print(f"测试数据：{len(TEST_DAYS)} 天 × 24 小时（real场景，最终评估）")
print(f"评估间隔: 每 {EVAL_INTERVAL} 个epoch评估一次")
print(f"早停耐心值: {EARLY_STOP_PATIENCE} 个epoch")
print()

# 早停机制变量
best_eval_score = float('-inf')
patience_counter = 0
best_actor_state = None
best_critic_state = None

for episode in range(NUM_EPISODES):
    # 每个episode遍历4天的所有时刻（与run_group_rl.py一致）
    episode_reward = 0.0
    episode_revenue = 0.0  # 累积4天的总revenue
    episode_deviation = 0.0  # 累积4天的总deviation
    
    # 遍历四天的所有时刻
    states_dict = None
    for step_idx, (day, t) in enumerate(train_samples):
        # 如果是新的一天，重置环境
        if step_idx == 0 or (step_idx > 0 and train_samples[step_idx-1][0] != day):
            env_train.reset(day=day)
        else:
            # 同一天的不同时刻，只切换day（如果day改变了）
            if env_train.day != day:
                env_train.set_day(day)
        
        # 获取当前状态（如果是第一个时刻或新的一天，需要初始化）
        if step_idx == 0 or (step_idx > 0 and train_samples[step_idx-1][0] != day):
            states_dict = {}
            for node_idx in range(NUM_NODES):
                states_dict[node_idx] = env_train.get_state(node_idx, bid[day, t], t)
            states_array = np.stack([states_dict[i].cpu().numpy() for i in range(NUM_NODES)], axis=0)
        
        # 获取当前时刻的bid
        total_bid = bid[day, t]
        
        # 每个节点选择动作（decentralized execution，使用共享actor）
        actions_dict = {}
        for node_idx in range(NUM_NODES):
            with torch.no_grad():
                action = shared_actor(states_dict[node_idx].unsqueeze(0))
            actions_dict[node_idx] = action.squeeze(0)
        
        actions_array = np.stack([actions_dict[i].cpu().numpy() for i in range(NUM_NODES)], axis=0)
        
        # 评估动作（计算reward）
        revenue, deviation = env_train.evaluate(actions_dict, total_bid, t)
        reward = revenue / 1e9 - LAMBDA * deviation / 1e9
        
        # 获取下一状态
        next_states_dict = {}
        for node_idx in range(NUM_NODES):
            next_t = t + 1 if t < 23 else t
            next_day = day if t < 23 else day
            next_states_dict[node_idx] = env_train.get_state(node_idx, bid[next_day, next_t], next_t)
        
        next_states_array = np.stack([next_states_dict[i].cpu().numpy() for i in range(NUM_NODES)], axis=0)
        
        # 存储经验
        done = 1.0 if (t == 23 and day == TRAIN_DAYS[-1]) else 0.0  # 只在最后一天最后一时刻设为done
        buffer.add(states_array, actions_array, reward.item(), next_states_array, done)
        
        # 更新状态
        states_dict = next_states_dict
        states_array = next_states_array
        
        # 防止NaN累积（累积4天的总和，与run_group_rl.py一致）
        reward_val = reward.item() if not (np.isnan(reward.item()) or np.isinf(reward.item())) else 0.0
        revenue_val = revenue.item() if not (np.isnan(revenue.item()) or np.isinf(revenue.item())) else 0.0
        deviation_val = deviation.item() if not (np.isnan(deviation.item()) or np.isinf(deviation.item())) else 0.0
        
        episode_reward += reward_val
        episode_revenue += revenue_val  # 累积4天的总revenue（与run_group_rl.py一致）
        episode_deviation += deviation_val  # 累积4天的总deviation（与run_group_rl.py一致）
    
    # 训练（centralized training）
    if buffer.size >= WARMUP_STEPS and episode % UPDATE_EVERY == 0:
        for _ in range(4):  # 多次更新
            batch = buffer.sample(BATCH_SIZE)
            
            # 检查输入是否包含NaN/Inf
            if torch.isnan(batch['states']).any() or torch.isinf(batch['states']).any():
                print(f"警告：batch['states']包含NaN/Inf，跳过更新")
                continue
            if torch.isnan(batch['actions']).any() or torch.isinf(batch['actions']).any():
                print(f"警告：batch['actions']包含NaN/Inf，跳过更新")
                continue
            if torch.isnan(batch['rewards']).any() or torch.isinf(batch['rewards']).any():
                print(f"警告：batch['rewards']包含NaN/Inf，跳过更新")
                continue
            
            # 计算当前价值
            current_values = critic(batch['states'], batch['actions'])
            
            # 检查current_values是否为NaN/Inf
            if torch.isnan(current_values).any() or torch.isinf(current_values).any():
                print(f"警告：current_values包含NaN/Inf，跳过更新")
                continue
            
            # 计算目标价值
            with torch.no_grad():
                next_actions_list = []
                for node_idx in range(NUM_NODES):
                    next_actions = shared_actor(batch['next_states'][:, node_idx, :])
                    next_actions_list.append(next_actions)
                next_actions = torch.stack(next_actions_list, dim=1)
                
                # 检查next_actions是否为NaN/Inf
                if torch.isnan(next_actions).any() or torch.isinf(next_actions).any():
                    print(f"警告：next_actions包含NaN/Inf，跳过更新")
                    continue
                
                next_values = critic_target(batch['next_states'], next_actions)
                
                # 检查next_values是否为NaN/Inf
                if torch.isnan(next_values).any() or torch.isinf(next_values).any():
                    print(f"警告：next_values包含NaN/Inf，跳过更新")
                    continue
                
                target_values = batch['rewards'] + GAMMA * (1 - batch['done']) * next_values
                
                # 检查target_values是否为NaN/Inf
                if torch.isnan(target_values).any() or torch.isinf(target_values).any():
                    print(f"警告：target_values包含NaN/Inf，跳过更新")
                    continue
            
            # Critic损失
            critic_loss = nn.MSELoss()(current_values, target_values)
            
            # 防止NaN损失
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                print(f"警告：Critic损失为NaN/Inf (current_values: {current_values.min().item():.2f}~{current_values.max().item():.2f}, "
                      f"target_values: {target_values.min().item():.2f}~{target_values.max().item():.2f})，跳过更新")
                continue
            
            # 更新Critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            
            # 更新Actor（共享网络）
            # 重新计算所有节点的动作（需要梯度）
            actions_list = []
            for node_idx in range(NUM_NODES):
                actions_list.append(shared_actor(batch['states'][:, node_idx, :]))
            actions = torch.stack(actions_list, dim=1)
            
            # Actor损失（最大化价值）
            actor_loss = -critic(batch['states'], actions).mean()
            
            # 防止NaN损失
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                print(f"警告：Actor 损失为NaN/Inf，跳过更新")
            else:
                # 更新Actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(shared_actor.parameters(), 1.0)
                actor_optimizer.step()
            
            # 软更新target critic
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    
    # 评估（验证阶段使用real场景的前4天）
    if (episode + 1) % EVAL_INTERVAL == 0:
        # 在验证数据上评估（real场景的前4天，与run_MILP.py对齐）
        eval_reward = 0.0
        eval_revenue = 0.0
        eval_deviation = 0.0
        
        for day in VAL_DAYS:
            env_eval.reset(day=day)  # 每天开始时重置环境
            states_dict = {}
            for node_idx in range(NUM_NODES):
                states_dict[node_idx] = env_eval.get_state(node_idx, bid[day, 0], 0)
            
            for t in range(24):
                total_bid = bid[day, t]
                
                # 选择动作
                actions_dict = {}
                for node_idx in range(NUM_NODES):
                    with torch.no_grad():
                        action = shared_actor(states_dict[node_idx].unsqueeze(0))
                    actions_dict[node_idx] = action.squeeze(0)
                
                # 评估（使用真实场景数据）
                revenue, deviation = env_eval.evaluate(actions_dict, total_bid, t)
                reward = revenue / 1e9 - LAMBDA * deviation / 1e9
                
                eval_reward += reward.item()
                eval_revenue += revenue.item()
                eval_deviation += deviation.item()
                
                # 更新状态
                if t < 23:
                    next_t = t + 1
                    for node_idx in range(NUM_NODES):
                        states_dict[node_idx] = env_eval.get_state(node_idx, bid[day, next_t], next_t)
        
        eval_score = eval_reward  # 使用reward作为评估指标
        
        print(f"Episode {episode+1}/{NUM_EPISODES} | "
              f"Train Reward (pred): {episode_reward:.3f} | "
              f"Train Revenue (pred): {episode_revenue:.3f} | "
              f"Train Deviation (pred): {episode_deviation:.3f} | "
              f"Val Reward (real): {eval_reward:.3f} | "
              f"Val Revenue (real): {eval_revenue:.3f} | "
              f"Val Deviation (real): {eval_deviation:.3f}")
        
        # 早停机制
        if eval_score > best_eval_score:
            best_eval_score = eval_score
            patience_counter = 0
            # 保存最佳模型
            best_actor_state = shared_actor.state_dict().copy()
            best_critic_state = critic.state_dict().copy()
            print(f"  ✓ 新的最佳评估分数: {best_eval_score:.3f}，保存模型")
        else:
            patience_counter += EVAL_INTERVAL
            print(f"  - 评估分数未提升，耐心计数: {patience_counter}/{EARLY_STOP_PATIENCE}")
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n早停触发！{EARLY_STOP_PATIENCE} 个epoch未提升，停止训练")
                # 恢复最佳模型
                if best_actor_state is not None:
                    shared_actor.load_state_dict(best_actor_state)
                    critic.load_state_dict(best_critic_state)
                    print("已恢复最佳模型参数")
                break
    else:
        # 普通训练日志
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | "
                  f"Total Reward (4 days): {episode_reward:.3f} | "
                  f"Total Revenue (4 days): {episode_revenue:.3f} | "
                  f"Total Deviation (4 days): {episode_deviation:.3f}")

# =========================
# 10. 最终测试评估（使用真实场景的前4天，与run_MILP.py对齐）
# =========================
print("\n" + "="*80)
print("开始最终测试评估（使用真实场景数据，前4天，与run_MILP.py对齐）...")
print("="*80)

# 使用真实数据评估（前4天，与run_MILP.py对齐）
final_eval_reward = 0.0
final_eval_revenue = 0.0
final_eval_deviation = 0.0

for day in TEST_DAYS:
    env_eval.reset(day=day)  # 每天开始时重置环境
    states_dict = {}
    for node_idx in range(NUM_NODES):
        states_dict[node_idx] = env_eval.get_state(node_idx, bid[day, 0], 0)
    
    for t in range(24):
        total_bid = bid[day, t]
        
        # 选择动作
        actions_dict = {}
        for node_idx in range(NUM_NODES):
            with torch.no_grad():
                action = shared_actor(states_dict[node_idx].unsqueeze(0))
            actions_dict[node_idx] = action.squeeze(0)
        
        # 评估（使用真实数据）
        revenue, deviation = env_eval.evaluate(actions_dict, total_bid, t)
        reward = revenue / 1e9 - LAMBDA * deviation / 1e9
        
        final_eval_reward += reward.item()
        final_eval_revenue += revenue.item()
        final_eval_deviation += deviation.item()
        
        # 更新状态
        if t < 23:
            next_t = t + 1
            for node_idx in range(NUM_NODES):
                states_dict[node_idx] = env_eval.get_state(node_idx, bid[day, next_t], next_t)

print(f"\n最终测试评估结果（真实场景，前4天，与run_MILP.py对齐）:")
print(f"  Total Reward: {final_eval_reward:.3f}")
print(f"  Total Revenue: {final_eval_revenue:.3f}")
print(f"  Total Deviation: {final_eval_deviation:.3f}")
print(f"  Average Reward per day: {final_eval_reward/len(TEST_DAYS):.3f}")
print(f"  Average Revenue per day: {final_eval_revenue/len(TEST_DAYS):.3f}")
print(f"  Average Deviation per day: {final_eval_deviation/len(TEST_DAYS):.3f}")

print("\n训练和评估完成！")
