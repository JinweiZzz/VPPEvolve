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
# 1. 数据读取模块
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
node2bid = {}
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
LAMBDA = 5e2                # deviation penalty
LR_HIGH = 1e-4              # 高层学习率（增大以加快收敛）
LR_LOW = 1e-4               # 低层学习率（增大以加快收敛）
MAX_GRAD_NORM = 1.0         # 梯度裁剪阈值
REWARD_SCALE = 1e-9         # Reward缩放因子（从1e-9改为1e-6，增大梯度）

# =========================
# 3. 辅助函数：计算设备灵活性
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
# 4. 环境（使用真实数据）
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
        
        # 初始化当前状态（current值）
        # current_storage, current_vehicle, current_AC, current_wash
        # 初始化为0.5倍容量（对于storage和vehicle）或0（对于AC和wash）
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
        
        # 1. p: electricity price
        p = self.prices[node_idx, time_idx] if time_idx < self.prices.shape[1] else self.prices[node_idx, -1]
        
        # 2. pv: photovoltaic generation
        pv = self.pv[node_idx, time_idx] if time_idx < self.pv.shape[1] else self.pv[node_idx, -1]
        
        # 3. wind: wind generation
        wind = self.wind[node_idx, time_idx] if time_idx < self.wind.shape[1] else self.wind[node_idx, -1]
        
        # 4-7. flex 和 current 值
        flex_storage, flex_vehicle, flex_AC, flex_wash, _, _ = self.node_flex[node_idx]
        current_storage = self.current_state[node_idx]['storage']
        current_vehicle = self.current_state[node_idx]['vehicle']
        current_AC = self.current_state[node_idx]['AC']
        current_wash = self.current_state[node_idx]['wash']
        
        # 8. t: time step (归一化到[0,1])
        t_normalized = time / 23.0
        
        # 9. total_bid
        total_bid_val = float(total_bid) if not isinstance(total_bid, torch.Tensor) else total_bid.item()
        
        # 构建状态向量
        state = np.array([
            p,                    # 0: price
            pv,                   # 1: pv
            wind,                 # 2: wind
            flex_storage,         # 3: flex_storage
            current_storage,      # 4: current_storage
            flex_vehicle,         # 5: flex_vehicle
            current_vehicle,      # 6: current_vehicle
            flex_AC,              # 7: flex_AC
            current_AC,           # 8: current_AC
            flex_wash,            # 9: flex_wash
            current_wash,         # 10: current_wash
            t_normalized,        # 11: t (归一化)
            total_bid_val,       # 12: total_bid
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
        
        # 确保total_bid是tensor并在正确的device上
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
            
            # 计算收益
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
# 5. 方法部分（保持不变）
# =========================
def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-9)

def feasible_alloc(logits, B_t, caps):
    w = softmax(logits)
    b = B_t * w
    b = np.minimum(b, caps)
    # 可选：再分配 clip 后剩余
    return b

class HierarchicalVPPWrapper:
    def __init__(self, sim, N, dt_high_min=15, dt_low_min=1):
        """
        sim: 环境对象（Env实例），提供get_state等方法
        N: 节点数量
        dt_high_min: 高层时间步长（分钟）
        dt_low_min: 低层时间步长（分钟）
        """
        self.sim = sim
        self.N = N
        self.dt_high = dt_high_min
        self.dt_low = dt_low_min
        self.K = int(dt_high_min // dt_low_min)
        assert self.dt_high % self.dt_low == 0

        # 低层策略由你传入（SAC/TD3），这里先占位
        self.low_policies = [None] * N
        
        # 当前时间和总bid（用于状态构建）
        self.current_time = 0
        self.current_total_bid = 0.0

    def get_obs_high(self, total_bid, time):
        """
        获取高层agent的观测状态（所有节点的状态信息）
        对应 alpha_score 的输入：p, pv, wind, flex_storage, current_storage, 
        flex_vehicle, current_vehicle, flex_AC, current_AC, flex_wash, current_wash, t
        
        Args:
            total_bid: 总bid值（用于低层，高层不需要）
            time: 当前时刻 (0-23)
        
        Returns:
            dict: 包含所有节点的状态信息
                - node_states: list of dict, 每个节点的状态
                - 每个节点状态包含: p, pv, wind, flex_storage, current_storage,
                  flex_vehicle, current_vehicle, flex_AC, current_AC, 
                  flex_wash, current_wash, t
        """
        node_states = []
        for i in range(self.N):
            # 获取完整状态（包含total_bid，但高层不使用）
            full_state = self.sim.get_state(i, total_bid, time)
            
            # 提取高层需要的状态（排除total_bid，即最后一个元素）
            state_array = full_state.cpu().numpy() if isinstance(full_state, torch.Tensor) else full_state
            
            # 根据Env.get_state()的定义，状态向量为：
            # [p, pv, wind, flex_storage, current_storage, flex_vehicle, current_vehicle,
            #  flex_AC, current_AC, flex_wash, current_wash, t, total_bid]
            node_state = {
                'p': state_array[0],              # price
                'pv': state_array[1],             # pv generation
                'wind': state_array[2],           # wind generation
                'flex_storage': state_array[3],   # flex_storage
                'current_storage': state_array[4], # current_storage
                'flex_vehicle': state_array[5],   # flex_vehicle
                'current_vehicle': state_array[6], # current_vehicle
                'flex_AC': state_array[7],        # flex_AC
                'current_AC': state_array[8],     # current_AC
                'flex_wash': state_array[9],      # flex_wash
                'current_wash': state_array[10],  # current_wash
                't': int(time),                   # time step (0-23)
                'n': i                            # node identifier
            }
            node_states.append(node_state)
        
        return {'node_states': node_states, 'time': time}
    
    def get_obs_low(self, node_idx, bq, time):
        """
        获取低层agent的观测状态（单个节点的状态信息 + bid分配）
        对应 device_allocation 的输入：p, bq, pv, wind, flex_storage, current_storage,
        flex_vehicle, current_vehicle, flex_AC, current_AC, flex_wash, current_wash, t
        
        Args:
            node_idx: 节点索引
            bq: 分配给该节点的bid quantity (kW)
            time: 当前时刻 (0-23)
        
        Returns:
            dict: 节点状态信息
                - p, bq, pv, wind, flex_storage, current_storage,
                  flex_vehicle, current_vehicle, flex_AC, current_AC,
                  flex_wash, current_wash, t, n
        """
        # 获取完整状态（使用bq作为total_bid参数）
        full_state = self.sim.get_state(node_idx, bq, time)
        state_array = full_state.cpu().numpy() if isinstance(full_state, torch.Tensor) else full_state
        
        # 构建低层状态（包含bq）
        obs_low = {
            'p': float(state_array[0]),           # price (normalized 0-1)
            'bq': float(bq),                      # bid quantity for this node
            'pv': float(state_array[1]),          # pv generation
            'wind': float(state_array[2]),        # wind generation
            'flex_storage': float(state_array[3]), # flex_storage
            'current_storage': float(state_array[4]), # current_storage
            'flex_vehicle': float(state_array[5]), # flex_vehicle
            'current_vehicle': float(state_array[6]), # current_vehicle
            'flex_AC': float(state_array[7]),     # flex_AC
            'current_AC': float(state_array[8]),  # current_AC
            'flex_wash': float(state_array[9]),   # flex_wash
            'current_wash': float(state_array[10]), # current_wash
            't': int(time),                       # time step (0-23)
            'n': node_idx                         # node identifier
        }
        
        return obs_low

    def reset(self, day=None, time=0, total_bid=0.0):
        """
        重置环境
        
        Args:
            day: 第几天（0-6）
            time: 初始时刻 (0-23)
            total_bid: 初始总bid值
        """
        if day is not None:
            self.sim.reset(day=day)
        else:
            self.sim.reset()
        self.current_time = time
        self.current_total_bid = total_bid
        return self.get_obs_high(total_bid, time)

    def step(self, high_scores, total_bid, time, lam=1.0, mu=1.0,
             alpha=1.0, beta=0.0, gamma=10.0, alpha_T=2.0):
        """
        执行一步分层决策
        
        Args:
            high_scores: 高层agent输出的分数列表，shape (N,)，对应alpha_score的输出
            total_bid: 总bid值 (kW)
            time: 当前时刻 (0-23)
            lam, mu: 高层reward的权重参数
            alpha, beta, gamma, alpha_T: 低层reward的权重参数
        
        Returns:
            s_next: 下一时刻的高层观测
            r_high: 高层reward
            done: 是否结束
            extra: 额外信息
        """
        self.current_time = time
        self.current_total_bid = total_bid
        
        # 1) 高层动作：scores -> 分配比例 -> b_i (kW)
        # high_scores是每个节点的alpha_score输出（正分数）
        # 通过softmax得到分配比例，再通过feasible_alloc得到b_i
        B_t = total_bid
        caps = np.array([float('inf')] * self.N)  # 如果没有容量约束，设为无穷大
        # 如果有容量约束，可以从sim获取
        if hasattr(self.sim, 'get_caps'):
            try:
                _, caps = self.sim.get_caps()
            except:
                pass
        
        # scores通过softmax得到分配比例
        allocation_probs = softmax(high_scores)  # shape (N,)
        # 计算每个节点的bid分配
        b = allocation_probs * B_t  # shape (N,)
        # 应用容量约束
        b = np.minimum(b, caps)  # shape (N,)
        
        # 2) 宏步目标能量 (kWh if dt in hours)
        DT_h = self.dt_high / 60.0
        dt_h = self.dt_low / 60.0
        E_target = b * DT_h

        E_delivered = np.zeros(self.N, dtype=float)
        H_terms = {"revenue": 0.0, "deviation": 0.0, "viol": 0.0}

        # 3) 细步 rollout
        for k in range(self.K):
            remE = E_target - E_delivered  # kWh remaining

            actions = {}
            obs_low_cache = {}
            for i in range(self.N):
                # 获取低层观测状态（包含bq = b[i]）
                obs_low_i = self.get_obs_low(i, b[i], time)
                obs_low_cache[i] = obs_low_i

                # 低层动作：输出6维动作向量
                # [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
                if self.low_policies[i] is not None:
                    # 将状态字典转换为输入格式（根据policy的实现要求）
                    # 这里假设policy接受状态字典或向量
                    if hasattr(self.low_policies[i], 'act'):
                        a_i = self.low_policies[i].act(obs_low_i)
                    else:
                        # 如果没有act方法，尝试直接调用
                        a_i = self.low_policies[i](obs_low_i)
                else:
                    # 默认动作：零输出
                    a_i = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                # 确保是6维向量
                if isinstance(a_i, dict):
                    # 如果是字典，转换为数组
                    a_i = np.array([
                        a_i.get('pv_ratio', 0.0),
                        a_i.get('wind_ratio', 0.0),
                        a_i.get('storage_ratio', 0.0),
                        a_i.get('vehicle_ratio', 0.0),
                        a_i.get('ac_ratio', 0.0),
                        float(a_i.get('wash_on_number', 0.0))
                    ])
                elif not isinstance(a_i, np.ndarray):
                    a_i = np.array(a_i)
                
                # 确保是6维
                if len(a_i) != 6:
                    a_i = np.pad(a_i, (0, max(0, 6 - len(a_i))), 'constant')[:6]
                
                actions[i] = a_i

            # 计算实际输出的功率（用于更新E_delivered和计算revenue/deviation）
            p_grid_by_node = {}
            local_cost_by_node = {}
            
            for i in range(self.N):
                a_i = actions[i]
                obs_low_i = obs_low_cache[i]
                
                # 根据device_allocation的输出计算实际功率
                # pv_supply = pv_ratio * pv
                pv_supply = a_i[0] * obs_low_i['pv']
                # wind_supply = wind_ratio * wind
                wind_supply = a_i[1] * obs_low_i['wind']
                
                # storage和vehicle需要max_power
                flex_storage, flex_vehicle, _, _, storage_max_power, vehicle_max_power = self.sim.node_flex[i]
                # storage_supply = storage_ratio * max_power
                storage_supply = a_i[2] * storage_max_power
                # vehicle_supply = vehicle_ratio * max_power
                vehicle_supply = a_i[3] * vehicle_max_power
                
                # AC和Wash是消耗
                # ac_consumption = ac_ratio * flex_AC
                ac_consumption = a_i[4] * obs_low_i['flex_AC']
                # wash_consumption = wash_on_number * 40.0 (每台40kW)
                wash_consumption = a_i[5] * 40.0
                
                # 总功率输出 = 发电 - 用电
                p_export = pv_supply + wind_supply + storage_supply + vehicle_supply - ac_consumption - wash_consumption
                p_grid_by_node[i] = p_export
                local_cost_by_node[i] = 0.0  # 占位

            # 计算violation（这里简化处理，实际需要从sim获取）
            # 需要在计算低层reward之前定义
            viol = {}  # dict i->float
            for i in range(self.N):
                viol[i] = 0.0  # 占位，实际需要从sim获取

            # delivered：对外净出力
            for i in range(self.N):
                p_export = max(0.0, p_grid_by_node[i])
                E_delivered[i] += p_export * dt_h  # kWh

                # 低层 reward（你在训练低层时用）
                remE_i = E_target[i] - E_delivered[i]
                r_low = (-alpha * abs(remE_i)
                         -beta * local_cost_by_node[i]
                         -gamma * viol[i])
                if k == self.K - 1:
                    r_low += -alpha_T * abs(remE_i)

                # 这里可把 (obs, act, r, obs_next) 存进低层 replay buffer
                # store_low(i, obs_low_cache[i], actions[i], r_low, next_obs)
            
            # 计算revenue和deviation（按时间比例，只在最后一个细步累加一次）
            # 注意：每个细步对应dt_low分钟，但revenue和deviation应该只计算一次（整个宏步）
            # 与run_MILP.py对齐：revenue是price * elec，deviation是|total_elec - total_bid|
            if k == self.K - 1:  # 只在最后一个细步计算
                # 计算总功率
                total_elec = sum(p_grid_by_node.values())
                
                # 计算revenue：所有节点的price * elec之和
                revenue = 0.0
                for i in range(self.N):
                    price_val = float(self.sim.prices[i, time] if time < self.sim.prices.shape[1] else self.sim.prices[i, -1])
                    revenue += price_val * p_grid_by_node[i]
                
                # 计算deviation：|total_elec - total_bid|
                deviation = abs(total_elec - total_bid)
                
                # 累计高层统计量（只累加一次，而不是15次）
                H_terms["revenue"] += revenue
                H_terms["deviation"] += deviation
                H_terms["viol"] += sum(viol.values())

        # 4) 高层 reward（宏步一次）
        r_high = H_terms["revenue"] - lam * H_terms["deviation"] - mu * H_terms["viol"]

        # 获取下一时刻的高层观测
        s_next = self.get_obs_high(total_bid, time)  # 注意：这里可能需要time+1，取决于实现
        done = False  # 占位，实际需要根据时间判断
        
        extra = {"b": b, "E_target": E_target, "E_delivered": E_delivered, **H_terms}
        return s_next, r_high, done, extra

# =========================
# 6. 创建固定环境（避免每次训练重新构建）
# =========================
print("\n创建固定环境...")
# 创建用于训练的环境（使用预测数据）
env_train = Env(price, pv_pred, wind_pred, node_list, day=0,
                node_device_mapping=node_device_mapping,
                base_dir="/data2/zengjinwei/VPP_multinode/config/device")

# 创建用于评估的环境（使用真实数据）
env_eval_real = Env(price, pv_real, wind_real, node_list, day=0,
                    node_device_mapping=node_device_mapping,
                    base_dir="/data2/zengjinwei/VPP_multinode/config/device")

# 创建用于评估的环境（使用预测数据）
env_eval_pred = Env(price, pv_pred, wind_pred, node_list, day=0,
                    node_device_mapping=node_device_mapping,
                    base_dir="/data2/zengjinwei/VPP_multinode/config/device")

print("固定环境创建完成！")

# =========================
# 7. Agent网络定义
# =========================
# 高层Agent：输出每个节点的score（用于bid分配）
class HighLevelAgent(nn.Module):
    """
    高层Agent：输入所有节点状态，输出每个节点的score
    对应 alpha_score 函数的功能
    """
    def __init__(self, state_dim=13):
        super().__init__()
        # 状态维度：13维（p, pv, wind, flex_storage, current_storage, flex_vehicle, 
        #         current_vehicle, flex_AC, current_AC, flex_wash, current_wash, t, total_bid）
        # 但高层不使用total_bid，所以实际使用12维（或者可以包含）
        self.node_feature_net = nn.Sequential(
            nn.Linear(state_dim - 1, 64),  # 排除total_bid
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # 输出每个节点的score
        self.score_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),  # 添加LayerNorm
            nn.ReLU(),
            nn.Dropout(0.1),   # 添加Dropout
            nn.Linear(16, 1),
            nn.ReLU(),  # 确保score为正数
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, node_states):
        """
        node_states: (num_nodes, state_dim) - 所有节点的状态向量
        返回: (num_nodes,) - 每个节点的score（正数）
        """
        # 提取每个节点的特征（排除total_bid）
        node_features = []
        for i in range(node_states.shape[0]):
            state_i = node_states[i, :-1]  # 排除最后一个元素（total_bid）
            feat_i = self.node_feature_net(state_i)
            score_i = self.score_net(feat_i)
            node_features.append(score_i)
        
        scores = torch.cat(node_features, dim=0)  # (num_nodes, 1) -> (num_nodes,)
        scores = scores.squeeze(-1) if scores.dim() > 1 else scores
        # 确保score为正数
        scores = torch.clamp(scores, min=1e-6)
        return scores
    
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# 低层Agent：每个节点一个，输出6维动作
class LowLevelAgent(nn.Module):
    """
    低层Agent：输入节点状态（包含bq），输出6维动作
    对应 device_allocation 函数的功能
    """
    def __init__(self, state_dim=13):
        super().__init__()
        # 输入包含bq的状态
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),  # 添加LayerNorm稳定训练
            nn.ReLU(),
            nn.Dropout(0.1),   # 添加Dropout防止过拟合
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # 权重初始化
        self._initialize_weights()
        
        # 分别的输出层，使用不同的激活函数
        # pv_ratio, wind_ratio, ac_ratio: [0, 1] -> Sigmoid
        self.pv_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.wind_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.ac_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        
        # storage_ratio, vehicle_ratio: [-1, 1] -> Tanh
        self.storage_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.vehicle_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        
        # wash_on_number: >=0 -> ReLU
        self.wash_head = nn.Sequential(nn.Linear(64, 1), nn.ReLU())
    
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
    
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# =========================
# 8. 初始化Agent和优化器
# =========================
print("\n初始化Agent和优化器...")

# 创建高层Agent
high_level_agent = HighLevelAgent(state_dim=13).to(device)
high_level_optimizer = optim.Adam(high_level_agent.parameters(), lr=LR_HIGH)

# 创建低层Agent（共享参数：所有节点共享同一个agent）
shared_low_level_agent = LowLevelAgent(state_dim=13).to(device)
low_level_optimizer = optim.Adam(shared_low_level_agent.parameters(), lr=LR_LOW)

# 创建HierarchicalVPPWrapper并设置低层策略
hierarchical_wrapper = HierarchicalVPPWrapper(
    sim=env_train,
    N=NUM_NODES,
    dt_high_min=15,
    dt_low_min=1
)

# 设置低层策略（将agent包装为策略对象）
class PolicyWrapper:
    """将Agent包装为策略接口"""
    def __init__(self, agent):
        self.agent = agent
    
    def act(self, obs_dict):
        """接受状态字典，返回动作向量"""
        if isinstance(obs_dict, dict):
            # 将字典转换为tensor
            state_list = [
                obs_dict['p'],
                obs_dict['bq'],
                obs_dict['pv'],
                obs_dict['wind'],
                obs_dict['flex_storage'],
                obs_dict['current_storage'],
                obs_dict['flex_vehicle'],
                obs_dict['current_vehicle'],
                obs_dict['flex_AC'],
                obs_dict['current_AC'],
                obs_dict['flex_wash'],
                obs_dict['current_wash'],
                float(obs_dict['t']) / 23.0,  # 归一化时间
            ]
            state_tensor = torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state_tensor = obs_dict if isinstance(obs_dict, torch.Tensor) else torch.tensor(obs_dict, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            action = self.agent(state_tensor)
        return action.squeeze(0).cpu().numpy()

# 设置低层策略（所有节点共享同一个agent）
for i in range(NUM_NODES):
    hierarchical_wrapper.low_policies[i] = PolicyWrapper(shared_low_level_agent)

print(f"初始化完成：")
print(f"  高层Agent: 1个")
print(f"  低层Agent: 1个（共享参数，用于{NUM_NODES}个节点）")
print(f"  设备: {device}")

# =========================
# 9. 训练循环（使用四天完整数据训练）
# =========================
print("\n开始训练...")
NUM_TRAIN_EPOCHS = int(1e7)  # 训练轮数
TRAIN_DAYS = [0, 1, 2, 3]  # 使用前4天进行训练

# 早停机制参数
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 1e-3
EARLY_STOPPING_EVAL_INTERVAL = 5
EARLY_STOPPING_METRIC = 'total_deviation'
EARLY_STOPPING_MODE = 'min'

# 准备训练数据：四天的所有时刻
train_samples = []
for day in TRAIN_DAYS:
    for t in range(24):
        train_samples.append((day, t))

print(f"训练数据：{len(TRAIN_DAYS)} 天 × 24 小时 = {len(train_samples)} 个样本")
print(f"训练轮数：{NUM_TRAIN_EPOCHS}")
print()

# 早停机制初始化
if EARLY_STOPPING_ENABLED:
    print("="*60)
    print("早停机制配置:")
    print(f"  启用: {EARLY_STOPPING_ENABLED}")
    print(f"  评估间隔: 每 {EARLY_STOPPING_EVAL_INTERVAL} 个epoch")
    print(f"  容忍度 (patience): {EARLY_STOPPING_PATIENCE} 个epoch")
    print(f"  最小提升阈值: {EARLY_STOPPING_MIN_DELTA}")
    print(f"  监控指标: {EARLY_STOPPING_METRIC}")
    print(f"  优化方向: {EARLY_STOPPING_MODE}")
    print("="*60)
    print()
    
    best_metric_value = float('-inf') if EARLY_STOPPING_MODE == 'max' else float('inf')
    patience_counter = 0
    best_epoch = -1

# 初始化TensorBoard
log_dir = f"runs/hierarchical_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard日志目录: {log_dir}")
print(f"查看日志: tensorboard --logdir={log_dir}\n")

# 创建模型保存目录
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

# 模型保存函数
def save_checkpoint(epoch, high_level_agent, shared_low_level_agent, high_level_optimizer, 
                    low_level_optimizer, metric_value, train_reward=None, 
                    train_revenue=None, train_deviation=None, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'metric_value': metric_value,
        'train_reward': train_reward,
        'train_revenue': train_revenue,
        'train_deviation': train_deviation,
        'high_level_agent_state_dict': high_level_agent.state_dict(),
        'high_level_optimizer_state_dict': high_level_optimizer.state_dict(),
        'shared_low_level_agent_state_dict': shared_low_level_agent.state_dict(),
        'low_level_optimizer_state_dict': low_level_optimizer.state_dict(),
    }
    
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    torch.save(checkpoint, latest_path)
    
    if is_best:
        torch.save(checkpoint, best_model_path)
        print(f"  ✓ 保存最佳模型 (epoch {epoch+1}, metric={metric_value:.6f})")
        if train_reward is not None:
            print(f"    训练集性能: Reward={train_reward:.3f}, Revenue={train_revenue:.3f}, Deviation={train_deviation:.3f}")

# =========================
# 训练辅助函数：run_experiment（需要在训练循环之前定义）
# =========================
def run_experiment(day, use_real_data=False):
    """
    运行一天的实验模拟
    day: 第几天（0-6）
    use_real_data: 是否使用真实数据（False用预测，True用真实）
    """
    # 选择数据源
    if use_real_data:
        env = env_eval_real
        data_type = "REAL"
    else:
        env = env_eval_pred
        data_type = "PRED"
    
    # 创建评估用的wrapper（使用评估环境）
    eval_wrapper = HierarchicalVPPWrapper(
        sim=env,
        N=NUM_NODES,
        dt_high_min=15,
        dt_low_min=1
    )
    # 设置低层策略（共享同一个agent）
    for i in range(NUM_NODES):
        eval_wrapper.low_policies[i] = PolicyWrapper(shared_low_level_agent)
    
    # 重置环境到指定的day
    env.reset(day=day)
    eval_wrapper.reset(day=day, time=0, total_bid=0.0)
    
    # 获取当天的bid数据
    bid_day = bid[day, :].tolist()
    
    total_revenue = 0.0
    total_deviation = 0.0
    rewards = []
    
    # 评估时不需要梯度
    with torch.no_grad():
        # 逐时刻运行
        for t in range(24):
            total_bid_t = bid_day[t]
            
            # 获取高层观测状态
            obs_high = eval_wrapper.get_obs_high(total_bid_t, t)
            node_states = []
            for node_state_dict in obs_high['node_states']:
                state_array = [
                    node_state_dict['p'],
                    node_state_dict['pv'],
                    node_state_dict['wind'],
                    node_state_dict['flex_storage'],
                    node_state_dict['current_storage'],
                    node_state_dict['flex_vehicle'],
                    node_state_dict['current_vehicle'],
                    node_state_dict['flex_AC'],
                    node_state_dict['current_AC'],
                    node_state_dict['flex_wash'],
                    node_state_dict['current_wash'],
                    float(node_state_dict['t']) / 23.0,
                    0.0,
                ]
                node_states.append(torch.tensor(state_array, dtype=torch.float32, device=device))
            
            node_states_tensor = torch.stack(node_states)
            
            # 高层Agent输出scores
            high_scores = high_level_agent(node_states_tensor)
            
            # 执行一步分层决策（使用评估wrapper）
            s_next, r_high, done, extra = eval_wrapper.step(
                high_scores.cpu().numpy(),
                total_bid_t,
                t,
                lam=1.0,
                mu=1.0
            )
            
            # 与训练时保持一致：使用env.evaluate()计算revenue和deviation
            # 这样确保使用正确的评估环境数据
            actions_dict = {}
            for i in range(NUM_NODES):
                obs_low = eval_wrapper.get_obs_low(i, extra['b'][i], t)
                # 构建状态tensor（评估时不需要梯度）
                state_list = [
                    obs_low['p'],
                    obs_low['bq'],
                    obs_low['pv'],
                    obs_low['wind'],
                    obs_low['flex_storage'],
                    obs_low['current_storage'],
                    obs_low['flex_vehicle'],
                    obs_low['current_vehicle'],
                    obs_low['flex_AC'],
                    obs_low['current_AC'],
                    obs_low['flex_wash'],
                    obs_low['current_wash'],
                    float(obs_low['t']) / 23.0,
                ]
                state_tensor = torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0)
                action = shared_low_level_agent(state_tensor)
                actions_dict[i] = action.squeeze(0).detach()  # 评估时不需要梯度
            
            # 使用env.evaluate计算revenue和deviation（与训练时一致）
            revenue_tensor, deviation_tensor = env.evaluate(actions_dict, total_bid_t, t)
            
            revenue = revenue_tensor.item() if isinstance(revenue_tensor, torch.Tensor) else float(revenue_tensor)
            deviation = deviation_tensor.item() if isinstance(deviation_tensor, torch.Tensor) else float(deviation_tensor)
            
            reward = revenue * REWARD_SCALE - LAMBDA * deviation * REWARD_SCALE
            
            total_revenue += revenue
            total_deviation += deviation
            rewards.append(reward)
    
    return {
        "day": day,
        "data_type": data_type,
        "total_revenue": total_revenue,
        "total_deviation": total_deviation,
        "rewards": rewards
    }

# 全局step计数器
global_step = 0

# 训练循环
for epoch in range(NUM_TRAIN_EPOCHS):
    # 累积四天的总和（保持梯度）
    epoch_total_reward_tensor = None
    epoch_total_revenue_tensor = None
    epoch_total_deviation_tensor = None
    
    # 标量值用于记录
    epoch_total_reward_scalar = 0.0
    epoch_total_revenue_scalar = 0.0
    epoch_total_deviation_scalar = 0.0
    
    # 遍历四天的所有时刻
    for step_idx, (day, t) in enumerate(train_samples):
        # 重置环境（如果是新的一天）
        if step_idx == 0 or (step_idx > 0 and train_samples[step_idx-1][0] != day):
            env_train.reset(day=day)
            hierarchical_wrapper.reset(day=day, time=0, total_bid=0.0)
        
        # 获取该时刻的bid
        total_bid = bid[day, t]
        
        # 获取高层观测状态
        obs_high = hierarchical_wrapper.get_obs_high(total_bid, t)
        node_states = []
        for node_state_dict in obs_high['node_states']:
            state_array = [
                node_state_dict['p'],
                node_state_dict['pv'],
                node_state_dict['wind'],
                node_state_dict['flex_storage'],
                node_state_dict['current_storage'],
                node_state_dict['flex_vehicle'],
                node_state_dict['current_vehicle'],
                node_state_dict['flex_AC'],
                node_state_dict['current_AC'],
                node_state_dict['flex_wash'],
                node_state_dict['current_wash'],
                float(node_state_dict['t']) / 23.0,  # 归一化时间
                0.0,  # total_bid占位（高层不使用）
            ]
            node_states.append(torch.tensor(state_array, dtype=torch.float32, device=device))
        
        node_states_tensor = torch.stack(node_states)  # (NUM_NODES, 13)
        
        # 高层Agent输出scores
        high_scores = high_level_agent(node_states_tensor)  # (NUM_NODES,)
        
        # 执行一步分层决策
        s_next, r_high, done, extra = hierarchical_wrapper.step(
            high_scores.detach().cpu().numpy(),
            total_bid,
            t,
            lam=1.0,
            mu=1.0
        )
        
        # 计算reward（保持梯度）
        revenue = extra['revenue'] if 'revenue' in extra else torch.tensor(0.0, device=device)
        deviation = extra['deviation'] if 'deviation' in extra else torch.tensor(0.0, device=device)
        
        # 使用Env.evaluate计算revenue和deviation（保持梯度）
        # 这里需要重新计算以保持梯度连接
        actions_dict = {}
        for i in range(NUM_NODES):
            obs_low = hierarchical_wrapper.get_obs_low(i, extra['b'][i], t)
            # 构建状态tensor
            state_list = [
                obs_low['p'],
                obs_low['bq'],
                obs_low['pv'],
                obs_low['wind'],
                obs_low['flex_storage'],
                obs_low['current_storage'],
                obs_low['flex_vehicle'],
                obs_low['current_vehicle'],
                obs_low['flex_AC'],
                obs_low['current_AC'],
                obs_low['flex_wash'],
                obs_low['current_wash'],
                float(obs_low['t']) / 23.0,
            ]
            state_tensor = torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0)
            action = shared_low_level_agent(state_tensor)
            actions_dict[i] = action.squeeze(0)
        
        # 使用Env.evaluate计算revenue和deviation（保持梯度）
        revenue_tensor, deviation_tensor = env_train.evaluate(actions_dict, total_bid, t)
        
        # 计算reward（使用改进的缩放因子）
        reward = revenue_tensor * REWARD_SCALE - LAMBDA * deviation_tensor * REWARD_SCALE
        
        # 累积总和（保持梯度）
        if epoch_total_reward_tensor is None:
            epoch_total_reward_tensor = reward
            epoch_total_revenue_tensor = revenue_tensor
            epoch_total_deviation_tensor = deviation_tensor
        else:
            epoch_total_reward_tensor = epoch_total_reward_tensor + reward
            epoch_total_revenue_tensor = epoch_total_revenue_tensor + revenue_tensor
            epoch_total_deviation_tensor = epoch_total_deviation_tensor + deviation_tensor
        
        # 获取标量值用于记录
        reward_val = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
        revenue_val = revenue_tensor.item() if isinstance(revenue_tensor, torch.Tensor) else float(revenue_tensor)
        deviation_val = deviation_tensor.item() if isinstance(deviation_tensor, torch.Tensor) else float(deviation_tensor)
        
        # 记录每个step的指标
        writer.add_scalar('Train/Step_Reward', reward_val, global_step)
        writer.add_scalar('Train/Step_Revenue', revenue_val, global_step)
        writer.add_scalar('Train/Step_Deviation', deviation_val, global_step)
        writer.add_scalar('Train/Step_TotalBid', float(total_bid), global_step)
        
        # 累积统计信息
        epoch_total_reward_scalar += reward_val
        epoch_total_revenue_scalar += revenue_val
        epoch_total_deviation_scalar += deviation_val
        
        global_step += 1
    
    # 使用四天的总和进行反向传播
    if epoch == 0:
        print(f"调试信息 (Epoch 0):")
        print(f"  Total Reward requires_grad: {epoch_total_reward_tensor.requires_grad}")
        print(f"  Total Revenue requires_grad: {epoch_total_revenue_tensor.requires_grad}")
        print(f"  Total Deviation requires_grad: {epoch_total_deviation_tensor.requires_grad}")
    
    # 计算loss（最大化总和reward）
    loss = -epoch_total_reward_tensor
    
    # 反向传播
    high_level_optimizer.zero_grad()
    low_level_optimizer.zero_grad()
    
    loss.backward()
    
    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(high_level_agent.parameters(), MAX_GRAD_NORM)
    torch.nn.utils.clip_grad_norm_(shared_low_level_agent.parameters(), MAX_GRAD_NORM)
    
    # 记录梯度信息
    has_gradient = False
    grad_norms = {}
    
    # 检查高层agent的梯度
    high_grad_norm = 0.0
    param_count = 0
    for param in high_level_agent.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            high_grad_norm += param_grad_norm.item() ** 2
            param_count += 1
            has_gradient = True
    high_grad_norm = high_grad_norm ** (1. / 2) if param_count > 0 else 0.0
    grad_norms['high_level'] = high_grad_norm
    writer.add_scalar('Gradient/HighLevel_grad_norm', high_grad_norm, epoch)
    
    # 检查低层agent的梯度（共享参数）
    low_grad_norm = 0.0
    param_count = 0
    for param in shared_low_level_agent.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            low_grad_norm += param_grad_norm.item() ** 2
            param_count += 1
            has_gradient = True
    low_grad_norm = low_grad_norm ** (1. / 2) if param_count > 0 else 0.0
    grad_norms['low_level_shared'] = low_grad_norm
    writer.add_scalar('Gradient/LowLevel_Shared_grad_norm', low_grad_norm, epoch)
    
    if epoch == 0:
        if not has_gradient:
            print("⚠️  警告: 反向传播后没有检测到梯度！")
        else:
            print(f"✓ 检测到梯度，梯度范数: {grad_norms}")
        print()
    
    # 更新优化器
    high_level_optimizer.step()
    low_level_optimizer.step()
    
    # 获取总和值
    total_reward_tensor_val = epoch_total_reward_tensor.item()
    total_revenue_tensor_val = epoch_total_revenue_tensor.item()
    total_deviation_tensor_val = epoch_total_deviation_tensor.item()
    
    # 记录epoch级别的指标
    writer.add_scalar('Train/Epoch_Total_Reward', total_reward_tensor_val, epoch)
    writer.add_scalar('Train/Epoch_Total_Revenue', total_revenue_tensor_val, epoch)
    writer.add_scalar('Train/Epoch_Total_Deviation', total_deviation_tensor_val, epoch)
    
    print(f"Epoch {epoch+1:03d}/{NUM_TRAIN_EPOCHS} | "
          f"Total Reward = {total_reward_tensor_val:.3f} | "
          f"Total Revenue = {total_revenue_tensor_val:.3f} | "
          f"Total Deviation = {total_deviation_tensor_val:.3f}")
    
    # 早停机制：定期评估
    should_evaluate = EARLY_STOPPING_ENABLED and (epoch + 1) % EARLY_STOPPING_EVAL_INTERVAL == 0
    should_stop = False
    
    if should_evaluate:
        print(f"\n[评估] Epoch {epoch+1}: 在训练集上评估模型（预测数据）...")
        
        # 在训练集上评估（使用预测数据，评估前4天）
        eval_total_reward = 0.0
        eval_total_revenue = 0.0
        eval_total_deviation = 0.0
        
        with torch.no_grad():
            for eval_day in TRAIN_DAYS:
                result = run_experiment(eval_day, use_real_data=False)
                eval_total_revenue += result['total_revenue']
                eval_total_deviation += result['total_deviation']
                eval_total_reward += (result['total_revenue'] * REWARD_SCALE - LAMBDA * result['total_deviation'] * REWARD_SCALE)
        
        # 监控deviation
        current_metric_value = eval_total_deviation
        EARLY_STOPPING_MODE = 'min'
        
        # 记录训练集评估指标
        writer.add_scalar('TrainSet_Eval/Total_Reward', eval_total_reward, epoch)
        writer.add_scalar('TrainSet_Eval/Total_Revenue', eval_total_revenue, epoch)
        writer.add_scalar('TrainSet_Eval/Total_Deviation', eval_total_deviation, epoch)
        
        print(f"  [训练集评估结果] Total Reward = {eval_total_reward:.3f} | "
              f"Total Revenue = {eval_total_revenue:.3f} | "
              f"Total Deviation = {eval_total_deviation:.3f}")
        
        # 检查是否有提升
        is_better = False
        is_first_eval = (best_epoch == -1)
        
        if is_first_eval:
            is_better = True
            improvement = 0.0
            relative_improvement = 0.0
        else:
            if current_metric_value < best_metric_value:
                improvement = best_metric_value - current_metric_value
                if abs(best_metric_value) > 1e-10:
                    relative_improvement = improvement / abs(best_metric_value)
                else:
                    relative_improvement = float('inf')
                
                if relative_improvement >= EARLY_STOPPING_MIN_DELTA:
                    is_better = True
        
        if is_better:
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(epoch, high_level_agent, shared_low_level_agent, high_level_optimizer,
                          low_level_optimizer, best_metric_value,
                          train_reward=eval_total_reward,
                          train_revenue=eval_total_revenue,
                          train_deviation=eval_total_deviation,
                          is_best=True)
            
            if is_first_eval:
                print(f"  ✓ 首次评估，保存初始最佳模型: {EARLY_STOPPING_METRIC} = {best_metric_value:.6f}")
            else:
                print(f"  ✓ 性能提升！{EARLY_STOPPING_METRIC}: {best_metric_value:.6f} "
                      f"(提升 {improvement:.6f}, 相对提升 {relative_improvement*100:.4f}%)")
        else:
            patience_counter += EARLY_STOPPING_EVAL_INTERVAL
            print(f"  ⚠ 无提升 (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                should_stop = True
                print(f"\n{'='*60}")
                print(f"早停触发！")
                print(f"  最佳 {EARLY_STOPPING_METRIC}: {best_metric_value:.6f} (epoch {best_epoch+1})")
                print(f"  当前 {EARLY_STOPPING_METRIC}: {current_metric_value:.6f} (epoch {epoch+1})")
                print(f"  连续 {patience_counter} 个epoch无提升")
                print(f"{'='*60}\n")
        
        print()
    
    # 定期刷新writer
    if (epoch + 1) % 10 == 0:
        writer.flush()
    
    # 早停检查
    if should_stop:
        print(f"训练在第 {epoch+1} 个epoch提前停止（早停机制）")
        break

# 训练结束处理
if EARLY_STOPPING_ENABLED and best_epoch >= 0:
    print(f"\n{'='*60}")
    print(f"训练总结:")
    print(f"  总训练轮数: {epoch+1}")
    print(f"  最佳epoch: {best_epoch+1}")
    print(f"  最佳 {EARLY_STOPPING_METRIC}: {best_metric_value:.6f}")
    print(f"  最佳模型已保存至: {best_model_path}")
    
    # 加载最佳模型
    print(f"\n加载最佳模型...")
    checkpoint = torch.load(best_model_path)
    high_level_agent.load_state_dict(checkpoint['high_level_agent_state_dict'])
    high_level_optimizer.load_state_dict(checkpoint['high_level_optimizer_state_dict'])
    shared_low_level_agent.load_state_dict(checkpoint['shared_low_level_agent_state_dict'])
    low_level_optimizer.load_state_dict(checkpoint['low_level_optimizer_state_dict'])
    
    # 更新wrapper中的策略
    for i in range(NUM_NODES):
        hierarchical_wrapper.low_policies[i] = PolicyWrapper(shared_low_level_agent)
    
    print("✓ 最佳模型已加载（包括高层Agent和所有低层Agent）")
    
    if 'train_reward' in checkpoint and checkpoint['train_reward'] is not None:
        print(f"\n最佳模型在训练集上的性能:")
        print(f"  Reward: {checkpoint['train_reward']:.3f}")
        print(f"  Revenue: {checkpoint['train_revenue']:.3f}")
        print(f"  Deviation: {checkpoint['train_deviation']:.3f}")
    print(f"{'='*60}\n")
else:
    # 如果没有早停，也加载最新模型
    print(f"\n加载最新模型...")
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    if os.path.exists(latest_path):
        checkpoint = torch.load(latest_path)
        high_level_agent.load_state_dict(checkpoint['high_level_agent_state_dict'])
        high_level_optimizer.load_state_dict(checkpoint['high_level_optimizer_state_dict'])
        shared_low_level_agent.load_state_dict(checkpoint['shared_low_level_agent_state_dict'])
        low_level_optimizer.load_state_dict(checkpoint['low_level_optimizer_state_dict'])
        
        # 更新wrapper中的策略
        for i in range(NUM_NODES):
            hierarchical_wrapper.low_policies[i] = PolicyWrapper(shared_low_level_agent)
        print("✓ 最新模型已加载\n")

print("\n训练完成！\n")

# 关闭训练阶段的writer
writer.flush()

# =========================
# 10. 在真实场景上测试（迁移测试）- 与run_MILP.py和run_group_rl.py对齐
# =========================
print("="*60)
print("开始真实场景迁移测试...")
print("="*60)
print("注意：以下测试使用真实数据，评估模型在真实场景下的迁移性能\n")

TEST_DAYS = [0, 1, 2, 3]  # 测试前4天（与run_MILP.py和run_group_rl.py对齐）

# 使用真实数据测试（迁移测试）
print("使用真实数据测试（迁移测试）...")
results_real = []

for day_idx, day in enumerate(TEST_DAYS):
    print(f"\n测试第 {day+1} 天（真实环境）...")
    
    # 使用真实数据测试（迁移测试）
    result_real = run_experiment(day, use_real_data=True)
    results_real.append(result_real)
    
    # 计算reward（用于记录）
    total_reward = result_real['total_revenue'] * REWARD_SCALE - LAMBDA * result_real['total_deviation'] * REWARD_SCALE
    
    print(f"[{result_real['data_type']}] Day {day+1} | "
          f"Revenue = {result_real['total_revenue']:.3f} | "
          f"Deviation = {result_real['total_deviation']:.3f} | "
          f"Reward = {total_reward:.3f}")

# 计算真实数据汇总
total_real_revenue = sum(r['total_revenue'] for r in results_real)
total_real_deviation = sum(r['total_deviation'] for r in results_real)
total_real_reward = total_real_revenue * REWARD_SCALE - LAMBDA * total_real_deviation * REWARD_SCALE
avg_real_revenue = total_real_revenue / len(TEST_DAYS)
avg_real_deviation = total_real_deviation / len(TEST_DAYS)
avg_real_reward = total_real_reward / len(TEST_DAYS)

print(f"\n{'='*60}")
print(f"{len(TEST_DAYS)} 天真实场景迁移测试汇总:")
print(f"{'='*60}")
print(f"总收益 (total_real_revenue): {total_real_revenue:.3f} | 平均收益: {avg_real_revenue:.3f}")
print(f"总偏差 (total_real_deviation): {total_real_deviation:.3f} | 平均偏差: {avg_real_deviation:.3f}")
print(f"总奖励 (total_real_reward): {total_real_reward:.3f} | 平均奖励: {avg_real_reward:.3f}")
print(f"{'='*60}")

# 打印每天的详细结果
print("\n每天详细结果（真实场景）:")
for r in results_real:
    day_reward = r['total_revenue'] * REWARD_SCALE - LAMBDA * r['total_deviation'] * REWARD_SCALE
    print(f"  Day {r['day']+1}: Revenue = {r['total_revenue']:.3f}, Deviation = {r['total_deviation']:.3f}, Reward = {day_reward:.3f}")
print(f"{'='*60}")
