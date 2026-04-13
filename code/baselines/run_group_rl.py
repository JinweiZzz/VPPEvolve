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
GROUP_SIZE = 5
NUM_GROUPS = 3

# STATE_DIM 包含 program_seed_joint.py 中的所有参数：
# p, pv, wind, flex_storage, current_storage, flex_vehicle, current_vehicle,
# flex_AC, current_AC, flex_wash, current_wash, t, total_bid
STATE_DIM = 13  # 13维状态向量
# ACTION_DIM = 6: 对应 program_seed_joint.py 的输出格式
# [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
ACTION_DIM = 6
LAMBDA = 1e3                # deviation penalty
LR = 1e-5
# REWARD_TRANSITION_EPOCH 已移除，改为基于deviation早停自动切换阶段

# =========================
# 3. 分组
# =========================
# 根据设备配置特征将节点分组
# 组1：储能+车辆为主（5个站点）
group_1_nodes = [
    '777008009',  # 特殊站点，有非常多的电动车（1000辆）
    '山西.古交发电厂/220kV.外母线',  # 车辆+储能（130车辆，200储能）
    '山西.开源路站/220kV.北母线',  # 中型站点，偏重储能（180储能，80车辆）
    '山西.温池站/220kV.南母线',  # 中型站点，风电和储能较多（14储能，60车辆）
    '山西.瑶河风电场/220kV.A母线',  # 小型站点，设备数量较少（100储能）
]

# 组2：用能设备为主（5个站点）
group_2_nodes = [
    '山西.云顶山站/220kV.B母线',  # 大型综合站点，仅有用能设备（80 wash，130 AC）
    '山西.冶峪站/220kV.B母线',  # 用能设备非常多（159 wash，210 AC）
    '山西.细米河风电场/220kV.A母线',  # 小型站点（40车辆，30 wash，50 AC）
    '山西.邑垣光伏电站/220kV.A母线',  # 大型综合站点，光伏为主（250 wash，140 AC）
    '山西.彤欧光伏电站/220kV.A母线',  # 超大型电动车站点（300车辆）
]

# 组3：综合型/特殊配置（5个站点）
group_3_nodes = [
    '华北.武乡/500kV.2母线',  # 特殊站点，储能为重（50储能）
    '山西.兆光电厂/220kV.东母线',  # 储能为重（30储能）
    '山西.海会站/220kV.D母线',  # 中型站点，光伏和空调较多（20储能，45车辆，40 wash，160 AC）
    '山西.禹王石光伏电站/220kV.A母线',  # 中型站点，各类设备均衡（120储能，80车辆，70 wash，110 AC）
    '山西.翠微站/220kV.西母线',  # 热电厂站点，储能和车辆较多（22储能）
]

# 将节点名称转换为索引
def get_node_indices(node_names, node_list):
    """根据节点名称列表获取对应的索引列表"""
    indices = []
    for node_name in node_names:
        if node_name in node_list:
            indices.append(node_list.index(node_name))
        else:
            print(f"警告: 节点 '{node_name}' 不在节点列表中")
    return indices

# 创建分组（使用索引）
groups = {
    0: get_node_indices(group_1_nodes, node_list),  # 组1：储能+车辆为主
    1: get_node_indices(group_2_nodes, node_list),  # 组2：用能设备为主
    2: get_node_indices(group_3_nodes, node_list),  # 组3：综合型/特殊配置
}

# 移除空组
groups = {g: nodes for g, nodes in groups.items() if len(nodes) > 0}
NUM_GROUPS = len(groups)

# 打印分组信息
print(f"\n节点分组完成！共 {NUM_GROUPS} 组")
for group_id, node_indices in groups.items():
    group_name = ['储能+车辆为主', '用能设备为主', '综合型/特殊配置'][group_id]
    node_names = [node_list[idx] for idx in node_indices]
    print(f"  组{group_id} ({group_name}): {len(node_indices)}个节点")
    for idx, node_name in zip(node_indices, node_names):
        print(f"    [{idx}] {node_name}")
print()

# =========================
# 4. Agent 定义
# =========================
class BidAllocatingNet(nn.Module):
    """
    第一层：Bid分配网络
    输入：所有节点的状态信息（用于了解各节点的能力）
    输出：每个节点分配的bid比例（使用softmax确保总和为1）
    """
    def __init__(self, state_dim, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 特征提取：处理所有节点的状态信息
        # 输入：每个节点的状态向量（state_dim维）
        # 输出：每个节点的特征向量
        self.node_feature_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # 全局特征聚合：将所有节点的特征聚合
        # 使用注意力机制或简单的平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 或者使用简单的mean
        
        # 分配网络：基于节点特征和全局特征决定分配比例
        # 输入：节点特征 + 全局特征
        self.allocating_net = nn.Sequential(
            nn.Linear(32 + 32, 64),  # 节点特征 + 全局特征
            nn.ReLU(),
            nn.Linear(64, 1),  # 输出每个节点的分配权重（未归一化）
        )
    
    def forward(self, node_states):
        """
        node_states: (num_nodes, state_dim) - 所有节点的状态向量
        返回: (num_nodes,) - 每个节点的bid分配比例（总和为1）
        """
        # 提取每个节点的特征
        node_features = self.node_feature_net(node_states)  # (num_nodes, 32)
        
        # 计算全局特征（所有节点的平均）
        global_feature = node_features.mean(dim=0, keepdim=True)  # (1, 32)
        
        # 为每个节点拼接节点特征和全局特征
        expanded_global = global_feature.expand(node_features.shape[0], -1)  # (num_nodes, 32)
        combined_features = torch.cat([node_features, expanded_global], dim=-1)  # (num_nodes, 64)
        
        # 计算每个节点的分配权重（未归一化）
        allocation_logits = self.allocating_net(combined_features)  # (num_nodes, 1)
        allocation_logits = allocation_logits.squeeze(-1)  # (num_nodes,)
        
        # 使用softmax确保总和为1
        allocation_probs = torch.softmax(allocation_logits, dim=0)  # (num_nodes,)
        
        return allocation_probs

class GroupAgent(nn.Module):
    """
    第二层：Actor网络
    输入：节点状态（包含分配的bid信息）
    输出：动作向量
    """
    def __init__(self):
        super().__init__()
        # 共享的特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # 分别的输出层，使用不同的激活函数
        # pv_ratio, wind_ratio, ac_ratio: [0, 1] -> Sigmoid
        self.pv_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.wind_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.ac_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        
        # storage_ratio, vehicle_ratio: [-1, 1] -> Tanh
        self.storage_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.vehicle_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        
        # wash_on_number: >=0 -> ReLU (保持浮点数以维持梯度)
        self.wash_head = nn.Sequential(nn.Linear(64, 1), nn.ReLU())

    def forward(self, s, allocated_bid=None):
        """
        输出动作向量，格式与 program_seed_joint.py 一致：
        [pv_ratio, wind_ratio, storage_ratio, vehicle_ratio, ac_ratio, wash_on_number]
        
        s: 状态向量（STATE_DIM维），如果allocated_bid不为None，则s中的total_bid会被替换为allocated_bid
        allocated_bid: 分配给该节点的bid值（可选）
        
        pv_ratio: [0, 1] - PV发电并网比例
        wind_ratio: [0, 1] - Wind发电并网比例
        storage_ratio: [-1, 1] - 储能充放电比例 (正=放电, 负=充电)
        vehicle_ratio: [-1, 1] - EV充放电比例 (正=放电, 负=充电)
        ac_ratio: [0, 1] - AC功率比例
        wash_on_number: >=0 - Wash开启数量 (浮点数，保持梯度)
        """
        # 如果提供了allocated_bid，替换状态中的total_bid（状态向量的最后一个元素）
        if allocated_bid is not None:
            s = s.clone()
            s[-1] = allocated_bid  # 替换total_bid
        
        features = self.feature_net(s)
        
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
# 5. 辅助函数：计算设备灵活性
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
# 6. 环境（使用真实数据）
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
# 6. 初始化 agent 和优化器
# =========================
# 创建Bid分配网络（全局共享，用于所有节点）
bid_allocating_net = BidAllocatingNet(STATE_DIM, NUM_NODES).to(device)
bid_allocating_optimizer = optim.Adam(bid_allocating_net.parameters(), lr=LR)

# 创建组Agent（每个组一个）
agents = {g: GroupAgent().to(device) for g in groups}
optimizers = {g: optim.Adam(agents[g].parameters(), lr=LR) for g in agents}

print(f"初始化了 Bid分配网络 (在 {device} 上)")
print(f"初始化了 {len(agents)} 个组Agent (在 {device} 上):")
for g, node_ids in groups.items():
    print(f"  Group {g}: {len(node_ids)} 个节点 {node_ids}")
print()

# =========================
# 7. 单步调度 + 训练
# =========================
def run_one_step(env, total_bid, time, current_epoch=None):
    """
    分层决策：
    1. 第一层：Bid分配网络决定每个节点分配多少bid
    2. 第二层：Actor网络根据分配的bid输出动作
    
    Args:
        env: 环境对象
        total_bid: 总bid值
        time: 当前时刻
        current_epoch: 当前训练epoch（用于动态调整reward计算方式）
    """
    actions = {}
    states = {}
    
    # 第一步：收集所有节点的状态（用于bid分配）
    all_node_states = []
    node_id_to_index = {}  # 映射：node_id -> 在all_node_states中的索引
    
    for node_id in range(NUM_NODES):
        s = env.get_state(node_id, total_bid, time)
        s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
        all_node_states.append(s_tensor)
        node_id_to_index[node_id] = len(all_node_states) - 1
    
    # 将所有节点状态堆叠成张量
    all_node_states_tensor = torch.stack(all_node_states)  # (NUM_NODES, STATE_DIM)
    
    # 第二步：使用Bid分配网络决定每个节点的bid分配比例
    allocation_probs = bid_allocating_net(all_node_states_tensor)  # (NUM_NODES,)
    
    # 将分配比例转换为实际的bid值
    total_bid_tensor = torch.tensor(total_bid, dtype=torch.float32, device=device)
    allocated_bids = allocation_probs * total_bid_tensor  # (NUM_NODES,)
    
    # 第三步：每个组的Agent根据分配的bid输出动作
    for g, node_ids in groups.items():
        agent = agents[g]
        for node_id in node_ids:
            s = env.get_state(node_id, total_bid, time)  # 获取状态（包含原始total_bid）
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device)
            
            # 获取分配给该节点的bid值
            allocated_bid = allocated_bids[node_id_to_index[node_id]]
            
            # 将分配的bid传递给Agent（Agent会替换状态中的total_bid）
            a = agent(s_tensor, allocated_bid=allocated_bid)
            actions[node_id] = a
            states[node_id] = s

    revenue, deviation = env.evaluate(actions, total_bid, time)
    
    # 动态reward计算：
    # - 第一阶段：只考虑deviation（最小化偏差），直到deviation无法下降触发早停
    # - 第二阶段：考虑revenue和deviation（最大化收益，同时控制偏差）
    # 通过全局变量phase2_started控制，而不是固定的epoch数
    if current_epoch is not None:
        # 检查是否已进入第二阶段（通过全局变量）
        if 'phase2_started' in globals() and phase2_started:
            # 第二阶段：考虑revenue和deviation
            reward = revenue / 1e9 - LAMBDA * deviation / 1e9
        else:
            # 第一阶段：只考虑deviation（负号表示要最小化deviation）
            reward = -LAMBDA * deviation / 1e9
    else:
        # 评估时，如果没有提供epoch，默认使用完整公式
        reward = revenue / 1e9 - LAMBDA * deviation / 1e9
    
    return reward, revenue, deviation

# =========================
# 8. 实验模拟模块
# =========================
def run_experiment(day, use_real_data=False):
    """
    运行一天的实验模拟
    day: 第几天（0-6）
    use_real_data: 是否使用真实数据（False用预测，True用真实）
    """
    # 选择数据源
    if use_real_data:
        pv_data = pv_real
        wind_data = wind_real
        data_type = "REAL"
    else:
        pv_data = pv_pred
        wind_data = wind_pred
        data_type = "PRED"
    
    # 使用固定环境（根据数据类型选择）
    if use_real_data:
        env = env_eval_real
    else:
        env = env_eval_pred
    
    # 重置环境到指定的day
    env.reset(day=day)
    
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
            reward, revenue, deviation = run_one_step(env, total_bid_t, t)
            
            total_revenue += revenue.item() if isinstance(revenue, torch.Tensor) else float(revenue)
            total_deviation += deviation.item() if isinstance(deviation, torch.Tensor) else float(deviation)
            rewards.append(reward.item() if isinstance(reward, torch.Tensor) else float(reward))
    
    return {
        "day": day,
        "data_type": data_type,
        "total_revenue": total_revenue,
        "total_deviation": total_deviation,
        "rewards": rewards
    }

# =========================
# 9. 创建固定环境（避免每次训练重新构建）
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
# 10. 训练循环（使用四天完整数据训练）
# =========================
print("\n开始训练...")
NUM_TRAIN_EPOCHS = int(1e7)  # 训练轮数
TRAIN_DAYS = [0, 1, 2, 3]  # 使用前4天进行训练

# 早停机制参数
EARLY_STOPPING_ENABLED = True  # 是否启用早停
EARLY_STOPPING_PATIENCE = 50  # 容忍多少个epoch没有提升
EARLY_STOPPING_MIN_DELTA = 1e-3  # 最小提升阈值（相对值，0.1%）
EARLY_STOPPING_EVAL_INTERVAL = 5  # 每N个epoch评估一次
EARLY_STOPPING_METRIC = 'total_deviation'  # 用于早停的指标：固定监控deviation
EARLY_STOPPING_MODE = 'min'  # 'min'表示越小越好（deviation越小越好）

# 准备训练数据：四天的所有时刻
train_samples = []
for day in TRAIN_DAYS:
    for t in range(24):
        train_samples.append((day, t))

print(f"训练数据：{len(TRAIN_DAYS)} 天 × 24 小时 = {len(train_samples)} 个样本")
print(f"训练轮数：{NUM_TRAIN_EPOCHS}")
print(f"Reward策略：第一阶段只考虑deviation，当deviation无法下降时自动切换到revenue+deviation联合优化")
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
    phase2_started = False  # 标志：是否已进入第二阶段（revenue+deviation联合优化）
else:
    # 如果早停未启用，也初始化phase2_started
    phase2_started = False

# 初始化TensorBoard
log_dir = f"runs/group_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard日志目录: {log_dir}")
print(f"查看日志: tensorboard --logdir={log_dir}\n")

# 创建模型保存目录
checkpoint_dir = os.path.join(log_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
best_model_path = os.path.join(checkpoint_dir, "best_model.pt")

# 模型保存函数
def save_checkpoint(epoch, bid_allocating_net, agents, bid_allocating_optimizer, optimizers, 
                    metric_value, train_reward=None, train_revenue=None, train_deviation=None, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'metric_value': metric_value,
        'train_reward': train_reward,  # 训练集上的reward
        'train_revenue': train_revenue,  # 训练集上的revenue
        'train_deviation': train_deviation,  # 训练集上的deviation
        'bid_allocating_net_state_dict': bid_allocating_net.state_dict(),
        'bid_allocating_optimizer_state_dict': bid_allocating_optimizer.state_dict(),
        'agents_state_dict': {g: agents[g].state_dict() for g in groups},
        'optimizers_state_dict': {g: optimizers[g].state_dict() for g in groups},
    }
    
    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    torch.save(checkpoint, latest_path)
    
    # 如果是最佳模型，保存为最佳模型
    if is_best:
        torch.save(checkpoint, best_model_path)
        print(f"  ✓ 保存最佳模型 (epoch {epoch+1}, metric={metric_value:.6f})")
        if train_reward is not None:
            print(f"    训练集性能: Reward={train_reward:.3f}, Revenue={train_revenue:.3f}, Deviation={train_deviation:.3f}")

# 全局step计数器
global_step = 0

for epoch in range(NUM_TRAIN_EPOCHS):
    # 累积四天的总和（保持梯度，用于反向传播）
    epoch_total_reward_tensor = None  # 保持梯度的tensor
    epoch_total_revenue_tensor = None
    epoch_total_deviation_tensor = None
    
    # 标量值用于记录（不保持梯度）
    epoch_total_reward_scalar = 0.0
    epoch_total_revenue_scalar = 0.0
    epoch_total_deviation_scalar = 0.0
    
    # 遍历四天的所有时刻
    for step_idx, (day, t) in enumerate(train_samples):
        # 使用固定环境，切换day并重置状态（如果是新的一天）
        if step_idx == 0 or (step_idx > 0 and train_samples[step_idx-1][0] != day):
            # 新的一天，重置环境
            env_train.reset(day=day)
        else:
            # 同一天的不同时刻，只切换day（如果day改变了）
            if env_train.day != day:
                env_train.set_day(day)
        
        # 获取该时刻的bid
        total_bid = bid[day, t]
        
        # 传递当前epoch信息，用于动态调整reward计算方式
        reward, revenue, deviation = run_one_step(env_train, total_bid, t, current_epoch=epoch)
        
        # 累积总和（保持梯度）
        if epoch_total_reward_tensor is None:
            epoch_total_reward_tensor = reward
            epoch_total_revenue_tensor = revenue
            epoch_total_deviation_tensor = deviation
        else:
            epoch_total_reward_tensor = epoch_total_reward_tensor + reward
            epoch_total_revenue_tensor = epoch_total_revenue_tensor + revenue
            epoch_total_deviation_tensor = epoch_total_deviation_tensor + deviation
        
        # 获取标量值用于记录
        reward_val = reward.item() if isinstance(reward, torch.Tensor) else float(reward)
        revenue_val = revenue.item() if isinstance(revenue, torch.Tensor) else float(revenue)
        deviation_val = deviation.item() if isinstance(deviation, torch.Tensor) else float(deviation)
        
        # 记录每个step的指标
        writer.add_scalar('Train/Step_Reward', reward_val, global_step)
        writer.add_scalar('Train/Step_Revenue', revenue_val, global_step)
        writer.add_scalar('Train/Step_Deviation', deviation_val, global_step)
        writer.add_scalar('Train/Step_TotalBid', float(total_bid), global_step)
        
        # 累积统计信息（标量值）
        epoch_total_reward_scalar += reward_val
        epoch_total_revenue_scalar += revenue_val
        epoch_total_deviation_scalar += deviation_val
        
        global_step += 1
    
    # 使用四天的总和进行反向传播
    # 检查reward是否有梯度（在反向传播前）
    if epoch == 0:
        print(f"调试信息 (Epoch 0):")
        print(f"  Total Reward requires_grad: {epoch_total_reward_tensor.requires_grad}")
        print(f"  Total Revenue requires_grad: {epoch_total_revenue_tensor.requires_grad}")
        print(f"  Total Deviation requires_grad: {epoch_total_deviation_tensor.requires_grad}")
    
    # 使用总和计算loss（最大化总和reward）
    loss = -epoch_total_reward_tensor  # maximize total reward
    
    # 反向传播（使用四天的总和）
    # 清零所有优化器的梯度（包括bid_allocating_net和所有组agents）
    bid_allocating_optimizer.zero_grad()
    for g in optimizers:
        optimizers[g].zero_grad()
    
    loss.backward()
    
    # 记录梯度信息并检查梯度是否存在
    has_gradient = False
    grad_norms = {}
    
    # 检查bid_allocating_net的梯度
    bid_allocating_grad_norm = 0.0
    bid_allocating_param_count = 0
    for param in bid_allocating_net.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2)
            bid_allocating_grad_norm += param_grad_norm.item() ** 2
            bid_allocating_param_count += 1
            has_gradient = True
    bid_allocating_grad_norm = bid_allocating_grad_norm ** (1. / 2) if bid_allocating_param_count > 0 else 0.0
    grad_norms['bid_allocating'] = bid_allocating_grad_norm
    writer.add_scalar('Gradient/BidAllocating_grad_norm', bid_allocating_grad_norm, epoch)
    
    # 检查每个组agent的梯度
    for g in groups:
        total_grad_norm = 0.0
        param_count = 0
        for param in agents[g].parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
                param_count += 1
                has_gradient = True
        total_grad_norm = total_grad_norm ** (1. / 2) if param_count > 0 else 0.0
        grad_norms[g] = total_grad_norm
        writer.add_scalar(f'Gradient/Group_{g}_grad_norm', total_grad_norm, epoch)
    
    # 在第一个epoch打印详细的梯度信息
    if epoch == 0:
        if not has_gradient:
            print("⚠️  警告: 反向传播后没有检测到梯度！")
        else:
            print(f"✓ 检测到梯度，梯度范数: {grad_norms}")
        print()
    
    # 更新所有优化器
    bid_allocating_optimizer.step()
    for g in optimizers:
        optimizers[g].step()
    
    # 使用四天的总和（而不是平均值）
    total_reward = epoch_total_reward_scalar  # 四天的总reward（标量值）
    total_revenue = epoch_total_revenue_scalar  # 四天的总revenue（标量值）
    total_deviation = epoch_total_deviation_scalar  # 四天的总deviation（标量值）
    
    # 获取tensor的总和值（用于记录）
    total_reward_tensor_val = epoch_total_reward_tensor.item() if isinstance(epoch_total_reward_tensor, torch.Tensor) else float(epoch_total_reward_tensor)
    total_revenue_tensor_val = epoch_total_revenue_tensor.item() if isinstance(epoch_total_revenue_tensor, torch.Tensor) else float(epoch_total_revenue_tensor)
    total_deviation_tensor_val = epoch_total_deviation_tensor.item() if isinstance(epoch_total_deviation_tensor, torch.Tensor) else float(epoch_total_deviation_tensor)
    
    # 记录epoch级别的指标（使用总和）
    writer.add_scalar('Train/Epoch_Total_Reward', total_reward_tensor_val, epoch)
    writer.add_scalar('Train/Epoch_Total_Revenue', total_revenue_tensor_val, epoch)
    writer.add_scalar('Train/Epoch_Total_Deviation', total_deviation_tensor_val, epoch)
    
    # 记录每个agent的参数统计
    for g in groups:
        for name, param in agents[g].named_parameters():
            writer.add_histogram(f'Parameters/Group_{g}_{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/Group_{g}_{name}', param.grad.data, epoch)
    
    # 显示reward计算方式
    if 'phase2_started' in globals() and phase2_started:
        reward_mode = "revenue + deviation (Phase 2)"
    else:
        reward_mode = "deviation only (Phase 1)"
    
    print(f"Epoch {epoch+1:03d}/{NUM_TRAIN_EPOCHS} | "
          f"Total Reward = {total_reward_tensor_val:.3f} | "
          f"Total Revenue = {total_revenue_tensor_val:.3f} | "
          f"Total Deviation = {total_deviation_tensor_val:.3f} | "
          f"Reward Mode: {reward_mode}")
    
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
                result = run_experiment(eval_day, use_real_data=False)  # 使用预测数据（训练集）
                eval_total_revenue += result['total_revenue']
                eval_total_deviation += result['total_deviation']
                eval_total_reward += (result['total_revenue'] / 1e9 - LAMBDA * result['total_deviation'] / 1e9)
        
        # 根据阶段选择监控指标
        # 第一阶段：监控deviation（最小化）
        # 第二阶段：监控reward（最大化）
        if not phase2_started:
            # 第一阶段：监控deviation
            current_metric_value = eval_total_deviation
            EARLY_STOPPING_MODE = 'min'  # 确保是最小化模式
        else:
            # 第二阶段：监控reward
            current_metric_value = eval_total_reward
            EARLY_STOPPING_MODE = 'max'  # 切换到最大化模式
        
        # 记录训练集评估指标
        writer.add_scalar('TrainSet_Eval/Total_Reward', eval_total_reward, epoch)
        writer.add_scalar('TrainSet_Eval/Total_Revenue', eval_total_revenue, epoch)
        writer.add_scalar('TrainSet_Eval/Total_Deviation', eval_total_deviation, epoch)
        
        print(f"  [训练集评估结果] Total Reward = {eval_total_reward:.3f} | "
              f"Total Revenue = {eval_total_revenue:.3f} | "
              f"Total Deviation = {eval_total_deviation:.3f}")
        
        # 检查是否有提升
        is_better = False
        is_first_eval = (best_epoch == -1)  # 是否是第一次评估
        
        if is_first_eval:
            # 第一次评估，直接保存为最佳模型
            is_better = True
            improvement = 0.0
            relative_improvement = 0.0
        else:
            # 后续评估，检查是否有提升
            if EARLY_STOPPING_MODE == 'max':
                # 对于最大化指标，检查是否提升超过阈值
                if current_metric_value > best_metric_value:
                    improvement = current_metric_value - best_metric_value
                    # 计算相对提升
                    if abs(best_metric_value) > 1e-10:  # 避免除零
                        relative_improvement = improvement / abs(best_metric_value)
                    else:
                        relative_improvement = float('inf')
                    
                    if relative_improvement >= EARLY_STOPPING_MIN_DELTA:
                        is_better = True
            else:  # 'min'
                # 对于最小化指标，检查是否降低超过阈值
                if current_metric_value < best_metric_value:
                    improvement = best_metric_value - current_metric_value
                    # 计算相对提升（降低）
                    if abs(best_metric_value) > 1e-10:  # 避免除零
                        relative_improvement = improvement / abs(best_metric_value)
                    else:
                        relative_improvement = float('inf')
                    
                    if relative_improvement >= EARLY_STOPPING_MIN_DELTA:
                        is_better = True
        
        if is_better:
            # 有提升，更新最佳值
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型（记录训练集上的性能）
            save_checkpoint(epoch, bid_allocating_net, agents, bid_allocating_optimizer, optimizers, 
                          best_metric_value, 
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
            # 没有提升，增加patience计数器
            patience_counter += EARLY_STOPPING_EVAL_INTERVAL
            print(f"  ⚠ 无提升 (patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
            # 检查是否应该触发阶段切换
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                if not phase2_started:
                    # 第一阶段结束：deviation无法下降，切换到第二阶段
                    phase2_started = True
                    patience_counter = 0  # 重置patience计数器
                    best_metric_value = float('-inf')  # 重置最佳值（第二阶段监控reward，越大越好）
                    best_epoch = -1  # 重置最佳epoch
                    print(f"\n{'='*60}")
                    print(f"第一阶段结束：Deviation无法继续下降")
                    print(f"  最佳 Deviation: {best_metric_value:.6f} (epoch {best_epoch+1})")
                    print(f"  当前 Deviation: {current_metric_value:.6f} (epoch {epoch+1})")
                    print(f"  连续 {patience_counter} 个epoch无提升")
                    print(f"")
                    print(f"切换到第二阶段：Revenue + Deviation 联合优化")
                    print(f"  重置patience计数器，继续训练...")
                    print(f"{'='*60}\n")
                else:
                    # 第二阶段也触发了早停，真正停止训练
                    should_stop = True
                    print(f"\n{'='*60}")
                    print(f"早停触发（第二阶段）！")
                    print(f"  最佳 {EARLY_STOPPING_METRIC}: {best_metric_value:.6f} (epoch {best_epoch+1})")
                    print(f"  当前 {EARLY_STOPPING_METRIC}: {current_metric_value:.6f} (epoch {epoch+1})")
                    print(f"  连续 {patience_counter} 个epoch无提升")
                    print(f"{'='*60}\n")
        
        print()  # 空行
    
    # 定期刷新writer（每10个epoch）
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
    
    # 加载最佳模型并显示训练集性能
    print(f"\n加载最佳模型...")
    checkpoint = torch.load(best_model_path)
    bid_allocating_net.load_state_dict(checkpoint['bid_allocating_net_state_dict'])
    bid_allocating_optimizer.load_state_dict(checkpoint['bid_allocating_optimizer_state_dict'])
    for g in groups:
        agents[g].load_state_dict(checkpoint['agents_state_dict'][g])
        optimizers[g].load_state_dict(checkpoint['optimizers_state_dict'][g])
    print("✓ 最佳模型已加载（包括Bid分配网络和所有组Agent）")
    
    # 显示训练集上的性能
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
        bid_allocating_net.load_state_dict(checkpoint['bid_allocating_net_state_dict'])
        bid_allocating_optimizer.load_state_dict(checkpoint['bid_allocating_optimizer_state_dict'])
        for g in groups:
            agents[g].load_state_dict(checkpoint['agents_state_dict'][g])
            optimizers[g].load_state_dict(checkpoint['optimizers_state_dict'][g])
        print("✓ 最新模型已加载\n")

print("\n训练完成！\n")

# 关闭训练阶段的writer
writer.flush()

# =========================
# 10. 在真实场景上测试（迁移测试）
# =========================
print("="*60)
print("开始真实场景迁移测试...")
print("="*60)
print("注意：以下测试使用真实数据，评估模型在真实场景下的迁移性能\n")

TEST_DAYS = [0, 1, 2, 3]  # 测试前4天

# 使用真实数据测试（迁移测试）
print("使用真实数据测试（迁移测试）...")
results_real = []

for day_idx, day in enumerate(TEST_DAYS):
    print(f"\n测试第 {day+1} 天（真实环境）...")
    
    # 使用真实数据测试（迁移测试）
    result_real = run_experiment(day, use_real_data=True)
    results_real.append(result_real)
    
    # 计算reward（用于记录）
    total_reward = result_real['total_revenue'] / 1e9 - LAMBDA * result_real['total_deviation'] / 1e9
    
    # 记录测试指标到TensorBoard
    writer.add_scalar('Test_Real/Day_Revenue', result_real['total_revenue'], day)
    writer.add_scalar('Test_Real/Day_Deviation', result_real['total_deviation'], day)
    writer.add_scalar('Test_Real/Day_Reward', total_reward, day)
    
    # 记录每天的每小时reward
    for t, reward in enumerate(result_real['rewards']):
        writer.add_scalar(f'Test_Real/Hourly_Reward/Day_{day+1}', reward, t)
    
    print(f"[{result_real['data_type']}] Day {day+1} | "
          f"Revenue = {result_real['total_revenue']:.3f} | "
          f"Deviation = {result_real['total_deviation']:.3f} | "
          f"Reward = {total_reward:.3f}")

# 计算真实数据汇总
total_real_revenue = sum(r['total_revenue'] for r in results_real)
total_real_deviation = sum(r['total_deviation'] for r in results_real)
total_real_reward = total_real_revenue / 1e9 - LAMBDA * total_real_deviation / 1e9
avg_real_revenue = total_real_revenue / len(TEST_DAYS)
avg_real_deviation = total_real_deviation / len(TEST_DAYS)
avg_real_reward = total_real_reward / len(TEST_DAYS)

# 记录真实数据汇总指标
writer.add_scalar('Test_Real/Total_Revenue', total_real_revenue, 0)
writer.add_scalar('Test_Real/Avg_Revenue', avg_real_revenue, 0)
writer.add_scalar('Test_Real/Total_Deviation', total_real_deviation, 0)
writer.add_scalar('Test_Real/Avg_Deviation', avg_real_deviation, 0)
writer.add_scalar('Test_Real/Total_Reward', total_real_reward, 0)
writer.add_scalar('Test_Real/Avg_Reward', avg_real_reward, 0)

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
    day_reward = r['total_revenue'] / 1e9 - LAMBDA * r['total_deviation'] / 1e9
    print(f"  Day {r['day']+1}: Revenue = {r['total_revenue']:.3f}, Deviation = {r['total_deviation']:.3f}, Reward = {day_reward:.3f}")
print(f"{'='*60}")

# 关闭TensorBoard writer
writer.close()
print(f"\nTensorBoard日志已保存到: {log_dir}")
print(f"查看日志: tensorboard --logdir={log_dir}")
