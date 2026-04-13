# ============================
# MIQCP-style VPP Optimizer
# (Minimal patch version)
# ============================

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


class OptimizeVPPMultiNodeMIQCP:
    """
    MIQCP-style risk-aware planning
    (structure compatible with your original pipeline)
    """

    def __init__(
        self,
        T,
        node_list,
        price_dict,
        total_bid,
        pv_dict,
        wind_dict,
        node_device_dict,
        base_dir,
        Sigma_price=None,   # LMP covariance: [T, N, N]
        beta=0.01,          # risk aversion
        timelimit=300,
        level=10,
        bid_penalty=100.0,
        ev_penalty=500.0,
        threads=8,
    ):
        self.T = T
        self.nodes = list(node_list)
        self.price = price_dict
        self.total_bid = np.asarray(total_bid, dtype=float)
        self.base_dir = base_dir
        self.level = level
        self.beta = beta
        self.bid_penalty = bid_penalty
        self.ev_penalty = ev_penalty

        self.model = gp.Model("VPP_MIQCP")
        self.model.setParam("TimeLimit", timelimit)
        self.model.setParam("Threads", threads)
        self.model.setParam("NonConvex", 2)

        self.N = len(self.nodes)

        # --- price covariance ---
        if Sigma_price is None:
            # simple diagonal covariance (can replace)
            self.Sigma = np.zeros((T, self.N, self.N))
            for t in range(T):
                for i in range(self.N):
                    self.Sigma[t, i, i] = 1.0
        else:
            self.Sigma = Sigma_price

        self.nd = {}
        for node in self.nodes:
            self.nd[node] = self._build_node(node, pv_dict[node], wind_dict[node], node_device_dict[node])

        self.total_bid_slack = self.model.addVars(T, lb=0.0, name="bid_slack")

        self._add_constraints()
        self._set_objective()

    # -----------------------------
    # node builder
    # -----------------------------
    def _build_node(self, node, pv_curve, wind_curve, device_dict):
        T, L = self.T, self.level

        storage_ids = device_dict.get("storage_id", [])
        vehicle_ids = device_dict.get("vehicle_id", [])
        wash_ids = device_dict.get("wash_id", [])
        ac_ids = device_dict.get("AC_id", [])
        
        storage_params = [load_device_yaml("storage", i, self.base_dir) for i in storage_ids]
        vehicle_params = [load_device_yaml("vehicle", i, self.base_dir) for i in vehicle_ids]
        wash_params = [load_device_yaml("wash", i, self.base_dir) for i in wash_ids]
        ac_params = [load_device_yaml("AC", i, self.base_dir) for i in ac_ids]

        pv_total = np.asarray(pv_curve, dtype=float).flatten()[:T]
        wind_total = np.asarray(wind_curve, dtype=float).flatten()[:T]

        pv_cur = self.model.addVars(T, lb=0, ub=1, name=f"{node}_pv_cur")
        wind_cur = self.model.addVars(T, lb=0, ub=1, name=f"{node}_wind_cur")

        storage = []
        for _ in storage_ids:
            storage.append({
                "ch": self.model.addVars(T, lb=0, ub=L, vtype=GRB.INTEGER),
                "dis": self.model.addVars(T, lb=0, ub=L, vtype=GRB.INTEGER),
                "f": self.model.addVars(T, vtype=GRB.BINARY),
            })

        vehicle = []
        for _ in vehicle_ids:
            vehicle.append({
                "ch": self.model.addVars(T, lb=0, ub=L, vtype=GRB.INTEGER),
                "dis": self.model.addVars(T, lb=0, ub=L, vtype=GRB.INTEGER),
                "f": self.model.addVars(T, vtype=GRB.BINARY),
                "slack": self.model.addVar(lb=0.0),
            })

        wash = [{"start": self.model.addVars(T, vtype=GRB.BINARY),
                 "on": self.model.addVars(T, vtype=GRB.BINARY)} for _ in wash_ids]

        ac = [self.model.addVars(T, lb=0, ub=L, vtype=GRB.INTEGER) for _ in ac_ids]

        elec = self.model.addVars(T, lb=-GRB.INFINITY, name=f"{node}_elec")

        # 🔥 MIQCP core variable
        U_tr = self.model.addVars(T, lb=-GRB.INFINITY, name=f"{node}_Utr")

        return dict(
            storage_ids=storage_ids, vehicle_ids=vehicle_ids, wash_ids=wash_ids, ac_ids=ac_ids,
            storage_params=storage_params, vehicle_params=vehicle_params,
            wash_params=wash_params, ac_params=ac_params,
            pv_total=pv_total, wind_total=wind_total,
            pv_cur=pv_cur, wind_cur=wind_cur,
            storage=storage, vehicle=vehicle, wash=wash, ac=ac,
            elec=elec, U_tr=U_tr,
        )

    # -----------------------------
    # constraints
    # -----------------------------
    def _add_constraints(self):
        T, L = self.T, self.level

        for node in self.nodes:
            D = self.nd[node]

            # storage SOC (prefix form)
            for k, dev in enumerate(D["storage_params"]):
                cap = float(dev["capacity"])
                pmax = float(dev["max_power"])
                soc = float(dev.get("SoC", 0.5 * cap))

                ch, dis, f = D["storage"][k]["ch"], D["storage"][k]["dis"], D["storage"][k]["f"]
                for t in range(T):
                    self.model.addConstr(ch[t] <= L * f[t], name=f"{node}_st_mutex_ch_{k}_{t}")
                    self.model.addConstr(dis[t] <= L * (1 - f[t]), name=f"{node}_st_mutex_dis_{k}_{t}")
                    soc = soc + (ch[t] - dis[t]) * (pmax / L)
                    self.model.addConstr(soc >= 0.0, name=f"{node}_st_soc_lb_{k}_{t}")
                    self.model.addConstr(soc <= cap, name=f"{node}_st_soc_ub_{k}_{t}")

            # vehicle SOC + soft demand
            for k, dev in enumerate(D["vehicle_params"]):
                cap = float(dev["capacity"])
                pmax = float(dev["max_power"])
                soc = float(dev.get("SoC", 0.5 * cap))
                demand = float(dev.get("capacity_demand", 0.0))

                ch, dis, f = D["vehicle"][k]["ch"], D["vehicle"][k]["dis"], D["vehicle"][k]["f"]
                for t in range(T):
                    self.model.addConstr(ch[t] <= L * f[t], name=f"{node}_ev_mutex_ch_{k}_{t}")
                    self.model.addConstr(dis[t] <= L * (1 - f[t]), name=f"{node}_ev_mutex_dis_{k}_{t}")
                    soc = soc + (ch[t] - dis[t]) * (pmax / L)
                    self.model.addConstr(soc >= 0.0, name=f"{node}_ev_soc_lb_{k}_{t}")
                    self.model.addConstr(soc <= cap, name=f"{node}_ev_soc_ub_{k}_{t}")

                self.model.addConstr(soc + D["vehicle"][k]["slack"] >= demand, name=f"{node}_ev_demand_{k}")
            
            # wash 约束
            for k, dev in enumerate(D["wash_params"]):
                t_in = max(0, int(dev.get("t_in", 0)) - 12)
                t_ter = int(dev.get("t_ter", 24))
                t_out = min(T, t_ter - 12 + 24)
                dur = int(dev.get("t_dur", 2))

                start, on = D["wash"][k]["start"], D["wash"][k]["on"]

                # 最多一次（不是必须一次）
                self.model.addConstr(gp.quicksum(start[t] for t in range(T)) <= 1, name=f"{node}_wash_max_once_{k}")

                for t in range(T):
                    if t < t_in or t > t_out - dur:
                        self.model.addConstr(start[t] == 0, name=f"{node}_wash_start_bound_{k}_{t}")

                for t in range(T):
                    self.model.addConstr(
                        on[t] == gp.quicksum(start[s] for s in range(max(0, t - dur + 1), t + 1)),
                        name=f"{node}_wash_on_{k}_{t}"
                    )

            # define node elec[t]
            for t in range(T):
                expr = 0.0

                # storage net injection
                for k, dev in enumerate(D["storage_params"]):
                    pmax = float(dev["max_power"])
                    expr += (D["storage"][k]["dis"][t] - D["storage"][k]["ch"][t]) * (pmax / L)

                # vehicle net injection
                for k, dev in enumerate(D["vehicle_params"]):
                    pmax = float(dev["max_power"])
                    expr += (D["vehicle"][k]["dis"][t] - D["vehicle"][k]["ch"][t]) * (pmax / L)
                
                # wash 消费
                for k, dev in enumerate(D["wash_params"]):
                    expr -= D["wash"][k]["on"][t] * float(dev["rate_power"])

                # AC 消费
                for k, dev in enumerate(D["ac_params"]):
                    expr -= D["ac"][k][t] * (float(dev["power_max"]) / L)

                # PV / wind injection (curtail ratio)
                expr += (1 - D["pv_cur"][t]) * float(D["pv_total"][t])
                expr += (1 - D["wind_cur"][t]) * float(D["wind_total"][t])

                self.model.addConstr(D["elec"][t] == expr, name=f"{node}_elec_def_{t}")
                # 🔥 market balance (MIQCP structure): U_tr == elec for risk term
                self.model.addConstr(D["U_tr"][t] == D["elec"][t], name=f"{node}_Utr_eq_{t}")

        # total bid on elec (aligned with MILP)
        for t in range(T):
            total_elec = gp.quicksum(self.nd[n]["elec"][t] for n in self.nodes)
            self.model.addConstr(total_elec - self.total_bid[t] <= self.total_bid_slack[t], name=f"bid_pos_{t}")
            self.model.addConstr(self.total_bid[t] - total_elec <= self.total_bid_slack[t], name=f"bid_neg_{t}")

    # -----------------------------
    # objective (MIQCP)
    # -----------------------------
    def _set_objective(self):
        obj = 0.0
        T = self.T

        # expected profit (use elec to align with MILP, U_tr == elec by constraint)
        for i, node in enumerate(self.nodes):
            D = self.nd[node]
            for t in range(T):
                obj += float(self.price[node][t]) * D["elec"][t]

        # risk term (mean-variance)
        for t in range(T):
            for i, ni in enumerate(self.nodes):
                for j, nj in enumerate(self.nodes):
                    obj -= self.beta * self.Sigma[t, i, j] * self.nd[ni]["U_tr"][t] * self.nd[nj]["U_tr"][t]

        # bid slack (quadratic)
        for t in range(T):
            obj -= self.bid_penalty * self.total_bid_slack[t] * self.total_bid_slack[t]

        # EV demand slack
        for node in self.nodes:
            for k in range(len(self.nd[node]["vehicle"])):
                obj -= self.ev_penalty * self.nd[node]["vehicle"][k]["slack"]

        self.model.setObjective(obj, GRB.MAXIMIZE)

    # -----------------------------
    def solve(self):
        self.model.optimize()
        if self.model.SolCount == 0:
            return None
        return self.model

    def get_strategy_by_node(self):
        T, L = self.T, self.level
        out = {}
        for node in self.nodes:
            D = self.nd[node]
            strategy = {"pv": [], "wind": [], "storage": [], "vehicle": [], "wash": [], "ac": []}
            strategy["pv"].append({"cur": [float(D["pv_cur"][t].X) for t in range(T)]})
            strategy["wind"].append({"cur": [float(D["wind_cur"][t].X) for t in range(T)]})

            for k in range(len(D["storage_ids"])):
                strategy["storage"].append({"p": [float(D["storage"][k]["dis"][t].X - D["storage"][k]["ch"][t].X) for t in range(T)]})

            for k in range(len(D["vehicle_ids"])):
                strategy["vehicle"].append({"p": [float(D["vehicle"][k]["dis"][t].X - D["vehicle"][k]["ch"][t].X) for t in range(T)]})

            for k in range(len(D["wash_ids"])):
                strategy["wash"].append({"p": [float(D["wash"][k]["on"][t].X) for t in range(T)]})

            for k in range(len(D["ac_ids"])):
                strategy["ac"].append({"p": [float(D["ac"][k][t].X) for t in range(T)]})

            out[node] = strategy
        return out

    def get_total_elec(self):
        T = self.T
        total = []
        for t in range(T):
            total.append(sum(float(self.nd[n]["elec"][t].X) for n in self.nodes))
        return total


class SimpleVPPSimulator:
    """
    以"net injection 正"为统一约定：
      + 表示向外卖电/注入
      - 表示从外用电/取电
    """

    def __init__(self, price, pv, wind, device_dict, level=10, base_dir="/data2/zengjinwei/VPP_multinode/config/device"):
        self.price = np.asarray(price, dtype=float)
        self.T = len(price)
        self.level = int(level)
        self.base_dir = base_dir

        # PV 和 Wind 作为节点的总功率，不需要设备ID
        # pv 和 wind 应该是长度为 T 的数组，表示每个时刻的总功率
        self.pv_total = np.asarray(pv, dtype=float).flatten()  # 确保是1D数组，长度为T
        self.wind_total = np.asarray(wind, dtype=float).flatten()  # 确保是1D数组，长度为T
        
        # 其他设备仍然需要设备ID
        self.storage_ids = list(device_dict.get("storage_id", []))
        self.vehicle_ids = list(device_dict.get("vehicle_id", []))
        self.wash_ids = list(device_dict.get("wash_id", []))
        self.ac_ids = list(device_dict.get("AC_id", []))

        self.storage_params = [load_device_yaml("storage", i, base_dir=self.base_dir) for i in self.storage_ids]
        self.vehicle_params = [load_device_yaml("vehicle", i, base_dir=self.base_dir) for i in self.vehicle_ids]
        self.wash_params = [load_device_yaml("wash", i, base_dir=self.base_dir) for i in self.wash_ids]
        self.ac_params = [load_device_yaml("AC", i, base_dir=self.base_dir) for i in self.ac_ids]

        self.reset()

    def reset(self, seed=None):
        self.t = 0
        self.done = False
        return np.zeros(1), {}

    def step(self, action):
        """
        action 由 execute_strategy 组装：
          pv_action: curtail ratio [0,1] (单个值，表示总功率的弃电比例)
          wind_action: curtail ratio [0,1] (单个值，表示总功率的弃电比例)
          storage_action: net injection fraction in [-1,1] * n_storage
          vehicle_action: net injection fraction in [-1,1] * n_vehicle
          wash_action: on {0,1} * n_wash
          ac_action: power fraction [0,1] * n_ac  (消费)
        """
        t = self.t
        idx = 0
        elec = 0.0
        penalty = 0.0

        # PV: 使用总功率和弃电比例
        pv_curtail_ratio = float(np.clip(action[idx], 0, 1)) if len(action) > idx else 0.0
        pv_gen = self.pv_total[t] if t < len(self.pv_total) else 0.0
        elec += (1 - pv_curtail_ratio) * pv_gen
        # PV 弃电的惩罚成本（简化处理，假设弃电成本为0或很小）
        penalty += (pv_curtail_ratio * pv_gen) * 0.0  # 可以后续调整
        idx += 1

        # Wind: 使用总功率和弃电比例
        wind_curtail_ratio = float(np.clip(action[idx], 0, 1)) if len(action) > idx else 0.0
        wind_gen = self.wind_total[t] if t < len(self.wind_total) else 0.0
        elec += (1 - wind_curtail_ratio) * wind_gen
        # Wind 弃电的惩罚成本（简化处理，假设弃电成本为0或很小）
        penalty += (wind_curtail_ratio * wind_gen) * 0.0  # 可以后续调整
        idx += 1

        # Storage: action in [-1,1] => net injection power
        for k, dev in enumerate(self.storage_params):
            frac = float(np.clip(action[idx], -1, 1))
            elec += frac * float(dev["max_power"])
            idx += 1

        # Vehicle
        for k, dev in enumerate(self.vehicle_params):
            frac = float(np.clip(action[idx], -1, 1))
            elec += frac * float(dev["max_power"])
            idx += 1

        # Wash: on {0,1} consume
        for k, dev in enumerate(self.wash_params):
            on = 1.0 if action[idx] > 0.5 else 0.0
            elec -= on * float(dev["rate_power"])
            idx += 1

        # AC: power frac [0,1] consume
        for k, dev in enumerate(self.ac_params):
            frac = float(np.clip(action[idx], 0, 1))
            elec -= frac * float(dev["power_max"])
            idx += 1

        reward = float(self.price[t]) * elec - penalty

        self.t += 1
        done = self.t >= self.T
        return np.zeros(1), reward, done, False, {"elec": elec, "penalty": penalty}


def execute_strategy(env, day, strategy, level=10):
    """
    把 MIQCP 解 strategy 映射为 env.step(action)
    约定：elec 正=注入；负=消费
    """
    revenue = 0.0  # Economic revenue = price * elec
    penalty = 0.0
    obs, _ = env.reset(seed=day)

    transitions = []
    elec_list = []
    penalty_list = []
    price_list = []
    
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
        revenue += float(r)  # r = price * elec
        penalty += float(info["penalty"])

        transitions.append([obs, next_obs, action, r, done, info])
        obs = next_obs
        elec_list.append(float(info["elec"]))
        penalty_list.append(float(info["penalty"]))
        if hasattr(env, 'price') and t < len(env.price):
            price_list.append(float(env.price[t]))
        else:
            price_list.append(0.0)

    # Get device states from simulator
    # These states are tracked by SimpleVPPSimulator with proper constraints
    # After 24 steps, env.storage_energy, env.vehicle_energy, env.wash_work_state contain final states
    
    # Calculate storage SoC from energy
    storage_soc = 0.5
    if hasattr(env, 'storage_energy') and env.storage_energy:
        total_energy = sum(env.storage_energy.values())
        total_capacity = sum(float(dev.get("capacity", 0.0)) for dev in env.storage_params)
        if total_capacity > 0:
            storage_soc = total_energy / total_capacity
    
    # Calculate vehicle SoC from energy
    vehicle_soc = {}
    if hasattr(env, 'vehicle_energy') and env.vehicle_energy:
        for vehicle_id, dev in zip(env.vehicle_ids, env.vehicle_params):
            energy = env.vehicle_energy.get(vehicle_id, 0.0)
            capacity = float(dev.get("capacity", 0.0))
            if capacity > 0:
                vehicle_soc[vehicle_id] = energy / capacity
            else:
                vehicle_soc[vehicle_id] = 0.5
    
    # Get wash work_state
    wash_work_state = {}
    if hasattr(env, 'wash_work_state') and env.wash_work_state:
        wash_work_state = dict(env.wash_work_state)

    return {
        "revenue": revenue,
        "transitions": transitions,
        "elec_list": elec_list,
        "penalty": penalty,
        "penalty_list": penalty_list,
        "price_list": price_list,
        "storage_soc": storage_soc,
        "vehicle_soc": vehicle_soc,
        "wash_work_state": wash_work_state,
    }


if __name__ == "__main__":
    import pandas as pd
    import pickle

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

    # 对每个node进行数据处理
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

    # 运行参数
    eval_days = [0, 1, 2, 3]
    T = 24
    LEVEL = 10
    BASE_DIR = "/data2/zengjinwei/VPP_multinode/config/device"
    
    # Initialize totals for 4 days
    total_pred_revenue = 0.0
    total_pred_deviation = 0.0
    total_real_revenue = 0.0
    total_real_deviation = 0.0
    
    bid_overall = bid
    
    for day in eval_days:
        print(f"\n==================== Day {day+1} ====================")
        
        # ---------- 组织当日输入 ----------
        price_day = {node: price[i, day, :].tolist() for i, node in enumerate(node_list)}

        bid_day = bid_overall[day, :].tolist()
        
        pv_day_pred = pv_pred[:, day, :].tolist()   # (N_nodes, 24)
        wind_day_pred = wind_pred[:, day, :].tolist()
        
        pv_day_real = pv_real[:, day, :].tolist()
        wind_day_real = wind_real[:, day, :].tolist()
        
        pv_dict_pred = {node: pv_day_pred[idx] for (idx, node) in enumerate(node_list)}
        wind_dict_pred = {node: wind_day_pred[idx] for (idx, node) in enumerate(node_list)}
        
        # 计算价格协方差矩阵（简化：使用对角矩阵）
        # 对于单节点场景，使用价格标准差
        node_price_std = {}
        for node in node_list:
            prices = np.array(price_day[node])
            node_price_std[node] = np.std(prices)

        # 构建协方差矩阵 [T, N, N]
        Sigma_price = np.zeros((T, len(node_list), len(node_list)))
        for t in range(T):
            for i, ni in enumerate(node_list):
                for j, nj in enumerate(node_list):
                    if i == j:
                        # 对角元素：使用节点价格方差
                        Sigma_price[t, i, j] = node_price_std[ni] ** 2
                    else:
                        # 非对角元素：假设相关性为0（可后续改进）
                        Sigma_price[t, i, j] = 0.0

        # ---------- 跑 multi-node MIQCP（用预测） ----------
        # Revenue优先优化：降低bid_penalty和beta，让优化器更关注revenue而不是deviation
        opt = OptimizeVPPMultiNodeMIQCP(
            T=24,
            node_list=node_list,
            price_dict=price_day,
            total_bid=bid_day,
            pv_dict=pv_dict_pred,
            wind_dict=wind_dict_pred,
            node_device_dict=node_device_mapping,
            base_dir=BASE_DIR,
            Sigma_price=Sigma_price,
            beta=0.001,          # 降低风险厌恶系数，更关注期望收益
            timelimit=600,
            level=LEVEL,
            bid_penalty=10.0,    # 大幅降低bid penalty（从300降到10），允许更大的deviation以换取更高revenue
            ev_penalty=500.0,    # 保持EV penalty不变，确保设备约束满足
            threads=8,
        )

        mdl = opt.solve()
        if mdl is None:
            print("❌ MIQCP infeasible / no solution")
            continue
                    
        # 检查求解状态
        status = mdl.Status
        print(f"[MIQCP] Solver status: {status}")
        if status == GRB.OPTIMAL:
            print(f"[MIQCP] Optimal solution found")
        elif status == GRB.TIME_LIMIT:
            print(f"[MIQCP] Time limit reached")
        elif status == GRB.SUBOPTIMAL:
            print(f"[MIQCP] Suboptimal solution")
        else:
            print(f"[MIQCP] Status: {status}")

        total_elec_plan = np.array(opt.get_total_elec(), dtype=float)
        total_bid = np.array(bid_day, dtype=float)
        plan_dev_l1 = np.abs(total_elec_plan - total_bid).sum()

        # Verify U_tr == elec constraint (for debugging)
        max_utr_elec_diff = 0.0
        for node in node_list:
            for t in range(T):
                utr_val = float(opt.nd[node]["U_tr"][t].X)
                elec_val = float(opt.nd[node]["elec"][t].X)
                diff = abs(utr_val - elec_val)
                max_utr_elec_diff = max(max_utr_elec_diff, diff)
        if max_utr_elec_diff > 1e-6:
            print(f"[WARNING] U_tr != elec constraint violation: max_diff = {max_utr_elec_diff:.6f}")

        # Calculate predicted scenario revenue
        pred_revenue = 0.0
        for node in node_list:
            for t in range(T):
                # Get elec[node][t] from MIQCP solution
                elec_node_t = float(opt.nd[node]["elec"][t].X)
                pred_revenue += price_day[node][t] * elec_node_t
        
        pred_deviation = plan_dev_l1
        
        # Debug: print bid slack values
        max_bid_slack = max([float(opt.total_bid_slack[t].X) for t in range(T)])
        print(f"[MIQCP-PRED] max bid_slack        : {max_bid_slack:.3f}")
        print(f"[MIQCP-PRED] predicted revenue    : {pred_revenue:.3f}")
        print(f"[MIQCP-PRED] predicted deviation  : {pred_deviation:.3f}")
            
        # Accumulate predicted scenario metrics
        total_pred_revenue += pred_revenue
        total_pred_deviation += pred_deviation
            
        # ---------- 拿到每个节点策略 ----------
        strategy_by_node = opt.get_strategy_by_node()

        # ---------- 用真实 pv/wind 做仿真评估（逐节点模拟后求和） ----------
        import os
        
        # Load evaluation configuration
        config_path = "/data2/zengjinwei/VPP_multinode/AlphaEvolve/config.yaml"
        try:
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
        
        total_revenue = 0.0  # Economic revenue = price * elec
        total_penalty_device = 0.0  # Device operation penalty
        total_penalty_bid = 0.0  # Bidding deviation penalty
        total_elec_real = np.zeros(T, dtype=float)  # 真实模拟下，每小时总出力
        
        # Collect device states for penalty calculation
        node_device_states = {}
            
        for (idx, node) in enumerate(node_list):
            # 真实模拟器：price 用当日价格；pv/wind 用真实
            env_sim = SimpleVPPSimulator(
                price=price_day[node],
                pv=pv_day_real[idx],
                wind=wind_day_real[idx],
                device_dict=node_device_mapping[node],
                level=LEVEL,
                base_dir=BASE_DIR,
            )
            
            result = execute_strategy(
                env_sim, day=day, strategy=strategy_by_node[node], level=LEVEL
            )
            
            print(f'result for node {node}:', result)
            total_revenue += float(result["revenue"])
            total_elec_real += np.array(result["elec_list"], dtype=float)
            
            # Store device states for penalty calculation
            node_device_states[node] = {
                "storage_soc": result["storage_soc"],
                "vehicle_soc": result["vehicle_soc"],
                "wash_work_state": result["wash_work_state"],
                "device_dict": node_device_mapping[node],
            }
        
        # Calculate bidding deviation penalty
        for t in range(T):
            elec_quan = total_elec_real[t]
            bid = total_bid[t]
            
            # Calculate threshold
            if abs(bid) < 50:
                thresh = 10
            else:
                thresh = abs(bidding_ratio * bid)
            
            # Calculate elec_clip and penalty_bid
            if elec_quan - bid > thresh:
                # Exceeded upper threshold
                avg_price = np.mean([price_day[n][t] for n in node_list])
                penalty_bid_t = (elec_quan - bid - thresh) * avg_price / normalize_ratio * bidding_penalty
                elec_clip = bid + thresh
            elif elec_quan - bid < -thresh:
                # Below lower threshold
                avg_price = np.mean([price_day[n][t] for n in node_list])
                penalty_bid_t = (bid - elec_quan - thresh) * avg_price / normalize_ratio * (1 + bidding_penalty)
                elec_clip = bid - thresh
            else:
                # Within threshold
                penalty_bid_t = 0.0
                elec_clip = elec_quan
            
            # Update revenue with clipped electricity
            avg_price = np.mean([price_day[n][t] for n in node_list])
            revenue_adjustment = (elec_clip - elec_quan) * avg_price
            total_revenue += revenue_adjustment
            total_penalty_bid += penalty_bid_t
        
        # Calculate device operation penalty at end of day
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
            
        # Calculate total penalty and final revenue
        total_penalty = device_penalty_effi * total_penalty_device + total_penalty_bid
        final_revenue = total_revenue - total_penalty
        
        # Deviation: L1 norm of (total_elec - total_bid) over 24 hours
        real_dev_l1 = np.abs(total_elec_real - np.array(total_bid, dtype=float)).sum()

        print(f"[SIM-REAL] real revenue       : {final_revenue:.3f}")
        print(f"[SIM-REAL] total penalty      : {total_penalty:.3f}")
        print(f"[SIM-REAL]   - penalty_device : {total_penalty_device:.3f}")
        print(f"[SIM-REAL]   - penalty_bid    : {total_penalty_bid:.3f}")
        print(f"[SIM-REAL] real deviation     : {real_dev_l1:.3f}")
        
        # Accumulate real scenario metrics
        total_real_revenue += final_revenue
        total_real_deviation += real_dev_l1
            
    # Print 4-day totals after loop
    print(f"\n{'='*60}")
    print(f"4-Day Summary:")
    print(f"{'='*60}")
    print(f"[PRED] Total predicted revenue   : {total_pred_revenue:.3f}")
    print(f"[PRED] Total predicted deviation : {total_pred_deviation:.3f}")
    print(f"[REAL] Total real revenue        : {total_real_revenue:.3f}")
    print(f"[REAL] Total real deviation      : {total_real_deviation:.3f}")
    print(f"{'='*60}")
