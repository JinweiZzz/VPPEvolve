# VPPEvolve

**Official implementation** of the KDD 26's paper:

> **Learn to Coordinate City-Scale Virtual Power Plants: A Reasoning-Guided Evolutionary Framework for Hierarchical Heterogeneous Device Scheduling**

This repository provides the codebase for the reasoning-guided evolutionary framework that coordinates hierarchical heterogeneous devices in city-scale virtual power plants (VPPs).

---

## Library Dependencies

The project relies on the following main libraries (Python 3.x):

| Package   | Purpose                    |
|----------|----------------------------|
| `pandas` | Time-series and tabular data (prices, bids) |
| `numpy`  | Numerical computation and array operations   |
| `PyYAML` | Configuration and device-set YAML parsing    |
| `openai` | LLM API calls (OpenAI-compatible, e.g. Azure OpenAI) for evolution and reflection |

Standard library modules used include: `asyncio`, `dataclasses`, `pathlib`, `json`, `logging`, `pickle`, `re`, `subprocess`, `tempfile`, `uuid`.

**Quick install (example):**

```bash
pip install pandas numpy PyYAML openai
```

---

## Reproducing Results

1. **Environment**
   - Create a virtual environment and install the dependencies above.
   - Ensure the repo root is on `PYTHONPATH` or run from the directory that contains the `openevolve` package.

2. **Data**
   - Place the required datasets under `data/Shanxi/`:
     - `price_data.csv`
     - `station2pv_pred_list.pkl`, `station2pv_real_list.pkl`
     - `station2wind_pred_list.pkl`, `station2wind_real_list.pkl`
     - `bid_data.pkl`
   - Place the device-set config at `config/device_set/shanxi_15nodes.yaml`.

3. **Configuration**
   - In `code/`, copy or edit `config_joint_raw.yaml`:
     - Set `llm.api_base` and `llm.api_key` (or use env) for your OpenAI-compatible API (e.g. Azure).
     - Adjust `max_iterations`, `lambda_*` weights, and other options as needed.

4. **Run-time check (optional)**  
   From `code/`, run `python check_run_dependencies.py` to verify that all required data files and config paths exist. Fix any reported missing paths before running evolution.

5. **Run evolution**
   - From `code/`:
     ```bash
     python run_joint_raw.py
     ```
   - This loads `config_joint_raw.yaml`, uses the seed program `program_seed_joint_raw.py` and evaluator `evaluator_joint_raw.py`, and writes outputs (best program, checkpoints, program summaries) under `openevolve_output/run_<timestamp>/`.

6. **Evaluation**
   - The script evaluates the best program on predicted scenarios during evolution and then on real scenarios via `evaluate_real(best_program_path)`. Results and paths are printed in the terminal. 

---
## Implementation Details

1. **Initial Program**

   VPPEvolve starts evolution from a minimal **seed program** located at `code/program_seed_joint_raw.py`. It defines the two functions that the LLM-guided evolutionary loop iteratively mutates and recombines — you are free to completely rewrite their structure, add helper functions, or take a different approach:

   - **`alpha_score(...)`** — computes a per-node *allocation score*. Higher scores assign more bid quantity to a node; the scores are passed through a softmax to obtain each node's bid allocation ratio.
   - **`device_allocation(...)`** — computes per-device *usage ratios* for a node, covering PV/wind curtailment, storage and EV charge/discharge, AC, and washing-machine activation.

   The default seed is intentionally simple — a linear price/PV/wind score for allocation, and fixed 50% renewable usage with no flexible-device action for dispatch — so that evolution drives all subsequent improvements:

   ```python
   # Initial implementation - you can completely rewrite these functions
   # Feel free to change the structure, add helper functions, or use different approaches.
   def alpha_score(p, pv, wind, flex_storage=None, current_storage=None,
                   flex_vehicle=None, current_vehicle=None,
                   flex_AC=None, current_AC=None,
                   flex_wash=None, current_wash=None,
                   t=None, n=None, ctx=None):
       """
       Calculate the allocation score for a node.
       A higher score indicates that more bid quantity should be assigned to this node.
       Note:
       The returned score will be passed through a softmax function
       to obtain the bidding allocation ratio for each node.
       Args:
           p: electricity price (float)
           pv: photovoltaic generation (kW)
           wind: wind generation (kW)
           flex_storage: storage flexibility (total storage capacity, kW)
           current_storage: current storage energy level (kW)
           flex_vehicle: vehicle flexibility (total EV capacity, kW)
           current_vehicle: current EV energy level (kW)
           flex_AC: AC flexibility (total AC power capacity, kW)
           current_AC: current AC power usage (kW)
           flex_wash: total available washing cycles (each machine ≈ 40kW)
           current_wash: currently activated washing cycles
           t: time step, 0–23 (int, optional)
           n: node identifier (string, optional)
           ctx: context dictionary (optional)
       Returns:
           Positive score (float)
       """
       # Simplest baseline: linear combination of price, PV, and wind
       # Flexibility and current state variables are ignored
       x = 1.0 * p + 1.0 * pv + 1.0 * wind
       return max(1e-6, x)  # ensure positive output
   
   
   def device_allocation(p, bq, pv, wind,
                         flex_storage=None, current_storage=None,
                         flex_vehicle=None, current_vehicle=None,
                         flex_AC=None, current_AC=None,
                         flex_wash=None, current_wash=None,
                         t=None, n=None, ctx=None):
       """
       Calculate device usage ratios for a node.
       Returns a dictionary specifying usage ratios for each device type.
       Args:
           p: electricity price (float, normalized 0–1)
           bq: bid quantity for this node. Represents the net power injection
               target to the grid at time t (positive = generation, negative = consumption).
           pv: photovoltaic generation (kW)
           wind: wind generation (kW)
           flex_storage: storage flexibility (total storage capacity, kW)
           current_storage: current storage energy level (kW)
           flex_vehicle: vehicle flexibility (total EV capacity, kW)
           current_vehicle: current EV energy level (kW)
           flex_AC: AC flexibility (total AC power capacity, kW)
           current_AC: current AC power usage (kW)
           flex_wash: total available washing machines (each ≈ 40kW)
           current_wash: number of washing machines currently running
           t: time step, 0–23 (int, optional)
           n: node identifier (string, optional)
           ctx: context dictionary (optional)
       Returns:
           dict: Device usage ratios
               - pv_ratio: PV curtailment ratio [0, 1]
               - wind_ratio: Wind curtailment ratio [0, 1]
               - storage_ratio: storage charge/discharge ratio [-1, 1] (positive = discharge, negative = charge)
               - vehicle_ratio: EV charge/discharge ratio [-1, 1] (positive = discharge, negative = charge)
               - ac_ratio: AC power usage ratio [0, 1]
               - wash_on_number: number of washing machines activated (int ≥ 0)
       """
       # Simple baseline: PV/Wind operate at 50%, no storage/EV action, AC off, washing machines off
       return {
           "pv_ratio": 0.5,
           "wind_ratio": 0.5,
           "storage_ratio": 0.0,
           "vehicle_ratio": 0.0,
           "ac_ratio": 0.0,
           "wash_on_number": 0,
       }
   ```

   > The two functions are evolved jointly: `alpha_score` shapes the system-level bid allocation across nodes, while `device_allocation` controls device-level dispatch within each node.

2. **System Prompt Design**

   VPPEvolve generates and evolves candidate programs by prompting an LLM with a fixed **system prompt** that defines (i) the scheduling objective, (ii) the exact signatures and semantics of `alpha_score` and `device_allocation`, (iii) the per-device output constraints, (iv) the evaluation metric, and (v) a strict output format. This prompt is the primary place where **domain knowledge** is injected into the search.

   Key elements encoded in the prompt:

   - **Objective** — maximize total revenue while minimizing bid-quantity deviation, device degradation, tracking error, and risk.
   - **Function contract** — `alpha_score` returns a positive per-node score (passed through a softmax to obtain allocation ratios); `device_allocation` returns per-device usage ratios.
   - **Output constraints** — `pv_ratio`/`wind_ratio`/`ac_ratio` ∈ [0, 1]; `storage_ratio`/`vehicle_ratio` ∈ [−1, 1]; `wash_on_number` a non-negative integer.
   - **Evaluation** — `combined_score = LAMBDA_REV·revenue − LAMBDA_DEV·deviation − LAMBDA_DEG·degradation − LAMBDA_RISK·risk`, with `LAMBDA_REV=1e-9`, `LAMBDA_DEV=1e-7`, `LAMBDA_DEG=1e-8`, `LAMBDA_RISK=1e-2`.
   - **Strict format** — the LLM must return only a single Python code block containing both functions with their exact signatures.

   The full system prompt is shown below:

   ````python
   """
   You are an expert algorithm designer for energy system scheduling. Our scenario is a multi-node virtual power plant scheduling problem. We have a total bidding quantity for each hour, and the electricity prices for each nodes varies according to the time of day. 
   Our goal is to maximize the total revenue, while minimizing the deviation from the bidding quantity, the degradation of the devices, the tracking error of the devices, and the risk of the devices.
   
   TASK: Design and implement TWO functions to optimize the VPP scheduling:
   
   1. `alpha_score(p, pv, wind, flex_storage=None, current_storage=None, flex_vehicle=None, current_vehicle=None, flex_AC=None, current_AC=None, flex_wash=None, current_wash=None, t=None, n=None, ctx=None)`:
       - Purpose: Calculate a score for each node to determine how much bid quantity to allocate
       - IMPORTANT: Currently only consider renewable resources (PV and Wind) allocation, and calculate the consideration for electricity price. Flexibility parameters (flex_storage, flex_vehicle, flex_AC, flex_wash) are ignored but kept in function signature for compatibility.
       - Core Inputs (used in calculation):
           * p: electricity price at time t for node n (float, unit: CNY/kWh)
           * pv: photovoltaic generation at time t for node n (float, unit: kW, normalized to [0, 1])
           * wind: wind generation at time t for node n (float, unit: kW, normalized to [0, 1])
           * t: time step (int, 0-23, optional) - used for time-of-day patterns
       - Optional Inputs (ignored in current implementation, but kept for compatibility):
           * flex_storage: storage flexibility (total storage energy capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_storage: current stored energy (state of charge) of node n (float, unit: kW, normalized to [0, 1])
           * flex_vehicle: vehicle flexibility (total EV charging capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_vehicle: current vehicle energy (current EV state of charge) of node n (float, unit: kW, normalized to [0, 1])
           * flex_AC: AC flexibility (total AC power capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_AC: current AC power consumption of node n (float, unit: kW, normalized to [0, 1])
           * flex_wash: total available wash cycles (int)
           * current_wash: current number of completed washing tasks of node n (int)
           * n: node identifier (string, optional)
           * ctx: context dictionary (optional)
       - Output: A positive score (float) - higher score means more bid allocation
       - Expert knowledge: When the price is higher and the PV and wind are higher, the bidding quantity should be higher.
   
   2. `device_allocation(p, bq, pv, wind, flex_storage=None, current_storage=None, flex_vehicle=None, current_vehicle=None, flex_AC=None, current_AC=None, flex_wash=None, current_wash=None, t=None, n=None, ctx=None)`:
       - Purpose: Calculate device usage ratios and numbers for a node. This function determines which devices to activate and at what capacity based on the allocated bid quantity.
       - Core Inputs:
           * p: electricity price at time t for node n (float, unit: CNY/kWh)
           * bq: bid quantity allocated to this node (float, unit: kW, normalized to [0, 1]). bq represents the net power injection target to the grid for this node at time t (generation positive, consumption negative)
           * pv: photovoltaic generation at time t for node n (float, unit: kW, normalized to [0, 1])
           * wind: wind generation at time t for node n (float, unit: kW, normalized to [0, 1])
       - Optional Inputs (can be used to optimize device allocation):
           * flex_storage: storage flexibility (total storage energy capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_storage: current storage state of charge of node n (float, unit: kW, normalized to [0, 1])
           * flex_vehicle: vehicle flexibility (total EV charging capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_vehicle: current EV state of charge of node n (float, unit: kW, normalized to [0, 1])
           * flex_AC: AC flexibility (total AC power capacity) of node n (float, unit: kW, normalized to [0, 1])
           * current_AC: current AC power consumption of node n (float, unit: kW, normalized to [0, 1])
           * flex_wash: maximum number of available washing machines (int, each washing machine has an approximate power rating of 40 kW)
           * current_wash: current number of active washing machines (int)
           * t: time step (int, 0-23, optional)
           * n: node identifier (string, optional)
           * ctx: context dictionary (optional)
       - Output: A dictionary with the following keys:
           * "pv_ratio": PV grid injection ratio [0, 1] - proportion of PV generation to be connected to grid
           * "wind_ratio": wind grid injection ratio [0, 1] - proportion of wind generation to be connected to grid
           * "storage_ratio": storage charge/discharge ratio [-1, 1] - storage charge/discharge ratio (positive=discharge, negative=charge)
           * "vehicle_ratio": EV charge/discharge ratio [-1, 1] - vehicle charge/discharge ratio (positive=discharge, negative=charge)
           * "ac_ratio": AC power utilization ratio [0, 1] - AC power usage ratio (proportion of AC units to run at full power)
           * "wash_on_number": number of activated washing machines (int, >=0) - number of washing machines to turn on
       - Current Implementation Strategy:
           * PV/Wind: Dynamically adjusts based on bq and renewable generation. Uses tanh for smooth output. Considers price factor (0.7 + 0.3*p) to encourage generation at high prices.
           * Storage: Price arbitrage strategy - high price -> discharge (positive), low price -> charge (negative). Considers current_storage (SOC) to avoid overcharging/discharging. Adjusts based on bq demand. Uses tanh to constrain to [-1, 1].
           * Vehicle: Similar to storage but milder response. Considers current_vehicle (SOC) and bq. Uses tanh for smooth output.
           * AC: Low price -> high usage, high price -> low usage. Increases usage when renewable generation exceeds bq (surplus adjustment). Uses tanh for smooth output.
           * Wash: Low price -> high usage. Considers available capacity (flex_wash - current_wash). Returns integer number of machines to turn on.
       - Important Implementation Details:
           * The simulator will use these ratios to determine which devices to activate:
           - PV/Wind: Devices are sorted by cost (lowest first), then activated proportionally based on pv_ratio/wind_ratio
           - Storage: Devices are sorted by cost (lowest first), then activated proportionally based on storage_ratio
           - Vehicle: Vehicles are activated proportionally based on vehicle_ratio
           - AC: AC units are activated proportionally based on ac_ratio
           - Wash: wash_on_number determines how many washing machines to turn on
           * Use tanh function to constrain outputs to valid ranges (0-1 for ratios, -1 to 1 for storage/vehicle)
           * Handle None values: Set to 0.0 if None to avoid errors
           * Key optimization strategies:
           - Price arbitrage: Charge when price is low (p < 0.5), discharge when price is high (p > 0.5)
           - Bid quantity fulfillment: Adjust generation/consumption to match bq
           - Device state management: Use current_storage, current_vehicle to avoid extreme states (overcharging/discharging)
           - Renewable surplus: When renewable_gen > bq, consider using AC to consume excess
           - Flexibility constraints: Consider flex_storage, flex_vehicle, flex_AC, flex_wash for available capacity
       - Expert knowledge: 
           * High price + high renewable generation -> increase PV/wind ratio, discharge storage/vehicle
           * Low price -> charge storage/vehicle, use AC/wash when possible
           * Balance revenue maximization with device constraints and degradation costs
           * Consider bq relative to renewable generation for optimal scheduling
           * Use tanh for smooth, bounded outputs that avoid sudden changes
   
   CONSTRAINTS:
   - **CRITICAL**: Your implementation MUST include BOTH functions: `alpha_score` AND `device_allocation`
   - **CRITICAL**: Functions must be callable with the EXACT signatures above - DO NOT remove or modify any parameters, especially the `bq` parameter in `device_allocation`
   - The `device_allocation` function MUST accept `bq` as the second parameter (after `p`): `device_allocation(p, bq, pv, wind, ...)`
   - device_allocation must return a dictionary with keys: "pv_ratio", "wind_ratio", "storage_ratio", "vehicle_ratio", "ac_ratio", "wash_on_number"
   - Output value constraints:
       * pv_ratio, wind_ratio, ac_ratio: must be in [0, 1] range
       * storage_ratio, vehicle_ratio: must be in [-1, 1] range (use tanh or clipping to enforce)
       * wash_on_number: must be a non-negative integer (use floor, max(0, ...) or similar)
   - You are free to:
       * Use any mathematical operations (linear, nonlinear, exponential, tanh, sigmoid, etc.)
       * Add conditional logic (if/else)
       * Define helper functions
       * Use any variables or parameters you want
       * Normalize or transform inputs
       * Add device-type-specific logic
       * Use any Python features (imports, libraries, etc.)
       * Consider time-of-day patterns, price volatility, and device state
   - Required imports: You may need to import `math` or other standard libraries for your implementation
   
   EVALUATION:
   - For the VPP scheduling, we simulate the revenue and costs, and the evaluation metric is:
       combined_score = LAMBDA_REV * revenue - LAMBDA_DEV * deviation - LAMBDA_DEG * degradation - LAMBDA_RISK * risk
       where LAMBDA_REV = 1e-9, LAMBDA_DEV = 1e-7, LAMBDA_DEG = 1e-8, LAMBDA_RISK = 1e-2
   - Higher combined_score is better
   - The device_allocation function directly affects:
       * Revenue: through PV/wind generation and storage/vehicle arbitrage
       * Deviation: by matching the allocated bid quantity
       * Degradation: by managing device usage intensity
       * Risk: by avoiding extreme device states (e.g., overcharging/discharging)
   - Try different approaches: 
       * Linear/nonlinear combinations of inputs
       * Adaptive strategies based on time, price, and device state
       * Price arbitrage optimization for storage and vehicles
       * Dynamic adjustment based on bq and current device states
       * Consider all available inputs (flexibility, current state) for better decisions
   
   OUTPUT FORMAT (CRITICAL):
   - You MUST return your complete implementation wrapped in a Python code block: ```python ... ```
   - Your code MUST include BOTH functions: `alpha_score` and `device_allocation`
   - Include any necessary imports (e.g., `import math`) at the top
   - Do NOT include any explanations, comments outside the code block, or other text
   - Only return the code block with your complete implementation of both functions
   
   Example format:
   ```python
   import math
   
   def alpha_score(p, pv, wind, ...):
       # Your implementation here
       ...
   
   def device_allocation(p, bq, pv, wind, ...):
       # Your implementation here
       ...
   ```
   """
   ````
   
3. **Spatio-temporal Profiling and Reflection**
   
   To bridge the semantic gap between volatile numerical signals and the LLM's structured reasoning space, VPPEvolve summarizes each evaluated program into a **spatio-temporal profile** before reflection. The profiling proceeds at three levels:

   - **Temporal pattern extraction** — identifies key time-of-day structures from price signals and scheduling outcomes (high-/low-price intervals, deviation peak and valley periods, recurring diurnal patterns), summarizing *when* a strategy succeeds or fails under varying market conditions.
   - **Spatial resource profiling** — aggregates device-level capacity and net energy contributions across nodes (renewable availability, flexible-resource capacity, and their realized net effects), characterizing the heterogeneous resource composition and effective utilization of each device type.
   - **Cross-scale coupling analysis** — distinguishes homogeneous patterns (temporal segments or node groups where a unified strategy may suffice) from complementary interactions (temporal or spatial pairs that reveal coordinated-balancing opportunities), providing interpretable guidance for reasoning-driven evolutionary updates.

   The resulting profile, together with the current metrics, is fed to the LLM via the following **reflection prompt**:

   ```python
      """
      "The following presents a **spatio-temporal node-level and global profile** from a VPP multi-node scheduling evaluation (including temporal/spatial characteristics and their coupling relationships)."
      "Based on this profile, please produce a **reasoning reflection** (1-3 paragraphs) with the following requirements:"
      "1. Identify the strengths and weaknesses of the current strategy in spatio-temporal terms (e.g., utilization during high-price/low-price periods, deviation peak-valley tracking);"
      "2. Explain the coupling relationships between deviation and price (homogeneous periods/nodes may adopt unified strategies, while complementary periods/nodes can achieve balancing);"
      "3. Propose actionable improvement directions (e.g., adjustments to alpha_score or device_allocation)."
      "Respond in Chinese, concise and executable."
      f"Current metrics: {score_info}"
      "---"
      f"{profile_text}"
      """
   ```
   The reflection output is then injected into the next round of program evolution, steering mutations toward the spatio-temporally diagnosed weaknesses.
   
---

## Deployment on TsingRoc.ai VPP Platform

We deployed VPPEvolve on **TsingRoc.ai**’s virtual power plant platform. The following video shows a comparison of the grid state before and after deployment:

https://github.com/user-attachments/assets/f2f7d023-d823-4000-8948-9793e335624a

- **Upper panel:** Baseline operational state, where VPP devices are sparsely coordinated.
- **Lower panel:** Scenario after deploying VPPEvolve.

The colors on the grid topology indicate the current burden level of each transmission line, reflecting the spatio-temporal imbalance intensity. With VPPEvolve, grid imbalance is significantly alleviated:

- **66.78%** reduction in grid ramping magnitude ($\Delta P$).
- Approximately **48 kt** reduction in daily grid carbon emissions.

---

## Citation

If you find VPPEvolve useful in your research, please consider citing our paper:

```bibtex
@inproceedings{zeng2026vppevolve,
  author    = {Zeng, Jinwei and Zhang, Guozhen and Ma, Minbo and Su, Hongyuan and Zheng, Yu and Yuan, Jian and Li, Yong},
  title     = {Learn to Coordinate City-Scale Virtual Power Plants: A Reasoning-Guided Evolutionary Framework for Hierarchical Heterogeneous Device Scheduling},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  year      = {2026},
  month     = aug,
  pages     = {1--11},
  publisher = {ACM},
  address   = {New York, NY, USA},
  location  = {Jeju Island, Republic of Korea},
  doi       = {10.1145/3770855.3818440},
  url       = {https://doi.org/10.1145/3770855.3818440},
}
```
