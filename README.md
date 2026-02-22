# VPPEvolve

**Official implementation** of the paper:

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
   - The script evaluates the best program on predicted scenarios during evolution and then on real scenarios via `evaluate_real(best_program_path)`. Results and paths are printed in the terminal. The evaluation uses the combined score (higher is better):
   $$\text{combined\_score} = \lambda_{\text{rev}} \cdot \text{revenue} - \lambda_{\text{dev}} \cdot \text{deviation} - \lambda_{\text{deg}} \cdot \text{degradation} - \lambda_{\text{risk}} \cdot \text{risk}$$
   with $\lambda$ coefficients set in `config_joint_raw.yaml`.

---

## Deployment on TsingRoc.ai VPP Platform

We deployed VPPEvolve on **TsingRoc.ai**’s virtual power plant platform. The following video shows a comparison of the grid state before and after deployment:

**[video/beijing_deployment.mp4](video/beijing_deployment.mp4)**

- **Upper panel:** Baseline operational state, where VPP devices are sparsely coordinated.
- **Lower panel:** Scenario after deploying VPPEvolve.

The colors on the grid topology indicate the current burden level of each transmission line, reflecting the spatio-temporal imbalance intensity. With VPPEvolve, grid imbalance is significantly alleviated:

- **66.78%** reduction in grid ramping magnitude ($\Delta P$).
- Approximately **48 kt** reduction in daily grid carbon emissions.

