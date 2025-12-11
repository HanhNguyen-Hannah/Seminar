# Supply Chain Resilience under Disruptions
## An Agent-Based Simulation with Mesa

This project implements a multi-tier supply chain agent-based model (ABM) to study resilience strategies under various disruption scenarios.

## Project Structure

```
├── agents.py                 # BaseFirm agent class for all supply chain tiers
├── model.py                  # MultiTierModel with network, disruptions, and KPIs
├── run.py                    # Experiment runner and data export
├── merge.ipynb               # Merge results from multiple seeds
├── generate_figures.ipynb    # Generate publication-quality figures
├── output_data_[seed]/       # Output folders per seed
│   ├── timeseries_all.csv
│   ├── agent_data.csv
│   ├── summary_raw.csv
│   ├── summary_aggregated.csv
│   └── fig_results.png
├── output_final/             # Merged data from all seeds
│   ├── timeseries_all.csv
│   ├── agent_data.csv
│   ├── summary_raw.csv
│   └── summary_aggregated.csv
└── figures/                  # Generated figures for paper/presentation
    ├── fig1_timeseries.png/pdf
    ├── fig2_ttr_by_scenario.png/pdf
    ├── fig3_ttr_horizontal.png/pdf
    ├── fig4_cost_service.png/pdf
    ├── fig5_heatmap.png/pdf
    ├── fig6_fill_drop.png/pdf
    ├── fig7_cost_breakdown.png/pdf
    ├── fig8_bullwhip_tier.png/pdf
    └── tables.tex
```

## Supply Chain Network

The model simulates a 4-tier supply chain with 4-2-2-4 configuration:

```
Suppliers (4) → Plants (2) → Distribution Centers (2) → Retailers (4)
```

Network connections use deterministic round-robin assignment to ensure reproducibility.

## Agent Behavior

Each agent (BaseFirm) follows a base-stock ordering policy with:
- Pipeline inventory tracking
- Safety stock adjustments based on backlog
- Bounded order quantities to prevent runaway amplification
- Stochastic demand at retailer level (Poisson distributed)

## Disruption Scenarios

| Scenario | Description | Affected Tier |
|----------|-------------|---------------|
| `capacity_loss` | Reduces production capacity by 10% | Supplier or Plant |
| `lead_time_surge` | Increases lead time by 8 periods | Plant or DC tier |
| `demand_spike` | Multiplies retailer demand by 5× | All Retailers |

## Resilience Strategies

| Strategy | Description |
|----------|-------------|
| `baseline` | No resilience measures |
| `dual_only` | Dual sourcing (2 suppliers per plant) |
| `safety_only` | Increased safety stock (factor 1.3) |
| `flex_only` | Flexible capacity (+50% at suppliers/plants) |
| `dynalloc_only` | Dynamic reallocation (prioritize high-backlog customers) |
| `all_combined` | All strategies enabled |

## Key Performance Indicators (KPIs)

- **Fill Rate**: Fulfilled demand / Total demand at retailers
- **Total Cost**: Holding cost + Backlog cost across all agents
- **Bullwhip Ratio**: Var(orders) / Var(demand) per tier and total chain
- **Time-to-Recover (TTR)**: Steps until fill rate recovers to 95% of baseline
- **Backlog Duration**: Average periods with positive backlog post-disruption

## Usage

### Step 1: Run Experiments

Edit `SEEDS` in `run.py` and execute for each seed separately:

```bash
# Edit run.py: SEEDS = [2411]
python run.py

# Edit run.py: SEEDS = [24]
python run.py

# Edit run.py: SEEDS = [200]
python run.py
```

> **Note:** Seeds must be run separately due to Mesa's random state handling.

### Step 2: Merge Results

Run `merge.ipynb` to combine data from all seeds:

```python
# Outputs to output_final/:
# - summary_raw.csv
# - summary_aggregated.csv
# - timeseries_all.csv
# - agent_data.csv
```

### Step 3: Generate Figures

Run `generate_figures.ipynb` to create publication-quality figures:

```python
# Outputs to figures/:
# - 8 figures (PNG + PDF)
# - LaTeX tables (tables.tex)
```

## Default Parameters

```python
assumptions = {
    "n_suppliers": 4,
    "n_plants": 2,
    "n_dcs": 2,
    "n_retailers": 4,
    "base_stock": {"supplier": 60, "plant": 50, "dc": 30, "retailer": 20},
    "capacity": {"supplier": 50, "plant": 60, "dc": 0, "retailer": 0},
    "lead_time": {"supplier": 3, "plant": 2, "dc": 2, "retailer": 1},
    "capacity_loss_frac": 0.1,
    "recovery_duration": 3,
    "holding_cost": 1.0,
    "backlog_cost": 5.0,
    "retailer_demand_mean": 7.0
}
```

## Dependencies

```
mesa
networkx
numpy
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install mesa networkx numpy pandas matplotlib seaborn
```

## Output Files

### Per-seed outputs (`output_data_[seed]/`)

| File | Description |
|------|-------------|
| `timeseries_all.csv` | Step-by-step metrics (fill rate, cost, bullwhip) |
| `agent_data.csv` | Final state of each agent |
| `summary_raw.csv` | Summary metrics per scenario/strategy |
| `summary_aggregated.csv` | Formatted summary with mean ± std |
| `fig_results.png` | Quick visualization |

### Merged outputs (`output_final/`)

Combined data from all seeds for statistical analysis.

### Figures (`figures/`)

| Figure | Description |
|--------|-------------|
| `fig1_timeseries` | Fill rate & cost over time (with error bands) |
| `fig2_ttr_by_scenario` | TTR grouped by scenario |
| `fig3_ttr_horizontal` | TTR averaged across scenarios |
| `fig4_cost_service` | Cost vs Fill Rate scatter (bubble = TTR) |
| `fig5_heatmap` | Strategy × Scenario performance matrix |
| `fig6_fill_drop` | Fill rate drop during disruption |
| `fig7_cost_breakdown` | Holding vs Backlog cost |
| `fig8_bullwhip_tier` | Bullwhip effect by supply chain tier |
| `tables.tex` | LaTeX tables for paper |

## References

- Mesa Documentation: https://mesa.readthedocs.io/
- Supply Chain Resilience Literature: See seminar paper