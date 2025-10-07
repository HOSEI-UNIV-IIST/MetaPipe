# MetaPipe Experiment Results

**Organized by Dataset** - Each dataset has its own directory with complete results.

## Directory Structure

```
RESULTS/
├── finance_stock_sp500/
│   ├── README.md              # Summary for this dataset
│   ├── data/
│   │   └── results.csv        # Raw numerical results
│   └── figures/
│       ├── comparison.png     # Bar charts (Quality, Cost, Latency)
│       ├── radar.png          # Radar chart (multi-metric)
│       └── ranking.png        # Ranking table
│
├── energy_electricity_load/
│   ├── README.md
│   ├── data/
│   │   └── results.csv
│   └── figures/
│       ├── comparison.png
│       ├── radar.png
│       └── ranking.png
│
└── README.md                  # This file
```

## Quick Access

### Finance Domain (Stock SP500)
- 📊 [Results Summary](finance_stock_sp500/README.md)
- 📈 [Comparison Chart](finance_stock_sp500/figures/comparison.png)
- 🎯 [Radar Chart](finance_stock_sp500/figures/radar.png)

### Energy Domain (Electricity Load)
- 📊 [Results Summary](energy_electricity_load/README.md)
- 📈 [Comparison Chart](energy_electricity_load/figures/comparison.png)
- 🎯 [Radar Chart](energy_electricity_load/figures/radar.png)

## Color Scheme

- **MetaPipe**: Red
- **Random**: Gray
- **Greedy-Cost**: Blue
- **Greedy-Quality**: Green
- **Thompson**: Orange
- **Static-Best**: Purple

## Metrics

- **Quality (SMAPE)**: Lower is better
- **Cost ($)**: Lower is better
- **Latency (ms)**: Lower is better

---

Generated: 2025-10-06 19:51:32