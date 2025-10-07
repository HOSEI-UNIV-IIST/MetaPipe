# HEALTHCARE Domain - mimic_vitals

## Results Summary

### 🏆 Winners

- **Best Quality**: MetaPipe (2.18 SMAPE)
- **Best Cost**: Greedy-Cost ($0.232)
- **Best Latency**: Thompson (973ms)

### 📊 Complete Rankings (by Quality)

| Rank | Method | Quality (SMAPE)↓ | Cost ($)↓ | Latency (ms)↓ |
|------|--------|------------------|-----------|---------------|
| 🥇 1 | **MetaPipe** | 2.18 | $0.256 | 1053 |
| 🥈 2 | **Thompson** | 2.39 | $0.281 | 973 |
| 🥉 3 | **Greedy-Cost** | 2.39 | $0.232 | 990 |
|    4 | **Random** | 2.43 | $0.242 | 1025 |
|    5 | **Greedy-Quality** | 2.50 | $0.274 | 1014 |
|    6 | **Static-Best** | 2.60 | $0.256 | 975 |

### 📁 Files

- `data/results.csv` - Complete numerical results
- `figures/comparison.png` - Bar chart comparison
- `figures/radar.png` - Radar chart (multi-metric)
- `figures/ranking.png` - Ranking table

### 🎯 Key Insight

**MetaPipe** achieves the best quality (2.18 SMAPE) on mimic_vitals.

---

Generated: 2025-10-06 19:51:32
