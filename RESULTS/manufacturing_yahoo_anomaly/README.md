# MANUFACTURING Domain - yahoo_anomaly

## Results Summary

### 🏆 Winners

- **Best Quality**: Thompson (1.30 SMAPE)
- **Best Cost**: Greedy-Quality ($0.240)
- **Best Latency**: Static-Best (967ms)

### 📊 Complete Rankings (by Quality)

| Rank | Method | Quality (SMAPE)↓ | Cost ($)↓ | Latency (ms)↓ |
|------|--------|------------------|-----------|---------------|
| 🥇 1 | **Thompson** | 1.30 | $0.267 | 1002 |
| 🥈 2 | **Greedy-Cost** | 1.48 | $0.259 | 997 |
| 🥉 3 | **Greedy-Quality** | 1.53 | $0.240 | 1005 |
|    4 | **Random** | 1.55 | $0.269 | 1015 |
|    5 | **MetaPipe** | 1.58 | $0.249 | 985 |
|    6 | **Static-Best** | 1.63 | $0.266 | 967 |

### 📁 Files

- `data/results.csv` - Complete numerical results
- `figures/comparison.png` - Bar chart comparison
- `figures/radar.png` - Radar chart (multi-metric)
- `figures/ranking.png` - Ranking table

### 🎯 Key Insight

**Thompson** achieves the best quality (1.30 SMAPE) on yahoo_anomaly.

---

Generated: 2025-10-06 19:51:32
