# Overall Performance - All Datasets Combined

## Results Summary (Averaged Across All 7 Datasets)

### 🏆 Overall Winners

- **Best Quality**: MetaPipe (8.97 SMAPE)
- **Best Cost**: MetaPipe ($0.247)
- **Best Latency**: Static-Best (971ms)

### 📊 Complete Rankings (by Quality)

| Rank | Method | Quality (SMAPE)↓ | Cost ($)↓ | Latency (ms)↓ |
|------|--------|------------------|-----------|---------------|
| 🥇 1 | **MetaPipe** | 8.97 | $0.247 | 993 |\n| 🥈 2 | **Static-Best** | 9.69 | $0.250 | 971 |\n| 🥉 3 | **Greedy-Quality** | 9.75 | $0.256 | 1000 |\n|    4 | **Random** | 10.39 | $0.253 | 1012 |\n|    5 | **Thompson** | 10.41 | $0.253 | 981 |\n|    6 | **Greedy-Cost** | 10.83 | $0.249 | 1011 |\n
### 📁 Files

- `data/aggregated_results.csv` - Averaged results across all datasets
- `figures/overall_comparison.png` - Bar chart comparison
- `figures/overall_radar.png` - Radar chart (multi-metric)
- `figures/overall_ranking.png` - Ranking table

### 🎯 Key Insight

**MetaPipe** achieves the best average quality (8.97 SMAPE) across all 7 datasets.

### 📂 Individual Dataset Results

For per-dataset analysis, see:
- [Finance (Stock SP500)](../finance_stock_sp500/README.md)
- [Energy (Electricity Load)](../energy_electricity_load/README.md)
- [Healthcare (MIMIC Vitals)](../healthcare_mimic_vitals/README.md)
- [Climate (Temperature)](../climate_temperature/README.md)
- [Traffic (PeMS Traffic)](../traffic_pems_traffic/README.md)
- [Manufacturing (Yahoo Anomaly)](../manufacturing_yahoo_anomaly/README.md)
- [Retail (Stock SP500)](../retail_stock_sp500/README.md)

---

Generated: 2025-10-06 19:57:07
