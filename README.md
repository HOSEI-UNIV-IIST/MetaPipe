# MetaPipe: Budget-Constrained Time-Series Routing with Meta-Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**MetaPipe** is a meta-learning framework for automatic time-series model selection with hard budget constraints on cost and latency.

[📖 Documentation](DESIGN.md) | [🚀 Quick Start](#quick-start) | [📝 Examples](sample/)

---

## 🎯 Overview

MetaPipe automatically selects the best forecasting model for each task while respecting **hard budget constraints**:

- ✅ **46% cost reduction** vs greedy baselines
- ✅ **Multi-horizon learning** (1 to 24 steps ahead)
- ✅ **Zero-shot transfer** to new domains
- ✅ **Theoretical guarantees** with 4 formal theorems

## 🏆 Novel Contributions

### 1. TCAR - Temporal Context-Aware Routing
Novel 79-feature extraction combining statistical, temporal, spectral, and meta-features with custom similarity kernel.

### 2. MAP - Multi-Horizon Adaptive Policy
Q-learning across 5 forecasting horizons with proven regret bound: R_T ≤ O(√(|H||A|T log T))

### 3. BCPR - Budget-Constrained Pareto Routing
Lagrangian primal-dual optimization with **hard** budget enforcement (not soft penalties).

### 4. UQE - Uncertainty-Quantified Escalation
Conformal prediction with finite-sample coverage guarantee: P(y ∈ C_α(x)) ≥ 1-α

### 5. CPTL - Cross-Pipeline Transfer Learning
MAML-inspired meta-learning for zero-shot domain adaptation.

## 🚀 Quick Start

### **Fastest Way: One Command**

```bash
./run_metapipe.sh
```

This runs all experiments and generates figures in **~1 minute**.

Results: `RESULTS/raw_data/*.csv` and `*.png`

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/metapipe.git
cd MetaPipe

# Install dependencies
pip install -r requirements.txt

# Run experiments
./run_metapipe.sh
```

### Basic Usage

```python
from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig

# Configure pipeline with budget constraints
config = PipelineConfig(
    cost_budget=1.0,           # Max $1 per prediction
    latency_budget=5000.0,     # Max 5 seconds
    escalation_enabled=True
)

# Define model pool
models = {
    'forecast': {
        'fast_model': lambda x, h: ...,
        'accurate_model': lambda x, h: ...,
    }
}

# Run prediction
pipeline = TimeSeriesPipeline(config, models)
result = pipeline.run(time_series, horizon=12, metadata={'domain': 'finance'})

print(f"Cost: ${result.cost:.4f}, Latency: {result.latency_ms:.2f}ms")
print(f"Selected: {result.selected_models['forecast']}")
```

## 📊 Results

### Main Performance

| Metric | MetaPipe | Best Baseline | Improvement |
|--------|----------|---------------|-------------|
| **Cost** | $0.18 ± 0.17 | $0.34 ± 0.10 | **46% lower** |
| **Quality** | 150.06 ± 43.28 | 160.82 ± 49.03 | Comparable |

### Component Contributions (Ablation Study)

| Configuration | Quality | Cost | Latency |
|---------------|---------|------|---------|
| **Full System** | **12.49** | **$0.44** | **1059ms** |
| -TCAR | 7.52 (-40%) | $0.76 (+73%) | 1291ms |
| -MAP | 10.33 (-17%) | $0.60 (+36%) | 1183ms |
| -UQE | 6.29 (-50%) | $0.57 (+30%) | 1468ms |

### Transfer Learning

Zero-shot transfer across domains:
- Finance → Healthcare: 79% efficiency
- Finance → Climate: 93% efficiency

## 📁 Project Structure

```
MetaPipe/
├── metapipe/                   # Core implementation
│   ├── features/              # TCAR feature extraction
│   ├── policy/                # MAP policy learning
│   ├── optimizer/             # BCPR budget optimization
│   ├── uncertainty/           # UQE escalation
│   ├── transfer/              # CPTL meta-learning
│   ├── data/                  # TSDB dataset loaders
│   └── runners/               # Pipeline orchestration
├── tests/                     # Comprehensive test suite
├── sample/                    # Example scripts
├── RESULTS/                   # Experimental results
├── DESIGN.md                  # Technical documentation
└── README.md                  # This file
```

## 🧪 Running Experiments

```bash
# Quick validation (30 seconds)
python quick_test.py

# Run experiments (generates all results)
python run_experiments.py --n_episodes 1000

# Generate professional figures
python generate_results.py

# View interactive dashboard
./view_results.sh

# Run test suite
bash run_tests.sh
```

## 📝 Examples

See [sample/](sample/) directory for complete examples:

- `basic_usage.py` - Simple introduction (~30s)
- `advanced_usage.py` - All 5 algorithms demo (~2min)
- `real_data_example.py` - Real dataset evaluation (~5min)

## 📚 Datasets

Integrated time-series datasets from:
- UCR Archive
- Monash Repository
- UCI, Yahoo, Numenta

Covering 7 domains: Finance, Energy, Healthcare, Climate, Traffic, Manufacturing, Retail

## 🎓 Theoretical Guarantees

### Theorem 1: MAP Regret Bound
R_T ≤ O(√(|H| |A| T log T)) - Sublinear regret for multi-horizon policy

### Theorem 2: BCPR Convergence
Convergence to ε-optimal in O(1/ε²) iterations

### Theorem 3: UQE Coverage
P(y ∈ C_α(x)) ≥ 1-α with finite-sample guarantee

### Theorem 4: CPTL Transfer Bound
L_target ≤ L_source + O(d_domain(S,T)) + ε_adapt

**[Full proofs in DESIGN.md](DESIGN.md)**

## 📄 Citation

```bibtex
@article{messou2025metapipe,
  title={MetaPipe: Budget-Constrained Time-Series Routing with Meta-Learning},
  author={Messou, Franck Junior Aboya and Chen, Jinhua and Liu, Tong and Zhang, Shilong and Tolba, Amr and Alfarraj, Osama and Yu, Keping},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2025},
  note={Open-source implementation available at \url{https://github.com/mesabo/XPipeline}}
}
```

**Authors**:
- Franck Junior Aboya Messou¹, Jinhua Chen¹, Tong Liu¹, Shilong Zhang¹
- Amr Tolba²*, Osama Alfarraj²
- Keping Yu¹*

**Affiliations**:
1. Graduate School of Science and Engineering, Hosei University, 3-7-2 Kajino-cho, Koganei-shi, Tokyo 184-8584, Japan
2. Department of Computer Science and Engineering, College of Applied Studies, King Saud University, Riyadh 11437, Saudi Arabia

**Corresponding Authors**:
- Keping Yu (keping.yu@ieee.org)
- Amr Tolba (atolba@ksu.edu.sa)

**Funding**: Ongoing Research Funding Program (ORF-2026-681), King Saud University, Riyadh, Saudi Arabia

## 🤝 Contributing

Contributions welcome! Areas of interest:
- New model adapters (Prophet, N-BEATS, TFT)
- Additional dataset integrations
- Performance optimizations
- Documentation improvements

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Technical Design**: [DESIGN.md](DESIGN.md)
- **Sample Code**: [sample/](sample/)

## ⭐ Acknowledgments

This work builds upon advances in:
- Meta-learning (MAML, Reptile)
- Conformal prediction
- Multi-objective optimization
- Time-series forecasting
