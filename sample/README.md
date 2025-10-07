# MetaPipe Sample Code

This directory contains example scripts and notebooks demonstrating MetaPipe usage.

## 📁 Contents

### 1. Basic Usage
**File**: `basic_usage.py`

Simple introduction to MetaPipe:
- Pipeline configuration
- Model pool definition
- Basic forecasting
- Budget constraints

```bash
python sample/basic_usage.py
```

**Output**: Demonstrates automatic model selection with 3 simple forecasting methods.

---

### 2. Advanced Usage
**File**: `advanced_usage.py`

Comprehensive examples of all MetaPipe features:
- **TCAR**: Feature extraction (79 features from time series)
- **MAP**: Multi-horizon policy learning (5 horizons)
- **UQE**: Uncertainty quantification with conformal prediction
- **CPTL**: Transfer learning across domains
- **Complete Pipeline**: End-to-end integration

```bash
python sample/advanced_usage.py
```

**Output**: Demonstrates all 5 novel algorithms in action.

---

### 3. Real Dataset Example
**File**: `real_data_example.py`

Using MetaPipe with real TSDB datasets:
- Loading UCR/Monash datasets
- Multi-dataset evaluation
- Cross-domain experiments

```bash
python sample/real_data_example.py
```

**Output**: Performance on real-world time-series datasets.

---

### 4. Jupyter Notebook (Interactive)
**File**: `metapipe_tutorial.ipynb`

Interactive tutorial covering:
- Installation and setup
- Step-by-step walkthrough
- Visualization of results
- Custom model integration

```bash
jupyter notebook sample/metapipe_tutorial.ipynb
```

**Requirements**: `pip install jupyter matplotlib pandas`

---

## 🚀 Quick Start

### Run All Examples
```bash
# Basic usage
python sample/basic_usage.py

# Advanced features
python sample/advanced_usage.py

# Real datasets
python sample/real_data_example.py
```

### Expected Output Structure

```
sample/
├── README.md                    # This file
├── basic_usage.py              # Simple introduction
├── advanced_usage.py           # All features demo
├── real_data_example.py        # Real dataset usage
├── metapipe_tutorial.ipynb     # Interactive notebook
└── outputs/                     # Generated results (auto-created)
    ├── basic_results.png
    ├── advanced_results.png
    └── real_data_results.png
```

---

## 📊 What You'll Learn

### From `basic_usage.py`:
✅ How to configure MetaPipe pipeline
✅ Define custom forecasting models
✅ Set budget constraints
✅ Run predictions with automatic model selection

### From `advanced_usage.py`:
✅ Extract TCAR features (statistical, temporal, spectral, meta)
✅ Train multi-horizon policies (MAP)
✅ Use conformal prediction for uncertainty (UQE)
✅ Transfer learning to new domains (CPTL)
✅ Integrate all components in end-to-end pipeline

### From `real_data_example.py`:
✅ Load TSDB datasets (UCR, Monash, UCI)
✅ Evaluate on multiple real datasets
✅ Compare against baselines
✅ Generate performance reports

### From `metapipe_tutorial.ipynb`:
✅ Interactive step-by-step walkthrough
✅ Visualize intermediate results
✅ Experiment with parameters
✅ Custom model integration guide

---

## 🎯 Use Cases

### Research Papers
Use `advanced_usage.py` and `real_data_example.py` to:
- Reproduce paper results
- Compare against your methods
- Generate figures for publications

### Production Deployment
Use `basic_usage.py` as template for:
- Integrating MetaPipe in your system
- Configuring budget constraints
- Custom model pools

### Learning & Education
Use `metapipe_tutorial.ipynb` for:
- Understanding meta-learning concepts
- Hands-on experimentation
- Teaching AutoML techniques

---

## 💡 Tips

1. **Start Simple**: Begin with `basic_usage.py` to understand core concepts
2. **Explore Features**: Run `advanced_usage.py` to see all capabilities
3. **Real Data**: Use `real_data_example.py` for practical applications
4. **Interactive**: Open `metapipe_tutorial.ipynb` for experimentation

---

## 🔧 Customization

### Add Your Own Model

```python
# Define custom model
def my_custom_model(x, horizon):
    # Your forecasting logic here
    return predictions

# Add to model pool
models = {
    'forecast': {
        'my_model': my_custom_model,
        # ... other models
    }
}
```

### Adjust Budgets

```python
config = PipelineConfig(
    cost_budget=2.0,        # Increase cost budget
    latency_budget=10000.0, # Increase latency budget
    escalation_enabled=True
)
```

### Custom Datasets

```python
from metapipe.data.tsdb_loader import TSDBDataset

# Load your dataset
dataset = TSDBDataset.load('your_dataset_name')
X, y = dataset.get_train_test_split()
```

---

## 📈 Expected Performance

Running examples should show:
- **Basic**: ~30 seconds, demonstrates routing
- **Advanced**: ~2 minutes, shows all features
- **Real Data**: ~5 minutes, evaluates on datasets
- **Notebook**: Interactive, variable time

---

## 🐛 Troubleshooting

**Issue**: Import errors
```bash
# Solution: Ensure MetaPipe is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue**: Missing dependencies
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

**Issue**: Notebook won't open
```bash
# Solution: Install Jupyter
pip install jupyter notebook
```

---

## 📚 Further Reading

- **Main README**: `../README.md` - Project overview
- **Design Doc**: `../DESIGN.md` - Technical details
- **Results**: `../RESULTS/` - Experimental results
- **Tests**: `../tests/` - Comprehensive test suite

---

## 🤝 Contributing

Found a useful example? Add it here!

1. Create new Python file in `sample/`
2. Follow existing structure
3. Add documentation
4. Update this README
5. Submit PR

---

**Happy Experimenting! 🎉**
