#!/usr/bin/env python3
"""
MetaPipe - Basic Usage Example
==============================

This example demonstrates the basic usage of MetaPipe for time-series forecasting
with automatic model selection and budget constraints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig

def simple_moving_average(x, horizon):
    """Simple moving average baseline"""
    return np.repeat(x[-5:].mean(), horizon)

def exponential_smoothing(x, horizon, alpha=0.3):
    """Exponential smoothing baseline"""
    forecast = []
    s = x[-1]
    for _ in range(horizon):
        forecast.append(s)
        s = alpha * s + (1 - alpha) * x[-1]
    return np.array(forecast)

def last_value(x, horizon):
    """Naive last value baseline"""
    return np.repeat(x[-1], horizon)

def main():
    print("=" * 70)
    print("MetaPipe - Basic Usage Example")
    print("=" * 70)

    # Step 1: Configure pipeline
    print("\n[Step 1] Configuring pipeline...")
    config = PipelineConfig(
        cost_budget=1.0,           # Maximum $1 per prediction
        latency_budget=5000.0,     # Maximum 5 seconds
        escalation_enabled=True    # Enable uncertainty-based escalation
    )
    print(f"  ✓ Budget: ${config.cost_budget} / {config.latency_budget}ms")
    print(f"  ✓ Escalation: {config.escalation_enabled}")

    # Step 2: Define model pool
    print("\n[Step 2] Defining model pool...")
    models = {
        'forecast': {
            'sma': simple_moving_average,
            'exp_smooth': exponential_smoothing,
            'naive': last_value
        }
    }
    print(f"  ✓ {len(models['forecast'])} models registered")

    # Step 3: Initialize pipeline
    print("\n[Step 3] Initializing MetaPipe...")
    pipeline = TimeSeriesPipeline(config, models)
    print("  ✓ Pipeline ready")

    # Step 4: Generate synthetic time series
    print("\n[Step 4] Generating synthetic data...")
    np.random.seed(42)
    t = np.linspace(0, 10, 200)
    trend = 0.5 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 5)
    noise = np.random.randn(200) * 2
    time_series = trend + seasonal + noise
    print(f"  ✓ Series length: {len(time_series)}")

    # Step 5: Run prediction
    print("\n[Step 5] Running prediction...")
    result = pipeline.run(
        time_series,
        horizon=12,
        metadata={'domain': 'synthetic', 'task_type': 'forecast'}
    )

    # Step 6: Display results
    print("\n[Step 6] Results:")
    print("=" * 70)
    print(f"  Prediction:     {result.prediction[:5]}... (first 5 points)")
    print(f"  Cost:           ${result.cost:.4f}")
    print(f"  Latency:        {result.latency_ms:.2f} ms")
    print(f"  Selected Model: {result.selected_models.get('forecast', 'N/A')}")
    print(f"  Within Budget:  Cost={result.cost <= config.cost_budget}, "
          f"Latency={result.latency_ms <= config.latency_budget}")
    print("=" * 70)

    # Step 7: Run multiple predictions to see adaptive routing
    print("\n[Step 7] Testing adaptive routing (5 runs)...")
    print("-" * 70)
    for i in range(5):
        # Add some variation to the series
        varied_series = time_series + np.random.randn(len(time_series)) * 0.5
        result = pipeline.run(
            varied_series,
            horizon=12,
            metadata={'domain': 'synthetic', 'task_type': 'forecast'}
        )
        print(f"  Run {i+1}: Model={result.selected_models['forecast']:12s} "
              f"Cost=${result.cost:.4f} Latency={result.latency_ms:6.1f}ms")

    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
