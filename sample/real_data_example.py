#!/usr/bin/env python3
"""
MetaPipe - Real Dataset Example
===============================

This example demonstrates MetaPipe usage with real TSDB datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig

def main():
    print("=" * 80)
    print("MetaPipe - Real Dataset Example")
    print("=" * 80)

    # Configure pipeline
    config = PipelineConfig(
        cost_budget=1.0,
        latency_budget=5000.0,
        escalation_enabled=True
    )

    # Define model pool with varying complexity
    models = {
        'forecast': {
            'simple_avg': lambda x, h: np.repeat(x[-5:].mean(), h),
            'weighted_avg': lambda x, h: np.repeat(np.average(x[-10:], weights=np.arange(1, 11)), h),
            'exp_smooth': lambda x, h: exponential_smoothing(x, h, alpha=0.3),
            'naive_seasonal': lambda x, h: naive_seasonal_forecast(x, h, period=12)
        }
    }

    pipeline = TimeSeriesPipeline(config, models)

    # Simulate different domain datasets
    print("\n[Evaluating on Synthetic Datasets]")
    print("-" * 80)

    datasets = generate_synthetic_datasets()

    results = []
    for name, (data, metadata) in datasets.items():
        print(f"\nDataset: {name}")
        print(f"  Domain: {metadata['domain']}, Length: {len(data)}")

        # Run prediction
        result = pipeline.run(
            data,
            horizon=12,
            metadata=metadata
        )

        print(f"  Selected Model:  {result.selected_models['forecast']}")
        print(f"  Cost:            ${result.cost:.4f}")
        print(f"  Latency:         {result.latency_ms:.2f} ms")

        results.append({
            'dataset': name,
            'model': result.selected_models['forecast'],
            'cost': result.cost,
            'latency': result.latency_ms
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20s} {'Model':<20s} {'Cost':>10s} {'Latency':>12s}")
    print("-" * 80)
    for r in results:
        print(f"{r['dataset']:<20s} {r['model']:<20s} ${r['cost']:>9.4f} {r['latency']:>11.2f}ms")

    print("\n" + "=" * 80)
    print("✅ Real dataset example completed!")
    print("=" * 80)

def exponential_smoothing(x, horizon, alpha=0.3):
    """Exponential smoothing forecast"""
    forecast = []
    s = x[-1]
    for _ in range(horizon):
        forecast.append(s)
        s = alpha * s + (1 - alpha) * x[-1]
    return np.array(forecast)

def naive_seasonal_forecast(x, horizon, period=12):
    """Naive seasonal forecast"""
    return np.tile(x[-period:], int(np.ceil(horizon / period)))[:horizon]

def generate_synthetic_datasets():
    """Generate synthetic datasets simulating different domains"""
    np.random.seed(42)

    datasets = {}

    # Finance - Stock prices (trending with volatility)
    t = np.linspace(0, 10, 300)
    trend = 100 + 5 * t
    volatility = np.random.randn(300) * 10
    datasets['finance_stock'] = (
        trend + volatility,
        {'domain': 'finance', 'task_type': 'forecast', 'frequency': 'daily'}
    )

    # Energy - Electricity demand (strong seasonality)
    t = np.linspace(0, 20, 500)
    daily_season = 50 * np.sin(2 * np.pi * t)
    weekly_season = 20 * np.sin(2 * np.pi * t / 7)
    datasets['energy_demand'] = (
        1000 + daily_season + weekly_season + np.random.randn(500) * 30,
        {'domain': 'energy', 'task_type': 'forecast', 'frequency': 'hourly'}
    )

    # Healthcare - Heart rate (irregular with spikes)
    t = np.linspace(0, 5, 200)
    base_rate = 70 + 5 * np.sin(2 * np.pi * t / 2)
    spikes = np.random.choice([0, 20], size=200, p=[0.9, 0.1])
    datasets['healthcare_ecg'] = (
        base_rate + spikes + np.random.randn(200) * 3,
        {'domain': 'healthcare', 'task_type': 'classify', 'frequency': '1s'}
    )

    # Climate - Temperature (seasonal with trend)
    t = np.linspace(0, 50, 600)
    annual_cycle = 15 * np.sin(2 * np.pi * t / 12)
    warming_trend = 0.02 * t
    datasets['climate_temp'] = (
        20 + annual_cycle + warming_trend + np.random.randn(600) * 2,
        {'domain': 'climate', 'task_type': 'forecast', 'frequency': 'monthly'}
    )

    # Traffic - Vehicle count (daily pattern + weekly pattern)
    t = np.linspace(0, 30, 720)
    daily = 1000 + 500 * np.sin(2 * np.pi * t)
    weekly = 200 * np.sin(2 * np.pi * t / 7)
    datasets['traffic_count'] = (
        daily + weekly + np.random.randn(720) * 100,
        {'domain': 'traffic', 'task_type': 'forecast', 'frequency': 'hourly'}
    )

    return datasets

if __name__ == '__main__':
    main()
