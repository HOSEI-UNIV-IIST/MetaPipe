#!/usr/bin/env python3
"""
Quick Test - Verify MetaPipe Installation and Basic Functionality
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("=" * 70)
print("MetaPipe Quick Test")
print("=" * 70)

# Test 1: TCAR
print("\n[1/5] Testing TCAR Feature Extraction...")
try:
    from metapipe.features.tcar import TCARExtractor

    extractor = TCARExtractor(seasonality_period=24)
    x = np.random.randn(200)
    features = extractor.extract(x, metadata={'domain': 'finance', 'task_type': 'forecast'})

    print(f"   ✓ Features extracted: {features.concat().shape}")
    print(f"   ✓ Statistical: {features.statistical.shape[0]} features")
    print(f"   ✓ Temporal: {features.temporal.shape[0]} features")
    print(f"   ✓ Spectral: {features.spectral.shape[0]} features")
    print(f"   ✓ Meta: {features.meta.shape[0]} features")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: MAP
print("\n[2/5] Testing MAP Policy...")
try:
    from metapipe.policy.map import MAPPolicy

    policy = MAPPolicy(
        feature_dim=79,  # from TCAR
        n_actions=5,
        horizons=[1, 3, 6, 12, 24]
    )

    action = policy.select_action(features.concat(), epsilon=0.0)
    print(f"   ✓ Policy initialized with {len(policy.horizons)} horizons")
    print(f"   ✓ Selected action: {action}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: BCPR
print("\n[3/5] Testing BCPR Optimizer...")
try:
    from metapipe.optimizer.bcpr import BCPROptimizer

    optimizer = BCPROptimizer(
        cost_budget=1.0,
        latency_budget=2000.0
    )

    reward, violation = optimizer.compute_augmented_reward(
        quality=0.85,
        cost=0.5,
        latency=1000.0
    )

    print(f"   ✓ Optimizer initialized")
    print(f"   ✓ Augmented reward: {reward:.4f}")
    print(f"   ✓ Cost violation: {violation.cost_violation:.2f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 4: UQE
print("\n[4/5] Testing UQE Escalation...")
try:
    from metapipe.uncertainty.uqe import ConformalEscalation

    escalator = ConformalEscalation(alpha=0.1)

    # Calibrate
    preds = np.random.randn(100)
    trues = preds + np.random.randn(100) * 0.5
    escalator.calibrate(preds, trues)

    # Predict
    conf_set = escalator.predict(10.0, 1.0)

    print(f"   ✓ Conformal predictor calibrated with {len(escalator.calibration_set)} samples")
    print(f"   ✓ Prediction interval: [{conf_set.lower:.2f}, {conf_set.upper:.2f}]")
    print(f"   ✓ Coverage level: {conf_set.coverage_level:.2%}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 5: Pipeline
print("\n[5/5] Testing End-to-End Pipeline...")
try:
    from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig

    config = PipelineConfig(
        cost_budget=1.0,
        latency_budget=5000.0,
        escalation_enabled=True
    )

    models = {
        'forecast': {
            'simple_avg': lambda x, horizon: np.repeat(x[-3:].mean(), horizon),
            'last_value': lambda x, horizon: np.repeat(x[-1], horizon)
        }
    }

    pipeline = TimeSeriesPipeline(config, models)

    # Run pipeline
    x = np.random.randn(100)
    result = pipeline.run(x, horizon=12, metadata={'domain': 'finance', 'task_type': 'forecast'})

    print(f"   ✓ Pipeline initialized")
    print(f"   ✓ Prediction shape: {result.prediction.shape}")
    print(f"   ✓ Cost: ${result.cost:.4f}")
    print(f"   ✓ Latency: {result.latency_ms:.2f} ms")
    print(f"   ✓ Selected model: {result.selected_models.get('forecast', 'N/A')}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ All Tests Passed!")
print("=" * 70)
print("\nMetaPipe is ready to use!")
print("\nNext steps:")
print("  1. Run full experiments: python run_experiments.py --quick")
print("  2. Run tests: bash run_tests.sh")
print("  3. Run demo: python main.py --mode demo")
print("=" * 70)
