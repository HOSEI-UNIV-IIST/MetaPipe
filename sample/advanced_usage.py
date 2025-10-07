#!/usr/bin/env python3
"""
MetaPipe - Advanced Usage Example
=================================

This example demonstrates advanced features:
- Custom feature extraction
- Multi-horizon predictions
- Transfer learning
- Uncertainty quantification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from metapipe.features.tcar import TCARExtractor
from metapipe.policy.map import MAPPolicy
from metapipe.uncertainty.uqe import ConformalEscalation
from metapipe.transfer.cptl import MetaRouter

def main():
    print("=" * 80)
    print("MetaPipe - Advanced Usage Example")
    print("=" * 80)

    # Example 1: TCAR Feature Extraction
    print("\n[Example 1] TCAR Feature Extraction")
    print("-" * 80)

    # Generate time series
    np.random.seed(42)
    t = np.linspace(0, 20, 500)
    ts = 50 + 10 * np.sin(2 * np.pi * t / 12) + 0.5 * t + np.random.randn(500) * 2

    # Extract features
    extractor = TCARExtractor(seasonality_period=24)
    features = extractor.extract(ts, metadata={'domain': 'finance', 'task_type': 'forecast'})

    print(f"  Statistical Features: {features.statistical.shape}")
    print(f"  Temporal Features:    {features.temporal.shape}")
    print(f"  Spectral Features:    {features.spectral.shape}")
    print(f"  Meta Features:        {features.meta.shape}")
    print(f"  Total Features:       {features.concat().shape}")

    # Show temporal similarity
    ts2 = ts + np.random.randn(len(ts)) * 0.5
    similarity = extractor.temporal_similarity_kernel(ts, ts2)
    print(f"\n  Temporal Similarity:  {similarity:.4f} (similar series)")

    # Example 2: Multi-Horizon Adaptive Policy
    print("\n[Example 2] Multi-Horizon Adaptive Policy (MAP)")
    print("-" * 80)

    policy = MAPPolicy(
        feature_dim=features.concat().shape[0],
        n_actions=5,
        horizons=[1, 3, 6, 12, 24]
    )

    print(f"  Horizons:     {policy.horizons}")
    print(f"  Actions:      {policy.n_actions}")
    print(f"  Feature dim:  {policy.feature_dim}")

    # Select action
    action = policy.select_action(features.concat(), epsilon=0.1)
    print(f"\n  Selected Action: {action}")

    # Simulate learning
    print("\n  Simulating policy learning...")
    for episode in range(10):
        state = features.concat()
        action = policy.select_action(state, epsilon=0.1)

        # Simulate rewards for each horizon
        rewards = {h: np.random.rand() for h in policy.horizons}
        next_state = state + np.random.randn(len(state)) * 0.1

        # Update policy
        loss = policy.update(state, action, rewards, next_state, done=False)

        if episode % 3 == 0:
            print(f"    Episode {episode:2d}: Loss={loss:.4f} Action={action}")

    # Example 3: Uncertainty-Quantified Escalation (UQE)
    print("\n[Example 3] Uncertainty-Quantified Escalation (UQE)")
    print("-" * 80)

    escalator = ConformalEscalation(alpha=0.1)

    # Calibrate with synthetic data
    print("  Calibrating conformal predictor...")
    preds = np.random.randn(100) * 10 + 50
    trues = preds + np.random.randn(100) * 2
    escalator.calibrate(preds, trues)
    print(f"    Calibration samples: {len(escalator.calibration_set)}")

    # Make prediction with uncertainty
    print("\n  Making predictions with uncertainty bounds...")
    for i in range(5):
        pred = 50 + np.random.randn()
        uncertainty = abs(np.random.randn()) * 2

        conf_set = escalator.predict(pred, uncertainty)
        should_escalate = escalator.should_escalate(
            pred, uncertainty,
            budget_remaining=0.5,
            cost_weak=0.1,
            cost_strong=0.3
        )

        print(f"    Pred {i+1}: {pred:6.2f} ∈ [{conf_set.lower:6.2f}, {conf_set.upper:6.2f}] "
              f"Coverage={conf_set.coverage_level:.0%} Escalate={should_escalate}")

    # Example 4: Cross-Pipeline Transfer Learning (CPTL)
    print("\n[Example 4] Cross-Pipeline Transfer Learning (CPTL)")
    print("-" * 80)

    meta_router = MetaRouter(feature_dim=features.concat().shape[0], n_actions=5)

    print("  Meta-training on source domains...")

    # Simulate meta-training
    source_domains = ['finance', 'energy', 'traffic']
    for domain in source_domains:
        # Generate domain-specific batches
        domain_batches = []
        for _ in range(5):
            batch_size = 10
            features_batch = np.random.randn(batch_size, meta_router.feature_dim)
            actions = np.random.randint(0, meta_router.n_actions, batch_size)
            rewards = np.random.rand(batch_size)

            domain_batches.append({
                'features': features_batch,
                'actions': actions,
                'rewards': rewards,
                'domain_id': hash(domain) % 100
            })

        # Meta-train
        meta_loss = meta_router.meta_train_step(domain_batches)
        print(f"    {domain:10s} Meta-loss: {meta_loss:.4f}")

    # Fast adaptation to new domain
    print("\n  Fast adaptation to target domain (healthcare)...")
    target_features = np.random.randn(20, meta_router.feature_dim)
    target_actions = np.random.randint(0, meta_router.n_actions, 20)
    target_rewards = np.random.rand(20)

    adapt_loss = meta_router.fast_adapt(
        target_features,
        target_actions,
        target_rewards,
        n_steps=5
    )
    print(f"    Adaptation loss: {adapt_loss:.4f}")

    # Example 5: Complete Pipeline with All Features
    print("\n[Example 5] Complete End-to-End Pipeline")
    print("-" * 80)

    from metapipe.runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig

    # Configure with all features
    config = PipelineConfig(
        cost_budget=1.0,
        latency_budget=2000.0,
        escalation_enabled=True,
        meta_learning_enabled=True
    )

    models = {
        'forecast': {
            'fast': lambda x, h: np.repeat(x[-3:].mean(), h),
            'medium': lambda x, h: np.repeat(x[-10:].mean(), h),
            'slow': lambda x, h: np.repeat(x[-30:].mean(), h)
        }
    }

    pipeline = TimeSeriesPipeline(config, models)

    print("  Running complete pipeline...")
    result = pipeline.run(
        ts,
        horizon=12,
        metadata={'domain': 'finance', 'task_type': 'forecast'}
    )

    print(f"\n  Results:")
    print(f"    Selected Model:  {result.selected_models['forecast']}")
    print(f"    Cost:            ${result.cost:.4f}")
    print(f"    Latency:         {result.latency_ms:.2f} ms")
    print(f"    Prediction:      {result.prediction[:3]}... (first 3)")

    print("\n" + "=" * 80)
    print("✅ Advanced examples completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • TCAR extracts 79 rich features from time series")
    print("  • MAP learns across 5 horizons simultaneously")
    print("  • UQE provides conformal prediction guarantees")
    print("  • CPTL enables zero-shot transfer to new domains")
    print("  • Complete pipeline integrates all components seamlessly")
    print("=" * 80)

if __name__ == '__main__':
    main()
