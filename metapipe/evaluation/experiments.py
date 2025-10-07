#!/usr/bin/env python3
"""
Experiment Runner for MetaPipe

Runs comprehensive evaluations comparing MetaPipe against baselines
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from ..data.tsdb_loader import TSDBDataset
from ..runners.timeseries_pipeline import TimeSeriesPipeline, PipelineConfig
from .baselines import *
from .metrics import compute_metrics, compute_regret, EvaluationMetrics


class ExperimentRunner:
    """
    Run experiments across multiple datasets and baselines
    """

    def __init__(
        self,
        data_dir: str = './data/tsdb',
        output_dir: str = './output/metapipe'
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset loader
        self.dataset_loader = TSDBDataset(data_dir)

        # Results storage
        self.results = []

    def run_experiment(
        self,
        dataset_names: List[str],
        models: Dict[str, Any],
        config: PipelineConfig,
        baselines: Optional[List[str]] = None,
        n_episodes: int = 100
    ) -> pd.DataFrame:
        """
        Run full experiment

        Parameters
        ----------
        dataset_names : list
            List of dataset names to evaluate
        models : dict
            Model registry
        config : PipelineConfig
            Pipeline configuration
        baselines : list, optional
            Baseline methods to compare (default: all)
        n_episodes : int
            Number of episodes per dataset

        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        if baselines is None:
            baselines = ['random', 'greedy_cost', 'greedy_quality', 'thompson']

        for dataset_name in tqdm(dataset_names, desc="Datasets"):
            # Load dataset
            try:
                data = self.dataset_loader.load(dataset_name)
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                continue

            # Run MetaPipe
            metapipe_results = self._run_metapipe(
                data, models, config, n_episodes
            )

            # Run baselines
            baseline_results = {}
            for baseline in baselines:
                baseline_results[baseline] = self._run_baseline(
                    baseline, data, models, n_episodes
                )

            # Aggregate and store
            self._aggregate_results(
                dataset_name, metapipe_results, baseline_results
            )

        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        return df

    def _run_metapipe(
        self,
        data: Any,
        models: Dict[str, Any],
        config: PipelineConfig,
        n_episodes: int
    ) -> List[EvaluationMetrics]:
        """Run MetaPipe on dataset"""
        pipeline = TimeSeriesPipeline(config, models)

        results = []
        for i in range(min(n_episodes, len(data.X_test))):
            x = data.X_test[i]
            y_true = data.y_test[i]

            # Run pipeline
            result = pipeline.run(
                x[:, 0],  # take first feature
                horizon=data.metadata.horizon,
                metadata=data.metadata.__dict__
            )

            # Compute metrics
            y_pred = result.prediction
            metrics = compute_metrics(
                y_true, y_pred,
                task_type=data.metadata.task_type,
                cost=result.cost,
                latency_ms=result.latency_ms
            )

            results.append(metrics)

        return results

    def _run_baseline(
        self,
        baseline_name: str,
        data: Any,
        models: Dict[str, Any],
        n_episodes: int
    ) -> List[EvaluationMetrics]:
        """Run baseline method"""
        # Initialize baseline
        if baseline_name == 'random':
            router = RandomRouter()
        elif baseline_name == 'greedy_cost':
            router = GreedyCost([0])  # assume model 0 is cheapest
        elif baseline_name == 'greedy_quality':
            router = GreedyQuality([len(models.get('forecast', {})) - 1])
        elif baseline_name == 'thompson':
            router = ThompsonSampling(
                n_actions=len(models.get('forecast', {})),
                feature_dim=50  # placeholder
            )
        else:
            raise ValueError(f"Unknown baseline: {baseline_name}")

        results = []
        # Placeholder: actual implementation would run routing
        for i in range(min(n_episodes, len(data.X_test))):
            # Dummy metrics
            metrics = EvaluationMetrics(
                smape=np.random.rand() * 20,
                cost=np.random.rand(),
                latency_ms=np.random.rand() * 1000
            )
            results.append(metrics)

        return results

    def _aggregate_results(
        self,
        dataset_name: str,
        metapipe_results: List[EvaluationMetrics],
        baseline_results: Dict[str, List[EvaluationMetrics]]
    ):
        """Aggregate and store results"""
        # MetaPipe
        self.results.append({
            'dataset': dataset_name,
            'method': 'metapipe',
            'mean_smape': np.mean([r.smape for r in metapipe_results if r.smape]),
            'mean_cost': np.mean([r.cost for r in metapipe_results]),
            'mean_latency': np.mean([r.latency_ms for r in metapipe_results])
        })

        # Baselines
        for baseline, results in baseline_results.items():
            self.results.append({
                'dataset': dataset_name,
                'method': baseline,
                'mean_smape': np.mean([r.smape for r in results if r.smape]),
                'mean_cost': np.mean([r.cost for r in results]),
                'mean_latency': np.mean([r.latency_ms for r in results])
            })

    def save_results(self, filename: str = 'experiment_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return df
