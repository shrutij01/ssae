#!/usr/bin/env python3
"""
Analyze metric values from metric_values.py
Print means and standard deviations organized by data type suffix
"""
import numpy as np
from data.metric_values import *
import inspect

def analyze_metrics():
    """Analyze all arrays in metric_values.py and print stats by dataset"""

    # Get all variables from metric_values module
    import data.metric_values as mv

    # Group arrays by dataset (prefix before metric type)
    datasets = {}

    for name in dir(mv):
        if not name.startswith('_') and name != 'np':  # Skip private vars and numpy
            value = getattr(mv, name)
            if isinstance(value, np.ndarray):
                # Extract dataset and metric type
                parts = name.split('_')
                if len(parts) >= 2:
                    metric_type = parts[-1]  # ds, sp, or mcc
                    dataset = '_'.join(parts[:-1])  # corr0, corr1, corrpt1, etc.

                    if dataset not in datasets:
                        datasets[dataset] = {}

                    datasets[dataset][metric_type] = value

    # Print results organized by dataset
    for dataset in sorted(datasets.keys()):
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")

        metrics = datasets[dataset]

        # Print each metric type for this dataset
        for metric_type in ['ds', 'sp', 'mcc']:  # Consistent ordering
            if metric_type in metrics:
                array = metrics[metric_type]
                mean_val = np.mean(array)
                std_val = np.std(array)

                print(f"{metric_type.upper()}:")
                print(f"  Array: {array}")
                print(f"  Mean:  {mean_val:.4f}")
                print(f"  Std:   {std_val:.4f}")
                print()

if __name__ == "__main__":
    analyze_metrics()