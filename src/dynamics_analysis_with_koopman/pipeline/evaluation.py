"""Utility functions for model evaluation and observable type extraction."""

from datafold import TSCDataFrame


def compute_mae(ground_truth: TSCDataFrame, predicted: TSCDataFrame) -> list[float]:
    """Compute mean absolute error between ground truth and predicted data."""
    diff = ground_truth.to_numpy() - predicted.to_numpy()
    return list(abs(diff).mean(axis=0))


def extract_data_column_types(tsc: TSCDataFrame) -> set[str]:
    """Extract unique column type suffixes from TSCDataFrame columns."""
    return {str(col).split('_')[-1] for col in tsc.columns}


def extract_observable_types(system_observables: list[str]) -> set[str]:
    """Extract unique observable type prefixes from system observables."""
    return {col.split('_')[0] for col in system_observables}
