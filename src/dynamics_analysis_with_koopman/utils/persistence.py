"""Utility functions for persisting experiment artifacts to filesystem."""

from pathlib import Path
from typing import Mapping, Optional, Sequence

import dill
import numpy as np
import pandas as pd
import yaml
from datafold import EDMD, TSCDataFrame

from ..config.configs import FullConfig
from ..pipeline.results import RunMetrics


def save_config(config: FullConfig, path: Path) -> None:
    """Save configuration to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config.get_full_config_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=True)


def save_metrics(metrics: RunMetrics, path: Path) -> None:
    """Save run metrics to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(metrics.to_dict(), f, default_flow_style=False, allow_unicode=True, sort_keys=True)


def save_timeseries_data(
    data_dict: dict[int, pd.DataFrame],
    enrichment: dict[int, pd.DataFrame],
    path: Path,
) -> None:
    """Save timeseries data and enrichment data to CSV files."""
    path.mkdir(parents=True, exist_ok=True)
    for key, df in data_dict.items():
        df.to_csv(path / f'timeseries_data_{key}.csv', index=False)
    for key, df in enrichment.items():
        df.to_csv(path / f'enrichment_ts_data_{key}.csv', index=False)


def save_position_timeseries(data_dict: dict[int, pd.DataFrame], path: Path) -> None:
    """Save per-series, grid-reduced (pre-observable) trajectory data to CSV."""
    path.mkdir(parents=True, exist_ok=True)
    for key, df in data_dict.items():
        df.to_csv(path / f'position_ts_{key}.csv', index=False)


def save_trained_model(model: EDMD, path: Path) -> None:
    """Pickle a trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        dill.dump(model, f)


def save_predictions(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    path: Path,
    is_test: bool = False,
) -> None:
    """Save ground-truth and predicted time series as gzip-compressed pickles."""
    path.mkdir(parents=True, exist_ok=True)
    prefix = 'test_' if is_test else 'train_'
    tsc.to_pickle(path / f'{prefix}tsc.pkl.gz')
    tsc_predicted.to_pickle(path / f'{prefix}tsc_predicted.pkl.gz')


def save_diffusion_coordinates(
    diff_maps_df: pd.DataFrame,
    diff_embedding_df: pd.DataFrame,
    path: Path,
) -> None:
    """Save diffusion map eigenvalues and coordinates to CSV."""
    path.mkdir(parents=True, exist_ok=True)
    diff_maps_df.to_csv(path / 'diffusion_map_eigenvalues.csv', index=False)
    diff_embedding_df.to_csv(path / 'diffusion_map_coordinates.csv', index=False)


def save_koopman_spectrum(
    eigenvalues: np.ndarray,
    residuals: Optional[np.ndarray],
    path: Path,
) -> None:
    """Save the Koopman eigenvalues, and their spectral residuals when computed, to CSV."""
    path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        np.expand_dims(eigenvalues, axis=0),
        columns=[f'lambda_{i + 1}' for i in range(len(eigenvalues))],
    ).to_csv(path / 'eigenvalues.csv', index=False)

    if residuals is not None:
        pd.DataFrame(
            np.expand_dims(residuals, axis=0),
            columns=[f'res_{i + 1}' for i in range(len(residuals))],
        ).to_csv(path / 'residuals.csv', index=False)


def save_koopman_modes(modes: Mapping[str, np.ndarray], selection: Sequence[int], path: Path) -> None:
    """Save the selected Koopman modes, one file per observable type and one column per mode.

    Writes ``selected_modes_<observable type>.csv.gz``, gzip-compressed (they are large).
    """
    path.mkdir(parents=True, exist_ok=True)
    for observable_type, observable_modes in modes.items():
        pd.DataFrame(observable_modes.T, columns=[f'v_{i}' for i in selection]).to_csv(
            path / f'selected_modes_{observable_type}.csv.gz', index=False, compression='gzip'
        )


def save_koopman_eigenfunctions(
    eigenfunctions: pd.DataFrame,
    initial_state_idx: int,
    path: Path,
) -> None:
    """Save the eigenfunction values at the initial condition to CSV, one row per series id."""
    path.mkdir(parents=True, exist_ok=True)
    # plain frame first: a TSCDataFrame rejects any index that is not the (ID, time) MultiIndex
    values = pd.DataFrame(eigenfunctions)
    if isinstance(values.index, pd.MultiIndex):
        # the time is the same initial state for every row, so the series id alone identifies a row
        values = values.droplevel(TSCDataFrame.tsc_time_idx_name)
    values.index.name = TSCDataFrame.tsc_id_idx_name
    values.to_csv(path / f'selected_eigenfunctions_at_{initial_state_idx}.csv', index=True)
