"""Resolve and load checkpoint artifacts by run registry entry or glob.

Shared by the paper notebooks that load a fitted model and its training data
directly from a `runs/` checkpoint directory, keyed by a JSON run registry
(e.g. `papers/clustering/juelich_run_registry.json`) or by a literal run id.
"""

import glob
from pathlib import Path
from typing import Optional, Tuple

import dill
import numpy as np
import pandas as pd
from datafold import EDMD, TSCDataFrame

from ..config.configs import PROJECT_ROOT, FullConfig
from ..data.checkpoint import load_checkpoint_series
from ..pipeline.mode_decompositions import create_train_test_tsc
from ..pipeline.results import RunMetrics
from ..plots.style import PlotStyle


def checkpoint_path_for(
    experiment: Optional[str] = None,
    run_id: Optional[str] = None,
    registry: Optional[dict] = None,
    run_selector: str = 'learning_run_dir',
    root_dir: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a checkpoint directory by experiment name (via the registry) or run id (via glob).

    Registry entries and glob matches are relative to the project root; the
    returned path is absolute, so it does not depend on the working directory.
    """
    if experiment is not None:
        if registry is None:
            raise ValueError('registry must be given to resolve a checkpoint by experiment name.')
        return PROJECT_ROOT / registry['runs'][experiment][run_selector]
    if run_id is not None:
        matches = glob.glob(f'runs/**/*{run_id}', root_dir=root_dir, recursive=True)
        return Path(root_dir) / matches[0]  # assumes unique run id
    raise ValueError('Either experiment or run_id must be given.')


def load_model_from_checkpoint(checkpoint_path: Path, model_name: str = 'edmd_model') -> EDMD:
    """Unpickle a trained model saved at a checkpoint."""
    with open(checkpoint_path / 'models' / f'{model_name}.pkl', 'rb') as f:
        return dill.load(f)


def load_train_tsc_from_checkpoint(
    checkpoint_path: Path,
    with_predictions: bool = False,
):
    """Load the pickled training ground truth, and optionally its stored predictions."""
    train_tsc = pd.read_pickle(checkpoint_path / 'data' / 'train_tsc.pkl.gz')
    if not with_predictions:
        return train_tsc
    train_tsc_predicted = pd.read_pickle(checkpoint_path / 'data' / 'train_tsc_predicted.pkl.gz')
    return train_tsc, train_tsc_predicted


def load_original_ts_from_checkpoint(
    checkpoint_path: Path,
    series_ids: list[int],
    system_observables: list[str],
) -> TSCDataFrame:
    """Load the original, pre-discretization per-series time series and assemble them into a TSCDataFrame.

    A preprocessing checkpoint has no fitted model and no `train_tsc.pkl.gz`
    (that is a learning-run artifact) -- only the raw, long-format per-cell
    CSVs under `original_ts/`, one per series id, in the same shape
    `create_train_test_tsc` already builds training data from.
    """
    data = load_checkpoint_series(checkpoint_path / 'original_ts', series_ids)
    train_tsc, _ = create_train_test_tsc(
        data=data,
        system_observables=system_observables,
        train_parameter=list(series_ids),
        test_parameter=[],
    )
    return train_tsc


def load_run_from_checkpoint(
    experiment: Optional[str] = None,
    run_id: Optional[str] = None,
    registry: Optional[dict] = None,
    run_selector: str = 'learning_run_dir',
    model_name: str = 'edmd_model',
):
    """Load a fitted model together with its run config, metrics and derived PlotStyle.

    Returns (edmd, run_config, run_metrics, style); ``style.path`` is the
    checkpoint directory itself (rather than the possibly different-machine
    root_path baked into config.yaml), since the checkpoint is the source of
    truth for where its own artifacts live.
    """
    checkpoint_path = checkpoint_path_for(
        experiment=experiment, run_id=run_id, registry=registry, run_selector=run_selector
    )
    edmd = load_model_from_checkpoint(checkpoint_path, model_name=model_name)
    run_config = FullConfig.from_yaml(checkpoint_path / 'config.yaml')
    run_metrics = RunMetrics(**FullConfig._load_yaml(checkpoint_path / 'metrics.yaml'))
    style = PlotStyle.from_run(run_config, run_metrics).with_(path=checkpoint_path)

    return edmd, run_config, run_metrics, style


def filter_eigenvalues_by_residual(
    eigenvalues: np.ndarray,
    residuals: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the n eigenvalues with the lowest spectral residual.

    Returns the kept eigenvalues together with their indices into the given
    (unfiltered) arrays, so callers can still label eigenvalues by their
    original position (e.g. lambda_i in plots).

    Complex eigenvalues come as conjugate pairs sharing the same residual, so
    a threshold exactly on such a pair could let n+1 through; one of that
    pair is then dropped to keep exactly n.
    """
    ev_threshold = np.partition(residuals, n - 1)[n - 1]
    mask_keep = residuals <= ev_threshold

    if np.sum(mask_keep) > n:
        last_pair = np.where(residuals == ev_threshold)[0]
        mask_keep[last_pair[-1]] = False

    return eigenvalues[mask_keep], np.where(mask_keep)[0]
