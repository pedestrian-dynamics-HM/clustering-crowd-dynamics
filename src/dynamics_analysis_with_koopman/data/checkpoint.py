"""Checkpoint data loading and processing utilities.

Handles loading timeseries and enrichment data from checkpoint directories,
including format compatibility and time window adjustments.
"""

import logging
from pathlib import Path

import dill
import pandas as pd

from ..config.configs import FullConfig

log = logging.getLogger(__name__)


def load_checkpoint_series(
    checkpoint_data_dir: Path,
    selected_series_ids: list[int],
) -> dict[int, pd.DataFrame]:
    """Load per-series CSVs first, then fallback to pickle for older checkpoints."""
    timeseries_data_dict: dict[int, pd.DataFrame] = {}

    for param in selected_series_ids:
        try:
            timeseries_data_dict[param] = pd.read_csv(checkpoint_data_dir / f'timeseries_data_{param}.csv')
        except FileNotFoundError:
            log.warning(
                'Could not load timeseries data for %s from %s directory.',
                param,
                checkpoint_data_dir,
            )

    if len(timeseries_data_dict) > 0:
        return timeseries_data_dict

    log.info('Try loading timeseries data from pickle file...')
    with open(checkpoint_data_dir / 'timeseries_data.pkl', 'rb') as file:
        timeseries_data_dict = dill.load(file)
        if not isinstance(timeseries_data_dict, dict):
            raise ValueError('Expected checkpoint pickle data to be a dictionary.')
        return timeseries_data_dict


def filter_observable_columns(
    timeseries_data_dict: dict[int, pd.DataFrame],
    system_observables: list[str],
) -> dict[int, pd.DataFrame]:
    """Keep only index columns plus selected observable columns."""
    filtered: dict[int, pd.DataFrame] = {}
    for key, df in timeseries_data_dict.items():
        cols_to_keep = ['timeStep', 'x_idx', 'y_idx']
        cols_to_keep += [col for col in df.columns if any(obs in col for obs in system_observables)]
        filtered[key] = df[cols_to_keep]
    return filtered


def apply_checkpoint_time_window(
    timeseries_data_dict: dict[int, pd.DataFrame],
    requested_tlim: list[int | None],
    checkpoint_tlim: list[int | None],
) -> dict[int, pd.DataFrame]:
    """Apply requested absolute tlim relative to checkpoint start."""
    start, end = requested_tlim
    checkpoint_start = checkpoint_tlim[0] or 0

    if start is not None:
        relative_start = start - checkpoint_start
        if relative_start < 0:
            raise ValueError(
                f"The requested tlim starts at timestep {start}, before the checkpoint's own start "
                f'({checkpoint_tlim[0]}). The checkpoint holds no data for that window -- raise tlim[0] '
                f'to at least {checkpoint_start}.'
            )
        for key in timeseries_data_dict:
            timeseries_data_dict[key] = timeseries_data_dict[key][
                timeseries_data_dict[key]['timeStep'] >= relative_start
            ]

    if end is not None:
        relative_end = end - checkpoint_start
        for key in timeseries_data_dict:
            timeseries_data_dict[key] = timeseries_data_dict[key][timeseries_data_dict[key]['timeStep'] <= relative_end]

    return timeseries_data_dict


def normalize_time_steps_to_zero(
    timeseries_data_dict: dict[int, pd.DataFrame],
) -> tuple[dict[int, pd.DataFrame], dict[int, int]]:
    """Shift each series to start at timestep zero and return per-series offsets."""
    timeshifts: dict[int, int] = {}
    for key, df in timeseries_data_dict.items():
        shift = int(df['timeStep'].min())
        timeshifts[key] = shift
        df = df.copy()
        df['timeStep'] = df['timeStep'] - shift
        timeseries_data_dict[key] = df
    return timeseries_data_dict, timeshifts


def load_checkpoint_enrichment_data(
    checkpoint_data_dir: Path,
    selected_series_ids: list[int],
    timeshifts: dict[int, int],
) -> dict[int, pd.DataFrame]:
    """Load optional enrichment data and align timesteps with shifted timeseries."""
    enrichment_counters: dict[int, pd.DataFrame] = {}
    for param in selected_series_ids:
        try:
            enrichment_df = pd.read_csv(checkpoint_data_dir / f'enrichment_ts_data_{param}.csv')
            enrichment_df['timeStep'] = enrichment_df['timeStep'] - timeshifts[param]
            enrichment_counters[param] = enrichment_df
        except FileNotFoundError:
            log.warning('Could not load enrichment data for %s', param)
    return enrichment_counters


def read_timeseries_from_checkpoint(
    config: FullConfig,
    checkpoint_path: Path,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """Read timeseries and enrichment data from a checkpoint directory.

    config.preprocessing.tlim must already be fully resolved by the time this is
    called (the current run's value, or inherited element-wise from the checkpoint
    where unset) -- see checkpoint_policy.merge_checkpoint_config, which
    resolve_checkpoint_config always runs before this. Only the checkpoint's own,
    unresolved tlim is still read here, needed to compute the relative time window
    its stored data was already trimmed to.
    """

    checkpoint_metrics = FullConfig._load_yaml(checkpoint_path / 'metrics.yaml')
    checkpoint_config = FullConfig._load_yaml(checkpoint_path / 'config.yaml')
    checkpoint_data_dir = checkpoint_path / 'data'

    available_timeseries = checkpoint_metrics['train_parameter'] + checkpoint_metrics['test_parameter']

    if config.data.train_parameter is None:
        raise ValueError('train_parameter must be set in the config to select series from a checkpoint.')

    missing_train = [p for p in config.data.train_parameter if p not in available_timeseries]
    if missing_train:
        raise ValueError(
            f'Train parameter {missing_train} is not stored in the checkpoint. '
            f'Available series ids: {available_timeseries}.'
        )

    missing_test = [p for p in config.data.test_parameter if p not in available_timeseries]
    if missing_test:
        raise ValueError(
            f'Test parameter {missing_test} is not stored in the checkpoint. '
            f'Available series ids: {available_timeseries}.'
        )

    missing_observables = [
        obs for obs in config.preprocessing.system_observables if obs not in checkpoint_metrics['system_observables']
    ]
    if missing_observables:
        raise ValueError(
            f'System observables {missing_observables} are not stored in the checkpoint. '
            f'Available observables: {checkpoint_metrics["system_observables"]}.'
        )

    selected_series_ids = config.data.train_parameter + config.data.test_parameter
    log.info('Loading timeseries data for series ids %s from checkpoint...', selected_series_ids)

    timeseries_data_dict = load_checkpoint_series(checkpoint_data_dir, selected_series_ids)
    timeseries_data_dict = filter_observable_columns(
        timeseries_data_dict,
        config.preprocessing.system_observables,
    )

    # The checkpoint's own (unresolved) tlim, needed only to compute the relative
    # time window its stored data was already trimmed to -- config.preprocessing.tlim
    # itself is already fully resolved (see the docstring above).
    checkpoint_tlim = FullConfig.from_dict(checkpoint_config).preprocessing.tlim

    timeseries_data_dict = apply_checkpoint_time_window(
        timeseries_data_dict,
        config.preprocessing.tlim,
        checkpoint_tlim,
    )
    # we want our time series to start at 0 to have them aligned
    timeseries_data_dict, timeshifts = normalize_time_steps_to_zero(timeseries_data_dict)

    enrichment_counters: dict[int, pd.DataFrame] = {}
    if config.data.enrich_timestep_data is not None:
        log.info('Loading enrichment data from checkpoint directory...')
        enrichment_counters = load_checkpoint_enrichment_data(
            checkpoint_data_dir,
            selected_series_ids,
            timeshifts,
        )

    return timeseries_data_dict, enrichment_counters
