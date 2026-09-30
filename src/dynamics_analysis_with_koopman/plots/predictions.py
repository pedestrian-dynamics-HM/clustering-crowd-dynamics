"""
Prediction visualization functions.

This module contains the entry points the pipeline calls after a model has
been fitted: the ground-truth figures, the prediction figures, and the mode
decomposition per measurement cell. Each builds the run's
:class:`~.style.PlotStyle` once and passes it to the individual plotting
functions.
"""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datafold import TSCDataFrame
from matplotlib.figure import Figure

from .config import MEASUREMENT_COLUMN_NAMES, TINY_SIZE
from .grid import _plot_single_grid, plot_features_in_grid, visualize_grid_for_series_id, visualize_error_grid
from .timeseries import plot_all_series_per_cell, plot_predictions_per_cell
from .animations import generate_timeseries_gif
from .layout import add_row_colorbars, apply_standard_layout
from .io_utils import save_figure
from .style import PlotStyle

log = logging.getLogger(__name__)


def _plot_mae_per_series(
    errors_dict: dict,
    observable: str,
    style: PlotStyle,
) -> Figure:
    """One subplot per series: the spatially averaged error over time."""
    num_ts = len(errors_dict)
    fig, axs = plt.subplots(nrows=num_ts, ncols=1, figsize=[6, num_ts * 3.5])
    if num_ts == 1:
        axs = np.expand_dims(axs, axis=0)

    for ax, (series_id, mae_error) in zip(axs, errors_dict.items()):
        ax.set_title(f'MAE in {observable} for series {series_id}')
        ax.set_xlabel('Time in seconds')
        ax.set_ylabel('Mean absolute error')
        time_values = (np.arange(0, len(mae_error)) + style.prediction_start) * style.temporal_resolution
        ax.plot(time_values, mae_error, c='red')

    # Smaller padding for tighter spacing between rows
    apply_standard_layout(fig, has_suptitle=False, has_colorbar=False, num_rows=num_ts, h_pad=1.5)

    return fig


def _plot_mae_with_std(
    errors_df: pd.DataFrame,
    observable: str,
    style: PlotStyle,
) -> Figure:
    """The error averaged across series, with a one-standard-deviation band."""
    error_avg = errors_df.mean(axis=1)
    error_std = errors_df.std(axis=1)

    fig, ax = plt.subplots(figsize=[5, 2.5])
    fig.suptitle(f'MAE for {observable} across all series', wrap=True)

    ax.set_xlabel('Time in seconds')
    ax.set_ylabel('Mean absolute error')
    ax.grid(True, alpha=0.3)
    ax.plot(np.arange(0, len(error_avg)) * style.temporal_resolution, error_avg, c='red')
    ax.fill_between(
        error_std.index * style.temporal_resolution,
        error_avg - error_std,
        error_avg + error_std,
        color='red',
        alpha=0.1,
    )

    apply_standard_layout(fig, has_suptitle=True, has_colorbar=False, num_rows=1)

    return fig


def _plot_mae_spatial_grids(
    prediction_error: pd.DataFrame,
    series_ids: List[int],
    observable: str,
    style: PlotStyle,
) -> Figure:
    """
    Where the error sits in space: mean error on top, its variance below.

    Both rows share one colour scale per row, so the panels of a row are
    comparable across series.
    """
    grid_shape = style.grid_shape
    mae_grids, var_grids = {}, {}

    for series_id in series_ids:
        error = prediction_error.loc[series_id].to_numpy().real
        mae_grids[series_id] = np.abs(error).mean(axis=0).reshape(-1, grid_shape[0], grid_shape[1])
        var_grids[series_id] = error.var(axis=0).reshape(-1, grid_shape[0], grid_shape[1])

    num_ts = len(series_ids)
    fig, axs = plt.subplots(nrows=2, ncols=num_ts, figsize=(num_ts * 3, 8))
    if num_ts == 1:
        axs = np.expand_dims(axs, axis=1)

    # Each row shares one colorbar, so every panel in a row must share one scale.
    rows = [
        (mae_grids, f'MAE in {observable}'),
        (var_grids, f'Variance in error for {observable}'),
    ]

    ims = [None, None]
    for i, (grids, _) in enumerate(rows):
        row_vmax = max(g.max() for g in grids.values())
        for ax, series_id in zip(axs[i], series_ids):
            _, ims[i] = _plot_single_grid(
                grid_data=grids[series_id],
                style=style,
                title=f'Series {series_id}',
                vmax=row_vmax,
                vmin=0,
                cmap='OrRd',
                ax=ax,
            )

    fig.suptitle(f'MAE and variance in error for {observable}', wrap=True)

    apply_standard_layout(fig, has_suptitle=True, has_colorbar=True, num_rows=2)

    add_row_colorbars(fig=fig, axs=axs, mappables=ims, labels=[label for _, label in rows])

    return fig


def plot_mae(
    tsc: TSCDataFrame, tsc_predicted: TSCDataFrame, observable: str, style: PlotStyle, is_test: bool = False
) -> Tuple[float, Optional[float]]:
    """
    Visualize the mean absolute error (MAE) for given time series, as three figures.

    ``mae_per_ts`` holds one error curve per series, ``mae_with_std`` their
    average with a spread band (only with more than one series), and
    ``mae_grid`` shows where in space the error sits.

    NOTE: The MAE may not be as representative since each cell contributes
    equally to the overall error.

    Parameters
    ----------
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    observable : str
        Observable to analyze.
    style : PlotStyle
        Supplies the grid geometry, the time axis and output settings.
    is_test : bool
        Whether this is test data (affects titles and file names only).

    Returns
    -------
    tuple
        (mean_mae, mean_ssim) -- the overall MAE; SSIM is not computed yet.
    """
    log.info(f'Visualizing MAE for {observable=}')
    postfix = 'test' if is_test else 'train'

    vis_columns = [column for column in tsc.columns if observable in column]
    prediction_error = (tsc_predicted[vis_columns] - tsc[vis_columns]).dropna()

    # MAE per series, averaged over space
    errors_dict = {}
    for series_id in tsc.ids:
        error = prediction_error.loc[series_id].to_numpy().real
        errors_dict[series_id] = np.abs(error).mean(axis=1)
        log.info(f'MAE in {observable} for series {series_id}: {errors_dict[series_id].mean()}')

    fig = _plot_mae_per_series(errors_dict, observable, style)
    save_figure(fig, style, 'evaluation', f'{observable}_mae_per_ts_{postfix}')

    # Average and spread across series -- only meaningful with more than one
    if len(errors_dict) > 1:
        try:
            errors_df = pd.DataFrame(dict(enumerate(errors_dict.values())))
        except ValueError:
            log.warning('An error occurred when trying to calculate MAE average and std.')
            log.warning('This was probably caused by different length time series in the tsc.')
        else:
            fig = _plot_mae_with_std(errors_df, observable, style)
            save_figure(fig, style, 'evaluation', f'{observable}_mae_with_std_{postfix}')
            log.info(f'Saved MAE plots for {observable=} to {style.path=}')

    mae_per_series = np.array([np.mean(v) for v in errors_dict.values()])
    log.info(f'Total MAE for all series: {mae_per_series.mean()}')

    fig = _plot_mae_spatial_grids(prediction_error, list(tsc.ids), observable, style)
    save_figure(fig, style, 'evaluation', f'{observable}_mae_grid_{postfix}')

    return float(mae_per_series.mean()), None


def visualize_ground_truth(
    ts_data_dict: dict,
    tsc: TSCDataFrame,
    run_config,
    run_metrics,
    observable_types: set,
    is_checkpoint: bool = False,
) -> None:
    """
    Visualize the input data, before any prediction is made.

    Creates the measurement-cell overview, the per-series grids, the per-cell
    time series and -- if the config asks for them -- the ground-truth gifs.

    Parameters
    ----------
    ts_data_dict : dict
        Dictionary of time series data by ID.
    tsc : TSCDataFrame
        Time series collection.
    run_config : FullConfig
        Run configuration object.
    run_metrics : RunMetrics
        Run metrics object.
    observable_types : set
        Set of observable types to visualize.
    is_checkpoint : bool
        A checkpoint's own run already produced the per-series figures, so
        only the measurement-cell overview is redrawn.
    """
    # Internal invariants: the pipeline fills these in before the plotting stage runs.
    assert run_metrics.grid_shape is not None, 'Grid shape is not available for visualization.'
    assert run_metrics.visualization_timesteps is not None, 'Visualization timesteps need to be selected'
    assert run_metrics.measurement_cells is not None, 'Measurement cells need to be selected'
    assert run_metrics.embedding_timeshift is not None, 'Embedding timeshift needs to be selected'
    assert isinstance(run_metrics.grid_resolution, list), 'Grid resolution needs to be a list of floats.'

    style = PlotStyle.from_run(run_config, run_metrics)

    for observable_type in observable_types:
        plot_features_in_grid(tsc=tsc, observable_type=observable_type, style=style)

        if not is_checkpoint:
            for series_id in tsc.ids:
                visualize_grid_for_series_id(
                    tsc=tsc,
                    series_id=series_id,
                    observable_type=observable_type,
                    style=style,
                )

            log.info(f'Create plot for training timeseries per measurement cell for {observable_type=}.')

    if not is_checkpoint:
        for observable in run_metrics.system_observables:
            plot_all_series_per_cell(tsc=tsc, observable=observable, style=style)

    # Generate GIFs at the end
    if run_config.visualization.generate_timeseries_gifs is not None:
        for observable_type in observable_types:
            for series_id in run_config.visualization.generate_timeseries_gifs:
                try:
                    generate_timeseries_gif(
                        tsc=tsc, observable_type=observable_type, series_id=series_id, style=style, is_ground_truth=True
                    )
                except KeyError:
                    log.warning(f'KeyError occurred for gif generation with series {series_id=}. No gif was generated.')
                    log.warning(f'Available series ids: {ts_data_dict.keys()}')


def _plot_measurement_area_predictions(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    col_str: str,
    measurement_area_ids: List[int],
    style: PlotStyle,
) -> Figure:
    """
    Compare predicted and true counts for the enriched measurement areas.

    These are the extra counting areas ``data.enrich_timestep_data`` adds; they
    are not grid cells, so they get their own line plot per area.
    """
    fig, axes = plt.subplots(nrows=len(measurement_area_ids), ncols=1, figsize=(10, 4 * len(measurement_area_ids)))
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    for area_id, ax in zip(measurement_area_ids, axes):
        for series_id in tsc_predicted.ids:
            series = tsc_predicted.loc[series_id][f'{area_id}_{col_str}']
            ax.plot(series.index * style.temporal_resolution, series.values, label=f'Series {series_id} (predicted)')

        # Reset the color cycle so a series keeps its colour for the true values
        ax.set_prop_cycle(None)

        for series_id in tsc.ids:
            series = tsc.loc[series_id][f'{area_id}_{col_str}']
            ax.plot(
                series.index * style.temporal_resolution,
                series.values,
                label=f'Series {series_id} (true values)',
                linestyle='--',
            )

        if style.embedding_timeshift > 0:
            shift_seconds = style.embedding_timeshift * style.temporal_resolution
            ax.axvspan(0, shift_seconds, color='gray', alpha=0.3)
            ax.text(
                shift_seconds / 2 + 3,
                0.8,
                f'Duration per state: {style.embedding_timeshift + 1} frames or '
                f'{(style.embedding_timeshift + 1) * style.temporal_resolution}s',
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=TINY_SIZE,
                color='black',
            )

        ax.set_xlabel('Time in seconds')
        ax.set_ylabel('Number of pedestrians (normalized)')
        ax.legend(loc='upper right')
        ax.set_title(f'Counting area {area_id}')
        ax.set_ylim(ymin=-0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle(f'Predictions and true values for {col_str} measurement areas', wrap=True)

    return fig


def _measurement_areas(tsc: TSCDataFrame) -> dict:
    """The enriched measurement-area columns present in the data, by column suffix."""
    measurements = {}
    for col_str in MEASUREMENT_COLUMN_NAMES.values():
        ids = [int(col.split('_')[0]) for col in tsc.columns if col_str in col]
        if len(ids) > 0:
            measurements[col_str] = ids
    return measurements


def visualize_model_predictions(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    run_config,
    run_metrics,
    observable_types: set,
    is_test: bool = False,
) -> List[float]:
    """
    Visualize model predictions for given time series and observables.

    Creates the error grid per series, the per-cell prediction plots, the MAE
    figures, and -- when the run enriched the data with counting areas -- their
    line plots.

    Parameters
    ----------
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    run_config : FullConfig
        Run configuration object.
    run_metrics : RunMetrics
        Run metrics object.
    observable_types : set
        Set of observable types to visualize.
    is_test : bool
        Whether this is test data.

    Returns
    -------
    list of float
        MAE values for each observable.
    """
    # Internal invariants: the pipeline fills these in before the plotting stage runs.
    assert run_metrics.grid_shape is not None, 'Grid shape is not available for visualization.'
    assert run_metrics.visualization_timesteps is not None, 'Visualization timesteps need to be selected'
    assert run_metrics.measurement_cells is not None, 'Measurement cells need to be selected'
    assert run_metrics.embedding_timeshift is not None, 'Embedding timeshift needs to be selected'
    assert isinstance(run_metrics.grid_resolution, list), 'Grid resolution needs to be a list of floats.'

    style = PlotStyle.from_run(run_config, run_metrics)
    data_set_prefix = 'testing' if is_test else 'training'
    postfix = 'test' if is_test else 'train'

    for observable_type in observable_types:
        log.info(f'Create error grid for {data_set_prefix} data with {observable_type=}.')
        for series_id in tsc.ids:
            log.debug(f'For time series id {series_id}:')
            visualize_error_grid(
                tsc=tsc,
                tsc_predicted=tsc_predicted,
                series_id=series_id,
                observable_type=observable_type,
                style=style,
                is_test=is_test,
            )

        if run_config.data.enrich_timestep_data is not None:
            for col_str, measurement_area_ids in _measurement_areas(tsc).items():
                fig = _plot_measurement_area_predictions(tsc, tsc_predicted, col_str, measurement_area_ids, style)
                save_figure(fig, style, 'evaluation', f'measurement_areas_{col_str}_{postfix}')

    mae_values = [0.0 for _ in run_metrics.system_observables]

    for i, observable in enumerate(run_metrics.system_observables):
        log.info(f'Create plot with predictions per cell for {data_set_prefix} data.')

        for series_id in tsc.ids:
            log.debug(f'For series id {series_id}:')
            if isinstance(tsc.loc[series_id], pd.DataFrame) and isinstance(tsc_predicted.loc[series_id], pd.DataFrame):
                plot_predictions_per_cell(
                    tsc=tsc,
                    tsc_predicted=tsc_predicted,
                    observable_type=observable,
                    style=style,
                    is_test=is_test,
                    series_id=series_id,
                )
            else:
                log.warning(f'Could not create plot with predictions per cell for series id {series_id}.')
                log.warning('Dataframes are not available for both ground truth and predictions.')

        log.info(f'Create error plot for {data_set_prefix} data.')
        mae_values[i], _ = plot_mae(
            tsc,
            tsc_predicted,
            observable=observable,
            style=style,
            is_test=is_test,
        )

    return mae_values


def visualize_mode_decomposition(
    run_config,
    run_metrics,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    model,
) -> Tuple[Optional[TSCDataFrame], Optional[int]]:
    """
    Visualize the mode decomposition and the iterative sum per measurement cell.

    Both figures use scalar values per cell, so they iterate over
    ``system_observables`` rather than over observable *types*.

    Parameters
    ----------
    run_config : FullConfig
        Run configuration object.
    run_metrics : RunMetrics
        Run metrics object.
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    model : EDMD
        Trained Koopman model.

    Returns
    -------
    eigenfunctions_per_series, initial_state_idx
        The evaluated eigenfunctions behind the figures, for the caller to persist.
    """
    # Internal invariant: the caller resolves measurement_cells from the config first.
    assert run_metrics.measurement_cells is not None, 'Measurement cells need to be selected for visualization.'

    # Import here to avoid circular dependencies
    from .koopman import plot_mode_decomposition_per_cell, plot_iterative_sum_per_cell

    style = PlotStyle.from_run(run_config, run_metrics)
    eigenfunctions, initial_state_idx = None, None

    for observable in run_metrics.system_observables:
        log.info(f'Create plot with mode decomposition per cell for {observable=}.')

        eigenfunctions, initial_state_idx = plot_mode_decomposition_per_cell(
            model,
            tsc=tsc,
            tsc_predicted=tsc_predicted,
            observable=observable,
            style=style,
        )

        plot_iterative_sum_per_cell(
            model,
            tsc=tsc,
            tsc_predicted=tsc_predicted,
            observable=observable,
            style=style,
        )

    return eigenfunctions, initial_state_idx
