"""
Time series visualization functions.

This module contains functions for visualizing time series data
as static plots (line plots for specific cells showing predictions vs true values).
For animated GIF generation, see animations.py.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from datafold import TSCDataFrame

from .layout import add_legend_outside, apply_layout_with_side_legend
from .io_utils import save_figure
from .style import PlotStyle

log = logging.getLogger(__name__)

# Plotting every measurement cell makes the figure unreadable; the first few
# are enough to see whether a prediction tracks the data.
MAX_CELLS_PER_FIGURE = 3


def _cells_to_plot(style: PlotStyle, observable: str) -> List[Tuple[str, str]]:
    """
    The (column, cell) pairs one per-cell figure shows, capped for readability.

    Column names are '{cell}_{observable}', e.g. '(0_0)_density' -- the format
    ``mode_decompositions._data_to_timeseries`` writes.
    """
    cells = style.measurement_cells[:MAX_CELLS_PER_FIGURE]
    return [(f'{cell}_{observable}', cell) for cell in cells]


def _finish_per_cell_figure(
    fig,
    axs: np.ndarray,
    style: PlotStyle,
    observable: str,
    suptitle: str,
    category: str,
    name: str,
) -> None:
    """Label the axes, add the side legends, lay the figure out and save it."""
    for ax in axs:
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{observable}')
        ax.grid(True, alpha=0.3)
        add_legend_outside(ax, position='right')

    fig.suptitle(suptitle, wrap=True)

    apply_layout_with_side_legend(fig, num_rows=len(axs), has_suptitle=True, right_margin=0.82)

    save_figure(fig, style, category, name)


def _per_cell_axes(n_cells: int) -> Tuple[plt.Figure, np.ndarray]:
    """One row of axes per measurement cell, always as an array."""
    fig, axs = plt.subplots(nrows=n_cells, ncols=1, figsize=(10, 4 * n_cells))
    return fig, axs if isinstance(axs, np.ndarray) else np.array([axs])


def plot_all_series_per_cell(
    tsc: TSCDataFrame,
    observable: str,
    style: PlotStyle,
) -> None:
    """
    Plot the input time series of every series id, one subplot per measurement cell.

    Parameters
    ----------
    tsc : TSCDataFrame
        Time series data.
    observable : str
        Observable to visualize (e.g. 'density', 'velocity_x').
    style : PlotStyle
        Supplies the measurement cells, time axis and output settings.
    """
    log.info(f'Plotting all series for {observable} at measurement cells')

    columns_and_cells = _cells_to_plot(style, observable)
    fig, axs = _per_cell_axes(len(columns_and_cells))

    colors = plt.cm.tab10(np.linspace(0, 1, len(tsc.ids)))

    for ax, (column, cell) in zip(axs, columns_and_cells):
        for idx, series_id in enumerate(tsc.ids):
            try:
                series_data = tsc.loc[series_id][column]
            except KeyError as e:
                log.error(f'Error accessing column {column}: {e}')
                log.error(f'Available columns: {list(tsc.columns)[:10]}...')
                log.error(f'Grid shape: {style.grid_shape}')
                raise

            ax.plot(
                series_data.index * style.temporal_resolution,
                series_data.values,
                label=f'Series {series_id}',
                color=colors[idx],
                alpha=0.8,
                linewidth=1.5,
            )

        ax.set_title(f'Cell {cell}')

    _finish_per_cell_figure(
        fig,
        axs,
        style,
        observable,
        suptitle=f'{observable.capitalize()} time series for all series at measurement cells',
        category='data',
        name=f'{observable}_all_series_per_cell',
    )

    log.info(f'Saved all series plot for {observable}')


def _plot_true_and_predicted_cell(
    ax,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    column: str,
    series_ids: List[int],
    color_map: Dict[int, np.ndarray],
    style: PlotStyle,
) -> None:
    """
    Draw one cell's true (faded) and predicted (opaque) lines, one colour per series.

    The grey band marks the timesteps the state embedding consumes before the
    first prediction exists.
    """
    for values, alpha, linewidth, labelled in (
        (tsc, 0.5, 1.0, False),
        (tsc_predicted, 1.0, 1.5, True),
    ):
        for series_id in series_ids:
            try:
                series_data = values.loc[series_id][column]
            except KeyError as e:
                log.error(f'Error accessing column {column}: {e}')
                log.error(f'Available columns: {list(values.columns)[:10]}...')
                log.error(f'Grid shape: {style.grid_shape}')
                raise

            ax.plot(
                series_data.index * style.temporal_resolution,
                series_data.values,
                color=color_map[series_id],
                alpha=alpha,
                linewidth=linewidth,
                linestyle='-',
                label=f'{series_id}' if labelled else None,
            )

    if style.embedding_timeshift > 0:
        duration_seconds = (style.embedding_timeshift + 1) * style.temporal_resolution
        ax.axvspan(
            0,
            style.embedding_timeshift * style.temporal_resolution,
            color='gray',
            alpha=0.3,
            label=f'state embedding depth ({duration_seconds}s)',
        )


def plot_predictions_per_cell(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    observable_type: str,
    style: PlotStyle,
    is_test: bool = False,
    series_id: Optional[int] = None,
) -> None:
    """
    Plot predicted against true values, one subplot per measurement cell.

    Parameters
    ----------
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    observable_type : str
        Observable to visualize.
    style : PlotStyle
        Supplies the measurement cells, time axis and output settings.
    is_test : bool
        Whether this is test data (affects title and file name only).
    series_id : int, optional
        Plot only this series. All series share one figure when omitted.
    """
    if series_id is not None:
        log.info(f'Plotting predictions for {observable_type} at measurement cells for series {series_id}')
    else:
        log.info(f'Plotting predictions for {observable_type} at measurement cells (all series)')

    columns_and_cells = _cells_to_plot(style, observable_type)
    series_ids = [series_id] if series_id is not None else list(tsc.ids)

    fig, axs = _per_cell_axes(len(columns_and_cells))

    colors = plt.cm.tab10(np.linspace(0, 1, len(series_ids)))
    color_map = {sid: colors[idx] for idx, sid in enumerate(series_ids)}

    for ax, (column, cell) in zip(axs, columns_and_cells):
        _plot_true_and_predicted_cell(ax, tsc, tsc_predicted, column, series_ids, color_map, style)
        ax.set_title(f'Cell {cell}')

    postfix = 'test' if is_test else 'train'
    if series_id is not None:
        suptitle = f'{observable_type.capitalize()} predictions vs true for series {series_id} ({postfix} data)'
        name = f'{observable_type}_per_cell_{series_id}_{postfix}'
    else:
        suptitle = f'{observable_type.capitalize()} predictions vs true for {postfix} data'
        name = f'{observable_type}_predictions_per_cell_{postfix}'

    _finish_per_cell_figure(
        fig,
        axs,
        style,
        observable_type,
        suptitle=suptitle,
        category='evaluation',
        name=name,
    )

    log.info(f'Saved prediction plot for {observable_type}')
