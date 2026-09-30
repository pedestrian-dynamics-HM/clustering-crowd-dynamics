"""
Koopman analysis visualization functions.

This module contains functions for visualizing Koopman operator analysis results,
including eigenvalues, eigenfunctions, modes, supermodes, and mode decomposition.

Each ``plot_*`` function draws figures only. The numbers behind them -- the
eigenvalues, the selected modes, the evaluated eigenfunctions -- are returned
rather than written to disk, so the pipeline decides what is persisted (see
``utils.persistence``) and a notebook can call these without leaving files
behind.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datafold import EDMD, TSCDataFrame
from matplotlib.figure import Figure
from scipy import stats

from .config import (
    MODES_CMAP,
    EV_TOLERANCE,
    MEDIUM_SIZE,
    DEFAULT_RIGHT_MARGIN,
    DEFAULT_RIGHT_MARGIN_LEGEND_COLORBAR,
    DEFAULT_GRID_ALPHA,
    DEFAULT_THIN_LINEWIDTH,
    DEFAULT_SCATTER_ALPHA,
    DEFAULT_SHADED_REGION_ALPHA,
    DEFAULT_LINE_ALPHA,
    SMALL_SIZE,
    DEFAULT_CBAR_WIDTH,
)
from .layout import (
    add_colorbar_next_to_grid,
    add_legend_outside,
    add_row_colorbars,
    add_row_subtitles,
    apply_layout_with_bottom_legend,
    apply_layout_with_side_legend,
    apply_standard_layout,
)
from .io_utils import save_figure
from .style import PlotStyle
from .grid import _plot_single_grid, _create_ts_plot_from_df, _parse_measurement_cell

log = logging.getLogger(__name__)


@dataclass
class KoopmanSpectrum:
    """The numbers the Koopman figures are drawn from, for the caller to persist.

    Fields are ``None`` (``modes``: empty) when the corresponding figure was not
    produced -- e.g. ``residuals`` stays None if no pseudospectrum was computed.
    """

    eigenvalues: np.ndarray
    residuals: Optional[np.ndarray] = None
    modes: Dict[str, np.ndarray] = field(default_factory=dict)
    selection: Optional[List[int]] = None
    eigenfunctions: Optional[TSCDataFrame] = None
    initial_state_idx: Optional[int] = None


#########################
### Helper Functions ####
#########################


def _get_ev_stability(evs: np.ndarray, tolerance: float = EV_TOLERANCE) -> List[int]:
    """
    Get the stability of the eigenvalues according to avila-2020-math.
    The stability depends on the magnitude of the eigenvalues:
    - (0) neutral: on the unit circle, i.e. abs(ev) = 1 ± tolerance
    - (1) stable: < 1 - tolerance
    - (2) unstable: > 1 + tolerance
    """
    evs_stability = []
    for ev in evs:
        if np.isclose(np.abs(ev), 1, rtol=tolerance):
            evs_stability.append(0)
        elif np.abs(ev) < 1:
            evs_stability.append(1)
        else:  # |ev| > 1
            evs_stability.append(2)
    log.info(
        f'Found {sum([ev == 0 for ev in evs_stability])} neutral, '
        + f'{sum([ev == 1 for ev in evs_stability])} stable '
        f'and {sum([ev == 2 for ev in evs_stability])} unstable eigenvalues.'
    )
    log.debug(evs)
    log.debug(evs_stability)
    return evs_stability


def _get_line_values_from_grid(grid, measurement_line):
    """Extract line values from a 2D grid based on measurement line specification."""
    if len(measurement_line) != 2:
        raise ValueError(
            f'A measurement line can only provide an x or a y coordinate, so it needs to be of length 2, '
            f'got {measurement_line}.'
        )
    # Internal invariant: the caller reshapes a Koopman mode into a 2d grid before calling this.
    assert len(grid.shape) == 2, (
        'Need a 2d grid with shape of length 3, first dimension either 1 or 2 for vector field.'
    )

    if measurement_line[0] is not None and measurement_line[1] is None:
        # this means we want a full column at a specific x position
        line_values = grid[:, measurement_line[0]]

    elif measurement_line[1] is not None and measurement_line[0] is None:
        # this means we want a full row at a specific y position
        line_values = grid[measurement_line[1], :]

    else:
        log.error(
            f'measurement line {measurement_line} does not have the right format. Can only have one value which is not None.'
        )
        raise ValueError

    return line_values


def _are_conjugates(z1, z2, tol=1e-7):
    """
    Check if two complex numbers z1 and z2 are conjugates of each other,
    considering numerical inaccuracies.
    """
    return math.isclose(z1.real, z2.real, abs_tol=tol) and math.isclose(z1.imag, -z2.imag, abs_tol=tol)


def _find_conjugate_pairs(complex_list, tol=1e-7):
    """
    Find all pairs of indices in the list where numbers are conjugates of each other.

    Parameters:
    - complex_list: List of complex numbers.
    - tol: Tolerance for floating-point comparison.

    Returns:
    - List of tuples (i, j) where complex_list[i] is a conjugate of complex_list[j].
    """
    pairs = []
    n = len(complex_list)

    for i in range(n):
        for j in range(i + 1, n):  # Compare each pair only once
            if _are_conjugates(complex_list[i], complex_list[j], tol):
                pairs.append((i, j))

    # get all the non-pairs
    non_pairs = [i for i in range(n) if i not in [pair[0] for pair in pairs] + [pair[1] for pair in pairs]]

    if len(non_pairs) > 1:  # we have a non-pair at the beginning
        if non_pairs[-1] == n - 1:  # we have a non-pair at the end
            non_pairs = non_pairs[:-1]  # remove it as an artifact

    return pairs, non_pairs


def _reshape_to_grid(mode: np.ndarray, grid_shape: List[int]) -> np.ndarray:
    """Reshape one flat mode into a (-1, rows, cols) grid, with a readable error if it does not fit."""
    try:
        return mode.reshape(-1, grid_shape[0], grid_shape[1])
    except (ValueError, AttributeError) as e:
        raise ValueError(f'Cannot reshape array of size {mode.shape} into {(grid_shape[0], grid_shape[1])}.') from e


def _plot_mode_row(
    axs,
    grid_real: np.ndarray,
    grid_imag: Optional[np.ndarray],
    mode_name: str,
    style: PlotStyle,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    include_negative: bool = False,
):
    """
    Plot one mode as a row of grids: real part, imaginary part, and optionally both negated.

    The colour scale is symmetric around zero and, unless ``vmin``/``vmax`` say
    otherwise, specific to this mode -- the scale differs a lot between modes
    and the pattern matters more than the absolute value.
    """
    # NOTE: do not overwrite vmax and vmin, they are checked for None per mode
    if vmax is not None:
        grid_max = vmax
    elif grid_imag is not None:
        grid_max = max(grid_real.max(), grid_imag.max())
    else:
        grid_max = grid_real.max()

    if vmin is not None:
        grid_min = vmin
    elif grid_imag is not None:
        grid_min = min(grid_real.min(), grid_imag.min())
    else:
        grid_min = grid_real.min()

    max_abs = max(abs(grid_max), abs(grid_min))  # symmetric scale for pairs of modes

    # Column index per panel, so a missing imaginary part leaves its column
    # empty rather than shifting the negated pair into it.
    panels = [(0, grid_real, f'{mode_name} (real)')]
    if grid_imag is not None:
        panels.append((1, grid_imag, f'{mode_name} (imag)'))

    if include_negative and axs.shape[0] >= 4:
        panels.append((2, (-1) * grid_real, f'- {mode_name} (real)'))
        panels.append((3, (-1) * grid_imag, f'- {mode_name} (imag)'))

    ax1, cax1 = None, None
    for i, grid, title in panels:
        ax, cax = _plot_single_grid(
            grid,
            style,
            title=title,
            vmin=-max_abs,
            vmax=max_abs,
            ax=axs[i],
            cmap=MODES_CMAP,
        )
        if i == 0:
            ax1, cax1 = ax, cax

    return ax1, cax1


def _get_koopman_triplet(
    tsc: Union[TSCDataFrame, None],
    model: EDMD,
    observable_type: str,
    selection: Union[List[int], None] = None,
    prediction_timeshift: Union[int, None] = None,
) -> Tuple[np.ndarray, np.ndarray, TSCDataFrame, int, List[int]]:
    """
    Get the Koopman triplet for the given system observable and series id.
    The triplet consists of the Koopman modes, eigenvalues, and eigenfunction for the initial condition.
    The initial condition is the first state of the time series respecting the temporal embedding length.
    """
    # get the modes
    modes = model.koopman_modes  # row per feature column, column per mode
    if modes is not None:
        modes = modes[modes.index.to_series().str.contains(observable_type)].to_numpy()  # shape (#cells, #modes)
    else:
        log.error(f'No Koopman modes available for {observable_type=}. Aborting...')
        raise ValueError(f'No Koopman modes available for {observable_type=}. Aborting...')

    # get the eigenvalues
    eigenvalues = model.koopman_eigenvalues.to_numpy()  # shape (75,)

    # select only the relevant triples
    if selection is None:
        selection = list(range(1, min(21, len(eigenvalues) + 1)))
    modes = modes[:, [v - 1 for v in selection]].T  # shape (len(selection), #grid cells)
    eigenvalues = eigenvalues[[v - 1 for v in selection]]  # (#selection,)

    # get the eigenfunctions for initial condition
    if tsc is not None:
        frames_for_initial_condition = model.n_samples_ic_ if model.n_samples_ic_ is not None else 1
        initial_state = 0 if prediction_timeshift is None else prediction_timeshift
        initial_state_idx = (
            initial_state + frames_for_initial_condition - 1
        )  # is 0 if we need exactly one state (no embedding)

        initial_conditions_per_series = tsc.select_time_values(
            np.arange(initial_state, initial_state_idx + 1)
        )  # expects TSCDataFrame or array
        # Internal invariants: datafold returns a TSCDataFrame for both of these calls.
        assert isinstance(initial_conditions_per_series, TSCDataFrame), (
            'Could not select the initial conditions from the TSCDataFrame.'
        )

        eval_eigenfunctions = model.koopman_eigenfunction(
            initial_conditions_per_series
        )  # expects TSCDataFrame or array
        assert isinstance(eval_eigenfunctions, TSCDataFrame), (
            'Could not evaluate the eigenfunctions for the initial conditions.'
        )
        eigenfunctions_per_series = eval_eigenfunctions.iloc[
            :, [v - 1 for v in selection]
        ]  # returns TSCDataFrame because series id remains
        eigenfunctions_per_series.columns = pd.Index(
            [f'koop_eigfunc{i}' for i in selection]
        )  # rename columns for selection
    else:
        initial_state_idx = None
        eigenfunctions_per_series = None

    return modes, eigenvalues, eigenfunctions_per_series, initial_state_idx, selection  # type: ignore


def _get_mode_ts(n_timesteps, grid_shape, initial_state_idx, measurement_cell, eigenvalue, mode, eigenfunction_value):
    """Calculate the time series contribution of a single mode for a specific cell."""
    x_idx, y_idx = _parse_measurement_cell(measurement_cell)
    mode_in_cell = mode.reshape(grid_shape[0], grid_shape[1])[y_idx, x_idx]  # scalar value
    # create timeseries with current eigenvalue and mode cell value
    mode_ts = eigenfunction_value * [
        mode_in_cell * power for power in (eigenvalue) ** (np.arange(0, n_timesteps - initial_state_idx))
    ]
    return mode_ts


#########################
### Eigenvalues #########
#########################


def _spectral_residuals(model: EDMD, evs: np.ndarray) -> Optional[np.ndarray]:
    """
    Residual per eigenvalue, the pseudospectrum measure of spectral pollution.

    Stored on the model when it filtered by residual during the fit, otherwise
    recomputed here. Returns None when the fitted DMD cannot produce them, in
    which case the eigenvalue plot falls back to its default colouring.
    """
    if model.dmd_model.compute_pseudospectrum is None:
        log.warning('Pseudospectrum is not available for this model.')
        return None

    try:
        return model.dmd_model.residuals_
    except AttributeError:
        log.debug(
            'Residuals were not stored for the model. This is only turned on when removing '
            'spectral pollution. Calculating them...'
        )

    try:
        return model.dmd_model._compute_spectral_residuals(
            eigenvalues=evs,
            eigenvectors_right=model.dmd_model.eigenvectors_right_,
        )
    except (TypeError, AttributeError):
        log.debug('Residuals could not be computed. Fall back to the standard color for all eigenvalues.')
        return None


def _plot_eigenvalue_spectrum(
    evs: np.ndarray,
    style: PlotStyle,
    residuals: Optional[np.ndarray] = None,
    xlim: Tuple[float, float] = (-1.4, 1.4),
    ylim: Tuple[float, float] = (-1.4, 1.4),
) -> Figure:
    """
    Scatter the eigenvalues in the complex plane, against the unit circle.

    Three colourings, in order of precedence: the selected triples highlighted
    in orange, the spectral residual on a colour scale, or the stability class
    (neutral / stable / unstable) of each eigenvalue.
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # unit circle for reference
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(-np.cos(t), np.sin(t), linewidth=1, zorder=0, color='black')

    colorbar_mappable = None

    if style.koopman_selection is not None:
        ax.scatter(evs.real, evs.imag, label='eigenvalue', s=3, zorder=5, alpha=DEFAULT_SCATTER_ALPHA, marker='x')
        selected = [s - 1 for s in style.koopman_selection]
        ax.scatter(evs[selected].real, evs[selected].imag, zorder=10, color='orange', label='selected eigenvalue')

    elif residuals is not None:
        log.info(f'Plot eigenvalues using residuals: {stats.describe(residuals)}')
        colorbar_mappable = ax.scatter(
            np.real(evs),
            np.imag(evs),
            s=20,
            c=np.real(residuals[: len(evs)]),
            cmap='jet',
            marker='.',
            zorder=5,
            vmax=1,
            vmin=0,
        )

    else:
        ev_stability = np.asarray(_get_ev_stability(evs))
        for code, (label, color) in enumerate([('neutral', 'steelblue'), ('stable', 'aqua'), ('unstable', 'teal')]):
            selected_evs = evs[ev_stability == code]
            ax.scatter(
                selected_evs.real,
                selected_evs.imag,
                label=label,
                color=color,
                s=3,
                zorder=5,
                alpha=DEFAULT_SCATTER_ALPHA,
            )

    ax.set_xlabel('$Re(\lambda)$')
    ax.set_ylabel('$Im(\lambda)$')
    ax.grid(True, alpha=DEFAULT_GRID_ALPHA, linestyle='--', linewidth=DEFAULT_THIN_LINEWIDTH)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', 'box')

    if style.run_name is not None:
        ax.set_title(f'{style.run_name} eigenvalues on the unit circle', wrap=True)

    # The residual colouring labels nothing, so it would get an empty legend
    # frame -- which now sits exactly where the colorbar goes.
    if ax.get_legend_handles_labels()[0]:
        add_legend_outside(ax, position='right')

    # Make room on the right BEFORE adding the colorbar, otherwise the two
    # right-hand elements end up misaligned.
    plt.tight_layout()
    fig.subplots_adjust(
        right=DEFAULT_RIGHT_MARGIN_LEGEND_COLORBAR if colorbar_mappable is not None else DEFAULT_RIGHT_MARGIN
    )

    if colorbar_mappable is not None:
        add_colorbar_next_to_grid(fig=fig, ax=ax, mappable=colorbar_mappable, label='Residual')

    return fig


def _plot_eigenvalue_evolution(evs: np.ndarray, t_max: int, style: PlotStyle) -> Figure:
    """Plot |lambda|^t for every eigenvalue, i.e. how fast each mode grows or decays."""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    # consider the embedding timeshift to have a matching time scale
    timesteps = np.arange(0, t_max - style.embedding_timeshift, step=1)
    for j, ev in enumerate(evs):
        ev_timeseries = np.abs([ev**t for t in timesteps])
        ax.plot(
            timesteps * style.temporal_resolution,
            ev_timeseries,
            linestyle='solid',
            linewidth=0.5,
            label=f'$\lambda_{{{j + 1}}}$',
        )

    ax.set_title('Koopman eigenvalues over time')
    ax.set_xlabel('time in seconds')
    ax.set_ylabel('Absolute eigenvalue')
    ax.grid(True, alpha=DEFAULT_GRID_ALPHA, linestyle='--', linewidth=DEFAULT_THIN_LINEWIDTH)

    # The layout helper handles tight_layout and the right margin
    apply_layout_with_side_legend(
        fig,
        num_rows=1,
        has_suptitle=False,
        right_margin=DEFAULT_RIGHT_MARGIN,
        h_pad=1.0,  # minimal vertical padding (single row)
        w_pad=0.5,  # reduce horizontal gap around plot
    )

    add_legend_outside(ax, position='right', ncol=3)

    return fig


def _plot_eigenvalue_magnitudes(evs: np.ndarray) -> Figure:
    """Bar chart of |lambda|, sorted, to see how quickly the spectrum decays."""
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)

    idx = np.argsort(np.abs(evs))
    ax.bar(range(len(evs)), np.abs(evs[idx[::-1]]), color='steelblue', alpha=0.7)

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Absolute value')
    ax.set_title('Eigenvalues sorted by magnitude')

    ax.set_xticks(range(len(evs)))
    ax.set_xticklabels(idx[::-1], rotation=70, fontsize=SMALL_SIZE)
    ax.grid(True, alpha=DEFAULT_GRID_ALPHA, linestyle='--', linewidth=DEFAULT_THIN_LINEWIDTH)

    apply_standard_layout(fig, has_suptitle=False, has_colorbar=False, num_rows=1)

    return fig


def plot_eigenvalues(
    model: EDMD,
    style: PlotStyle,
    t_max: Optional[int] = None,
    xlim: Tuple[float, float] = (-1.4, 1.4),
    ylim: Tuple[float, float] = (-1.4, 1.4),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Visualize the eigenvalues of the Koopman operator, as three figures.

    ``eigenvalues`` places them in the complex plane against the unit circle,
    ``eigenvalues_timeseries`` shows |lambda|^t per eigenvalue, and
    ``eigenvalues_abs`` ranks them by magnitude.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    style : PlotStyle
        Supplies the selection, the time axis and output settings.
    t_max : int, optional
        Last timestep of the evolution plot. Skipped when omitted.
    xlim, ylim : tuple of float
        Axis limits of the complex plane.

    Returns
    -------
    eigenvalues, residuals
        The plotted eigenvalues and, when computed, their spectral residuals.
        The caller persists them; this function only draws.
    """
    log.info('Visualizing eigenvalues on the unit circle')

    evs = model.koopman_eigenvalues.to_numpy()

    # Only computed when they are what the plot colours by: an explicit
    # selection takes precedence over the residuals, and computing them is
    # expensive.
    residuals = None
    if style.compute_pseudospectrum and style.koopman_selection is None:
        residuals = _spectral_residuals(model, evs)

    fig = _plot_eigenvalue_spectrum(evs, style, residuals=residuals, xlim=xlim, ylim=ylim)
    save_figure(fig, style, 'interpretation', 'eigenvalues')

    if t_max is not None:
        fig = _plot_eigenvalue_evolution(evs, t_max, style)
        save_figure(fig, style, 'interpretation', 'eigenvalues_timeseries')

    fig = _plot_eigenvalue_magnitudes(evs)
    save_figure(fig, style, 'interpretation', 'eigenvalues_abs')

    log.info(f'Saved eigenvalue plots to {style.path}')
    return evs, residuals


#########################
### Modes ###############
#########################


def plot_modes(
    model: EDMD,
    tsc: Union[TSCDataFrame, None],
    observable_type: str,
    style: PlotStyle,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_width: float = DEFAULT_CBAR_WIDTH,
):
    """
    Visualize the selected Koopman modes as a grid per real and imaginary part.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc : TSCDataFrame or None
        Time series the eigenfunctions are evaluated on. May be None; then no
        eigenfunctions are returned.
    observable_type : str
        Observable whose mode rows are plotted.
    style : PlotStyle
        Supplies the selection, grid geometry and output settings.
    vmin, vmax : float, optional
        Shared colour limits. Derived per mode when omitted.
    cbar_width : float, default=DEFAULT_CBAR_WIDTH
        Width of the row colorbars, in figure coordinates.

    Returns
    -------
    selection, modes, eigenvalues, eigenfunctions_per_series, initial_state_idx
        The Koopman triple behind the figure, for the caller to persist or reuse.
    """
    log.info(f'Visualizing Koopman modes for {observable_type=}')

    grid_shape = style.grid_shape
    resolution_x, resolution_y = style.grid_resolution

    # we do not use the eigenfunction values here, because these are series specific
    modes, eigenvalues, eigenfunctions_per_series, initial_state_idx, selection = _get_koopman_triplet(
        tsc=tsc,
        model=model,
        observable_type=observable_type,
        selection=style.koopman_selection,
        prediction_timeshift=style.prediction_timeshift,
    )

    # one row per mode, each with real and imaginary part (and their negatives)
    # Use same sizing logic as error_grid for consistent layout
    fig, axs = plt.subplots(
        nrows=len(selection),
        ncols=4,
        figsize=((4 * grid_shape[1] * resolution_x) * 0.4, (grid_shape[0] * resolution_y) * len(selection) * 0.4 + 5),
    )

    if len(selection) == 1:
        axs = np.expand_dims(axs, axis=0)

    images = []
    for i, (j, mode, ev) in enumerate(zip(selection, modes, eigenvalues)):
        grid_real = _reshape_to_grid(mode.real, grid_shape)
        grid_imag = _reshape_to_grid(mode.imag, grid_shape)

        ax1, cax1 = _plot_mode_row(
            axs=axs[i, :],
            grid_real=grid_real,
            grid_imag=grid_imag,
            mode_name=f'$v_{{{j}}}$',
            style=style,
            vmin=vmin,
            vmax=vmax,
            include_negative=True,
        )

        ax1.set_ylabel(f'$\lambda_{{{j}}}$ = {ev.real:.2f} + {ev.imag:.2f}i')
        images.append(cax1)  # color scale is the same due to setting vmax and vmin

    # Suptitle first, then the layout manager, then the colorbars
    fig.suptitle(f'{observable_type.capitalize()} Koopman modes', wrap=True)

    apply_standard_layout(fig, has_suptitle=True, has_colorbar=True, num_rows=len(selection))

    add_row_colorbars(fig=fig, axs=axs, mappables=images, cbar_width=cbar_width)

    save_figure(fig, style, 'interpretation', f'{observable_type}_modes')

    log.info(f'Saved plot of Koopman modes for {observable_type=} to {style.path=}')

    return selection, modes, eigenvalues, eigenfunctions_per_series, initial_state_idx


def _plot_mode_surface_pair(
    axis01,
    mode_line: np.ndarray,
    grid_values: np.ndarray,
    line_resolution: float,
    t_max: int,
    observable: str,
    style: PlotStyle,
    azim: int,
) -> None:
    """
    Draw one mode's space-time surface twice, the second copy flipped upside down.

    Two views of the same surface make it easier to see both crests and troughs
    of a travelling pattern.
    """
    x_idx = np.arange(0, mode_line.shape[-1]) * line_resolution
    y_idx = np.arange(0, t_max) * style.temporal_resolution  # time axis
    xx, yy = np.meshgrid(x_idx, y_idx)

    for ax in axis01:
        surf = ax.plot_surface(
            xx,
            yy,
            grid_values,
            cmap=MODES_CMAP,
            linewidth=0,
            antialiased=True,
            vmin=np.nanmin(grid_values.real),
            vmax=np.nanmax(grid_values.real),
            rcount=200,
            ccount=200,
        )

        ax.set_ylabel('time in s')
        ax.set_zlabel(f'{observable.replace("_", " ")}')
        ax.view_init(azim=azim, elev=10)
        ax.set_zticks([])

    axis01[0].figure.colorbar(surf, shrink=0.5, aspect=5)
    axis01[1].invert_zaxis()  # flip one upside down


def _mode_line_along(
    mode: np.ndarray,
    measurement_line: List[int],
    style: PlotStyle,
) -> Tuple[np.ndarray, float, str]:
    """
    Take one mode's values along the requested grid row or column.

    Returns the line, the cell size along it, and a label naming its position
    in meters.
    """
    grid_shape = style.grid_shape
    try:
        mode = mode.reshape(grid_shape[0], grid_shape[1])
    except (ValueError, AttributeError) as e:
        log.error(f'Cannot reshape array of size {mode.shape} into {(grid_shape[0], grid_shape[1])}.')
        raise ValueError(f'Cannot reshape array of size {mode.shape} into {(grid_shape[0], grid_shape[1])}.') from e

    try:
        mode_line = _get_line_values_from_grid(mode, measurement_line)
    except IndexError:
        log.error(f'Could not access measurement line {measurement_line} in array of shape {mode.shape}')
        raise

    if measurement_line[1] is not None:
        line_resolution = style.grid_resolution[0]
        label = f'y={measurement_line[1] * line_resolution:.1f}m'
    else:
        line_resolution = style.grid_resolution[1]
        label = f'x={measurement_line[0] * line_resolution:.1f}m'

    return mode_line, line_resolution, label


def plot_3d_modes(
    model: EDMD,
    tsc: Union[TSCDataFrame, None],
    t_max: int,
    measurement_line: List[int],
    observable: str,
    style: PlotStyle,
    series_id: Optional[int] = None,
    eigenfunctions_tsc=None,
    normalize_ev: bool = False,
    azim: int = 30,
) -> None:
    """
    Visualize the selected modes along one grid line as a surface over time.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc : TSCDataFrame or None
        Time series used to evaluate the initial condition.
    t_max : int
        Last timestep to show.
    measurement_line : list of int
        Exactly one of [x, None] / [None, y]: the grid column or row to follow.
    observable : str
        Observable to visualize.
    style : PlotStyle
        Supplies the selection, grid geometry, the time axis and output settings.
    series_id : int, optional
        Weight the mode by this series' eigenfunction values instead of by
        powers of its eigenvalue.
    eigenfunctions_tsc : TSCDataFrame, optional
        Eigenfunction values, required when ``series_id`` is given.
    normalize_ev : bool
        Project eigenvalues onto the unit circle before evolving them.
    azim : int
        Azimuth of the 3d view.
    """
    log.info(f'Visualizing Koopman modes over time for {observable=} and {measurement_line=}')

    # we do not use the eigenfunction values here, because these are series specific
    modes, eigenvalues, _, _, selection = _get_koopman_triplet(
        tsc=tsc,
        model=model,
        observable_type=observable,  # checks for contains, can also be observable
        selection=style.koopman_selection,
        prediction_timeshift=style.prediction_timeshift,
    )

    fig, axs = plt.subplots(
        nrows=len(selection), ncols=2, figsize=(7 * 2, 5.5 * len(selection)), subplot_kw=dict(projection='3d')
    )

    if len(selection) == 1:
        axs = np.expand_dims(axs, axis=0)

    for axis01, j, mode, ev in zip(axs, selection, modes, eigenvalues):
        if normalize_ev:
            ev = ev / abs(ev)

        mode_line, line_resolution, line_label = _mode_line_along(mode, measurement_line, style)

        if series_id is not None:
            # starts at the first full initial condition!
            eigenfunction_values = eigenfunctions_tsc.loc[series_id][f'koop_eigfunc{j - 1}'].to_numpy()
            t_max = min(t_max, len(eigenfunction_values))
            grid_values = np.array([zeta * mode_line for zeta in eigenfunction_values])[:t_max]
            axis01[0].set_title(
                f'$v_{{{j}}} \cdot \zeta_{{{j}}}(x_{{t}})$ over time $t$ for {line_label}\n series {series_id}'
            )
        else:
            grid_values = np.array([mode_line * ev**i for i in range(t_max)])
            axis01[0].set_title(f'$v_{{{j}}} \cdot \lambda_{{{j}}}^{{t}}$ over time $t$ for {line_label}')

        _plot_mode_surface_pair(axis01, mode_line, grid_values, line_resolution, t_max, observable, style, azim)

        for ax in axis01:
            ax.set_xlabel(f'{"x" if measurement_line[1] is not None else "y"} position in m')

    plt.tight_layout()

    series_postfix = f'_{series_id}' if series_id is not None else ''
    save_figure(fig, style, 'interpretation', f'{observable}_3d_modes_over_time{series_postfix}')

    log.info(f'Saved plot of Koopman modes for {observable=} to {style.path=}')


#########################
### Supermodes ##########
#########################


def _build_supermodes(
    modes: np.ndarray,
    selection: List[int],
    eigenfunction_for_timestep: pd.DataFrame,
    eigenvalue_pairs: List[Tuple[int, int]],
    non_pairs: List[int],
    timestep: int,
    style: PlotStyle,
) -> List[dict]:
    """
    Combine modes with their eigenfunction values into the supermodes of one timestep.

    A conjugate pair of modes only becomes a real-valued spatial pattern once
    both are weighted by their own eigenfunction and added; unpaired modes are
    already real. Both cases produce one row of the supermode figure.

    Returns
    -------
    list of dict
        One entry per row, with the key it is stored under, the real and
        imaginary grids, the mode name and the subtitle for that row.
    """
    rows = []

    for m in non_pairs:  # m is the list index, starting at 0
        j = selection[m]
        ef = eigenfunction_for_timestep[f'koop_eigfunc{j - 1}'].to_numpy()[0]
        supermode = _reshape_to_grid(modes[m] * ef, style.grid_shape)

        rows.append(
            dict(
                key=m,
                grid_real=supermode.real,
                grid_imag=supermode.imag,  # is all zero, just for sanity check
                mode_name=f'$v_{{{j}}}$',
                subtitle=(
                    f'$v_{{{j}}} \cdot \zeta_{{{j}}}(x^{{td}}_{{{timestep}}})$ '
                    f'= {ef.real:.2f} + {ef.imag:.2f}i '
                    f'at t = {timestep * style.temporal_resolution:.1f}s'
                ),
            )
        )

    for m, n in eigenvalue_pairs:  # list indices; the true number is selection[.]
        j1, j2 = selection[m], selection[n]
        ef1 = eigenfunction_for_timestep[f'koop_eigfunc{j1 - 1}'].to_numpy()[0]
        ef2 = eigenfunction_for_timestep[f'koop_eigfunc{j2 - 1}'].to_numpy()[0]

        # No shift by the eigenvalues here: lambda^0 = 1 at the initial condition
        supermode = _reshape_to_grid((modes[m] * ef1) + (modes[n] * ef2), style.grid_shape)

        rows.append(
            dict(
                key=(m, n),
                grid_real=supermode.real,
                grid_imag=supermode.imag,  # is all zero, just for sanity check
                mode_name=f'$v_{{{j1}}}$ + $v_{{{j2}}}$',
                # Manual line break to keep auto-wrap out of the math environment
                subtitle=(
                    f'$v_{{{j1}}} \cdot \zeta_{{{j1}}}(x^{{td}}_{{{timestep}}}) '
                    f'+ v_{{{j2}}} \cdot \zeta_{{{j2}}}(x^{{td}}_{{{timestep}}})$\n'
                    f'$= v_{{{j1}}} \cdot$ ({ef1.real:.2f} + {ef1.imag:.2f}i) '
                    f'$+ v_{{{j2}}} \cdot$ ({ef2.real:.2f} + {ef2.imag:.2f}i) '
                    f'at t = {timestep * style.temporal_resolution:.1f}s'
                ),
            )
        )

    return rows


def _plot_supermodes_at_timestep(
    supermode_rows: List[dict],
    nrows: int,
    series_id: int,
    timestep: int,
    observable_type: str,
    style: PlotStyle,
    vmin: Optional[float],
    vmax: Optional[float],
) -> None:
    """Draw and save one supermode figure: one row per (super)mode of a single timestep."""
    grid_shape = style.grid_shape
    resolution_x, resolution_y = style.grid_resolution

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=((2 * grid_shape[1] * resolution_x) * 0.4, (grid_shape[0] * resolution_y) * nrows * 0.5 + 5),
    )
    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)

    images = []
    for row_idx, row in enumerate(supermode_rows):
        ax1, cax1 = _plot_mode_row(
            axs=axs[row_idx, :],
            grid_real=row['grid_real'],
            grid_imag=row['grid_imag'],
            mode_name=row['mode_name'],
            style=style,
            vmin=vmin,
            vmax=vmax,
            include_negative=False,
        )
        ax1.set_ylabel(f'{observable_type}')
        images.append(cax1)

    fig.suptitle(
        f'{observable_type.capitalize()} Koopman supermodes w.r.t. eigenvalue pairs \n'
        f' for series {series_id} at t={timestep * style.temporal_resolution:.1f}s',
        wrap=False,
    )  # Disable auto-wrap to prevent breaking math formulas

    apply_standard_layout(
        fig,
        has_suptitle=True,
        has_colorbar=True,
        num_rows=nrows,
        h_pad=8.0,  # Large padding between rows for subtitles
    )

    # The subtitles sit between the rows, so the bottom margin has to make room
    # for them BEFORE they and the colorbars are placed at their final positions.
    fig.subplots_adjust(bottom=max(0.02, _lowest_subtitle_position(axs) - 0.03))

    add_row_subtitles(fig=fig, axs=axs, subtitles=[row['subtitle'] for row in supermode_rows], fontsize=MEDIUM_SIZE)

    add_row_colorbars(fig=fig, axs=axs, mappables=images, labels=[f'{observable_type}'] * len(images))

    save_figure(fig, style, 'interpretation', f'{observable_type}_supermodes_{series_id}_t{timestep}')

    log.info(f'Saved plot of supermodes for {observable_type=} in series {series_id} to {style.path=}')


def _lowest_subtitle_position(axs: np.ndarray) -> float:
    """Figure-fraction y of the lowest row subtitle, which sets the bottom margin."""
    min_y_pos = 1.0
    for row_idx, ax_row in enumerate(axs):
        pos_left = ax_row[0].get_position()
        if row_idx < len(axs) - 1:
            next_row_top = axs[row_idx + 1][0].get_position().y1
            y_pos = (pos_left.y0 + next_row_top) / 2
        else:
            row_height = pos_left.y1 - pos_left.y0
            y_pos = pos_left.y0 - row_height * 0.15
        min_y_pos = min(min_y_pos, y_pos)
    return min_y_pos


def _ground_truth_grids(
    tsc: TSCDataFrame,
    series_id: int,
    observable_type: str,
    timesteps: List[int],
    style: PlotStyle,
) -> List[np.ndarray]:
    """The observable's ground-truth grids at the given timesteps, for one series."""
    df = tsc.loc[:, tsc.columns.str.contains(observable_type)].select_time_values(np.array(timesteps)).loc[series_id]  # type: ignore
    return [_reshape_to_grid(values.to_numpy(), style.grid_shape) for _, values in df.iterrows()]


def _iterative_indices(
    eigenvalue_pairs: List[Tuple[int, int]],
    non_pairs: List[int],
) -> List[Tuple[int, ...]]:
    """Conjugate pairs and single modes, interleaved back into mode order."""
    return sorted(eigenvalue_pairs + [(x,) for x in non_pairs], key=lambda item: item[0])


def _plot_supermode_iterative_sum(
    supermodes: dict,
    tsc: TSCDataFrame,
    modes: np.ndarray,
    iterative_indices: List[Tuple[int, ...]],
    series_id: int,
    observable_type: str,
    style: PlotStyle,
    vmax: Optional[float],
) -> None:
    """
    Add the supermodes up one at a time, one row per partial sum, plus an error row.

    Reading the figure top to bottom shows how many modes it takes before the
    sum resembles the ground truth, which the last row compares it against.
    """
    grid_shape = style.grid_shape
    timesteps = list(supermodes.keys())  # only the timesteps this series has data for
    resolution_x, resolution_y = style.grid_resolution

    nrows = len(iterative_indices) + 1  # +1 for the error row
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=len(timesteps),
        figsize=(
            len(timesteps) * grid_shape[1] * resolution_x * 0.4 + 5,
            (grid_shape[0] * resolution_y) * nrows * 0.5 + 5,
        ),
    )
    # plt.subplots squeezes out any axis of length 1 (e.g. a single timestep),
    # so axs is not always 2d -- normalize it back to (nrows, ncols) for axs[j, :] below
    axs = np.array(axs).reshape(nrows, len(timesteps))
    ims = np.empty(shape=(nrows), dtype=object)

    grid_data_list = [np.zeros_like(_reshape_to_grid(modes[0], grid_shape)) for _ in timesteps]

    for j, ev_idx in enumerate(iterative_indices):
        if len(ev_idx) == 1:  # then we have (m, )
            m = ev_idx[0]
            key, title = m, (f'$\omega_{{{m + 1}}}$' if j == 0 else f' + $\omega_{{{m + 1}}}$')
        else:  # a conjugate pair
            m, n = ev_idx
            key, title = (m, n), f'$ + \omega_{{({m + 1},{n + 1})}}$'

        for i, timestep in enumerate(timesteps):
            grid_data_list[i] += supermodes[timestep][key]

        # convert to real explicitly (should only contain real numbers anyway)
        _, _, ims[j] = _create_ts_plot_from_df(
            [grid.real for grid in grid_data_list],
            timesteps,
            style,
            plot=(fig, axs[j, :]),
            vmax=vmax,
            remove_axis_labels=True,
        )
        axs[j, 0].set_ylabel(title)

    # one last error row for the final sum
    ground_truth_list = _ground_truth_grids(tsc, series_id, observable_type, timesteps, style)
    error_grid_list = [
        np.abs(sum_grid - ground_truth_grid) for sum_grid, ground_truth_grid in zip(grid_data_list, ground_truth_list)
    ]
    _, _, ims[-1] = _create_ts_plot_from_df(
        error_grid_list, timesteps, style, plot=(fig, axs[-1, :]), cmap='OrRd', remove_axis_labels=True
    )
    axs[-1, 0].set_ylabel('Absolute error', wrap=True)

    fig.suptitle(f'{observable_type.capitalize()} Koopman iterative sum for series {series_id}', wrap=True)

    apply_standard_layout(
        fig,
        has_suptitle=True,
        has_colorbar=True,
        num_rows=nrows,
        w_pad=0.5,  # tighter horizontal spacing keeps each subplot readable
    )

    add_row_colorbars(fig=fig, axs=axs, mappables=ims, labels=[f'{observable_type}'] * len(ims))

    save_figure(fig, style, 'interpretation', f'{observable_type}_iterative_sum_supermodes_{series_id}')


def plot_supermodes(
    model: EDMD,
    tsc: TSCDataFrame,
    observable_type: str,
    style: PlotStyle,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Visualize Koopman supermodes: conjugate mode pairs combined into real patterns.

    Per series this produces one figure per visualization timestep, plus one
    figure that sums the supermodes up iteratively and compares them against
    the ground truth.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc : TSCDataFrame
        Time series the eigenfunctions are evaluated on.
    observable_type : str
        Observable to visualize.
    style : PlotStyle
        Supplies the selection, timesteps, grid geometry and output settings.
    vmin, vmax : float, optional
        Shared colour limits. Derived per supermode when omitted.
    """
    modes, eigenvalues, _, _, selection = _get_koopman_triplet(
        tsc=tsc,
        model=model,
        observable_type=observable_type,
        selection=style.koopman_selection,
        prediction_timeshift=style.prediction_timeshift,
    )

    eigenvalue_pairs, non_pairs = _find_conjugate_pairs(eigenvalues)
    log.info(
        f'Found {len(eigenvalue_pairs)} pairs of conjugate eigenvalues and '
        f'{len(non_pairs)} eigenvalue(s) with no imaginary part.'
    )
    log.debug(f'Conjugate pairs: {eigenvalue_pairs}')
    log.debug(f'Non-conjugate modes: {non_pairs}')

    eval_eigenfunctions = model.koopman_eigenfunction(tsc)  # returns TSCDataFrame
    timesteps = _available_timesteps(eval_eigenfunctions, style.visualization_timesteps)
    if not timesteps:
        return

    eval_eigenfunctions = eval_eigenfunctions.select_time_values(timesteps)
    nrows = len(eigenvalue_pairs) + len(non_pairs)

    # per series, because the supermodes use series-specific eigenfunction values
    for series_id in tsc.ids:
        eigenfunction_values = eval_eigenfunctions.loc[series_id]
        supermodes = {}

        for timestep in timesteps:
            eigenfunction_for_timestep = eigenfunction_values[
                eigenfunction_values.index.get_level_values('time') == timestep
            ]

            if len(eigenfunction_for_timestep) == 0:
                continue  # this series has no data for this timestep

            supermode_rows = _build_supermodes(
                modes,
                selection,
                eigenfunction_for_timestep,
                eigenvalue_pairs,
                non_pairs,
                timestep,
                style,
            )
            # keep the real part around for the iterative sum below
            supermodes[timestep] = {row['key']: row['grid_real'] for row in supermode_rows}

            _plot_supermodes_at_timestep(supermode_rows, nrows, series_id, timestep, observable_type, style, vmin, vmax)

        if supermodes:
            _plot_supermode_iterative_sum(
                supermodes,
                tsc,
                modes,
                _iterative_indices(eigenvalue_pairs, non_pairs),
                series_id,
                observable_type,
                style,
                vmax,
            )


def _available_timesteps(eigenfunctions: TSCDataFrame, requested: List[int]) -> List[int]:
    """
    Keep only the requested timesteps the eigenfunctions actually cover.

    With a temporal embedding the eigenfunctions start at the first full
    initial condition, so early visualization timesteps do not exist for them.
    Selecting one anyway raises a bare KeyError out of pandas.
    """
    available = set(eigenfunctions.index.get_level_values('time'))
    timesteps = [t for t in requested if t in available]

    if not timesteps:
        log.warning(
            'Skipping supermode visualization: none of the requested timesteps %s exist in the '
            'eigenfunction values, which start at %s.',
            requested,
            min(available) if available else None,
        )
    elif len(timesteps) != len(requested):
        log.warning(
            'Adjusted supermode timesteps from %s to %s: the eigenfunction values start at %s.',
            requested,
            timesteps,
            min(available),
        )
    return timesteps


#########################
### Eigenfunctions ######
#########################


def _plot_eigenfunctions_per_series(
    eval_eigenfunctions: TSCDataFrame,
    selection: List[int],
    columns: List[str],
    style: PlotStyle,
) -> Figure:
    """One subplot per series, with one line per selected eigenfunction."""
    log.info('Visualizing eigenfunctions across all indices per timeseries')

    series_ids = eval_eigenfunctions.ids
    fig, axs = plt.subplots(ncols=1, nrows=len(series_ids), figsize=(10, 6 * len(series_ids) + 2))
    axs = axs if len(series_ids) > 1 else [axs]  # make it iterable

    for ax, series_id in zip(axs, series_ids):
        series = eval_eigenfunctions.loc[series_id]
        for j, column in zip(selection, columns):
            ax.plot(series.index * style.temporal_resolution, series[column].to_numpy().real, label=f'$\phi_{{{j}}}$')

        ax.axhline(0, color='black', linestyle='dotted', linewidth=DEFAULT_THIN_LINEWIDTH, alpha=DEFAULT_LINE_ALPHA)
        ax.set_title(f'Series {series_id}')

    apply_layout_with_bottom_legend(
        fig=fig,
        num_rows=len(series_ids),
        num_legend_items=len(selection),
        legend_position='bottom',
        has_suptitle=False,
        w_pad=0.5,
    )

    left, bottom, _, _ = axs[-1].get_position().bounds  # (left, bottom, width, height)
    axs[-1].legend(
        loc='upper left', bbox_to_anchor=(left - 0.15, bottom - 0.6, 0, 0), ncol=4
    )  # last axis gets the legend

    return fig


def _plot_eigenfunctions_per_triple(
    eval_eigenfunctions: TSCDataFrame,
    evs: np.ndarray,
    selection: List[int],
    columns: List[str],
    style: PlotStyle,
) -> Figure:
    """
    One subplot per eigenfunction, with every series in it.

    Each series is shadowed by a dotted ``lambda^t * zeta(x_0)`` line: where
    the two drift apart, the triple is likely spectral pollution
    (colbrook-2023-math).
    """
    log.info('Visualizing eigenfunctions across all series per index')

    fig, axs = plt.subplots(ncols=1, nrows=len(columns), figsize=(10, 6 * len(columns)))
    axs = axs if len(columns) > 1 else [axs]  # make it iterable

    for ax, column, j in zip(axs, columns, selection):
        for series_id in eval_eigenfunctions.ids:
            series = eval_eigenfunctions.loc[series_id]
            timesteps = series.index
            values = series[column].to_numpy().real

            ax.plot(timesteps * style.temporal_resolution, values, label=rf'$\zeta_{{{j}}}(x_0)$ ({series_id})')

            initial_amplitude = values[0]
            ev_timeseries = initial_amplitude * evs[j - 1] ** (np.arange(0, len(timesteps)))
            ax.plot(
                timesteps * style.temporal_resolution,
                ev_timeseries,
                linestyle='dotted',
                color='black',
                label=rf'$\lambda_{{{j}}}^t \cdot \zeta_{{{j}}}(x_0)$ ({series_id})',
            )

        ax.axhline(0, color='black', linestyle='dotted', linewidth=DEFAULT_THIN_LINEWIDTH, alpha=DEFAULT_LINE_ALPHA)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Eigenfunction value')
        ax.set_title(
            rf'Koopman eigenfunction $\zeta_{{{j}}}$ with '
            rf'$\lambda_{{{j}}} = {evs[j - 1].real:.2f} + {evs[j - 1].imag:.2f}i$',
            wrap=True,
        )
        ax.grid(True, alpha=DEFAULT_GRID_ALPHA)
        add_legend_outside(ax, position='right')

    fig.suptitle('Koopman eigenfunctions per unique index', wrap=True)

    apply_layout_with_side_legend(
        fig,
        num_rows=len(columns),
        has_suptitle=True,
        right_margin=DEFAULT_RIGHT_MARGIN,
        h_pad=1.5,  # Reduced padding between rows for tighter spacing
        w_pad=0.5,  # shrink horizontal gap between subplots and legend
    )

    return fig


def plot_eigenfunctions(model: EDMD, tsc: TSCDataFrame, style: PlotStyle) -> None:
    """
    Visualize the Koopman eigenfunctions as line plots, two figures.

    The eigenfunctions are not available analytically but numerically, i.e. we
    have their output value for each individual state of each time series.
    ``eigenfunctions_per_series`` groups the lines by series,
    ``eigenfunctions_per_triple`` by eigenfunction.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc : TSCDataFrame
        Time series the eigenfunctions are evaluated on.
    style : PlotStyle
        Supplies the selection, the time axis and output settings.
    """
    log.info('Getting all eigenfunction values per series...')

    eval_eigenfunctions = model.koopman_eigenfunction(tsc)  # returns TSCDataFrame

    if not isinstance(eval_eigenfunctions, TSCDataFrame):
        log.error('Could not evaluate the eigenfunctions. Aborting...')
        raise ValueError('Could not evaluate the eigenfunctions. Aborting...')

    selection = style.koopman_selection
    if selection is None:
        selection = list(range(1, min(11, eval_eigenfunctions.shape[1] + 1)))
    columns = [f'koop_eigfunc{i - 1}' for i in selection]

    fig = _plot_eigenfunctions_per_series(eval_eigenfunctions, selection, columns, style)
    save_figure(fig, style, 'interpretation', 'eigenfunctions_per_series')
    log.info(f'Saved plot of eigenfunctions per series to {style.path}')

    evs = model.koopman_eigenvalues.to_numpy()
    fig = _plot_eigenfunctions_per_triple(eval_eigenfunctions, evs, selection, columns, style)
    save_figure(fig, style, 'interpretation', 'eigenfunctions_per_triple')
    log.info(f'Saved plot of eigenfunctions per triple to {style.path}')


#########################
### Per-cell plots ######
#########################


@dataclass(frozen=True)
class _PerCellSpec:
    """The parts that differ between the two per-cell contribution figures."""

    cumulative: bool
    """ Plot the running sum over modes instead of each mode on its own."""

    mode_label: Callable[[int, int], str]
    """ Legend entry for one mode, given its number and the initial state index."""

    plot_total: bool
    """ Add a separate line for the sum of all selected modes."""

    true_color: str
    height_per_cell: float
    height_per_mode: float
    extra_legend_items: int
    legend_w_pad: float
    title: str
    """ Suptitle, formatted with observable and series_id."""

    name: str
    """ File name stem, formatted with observable and series_id."""


_DECOMPOSITION = _PerCellSpec(
    cumulative=False,
    mode_label=lambda i, idx: f'$\lambda_{{{i}}}^t \cdot v_{{{i}}} \cdot \zeta_{{{i}}}(x^{{td}}_{{{idx}}})$',
    plot_total=True,
    true_color='red',
    height_per_cell=4.0,
    height_per_mode=0.3,
    extra_legend_items=3,  # selection + sum + true + predicted
    legend_w_pad=4.0,
    title='{observable} decomposition for series {series_id}',
    name='{observable}_decomposition_per_cell_{series_id}',
)

_ITERATIVE_SUM = _PerCellSpec(
    cumulative=True,
    mode_label=lambda i, idx: (
        f'$\sum_{{j=1}}^{{{i}}}\lambda_{{j}}^t \cdot v_{{j}} \cdot \zeta_{{j}}(x^{{td}}_{{{idx}}}$'
    ),
    plot_total=False,
    true_color='black',
    height_per_cell=3.0,
    height_per_mode=0.0,
    extra_legend_items=2,  # selection + true + predicted
    legend_w_pad=1.5,
    title='{observable} value iterative sum for series {series_id}',
    name='{observable}_iterative_sum_per_cell_{series_id}',
)


@dataclass(frozen=True)
class _SeriesContributions:
    """The per-series numbers both per-cell figures are built from."""

    series_id: int
    n_timesteps: int
    initial_state_idx: int
    """ Index of the first state a prediction exists for."""

    selection: List[int]
    eigenvalues: np.ndarray
    modes: np.ndarray
    eigenfunction_values: np.ndarray

    def per_mode(self, cell: str, grid_shape: List[int]):
        """Yield (mode number, its real contribution over time) for one cell."""
        for i, eigenvalue, mode, eigenfunction_value in zip(
            self.selection, self.eigenvalues, self.modes, self.eigenfunction_values
        ):
            value_ts = _get_mode_ts(
                self.n_timesteps,
                grid_shape,
                self.initial_state_idx,
                measurement_cell=cell,
                eigenvalue=eigenvalue,
                mode=mode,
                eigenfunction_value=eigenfunction_value,
            )
            yield i, value_ts.real


def _draw_cell_panel(
    ax,
    cell: str,
    series: _SeriesContributions,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    observable: str,
    style: PlotStyle,
    spec: _PerCellSpec,
) -> None:
    """Draw one cell's panel: each mode's contribution, the truth and the prediction."""
    time_axis = np.arange(series.initial_state_idx, series.n_timesteps) * style.temporal_resolution

    value_sum = np.zeros(series.n_timesteps - series.initial_state_idx)
    for i, contribution in series.per_mode(cell, style.grid_shape):
        value_sum += contribution

        ax.plot(
            time_axis,
            value_sum if spec.cumulative else contribution,
            label=spec.mode_label(i, series.initial_state_idx),
            linestyle='solid',
            linewidth=DEFAULT_THIN_LINEWIDTH if spec.cumulative else 0.7,
        )

    ax.axvspan(
        0,
        series.initial_state_idx * style.temporal_resolution,
        color='gray',
        alpha=DEFAULT_SHADED_REGION_ALPHA,
        label='state embedding depth',
    )

    if spec.plot_total:
        ax.plot(
            time_axis,
            value_sum,
            label='Values over time for selected Koopman triples',
            color='black',
            linestyle='solid',
            linewidth=DEFAULT_THIN_LINEWIDTH,
        )

    # f'{cell}_{observable}' is the true key: the caller passes an entry of
    # run_metrics.system_observables, not an observable *type*.
    ax.plot(
        tsc.loc[series.series_id].index * style.temporal_resolution,
        tsc[f'{cell}_{observable}'].loc[series.series_id],
        label='True values',
        color=spec.true_color,
        linewidth=0.5,
    )

    # the prediction, to see how much of it the selected modes account for
    ax.plot(
        tsc_predicted.loc[series.series_id].index * style.temporal_resolution,
        tsc_predicted[f'{cell}_{observable}'].loc[series.series_id],
        label='Predicted values',
        color='black',
        linewidth=0.8,
        linestyle='dotted',
    )

    ax.set_xlabel('time in seconds')
    ax.set_ylabel('values per mode')
    ax.set_title(f'Grid cell {cell}')


def _plot_per_cell_figures(
    model: EDMD,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    observable: str,
    style: PlotStyle,
    spec: _PerCellSpec,
) -> Tuple[Optional[TSCDataFrame], Optional[int]]:
    """
    Draw one per-cell contribution figure per series, following ``spec``.

    Shared by :func:`plot_mode_decomposition_per_cell` and
    :func:`plot_iterative_sum_per_cell`, which differ only in whether each mode
    is drawn on its own or added to a running sum.

    Returns
    -------
    eigenfunctions_per_series, initial_state_idx
        The evaluated eigenfunctions behind the figures, for the caller to persist.
    """
    measurement_cells = style.measurement_cells

    # the full Koopman triple, to recalculate the prediction; initial_state_idx
    # is the number of states the initial condition consumes, skipped ones included
    modes, eigenvalues, eigenfunctions_per_series, initial_state_idx, selection = _get_koopman_triplet(
        tsc=tsc,
        model=model,
        observable_type=observable,
        selection=style.koopman_selection,
        prediction_timeshift=style.prediction_timeshift,
    )

    for series_id in tsc.ids.intersection(tsc_predicted.ids):  # only use the common ids
        log.debug(f'Visualizing per-cell contributions for series {series_id}')
        eigenfunction_values = eigenfunctions_per_series.loc[series_id].to_numpy().T  # need the inner array

        fig, axs = plt.subplots(
            nrows=len(measurement_cells),
            ncols=1,
            figsize=[5, len(measurement_cells) * spec.height_per_cell + spec.height_per_mode * len(selection)],
        )

        if len(measurement_cells) == 1:
            axs = np.expand_dims(axs, axis=0)

        series = _SeriesContributions(
            series_id=series_id,
            n_timesteps=len(tsc.loc[series_id].index),
            initial_state_idx=initial_state_idx,
            selection=selection,
            eigenvalues=eigenvalues,
            modes=modes,
            eigenfunction_values=eigenfunction_values,
        )

        for ax, cell in zip(axs, measurement_cells):
            _draw_cell_panel(ax, cell, series, tsc, tsc_predicted, observable, style, spec)

        # Set before the layout: it measures the suptitle to reserve room for it.
        fig.suptitle(spec.title.format(observable=observable.capitalize(), series_id=series_id), wrap=True)

        apply_layout_with_bottom_legend(
            fig=fig,
            num_rows=len(measurement_cells),
            num_legend_items=len(selection) + spec.extra_legend_items,
            legend_position='bottom',
            has_suptitle=True,
            w_pad=spec.legend_w_pad,
        )

        left, bottom, _, _ = axs[-1].get_position().bounds  # (left, bottom, width, height)
        leg = axs[-1].legend(
            loc='upper left', bbox_to_anchor=(left - 0.6, bottom - 1.2, 0, 0), ncol=2
        )  # last axis gets the legend
        for line in leg.get_lines():
            line.set_linewidth(3.0)

        save_figure(fig, style, 'interpretation', spec.name.format(observable=observable, series_id=series_id))

        log.info(f'Saved per-cell plot for {observable} in series {series_id} to {style.path}')

    return eigenfunctions_per_series, initial_state_idx


def plot_mode_decomposition_per_cell(
    model: EDMD,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    observable: str,
    style: PlotStyle,
) -> Tuple[Optional[TSCDataFrame], Optional[int]]:
    """
    Show what each Koopman triple contributes to a cell's value over time.

    One line per selected triple, plus their sum, the true values and the
    prediction -- so the reader can see which mode carries which part of the
    signal.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    observable : str
        Observable to decompose. Scalar per cell, so this is an entry of
        ``system_observables``, not an observable type.
    style : PlotStyle
        Supplies the measurement cells, the selection and output settings.

    Returns
    -------
    eigenfunctions_per_series, initial_state_idx
        The evaluated eigenfunctions behind the figures, for the caller to persist.
    """
    log.info(f'Visualizing value decomposition for {observable} in {style.measurement_cells}')
    return _plot_per_cell_figures(model, tsc, tsc_predicted, observable, style, _DECOMPOSITION)


def plot_iterative_sum_per_cell(
    model: EDMD,
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    observable: str,
    style: PlotStyle,
) -> Tuple[Optional[TSCDataFrame], Optional[int]]:
    """
    Show the running sum over Koopman triples for a cell's value over time.

    One line per partial sum -- the first mode, the first two, and so on -- so
    the reader can see how many triples the prediction really needs.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    observable : str
        Observable to sum up. Scalar per cell, so this is an entry of
        ``system_observables``, not an observable type.
    style : PlotStyle
        Supplies the measurement cells, the selection and output settings.

    Returns
    -------
    eigenfunctions_per_series, initial_state_idx
        The evaluated eigenfunctions behind the figures, for the caller to persist.
    """
    log.info(f'Visualizing iterative sum for {observable} in {style.measurement_cells}')
    return _plot_per_cell_figures(model, tsc, tsc_predicted, observable, style, _ITERATIVE_SUM)


#########################
### Entry point #########
#########################


def visualize_koopman_triple(
    run_config, run_metrics, tsc: TSCDataFrame, model: EDMD, observable_types, time_values
) -> KoopmanSpectrum:
    """
    Visualize the Koopman triple: eigenvalues, eigenfunctions, and modes.

    Entry point for the pipeline: builds the run's :class:`PlotStyle` once and
    calls the individual plotting functions with it. The numbers behind the
    figures come back in a :class:`KoopmanSpectrum` for the caller to persist.

    Parameters
    ----------
    run_config : FullConfig
        Run configuration -- also decides *whether* supermodes and mode gifs
        are produced.
    run_metrics : RunMetrics
        Run metrics, for the grid geometry and timeshifts.
    tsc : TSCDataFrame
        Time series the model was fitted on.
    model : EDMD
        Fitted model.
    observable_types : iterable of str
        Observable types to plot modes for.
    time_values : list of int
        Timesteps that were predicted; the last one bounds the mode gifs.
    """
    style = PlotStyle.from_run(run_config, run_metrics)

    # NOTE: eigenvalues are not specific to the system observable (they are the weights for the modes)
    eigenvalues, residuals = plot_eigenvalues(
        model=model,
        style=style,
        t_max=style.grid_shape[0],  # first dimension is the number of timesteps
    )
    spectrum = KoopmanSpectrum(eigenvalues=eigenvalues, residuals=residuals)

    # plot eigenfunctions, needed as the amplitude of the initial condition
    # NOTE: the eigenfunctions are not specific to the system observable, i.e. one corresponding with each eigenvalue
    plot_eigenfunctions(model=model, tsc=tsc, style=style)

    # NOTE: the modes can be split into subsets for the different system observable types
    # this is because each mode contains one value per input column in the training set
    for observable_type in sorted(observable_types):
        selection, modes, _, _, _ = plot_modes(
            model=model,
            tsc=tsc,
            observable_type=observable_type,
            style=style,
        )
        spectrum.modes[observable_type], spectrum.selection = modes, selection

        if run_config.visualization.plot_supermodes:
            plot_supermodes(
                model=model,
                tsc=tsc,
                observable_type=observable_type,
                style=style,
            )

        # plot the modes as gif with their eigenvalue
        if run_config.visualization.generate_mode_gifs is not None:
            # Import here to avoid circular dependency
            from .animations import generate_modes_gif

            generate_modes_gif(
                model=model,
                selection=run_config.visualization.generate_mode_gifs,
                observable_type=observable_type,
                tmax=time_values[-1],  # time_values are exactly the steps we predict
                style=style,
            )

    return spectrum


def plot_modes_as_image(
    model: EDMD,
    nrows: int,
    ncols: int,
    style: PlotStyle,
) -> None:
    """
    Plot a compact image grid of modes with no labels or spacing.

    Real and imaginary part of each mode sit side by side; the result is meant
    as a figure for a paper, not for reading off values.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    nrows, ncols : int
        Shape of the image grid. Two panels are used per mode.
    style : PlotStyle
        Supplies the grid geometry and output settings.
    """
    log.info('Creating modes image grid')

    grid_shape = style.grid_shape
    num_modes = int(np.ceil((ncols * nrows) / 2))

    modes = model.koopman_modes
    if modes is None:
        raise ValueError('No Koopman modes available. Aborting...')
    if modes.shape[-1] <= num_modes:
        raise ValueError(
            f'The requested grid of {nrows}x{ncols} images needs {num_modes} modes, but the model '
            f'only has {modes.shape[-1]}. Reduce nrows/ncols.'
        )

    real_modes = modes.to_numpy().real[:, :num_modes]
    imag_modes = modes.to_numpy().imag[:, :num_modes]

    fig = plt.figure(figsize=(ncols * grid_shape[1] / grid_shape[0], nrows))

    # Exact subplot width and height in figure coordinates
    img_w = 1.0 / ncols
    img_h = 1.0 / nrows

    for j, (real, imag) in enumerate(zip(real_modes.T, imag_modes.T)):
        row = (2 * j) // ncols
        col_real = (2 * j) % ncols
        col_imag = col_real + 1

        im_max = max(max(real), max(imag))
        im_min = min(min(real), min(imag))
        max_abs = max(abs(im_max), abs(im_min))  # Symmetric color scale

        for col, values in ((col_real, real), (col_imag, imag)):
            ax = fig.add_axes([col * img_w, 1 - (row + 1) * img_h, img_w, img_h * 1.05])
            ax.imshow(values.reshape(grid_shape[0], grid_shape[1]), cmap=MODES_CMAP, vmin=-max_abs, vmax=max_abs)
            ax.axis('off')

    plt.tight_layout()

    # Save without any padding or whitespace
    save_figure(fig, style, 'interpretation', 'modes_image', bbox_inches='tight', pad_inches=0)
    log.info(f'Saved modes as image to {style.path}')
