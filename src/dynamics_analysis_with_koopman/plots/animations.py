"""
Animation generation functions for Koopman analysis.

This module contains functions for generating animated GIFs of:
- Time series evolution (ground truth and predictions)
- Koopman mode evolution over time
- Eigenfunction-weighted mode time series
- Complex-valued mode visualizations (real and imaginary parts)

All animations visualize the temporal evolution of spatial patterns
in the dynamical system, helping to understand the underlying dynamics
captured by the Koopman operator.
"""

import logging
from functools import partial
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datafold import TSCDataFrame, EDMD
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver
from matplotlib.ticker import FormatStrFormatter

from .config import (
    DEFAULT_MODES_FPS,
    MODES_CMAP,
    MODES_ABS_CMAP,
    DEFAULT_GIF_FPS,
    DEFAULT_TOP_MARGIN,
    DEFAULT_TOP_MARGIN_SUBTITLE,
    DEFAULT_MIN_GIF_FIG_WIDTH,
    DEFAULT_MIN_GIF_FIG_HEIGHT,
    COLORBAR_TICK_FORMAT,
)
from .layout import apply_standard_layout
from .grid import _plot_single_grid, _rotate_vector_clockwise
from .io_utils import save_animation
from .style import PlotStyle

log = logging.getLogger(__name__)


def _update_quiver_data(
    i: int,
    grid_data: np.ndarray,
    image: Quiver,
    ax: Axes,
    style: PlotStyle,
    start_seconds: float,
) -> tuple:
    """
    Redraw one animation frame of a vector field.

    Parameters
    ----------
    i : int
        Frame index.
    grid_data : array
        Grid data for all frames.
    image : Quiver
        The quiver to update in place.
    ax : Axes
        Axes holding the quiver, whose x label carries the time stamp.
    style : PlotStyle
        Supplies grid resolution, rotation and the time axis.
    start_seconds : float
        Time of frame 0, in seconds.

    Returns
    -------
    tuple
        The updated quiver, as ``FuncAnimation`` expects.
    """
    ax.set_xlabel(f'at {i * style.temporal_resolution + start_seconds:.1f} seconds')

    resolution_x, resolution_y = style.grid_resolution
    x_velocity = grid_data[i][0] * resolution_x
    y_velocity = grid_data[i][1] * resolution_y

    x_velocity, y_velocity = _rotate_vector_clockwise(x_velocity, y_velocity, style.rotation)

    angles = np.arctan2(x_velocity, y_velocity)
    norm_angles = (angles + np.pi) / (2 * np.pi)

    image.set_UVC(x_velocity, y_velocity, norm_angles)

    return (image,)


def _add_gif_colorbar(
    fig: Figure,
    ax: Axes,
    image,
    cbar_width: float,
    cbar_pad: float,
) -> None:
    """
    Put a colorbar in the space the layout reserved to the right of ``ax``.

    Quiver plots encode the flow direction, so their bar is labelled in
    radians and its limits are pinned; image plots keep the data limits they
    were drawn with, which stops matplotlib from rescaling colours per frame.
    """
    # Apply the fixed data aspect before measuring
    # Otherwise the color bar takes up space and shrinkts the grid plot
    ax.apply_aspect()
    pos = ax.get_position()
    cbar_ax = fig.add_axes([pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height])

    if isinstance(image, Quiver):
        image.set_clim(0.0, 1.0)
        cbar = fig.colorbar(image, cax=cbar_ax, ticks=[0.0, 0.5, 1.0])
        cbar.ax.set_yticklabels(['$- \\pi$', '0', '$\\pi$'])
    else:
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(COLORBAR_TICK_FORMAT))


def _extract_selected_modes(
    model: EDMD,
    observable_type: str,
    selection: List[int],
) -> np.ndarray:
    """
    Pull the selected Koopman modes for one observable out of a fitted model.

    Parameters
    ----------
    model : EDMD
        Fitted model.
    observable_type : str
        Only the mode rows belonging to this observable are kept.
    selection : list of int
        Mode numbers, 1-indexed as everywhere else in the config.

    Returns
    -------
    np.ndarray
        Shape (len(selection), #grid cells).
    """
    modes = model.koopman_modes  # row per feature column, column per mode
    if not isinstance(modes, pd.DataFrame):
        raise ValueError('Could not extract Koopman modes from the model -- the fitted DMD has no spectral mode.')
    if max(selection) > modes.shape[-1]:
        raise ValueError(
            f'The requested mode selection {selection} is out of bounds: the model has {modes.shape[-1]} modes.'
        )

    modes = modes[modes.index.to_series().str.contains(observable_type)].to_numpy()
    return modes[:, [v - 1 for v in selection]].T


def _modes_in_both_signs(
    modes: np.ndarray,
    selection: List[int],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Append the negated modes, so each pattern is animated in both signs.

    A Koopman mode is an eigenvector and therefore only defined up to a scalar
    factor: ``v`` and ``-v`` are the same spatial pattern, and which of the two
    the eigendecomposition returns is arbitrary. Animating both means the
    reader sees the pattern either way round.

    Returns
    -------
    modes, selection, signs
        The doubled modes and their (repeated) mode numbers, plus a
        'pos'/'neg' tag per entry that goes into the file name.
    """
    return (
        np.concatenate((modes, -modes), axis=0),
        selection + selection,
        ['pos'] * len(selection) + ['neg'] * len(selection),
    )


def _as_grids(mode_ts: List[np.ndarray], grid_shape: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split a complex mode time series into real and imaginary (-1, rows, cols) grids."""
    return (
        [m.real.reshape(-1, grid_shape[0], grid_shape[1]) for m in mode_ts],
        [m.imag.reshape(-1, grid_shape[0], grid_shape[1]) for m in mode_ts],
    )


def _timeseries_animator(
    image,
    ax: Axes,
    grid_data: np.ndarray,
    style: PlotStyle,
    start_seconds: float,
) -> Optional[Callable[[int], tuple]]:
    """
    The frame update for a time series animation, scalar or vector.

    Returns None for grid data that is neither, which the caller reports.
    """
    if isinstance(image, AxesImage):

        def animate(i: int):
            ax.set_xlabel(f'at {i * style.temporal_resolution + start_seconds:.1f} seconds')
            image.set_data(grid_data[i][0])
            return (image,)

        return animate

    if isinstance(image, Quiver):
        return partial(
            _update_quiver_data, grid_data=grid_data, image=image, ax=ax, style=style, start_seconds=start_seconds
        )

    return None


def generate_timeseries_gif(
    tsc: TSCDataFrame,
    observable_type: str,
    series_id: int,
    style: PlotStyle,
    fps: int = DEFAULT_GIF_FPS,
    is_ground_truth: bool = True,
) -> None:
    """
    Animate the spatial evolution of one time series.

    NOTE: GIF generation can be computationally expensive!

    Parameters
    ----------
    tsc : TSCDataFrame
        Time series data.
    observable_type : str
        Observable to animate.
    series_id : int
        Series to animate.
    style : PlotStyle
        Supplies grid geometry, rotation, the time axis and output settings.
    fps : int
        Frames per second.
    is_ground_truth : bool
        Ground truth starts at t=0; a prediction starts after the embedding.
    """
    log.info(f'Generating gif for {observable_type} at {series_id}')

    grid_shape = style.grid_shape
    vis_columns = [col for col in tsc.columns if observable_type in col]

    try:
        gif_data_values = tsc[vis_columns].loc[series_id].to_numpy()
        grid_data = gif_data_values.reshape(gif_data_values.shape[0], -1, grid_shape[0], grid_shape[1])
    except ValueError as err:
        log.error('Could not select gif data from tsc DataFrame')
        raise ValueError('Could not select gif data from tsc DataFrame') from err

    if is_ground_truth:
        prefix = 'ground_truth'
        start_seconds = 0.0
    else:
        prefix = 'predicted'
        start_seconds = style.embedding_timeshift * style.temporal_resolution

    resolution_x, resolution_y = style.grid_resolution

    # Enforce a minimum size: for small grids the data-driven size otherwise
    # leaves too little room for the suptitle, forcing it to wrap into the
    # top-positioned axis ticks.
    fig, ax = plt.subplots(
        figsize=(
            max(DEFAULT_MIN_GIF_FIG_WIDTH, grid_shape[1] * resolution_x * 0.5),
            max(DEFAULT_MIN_GIF_FIG_HEIGHT, grid_shape[0] * resolution_y * 0.5 + 2),
        )
    )
    # Add title first (for proper layout)
    fig.suptitle(f'{prefix.capitalize()} {observable_type} for series {series_id}', wrap=True)

    # Fixed colour limits across all frames, so the animation is comparable
    vmin, vmax = np.floor(grid_data.min()), grid_data.max()
    _, image = _plot_single_grid(
        grid_data[0],
        style,
        vmax=vmax,
        vmin=vmin,
        ax=ax,
    )

    # Extra top margin: the grid plot's x-axis ticks sit above the axes and
    # would otherwise crowd the suptitle.
    apply_standard_layout(
        fig,
        has_suptitle=True,
        has_colorbar=True,  # Always reserve space
        num_rows=1,
        top_margin=DEFAULT_TOP_MARGIN,
    )

    _add_gif_colorbar(fig, ax, image, cbar_width=0.03, cbar_pad=0.03)

    if isinstance(image, AxesImage):
        # Stop matplotlib from rescaling the colours once the data changes
        image.set_clim(vmin=vmin, vmax=vmax)
        ax.autoscale(enable=False)

    animate = _timeseries_animator(image, ax, grid_data, style, start_seconds)
    if animate is None:
        log.error(f'Unexpected grid dimension: {grid_data[0].shape[0]}')
        return

    ani = FuncAnimation(fig, animate, blit=False, frames=grid_data.shape[0])

    save_animation(ani, fig, style, f'{prefix}_{observable_type}_{series_id}', fps=fps)

    log.info(f'Saved animation for {observable_type=} with {series_id=}')


def generate_modes_gif(
    model: EDMD,
    selection: List[int],
    observable_type: str,
    tmax: int,
    style: PlotStyle,
    fps: int = DEFAULT_MODES_FPS,
    normalize_ev: bool = False,
) -> None:
    """
    Animate how each selected Koopman mode evolves under its own eigenvalue.

    No initial condition is involved: frame t shows ``mode * eigenvalue**t``.

    NOTE: The EDMD prediction in each step only uses the real part in the sum.
    However, the imaginary part is very relevant for evolving the system due to
    multiplication with the complex eigenvalue (and i^2 = -1).

    Parameters
    ----------
    model : EDMD
        Fitted EDMD model containing Koopman modes and eigenvalues.
    selection : list of int
        Mode indices to animate (1-indexed).
    observable_type : str
        Observable to animate (e.g. 'density', 'velocity').
    tmax : int
        Last timestep to animate.
    style : PlotStyle
        Supplies grid geometry, rotation, the time axis and output settings.
    fps : int
        Frames per second.
    normalize_ev : bool
        Project eigenvalues onto the unit circle, which removes the transient
        growth or decay and leaves only the pattern.
    """
    log.info(f'Generating gifs for modes {selection} and {observable_type=}')

    modes = _extract_selected_modes(model, observable_type, selection)

    eigenvalues = model.koopman_eigenvalues.to_numpy()
    eigenvalues = eigenvalues[[v - 1 for v in selection]]

    if style.abs_value:
        # |v| and |-v| are the same pattern, so animating both signs would
        # just produce two identical gifs.
        signs: List[Optional[str]] = [None] * len(selection)
    else:
        eigenvalues = np.concatenate((eigenvalues, eigenvalues), axis=0)
        modes, selection, signs = _modes_in_both_signs(modes, selection)

    # The animation starts where the predictions do
    start_shift = style.prediction_start
    start_seconds = start_shift * style.temporal_resolution

    for i, mode, eigenvalue, sign in zip(selection, modes, eigenvalues, signs):
        if normalize_ev:
            eigenvalue = eigenvalue / np.abs(eigenvalue)

        # cut off the initial condition
        mode_ts = [mode * power for power in eigenvalue ** np.arange(0, tmax - start_shift)]
        gif_data_real, gif_data_imag = _as_grids(mode_ts, style.grid_shape)

        title = (
            f'{observable_type.capitalize()} Koopman mode $v_{{{i}}}$ with $\lambda_{{{i}}}$'
            + f' = {eigenvalue.real:.3f} + {eigenvalue.imag:.3f}i'
            + f'{" (normalized)" if normalize_ev else ""}'
        )

        fig, ani = _generate_complex_valued_gif(gif_data_real, gif_data_imag, title, style, start_seconds)

        gif_name = (
            f'{observable_type}_mode_{i}'
            f'{f"_{sign}" if sign else ""}'
            f'{"_norm" if normalize_ev else ""}'
            f'{"_contour" if style.contour else ""}'
            f'{"_abs" if style.abs_value else ""}'
        )
        save_animation(ani, fig, style, gif_name, fps=fps)


def _generate_complex_valued_gif(
    gif_data_real: List[np.ndarray],
    gif_data_imag: List[np.ndarray],
    title: str,
    style: PlotStyle,
    start_seconds: float,
) -> Tuple[Figure, FuncAnimation]:
    """
    Animate the real and imaginary part of complex grid data side by side.

    Shared by the two mode animations, which differ only in how they build the
    time series they hand in.

    Parameters
    ----------
    gif_data_real, gif_data_imag : list of np.ndarray
        One grid per frame, for each part.
    title : str
        Suptitle of the animation.
    style : PlotStyle
        Supplies grid resolution, rotation, vector scaling and the time axis.
    start_seconds : float
        Time of frame 0, in seconds.

    Returns
    -------
    fig, ani
        The figure the animation draws on, and the animation itself.
    """
    # Mode-specific colour limits, so a weak pattern stays visible; symmetric,
    # so a conjugate pair of modes shares one scale. style.abs_value plots the
    # magnitude instead, pinned at vmin=0 on a grayscale colormap.
    all_data = gif_data_real + gif_data_imag  # this is a sum of lists!
    if style.abs_value:
        vmin, vmax = 0, max(np.max(np.abs(m)) for m in all_data)
        cmap = MODES_ABS_CMAP
    else:
        raw_vmin = min(np.min(m) for m in all_data)
        raw_vmax = max(np.max(m) for m in all_data)
        max_abs = max(abs(raw_vmin), abs(raw_vmax))
        vmin, vmax = -max_abs, max_abs
        cmap = MODES_CMAP

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(6, 5))  # width and height
    fig.suptitle(title, wrap=True)

    images = []
    for ax in axs:
        # start from an all-zero grid; the animation fills in the frames
        _, image = _plot_single_grid(
            gif_data_real[0] * 0,
            style,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
        )
        images.append(image)

    # Extra top margin: each subplot has both a top-positioned x-axis and its
    # own "real part"/"imaginary part" subtitle, which need more headroom
    # below the suptitle than the standard margin leaves.
    apply_standard_layout(
        fig,
        has_suptitle=True,
        has_colorbar=True,  # Reserve space on the right
        num_rows=1,
        top_margin=DEFAULT_TOP_MARGIN_SUBTITLE,
    )

    # One colorbar for both subplots, in its own axes so neither shrinks
    _add_gif_colorbar(fig, axs[1], images[-1], cbar_width=0.02, cbar_pad=0.02)

    if isinstance(images[-1], Quiver):
        animate = _quiver_pair_animator(axs, images, gif_data_real, gif_data_imag, style, start_seconds)
    else:
        animate = _image_pair_animator(axs, images, gif_data_real, gif_data_imag, style, start_seconds)

    return fig, FuncAnimation(fig, animate, blit=False, frames=len(gif_data_real))


def _quiver_pair_animator(
    axs: np.ndarray,
    images: List[Quiver],
    gif_data_real: List[np.ndarray],
    gif_data_imag: List[np.ndarray],
    style: PlotStyle,
    start_seconds: float,
) -> Callable[[int], tuple]:
    """Frame update for the vector-field version of the real/imaginary pair."""
    update_real = partial(
        _update_quiver_data,
        grid_data=gif_data_real,
        image=images[0],
        ax=axs[0],
        style=style,
        start_seconds=start_seconds,
    )

    update_imag = partial(
        _update_quiver_data,
        grid_data=gif_data_imag,
        image=images[1],
        ax=axs[1],
        style=style,
        start_seconds=start_seconds,
    )

    def animate(i: int):
        axs[0].set_title('real part')
        axs[1].set_title('imaginary part')
        return update_real(i=i) + update_imag(i=i)

    return animate


def _image_pair_animator(
    axs: np.ndarray,
    images: List[AxesImage],
    gif_data_real: List[np.ndarray],
    gif_data_imag: List[np.ndarray],
    style: PlotStyle,
    start_seconds: float,
) -> Callable[[int], list]:
    """Frame update for the scalar-image version of the real/imaginary pair."""

    def animate(i: int):
        updated_images = []
        for ax, data, img, prefix in zip(axs, [gif_data_real, gif_data_imag], images, ['real', 'imaginary']):
            ax.set_title(f'{prefix} part')
            frame = data[i][0, :, :] if len(data[i].shape) == 3 else data[i]
            if style.abs_value:
                frame = np.abs(frame)
            ax.set_xlabel(f'at {i * style.temporal_resolution + start_seconds:.1f} seconds')

            img.set_data(frame)
            updated_images.append(img)
        return updated_images

    return animate


def generate_eigenfunction_modes_gif(
    model: EDMD,
    eigenfunctions_tsc: TSCDataFrame,
    selection: List[int],
    series_ids: List[int],
    observable_type: str,
    style: PlotStyle,
    fps: int = DEFAULT_MODES_FPS,
) -> None:
    """
    Animate each selected mode weighted by its eigenfunction along a real series.

    Unlike :func:`generate_modes_gif`, the time series comes from the data --
    frame t shows ``mode * eigenfunction(x_t)`` -- so no eigenvalue is needed.

    Parameters
    ----------
    model : EDMD
        Fitted EDMD model containing Koopman modes.
    eigenfunctions_tsc : TSCDataFrame
        Eigenfunction values over time, per series.
    selection : list of int
        Mode indices to animate (1-indexed).
    series_ids : list of int
        Series to animate.
    observable_type : str
        Observable to animate (e.g. 'density', 'velocity').
    style : PlotStyle
        Supplies grid geometry, rotation, the time axis and output settings.
    fps : int
        Frames per second.
    """
    log.info(f'Generating gifs for modes {selection} and {observable_type=}')

    modes = _extract_selected_modes(model, observable_type, selection)

    if style.abs_value:
        # |v| and |-v| are the same pattern, so animating both signs would
        # just produce two identical gifs.
        signs: List[Optional[str]] = [None] * len(selection)
    else:
        modes, selection, signs = _modes_in_both_signs(modes, selection)

    start_seconds = style.prediction_start * style.temporal_resolution

    for i, mode, sign in zip(selection, modes, signs):
        for series_id in series_ids:
            # starts at the first full initial condition
            eigenfunction_values = eigenfunctions_tsc.loc[series_id][f'koop_eigfunc{i - 1}'].to_numpy()
            mode_ts = [zeta * mode for zeta in eigenfunction_values]
            gif_data_real, gif_data_imag = _as_grids(mode_ts, style.grid_shape)

            title = (
                f'{observable_type.capitalize()} timeseries for $v_{{{i}}}*\\zeta_{{{i}}}(t)$ for series {series_id}'
            )

            fig, ani = _generate_complex_valued_gif(gif_data_real, gif_data_imag, title, style, start_seconds)

            gif_name = f'{observable_type}_mode_{i}{f"_{sign}" if sign else ""}_{series_id}'
            save_animation(ani, fig, style, gif_name, fps=fps)
