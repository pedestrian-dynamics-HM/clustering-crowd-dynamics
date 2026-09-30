"""
Grid visualization functions.

This module contains functions for visualizing data on spatial grids,
including single grids, time series of grids, and error visualizations.
"""

import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datafold import TSCDataFrame
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver
from matplotlib.transforms import Affine2D

from .config import (
    CUSTOM_CMAP,
    CYCLIC_CMAP,
    DEFAULT_VMIN,
    DEFAULT_VMAX,
    T_FOR_FEATURE_PLOT,
    ERROR_GRID_LABELS,
    TINY_SIZE,
    DEFAULT_RIGHT_MARGIN_COLORBAR,
    DEFAULT_SCATTER_ALPHA,
    DEFAULT_SHAFT_WIDTH,
    MODES_ABS_CMAP,
)
from .layout import add_colorbar_next_to_grid, add_row_colorbars, apply_standard_layout, format_grid_axes
from .io_utils import save_figure
from .style import PlotStyle

log = logging.getLogger(__name__)


def _compute_grid_rotation(
    x_length: float,
    y_length: float,
    rotation: int,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute the geometry shared by every rotation-aware grid plot.

    A grid rotation always rotates clockwise by ``rotation`` degrees around
    the grid center, then shifts the result so its bounding box starts at
    (0, 0). Used to keep the image transform, vector field positions, and
    measurement-cell labels in sync.

    Returns
    -------
    rotation_matrix : ndarray, shape (2, 2)
    center : tuple of float
        (center_x, center_y) of the unrotated grid
    shift : tuple of float
        (shift_x, shift_y) applied after rotation
    extent : tuple of float
        (width, height) of the rotated, shifted bounding box
    """
    center_x, center_y = x_length / 2, y_length / 2
    theta = np.deg2rad(rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    # Rotation matrix for clockwise rotation
    rotation_matrix = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

    corners = np.array([[0, 0], [x_length, 0], [x_length, y_length], [0, y_length]])
    rotated_corners = (corners - [center_x, center_y]) @ rotation_matrix.T + [center_x, center_y]
    xmin, xmax = rotated_corners[:, 0].min(), rotated_corners[:, 0].max()
    ymin, ymax = rotated_corners[:, 1].min(), rotated_corners[:, 1].max()

    return rotation_matrix, (center_x, center_y), (-xmin, -ymin), (xmax - xmin, ymax - ymin)


def _rotate_grid_points(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    rotation_matrix: np.ndarray,
    center: Tuple[float, float],
    shift: Tuple[float, float],
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Rotate point(s) about ``center`` using ``rotation_matrix``, then translate by ``shift``."""
    center_x, center_y = center
    shift_x, shift_y = shift
    dx, dy = x - center_x, y - center_y
    rotated_x = center_x + dx * rotation_matrix[0, 0] + dy * rotation_matrix[0, 1]
    rotated_y = center_y + dx * rotation_matrix[1, 0] + dy * rotation_matrix[1, 1]
    return rotated_x + shift_x, rotated_y + shift_y


def _rotate_vector_clockwise(
    vx: np.ndarray,
    vy: np.ndarray,
    rotation: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate direction vector(s) clockwise by ``rotation`` degrees (no translation)."""
    if rotation == 0:
        return vx, vy
    theta = np.deg2rad(rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotated_vx = vx * cos_theta + vy * sin_theta
    rotated_vy = vx * (-sin_theta) + vy * cos_theta
    return rotated_vx, rotated_vy


def _rotate_point_for_grid_plot(
    x: float,
    y: float,
    x_length: float,
    y_length: float,
    rotation: int,
) -> Tuple[float, float]:
    """
    Map a point from unrotated grid data coordinates to rotated coordinates.

    Matches the clockwise ``rotation`` and re-centering shift that
    ``_plot_single_grid`` applies to the grid image.
    """
    if rotation == 0:
        return x, y

    rotation_matrix, center, shift, _ = _compute_grid_rotation(x_length, y_length, rotation)
    return _rotate_grid_points(x, y, rotation_matrix, center, shift)


def _parse_measurement_cell(cell: str) -> Tuple[int, int]:
    """
    Parse a measurement-cell key such as ``'(4_7)'`` into its grid indices.

    The key format is produced by
    ``mode_decompositions._data_to_timeseries``, which builds column names as
    ``f'({x_idx}_{y_idx})_{feature}'`` — so a trailing ``_<feature>`` suffix is
    tolerated too (only the first two underscore-separated tokens are used).

    Parameters
    ----------
    cell : str
        Cell key, e.g. ``'(4_7)'`` or ``'(4_7)_velocity_x'``.

    Returns
    -------
    x_idx, y_idx : int
    """
    x_idx, y_idx = map(int, cell.replace('(', '').replace(')', '').split('_')[:2])
    return x_idx, y_idx


def _draw_scalar_grid(
    grid_data: np.ndarray,
    ax: Axes,
    vmin: float,
    vmax: float,
    cmap: Union[Colormap, str],
    style: PlotStyle,
) -> Tuple[AxesImage, Tuple[int, int], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """
    Draw one scalar value per cell as an image.

    Honours ``style.contour`` (bicubic interpolation instead of flat cells),
    ``style.abs_value`` (magnitude on a grayscale colormap, vmin pinned to 0)
    and ``style.color_levels`` (quantize the colormap into discrete bands),
    together giving the black-and-white contour look used for mode plots.

    Returns the image, the (x_length, y_length) of the unrotated grid and the
    axis limits the rotation needs (None when the grid is not rotated).
    """
    resolution_x, resolution_y = style.grid_resolution

    if style.abs_value:
        grid_data = np.abs(grid_data)
        vmin = 0
        cmap = MODES_ABS_CMAP

    if style.color_levels is not None:
        cmap = plt.get_cmap(cmap if isinstance(cmap, str) else cmap.name, style.color_levels)

    im = ax.imshow(
        grid_data,
        cmap=cmap,
        origin='lower',
        extent=(0, grid_data.shape[1], 0, grid_data.shape[0]),
        aspect=resolution_y / resolution_x,
        interpolation='bicubic' if style.contour else 'none',
        vmin=vmin,
        vmax=vmax,
    )
    x_length, y_length = grid_data.shape[1], grid_data.shape[0]

    if style.rotation == 0:
        return im, (x_length, y_length), None, None

    # Rotate the image around the center of the original grid
    _, center, shift, extent = _compute_grid_rotation(x_length, y_length, style.rotation)
    im.set_transform(Affine2D().rotate_deg_around(*center, -style.rotation).translate(*shift) + ax.transData)

    xlim_range, ylim_range = (0, extent[0]), (0, extent[1])
    ax.set_xlim(*xlim_range)
    ax.set_ylim(*ylim_range)
    return im, (x_length, y_length), xlim_range, ylim_range


def _draw_vector_grid(
    grid_data: np.ndarray,
    ax: Axes,
    style: PlotStyle,
) -> Tuple[Quiver, Tuple[int, int], Tuple[float, float], Tuple[float, float]]:
    """
    Draw one (x, y) vector per cell as a quiver field, coloured by direction.

    Returns the quiver, the (x_length, y_length) of the unrotated grid and the
    axis limits, which the rotated bounding box widens.
    """
    resolution_x, resolution_y = style.grid_resolution

    x_idx = np.arange(0, grid_data.shape[2])
    y_idx = np.arange(0, grid_data.shape[1])
    xx, yy = np.meshgrid(x_idx, y_idx)

    x_velocity = grid_data[0, :, :] * resolution_x
    y_velocity = grid_data[1, :, :] * resolution_y

    y_length, x_length = grid_data.shape[1:]
    xlim_range, ylim_range = (0, x_length), (0, y_length)
    bbox_width, bbox_height = x_length, y_length

    if style.rotation != 0:
        rotation_matrix, center, shift, extent = _compute_grid_rotation(x_length, y_length, style.rotation)
        xx, yy = _rotate_grid_points(xx, yy, rotation_matrix, center, shift)
        x_velocity, y_velocity = _rotate_vector_clockwise(x_velocity, y_velocity, style.rotation)

        xlim_range, ylim_range = (0, extent[0]), (0, extent[1])
        ax.set_xlim(*xlim_range)
        ax.set_ylim(*ylim_range)
        bbox_width, bbox_height = extent

    ax.set_aspect(resolution_x / resolution_y, adjustable='box')
    ax.set_box_aspect(bbox_height / bbox_width)

    angles = np.arctan2(x_velocity, y_velocity)
    norm_angles = (angles + np.pi) / (2 * np.pi)

    im = ax.quiver(
        xx,
        yy,
        x_velocity,
        y_velocity,
        norm_angles,
        cmap=CYCLIC_CMAP,
        scale=style.vector_scaling,
        scale_units='xy',
        angles='xy',
        pivot='mid',
        width=DEFAULT_SHAFT_WIDTH,
    )
    return im, (x_length, y_length), xlim_range, ylim_range


def _plot_single_grid(
    grid_data: np.ndarray,
    style: PlotStyle,
    title: str = '',
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    ax: Optional[Axes] = None,
    cmap: Union[Colormap, str] = CUSTOM_CMAP,
    remove_axis_labels: bool = False,
) -> Tuple[Axes, Union[AxesImage, Quiver]]:
    """
    Plot a single grid from a numpy array -- the panel every grid figure is built from.

    Scalar data (one value per cell) is drawn as an image, vector data (two
    values per cell) as a quiver field; both honour ``style.rotation`` and
    ``style.grid_resolution``.

    Parameters
    ----------
    grid_data : np.ndarray
        Shape (y, x) or (1, y, x) for scalars, (2, y, x) for vectors.
    style : PlotStyle
        Supplies grid resolution, rotation and vector scaling.
    title : str, default=''
        Title for this panel.
    vmin, vmax : float
        Colour limits (ignored for quiver plots, which colour by angle).
    ax : Axes, optional
        Axes to draw on. Defaults to the current axes.
    cmap : Colormap or str
        Colormap for scalar data.
    remove_axis_labels : bool
        Drop the tick labels, e.g. for inner panels of a multi-row figure.

    Returns
    -------
    ax : Axes
    im : AxesImage or Quiver
        The mappable, for attaching a colorbar.
    """
    ax = ax or plt.gca()
    resolution_x, resolution_y = style.grid_resolution

    is_scalar = len(grid_data.shape) == 2 or (len(grid_data.shape) == 3 and grid_data.shape[0] == 1)
    is_vector = len(grid_data.shape) == 3 and grid_data.shape[0] == 2

    if is_scalar:
        grid_data = grid_data[0, :, :] if len(grid_data.shape) == 3 else grid_data
        im, (x_length, y_length), xlim_range, ylim_range = _draw_scalar_grid(grid_data, ax, vmin, vmax, cmap, style)
    elif is_vector:
        im, (x_length, y_length), xlim_range, ylim_range = _draw_vector_grid(grid_data, ax, style)
    else:
        log.error(f'Could not plot grid data with shape {grid_data.shape}')
        raise ValueError(f'Unsupported grid data shape: {grid_data.shape}')

    ax.set_title(title)
    format_grid_axes(
        ax,
        x_length,
        y_length,
        resolution_x,
        resolution_y,
        rotation=style.rotation,
        remove_labels=remove_axis_labels,
        xlim=xlim_range,
        ylim=ylim_range,
    )

    # Disable grid for spatial visualizations (imshow/quiver plots)
    ax.grid(False)

    log.debug(f'Plotted grid for single timestep with {title=}')
    return ax, im


def _as_grid_list(
    grid_data: Union[pd.DataFrame, List[np.ndarray]],
    grid_shape: List[int],
) -> List[np.ndarray]:
    """Reshape a frame of flat cell columns, or a list of flat arrays, into (-1, rows, cols) grids."""
    if isinstance(grid_data, pd.DataFrame):
        return [values.to_numpy().reshape(-1, grid_shape[0], grid_shape[1]) for _, values in grid_data.iterrows()]
    if isinstance(grid_data, list):
        return [grid.reshape(-1, grid_shape[0], grid_shape[1]) for grid in grid_data]
    raise TypeError('grid_data must be pandas.DataFrame or list of numpy.ndarray')


def _resolve_color_limits(
    grid_data_list: List[np.ndarray],
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[float, float]:
    """Fill in whichever colour limit was left open from the data itself."""
    if vmax is None:
        vmax = max([grid.max() for grid in grid_data_list])
        if vmax is None or np.isnan(vmax):
            raise ValueError('Could not derive vmax: the grid data is empty or contains only NaN.')

    if vmin is None:
        vmin = min([grid.min() for grid in grid_data_list])
        if vmin is None or np.isnan(vmin):
            raise ValueError('Could not derive vmin: the grid data is empty or contains only NaN.')

    return vmin, vmax


def _create_ts_plot_from_df(
    grid_data: Union[pd.DataFrame, List[np.ndarray]],
    timesteps: List[int],
    style: PlotStyle,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    plot: Tuple[Optional[Figure], Optional[Axes]] = (None, None),
    title: str = '',
    cmap: Union[Colormap, str] = CUSTOM_CMAP,
    remove_title: bool = False,
    remove_axis_labels: bool = False,
) -> Tuple[Figure, np.ndarray, Union[AxesImage, Quiver]]:
    """
    Draw one row of grid panels, one panel per timestep.

    Called either on its own -- then it creates the figure -- or with ``plot``
    set to an existing (figure, row of axes), which is how the multi-row
    figures (error grid, iterative sum) fill one row at a time.

    Parameters
    ----------
    grid_data : DataFrame or list of ndarray
        One entry per timestep, each flat over the grid cells.
    timesteps : list of int
        Timestep indices, used for the panel titles.
    style : PlotStyle
        Supplies grid shape/resolution, rotation, vector scaling and the time axis.
    vmin, vmax : float, optional
        Colour limits shared by the whole row. Derived from the data if omitted.
    plot : tuple, optional
        Existing (Figure, Axes row) to draw into.
    title : str
        Figure suptitle. Only set when this call creates the figure.
    cmap : Colormap or str
        Colormap for scalar data.
    remove_title, remove_axis_labels : bool
        Drop the per-panel timestep title / the tick labels.

    Returns
    -------
    fig : Figure
    axs : array of Axes
    im : AxesImage or Quiver
        The last panel's mappable, for the row colorbar.
    """
    grid_data_list = _as_grid_list(grid_data, style.grid_shape)
    vmin, vmax = _resolve_color_limits(grid_data_list, vmin, vmax)

    resolution_x, resolution_y = style.grid_resolution
    grid_shape = style.grid_shape

    if plot[0] is not None and plot[1] is not None:
        fig, axs = plot
    else:
        fig, axs = plt.subplots(
            nrows=1,
            ncols=len(timesteps),
            figsize=((len(timesteps) * grid_shape[1] * resolution_x) * 0.3, grid_shape[0] * resolution_y * 0.52 + 2),
        )

    if fig is None or axs is None:
        raise ValueError('Could not create figure and axes for time series plot')

    axs = axs if isinstance(axs, np.ndarray) else np.array([axs])

    for i, data in enumerate(grid_data_list):
        _, im = _plot_single_grid(
            grid_data=data,
            style=style,
            title='' if remove_title else f't = {timesteps[i] * style.temporal_resolution:.1f} s',
            vmin=vmin,
            vmax=vmax,
            ax=axs[i],
            cmap=cmap,
            remove_axis_labels=remove_axis_labels,
        )

    if title:
        fig.suptitle(title, wrap=True)

    # With a plot tuple we are filling a row of an existing figure and must
    # **not** call tight_layout again (it would shrink the other rows).
    if plot[0] is None or plot[1] is None:
        apply_standard_layout(fig, has_suptitle=bool(title), has_colorbar=False, num_rows=1)

    log.debug(f'Created time series plot for {timesteps=}')
    return fig, axs, im


def plot_features_in_grid(
    tsc: TSCDataFrame,
    observable_type: str,
    style: PlotStyle,
) -> None:
    """
    Show where the measurement cells sit in the measurement area grid.

    Parameters
    ----------
    tsc : TSCDataFrame
        Time series data.
    observable_type : str
        Observable whose grid is used as the background.
    style : PlotStyle
        Supplies the measurement cells, grid geometry and output settings.
    """
    log.info('Visualizing measurement cells in grid')

    grid_shape = style.grid_shape
    measurement_cells = style.measurement_cells

    # Select data for visualization
    vis_columns = [col for col in tsc.columns if observable_type in col]
    dataframe = tsc[vis_columns]

    if not isinstance(dataframe, TSCDataFrame):
        raise ValueError('Could not select time values from TSCDataFrame')

    grid_data = dataframe.select_time_values(T_FOR_FEATURE_PLOT).loc[dataframe.ids[0]].to_numpy()
    grid_data = grid_data.reshape(-1, grid_shape[0], grid_shape[1])[0]

    # Mark measurement cells
    try:
        for cell in measurement_cells:
            x_idx, y_idx = _parse_measurement_cell(cell)
            grid_data[y_idx, x_idx] = 1
    except IndexError as e:
        log.error(f'Error plotting measurement cells: {e}')
        log.error(f'Validate {measurement_cells=} w.r.t. {grid_shape=}')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    _plot_single_grid(
        grid_data,
        style,
        title=f'Measurement cells (black) at {T_FOR_FEATURE_PLOT * style.temporal_resolution:.1f}s',
        vmax=1,
        ax=ax,
    )

    # Label each marked cell, at the position the rotation moved it to
    x_length, y_length = grid_shape[1], grid_shape[0]
    for cell in measurement_cells:
        x_idx, y_idx = _parse_measurement_cell(cell)
        label_x, label_y = _rotate_point_for_grid_plot(x_idx + 0.8, y_idx + 0.8, x_length, y_length, style.rotation)
        ax.text(
            label_x,
            label_y,
            cell,
            fontsize=TINY_SIZE,
            color='black',
            fontweight='bold',
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=DEFAULT_SCATTER_ALPHA),
        )

    apply_standard_layout(fig, has_suptitle=False, has_colorbar=False)

    save_figure(fig, style, 'data', f'{observable_type}_grid_measurement_points')

    log.info(f'Saved visualization of measurement cells to {style.path=}')


def visualize_grid_for_series_id(
    tsc: TSCDataFrame,
    series_id: int,
    observable_type: str,
    style: PlotStyle,
    vmax: Optional[float] = None,
) -> None:
    """
    Show the ground truth grid of one series at ``style.visualization_timesteps``.

    Parameters
    ----------
    tsc : TSCDataFrame
        Time series data.
    series_id : int
        Series to visualize.
    observable_type : str
        Observable to visualize.
    style : PlotStyle
        Supplies the timesteps, grid geometry and output settings.
    vmax : float, optional
        Upper colour limit. Derived from the data if omitted.
    """
    log.info(f'Visualizing grid for series {series_id} and {observable_type=}')

    timesteps = style.visualization_timesteps

    vis_columns = [col for col in tsc.columns if observable_type in col]
    dataframe = tsc[vis_columns]

    if not isinstance(dataframe, TSCDataFrame):
        raise ValueError('Could not select time values from TSCDataFrame')

    dataframe = dataframe.select_time_values(np.array(timesteps)).loc[series_id]
    # Internal invariant: selecting a single series id off a TSCDataFrame yields a DataFrame.
    assert isinstance(dataframe, pd.DataFrame)

    fig, axs, im = _create_ts_plot_from_df(
        dataframe,
        timesteps,
        style,
        vmax=vmax,
        title=f'{observable_type.capitalize()} ground truth for series {series_id}',
    )

    # Make room on the right BEFORE adding the colorbar. The top is left as
    # _create_ts_plot_from_df's layout measured it against the suptitle.
    fig.subplots_adjust(right=DEFAULT_RIGHT_MARGIN_COLORBAR)

    # Add the colorbar last, so it is placed against the final margins
    add_colorbar_next_to_grid(fig, axs[-1], im, label=f'{observable_type}')

    save_figure(fig, style, 'data', f'{observable_type}_grid_{series_id}')

    log.info(f'Saved series {series_id} for {observable_type=} to {style.path=}')


def visualize_error_grid(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    series_id: int,
    observable_type: str,
    style: PlotStyle,
    is_test: bool = False,
    vmax: Optional[float] = None,
    remove_axis_labels: bool = ERROR_GRID_LABELS,
) -> None:
    """
    Compare truth and prediction for one series as three rows of grids.

    Row 1 is the ground truth, row 2 the absolute error, row 3 the prediction;
    each row has one panel per timestep in ``style.visualization_timesteps``.

    Parameters
    ----------
    tsc, tsc_predicted : TSCDataFrame
        True and predicted time series data.
    series_id : int
        Series to visualize.
    observable_type : str
        Observable to visualize.
    style : PlotStyle
        Supplies the timesteps, grid geometry and output settings.
    is_test : bool
        Whether this is test data (affects title and file name only).
    vmax : float, optional
        Upper colour limit for the truth/prediction rows.
    remove_axis_labels : bool
        Drop the tick labels on every panel.
    """
    log.info(f'Visualizing error grid for {observable_type} at {series_id}')

    timesteps = style.visualization_timesteps
    grid_shape = style.grid_shape
    resolution_x, resolution_y = style.grid_resolution

    df = tsc.select_time_values(np.array(timesteps)).loc[series_id]
    df_predicted = tsc_predicted.select_time_values(np.array(timesteps)).loc[series_id]

    vis_columns = [col for col in df.columns if observable_type in col]
    df = df[vis_columns]
    df_predicted = df_predicted[vis_columns]

    if vmax is None:
        vmax = max(df.max().max(), df_predicted.max().max())
    vmin = min(df.min().min(), df_predicted.min().min())

    error = np.abs(df_predicted - df)

    fig, axs = plt.subplots(
        nrows=3,
        ncols=len(timesteps),
        figsize=((len(timesteps) * grid_shape[1] * resolution_x) * 0.4, (grid_shape[0] * resolution_y) * 1 + 5),
    )

    # One row per quantity: (row label, data, colorbar label, extra plot options)
    rows = [
        ('True values', df, observable_type, dict(vmin=vmin, vmax=vmax)),
        ('Absolute error', error, 'error', dict(vmin=0, cmap='OrRd', remove_title=True)),
        ('Predicted values', df_predicted, observable_type, dict(vmin=vmin, vmax=vmax, remove_title=True)),
    ]

    ims = np.empty(shape=(3,), dtype=object)
    for i, (_, row_data, _, options) in enumerate(rows):
        _, _, ims[i] = _create_ts_plot_from_df(
            row_data, timesteps, style, plot=(fig, axs[i]), remove_axis_labels=remove_axis_labels, **options
        )

    postfix = 'test' if is_test else 'train'
    fig.suptitle(f'{observable_type.capitalize()} error grid for {postfix} data, series {series_id}', wrap=True)

    apply_standard_layout(fig, has_suptitle=True, has_colorbar=True, num_rows=3)

    add_row_colorbars(fig=fig, axs=axs, mappables=ims, labels=[label for _, _, label, _ in rows])

    # After the colorbars: an ylabel set earlier changes what tight_layout
    # reserves and shifts every panel.
    for i, (row_label, _, _, _) in enumerate(rows):
        axs[i, 0].set_ylabel(row_label)

    save_figure(fig, style, 'evaluation', f'{observable_type}_error_grid_{series_id}_{postfix}')

    log.info(f'Saved error grid for {observable_type=} with {series_id=} to {style.path=}')
