"""
Layout management utilities for consistent plot formatting.

This module provides utilities to fix overlapping title issues and
ensure consistent layout across all visualizations.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver
from matplotlib.ticker import FormatStrFormatter

from .config import (
    DEFAULT_CBAR_WIDTH,
    DEFAULT_CBAR_PAD,
    DEFAULT_SUPTITLE_PAD,
    MAX_SUPTITLE_BAND_GROWTH,
    MAX_SUPTITLE_BAND_PASSES,
    DEFAULT_RIGHT_MARGIN,
    DEFAULT_RIGHT_MARGIN_COLORBAR,
    DEFAULT_BOTTOM_MARGIN,
    DEFAULT_LEFT_MARGIN,
    DEFAULT_TIGHT_W_PAD,
    DEFAULT_TIGHT_PAD,
    DEFAULT_FRAMEALPHA,
    COLORBAR_TICK_FORMAT,
)

log = logging.getLogger(__name__)


def reserve_suptitle_band(fig: Figure) -> None:
    """
    Grow the figure until the suptitle has a clear band above the content.

    ``tight_layout`` reserves room for each axes' own title and tick labels but
    knows nothing about the suptitle, so the strip at the top of the figure
    gets claimed twice and the two titles collide. The room is added to the
    canvas rather than taken from the axes: shrinking them instead only moves
    the collision one row down, into the axes' own labels.

    The gap is measured, not calculated, because too much of it is decided at
    draw time -- suptitles here are drawn with ``wrap=True``, so the same
    string is one line in a wide figure and three in a narrow one, and grid
    panels only take their final height once their aspect is applied. Growing
    moves the suptitle up with the figure top, so measuring and growing repeat
    until the gap holds.

    Does nothing when the figure has no suptitle yet -- which means the caller
    applied the layout before setting it, and gets a warning to that effect.
    """
    suptitle = getattr(fig, '_suptitle', None)
    if suptitle is None or not suptitle.get_text():
        log.warning(
            'Layout was asked to leave room for a suptitle, but the figure has none yet. '
            'Set the suptitle before applying the layout, or the two will overlap.'
        )
        return

    original_height = fig.bbox.height
    max_height = original_height * (1 + MAX_SUPTITLE_BAND_GROWTH)

    for _ in range(MAX_SUPTITLE_BAND_PASSES):
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            suptitle_bottom = suptitle.get_window_extent(renderer).y0
            content_top = max(
                (bbox.y1 for bbox in (ax.get_tightbbox(renderer) for ax in fig.axes) if bbox is not None), default=0.0
            )
        except (AttributeError, ValueError) as e:
            log.warning(f'Could not measure the suptitle ({e}); leaving the layout as it is.')
            return

        missing = DEFAULT_SUPTITLE_PAD * original_height - (suptitle_bottom - content_top)
        if missing <= 0:
            break

        # Round up to an even pixel count: the ffmpeg writers behind the
        # video formats in animation_writters_by_format reject odd dimensions.
        new_height = min(fig.bbox.height + missing, max_height)
        new_height = 2 * np.ceil(new_height / 2)
        if new_height <= fig.bbox.height:
            log.warning(
                'The suptitle still overlaps the content after growing the figure by '
                f'{MAX_SUPTITLE_BAND_GROWTH:.0%}; leaving it at that.'
            )
            break

        # Keep every axes where it is, in inches: only the new strip is added.
        scale = fig.bbox.height / new_height
        top, bottom = fig.subplotpars.top, fig.subplotpars.bottom
        fig.set_figheight(new_height / fig.dpi)
        fig.subplots_adjust(top=top * scale, bottom=bottom * scale)

    log.debug(f'Reserved a suptitle band: height {original_height:.0f} -> {fig.bbox.height:.0f} px')


def apply_standard_layout(
    fig: Figure,
    has_suptitle: bool = True,
    has_colorbar: bool = False,
    num_rows: int = 1,
    colorbar_position: str = 'right',
    h_pad: Optional[float] = None,
    w_pad: Optional[float] = None,
    top_margin: Optional[float] = None,
) -> None:
    """
    Apply consistent layout with proper spacing to avoid overlapping titles.

    This method fixes the common pattern of overlapping titles by:
    1. Assuming suptitle was already added (caller should add it first)
    2. Calling tight_layout with dynamic padding based on num_rows
    3. Fine-tuning with subplots_adjust for consistent margins

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to apply layout to
    has_suptitle : bool, default=True
        Whether the figure has a suptitle that needs space
    has_colorbar : bool, default=False
        Whether the figure has a colorbar on the side
    num_rows : int, default=1
        Number of rows of subplots (used for dynamic h_pad)
    colorbar_position : str, default='right'
        Position of colorbar ('right' or 'bottom')
    h_pad : float, optional
        Custom vertical padding between rows. If None, calculates automatically.
    top_margin : float, optional
        Override for the top margin fraction. Use this when the default
        leaves too little room, e.g. for axes with per-axes subtitles or
        top-positioned tick labels. If None, uses the standard 0.92/0.98
        split based on ``has_suptitle``. Room for the suptitle itself is
        added by :func:`reserve_suptitle_band` and needs no override.
    """
    # Calculate dynamic spacing based on number of rows
    # More rows need more vertical spacing to avoid crowding
    # Increased multiplier from 0.5 to 1.0 for better spacing between rows
    if h_pad is None:
        h_pad = max(3.0, 1.0 + num_rows * 1.0)

    # Apply tight_layout first with dynamic padding
    try:
        fig.tight_layout(h_pad=h_pad, w_pad=w_pad if w_pad is not None else DEFAULT_TIGHT_W_PAD, pad=DEFAULT_TIGHT_PAD)
    except ValueError as e:
        log.warning(f'tight_layout failed: {e}. Continuing with subplots_adjust only.')

    # Adjust for suptitle and colorbar with consistent margins
    top = top_margin if top_margin is not None else (0.92 if has_suptitle else 0.98)
    right = DEFAULT_RIGHT_MARGIN_COLORBAR if (has_colorbar and colorbar_position == 'right') else 0.98
    bottom = 0.15 if (has_colorbar and colorbar_position == 'bottom') else DEFAULT_BOTTOM_MARGIN

    fig.subplots_adjust(top=top, right=right, bottom=bottom, left=DEFAULT_LEFT_MARGIN)

    if has_suptitle:
        reserve_suptitle_band(fig)

    log.debug(f'Applied standard layout: h_pad={h_pad}, top={top}, right={right}')


def add_colorbar_next_to_grid(
    fig: Figure,
    ax: Axes,
    mappable,
    label: str = '',
    cbar_width: float = DEFAULT_CBAR_WIDTH,
    cbar_pad: float = DEFAULT_CBAR_PAD,
) -> None:
    """
    Add a colorbar next to ``ax`` without changing the size of ``ax``.

    The colorbar gets its own figure-level axes placed against the drawn
    bounds of ``ax``. Quiver plots colour by direction rather than magnitude,
    so their bar is pinned to [0, 1] and labelled in radians.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure
    ax : Axes
        The axes containing the mappable
    mappable
        The mappable object (e.g., result of imshow, quiver)
    label : str, default=''
        Label for the colorbar
    cbar_width : float, default=DEFAULT_CBAR_WIDTH
        Width of the colorbar in figure coordinates
    cbar_pad : float, default=DEFAULT_CBAR_PAD
        Padding between the axes and the colorbar in figure coordinates
    """
    # An aspect-constrained axes only takes its final box once the aspect has
    # been applied -- which normally happens at draw time, i.e. too late to
    # measure here.
    ax.apply_aspect()
    pos = ax.get_position()

    # Colorbar axes: [left, bottom, width, height]
    cbar_ax = fig.add_axes(
        [
            pos.x1 + cbar_pad,  # Left edge (right of the axes + padding)
            pos.y0,  # Bottom edge (aligned with the axes)
            cbar_width,  # Width
            pos.height,  # Height (same as the axes)
        ]
    )

    if isinstance(mappable, Quiver):
        mappable.set_clim(0.0, 1.0)
        cbar = fig.colorbar(mappable, cax=cbar_ax, ticks=[0.0, 0.5, 1.0])
        cbar.ax.set_yticklabels(['$- \\pi$', '0', '$\\pi$'])
    else:
        cbar = fig.colorbar(mappable, cax=cbar_ax, label=label)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(COLORBAR_TICK_FORMAT))

    log.debug(f'Added colorbar next to axes at {pos.x1:.2f}, {pos.y0:.2f}')


def add_row_colorbars(
    fig: Figure,
    axs: np.ndarray,
    mappables: List,
    labels: Optional[List[str]] = None,
    cbar_width: float = DEFAULT_CBAR_WIDTH,
    cbar_pad: float = DEFAULT_CBAR_PAD,
) -> None:
    """
    Add one colorbar per row, at the right of the row's last subplot.

    A loop over :func:`add_colorbar_next_to_grid`, so the rows of an error
    grid and a single grid figure place their colorbars the same way.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure
    axs : np.ndarray
        2D array of axes (rows x cols)
    mappables : list
        List of mappable objects (one per row)
    labels : list of str, optional
        Labels for each colorbar (one per row)
    cbar_width : float, default=DEFAULT_CBAR_WIDTH
        Width of colorbar in figure coordinates
    cbar_pad : float, default=DEFAULT_CBAR_PAD
        Padding between subplot and colorbar in figure coordinates
    """
    if labels is None:
        labels = [''] * len(mappables)

    for ax_row, mappable, label in zip(axs, mappables, labels):
        last_ax = ax_row[-1] if isinstance(ax_row, np.ndarray) else ax_row
        add_colorbar_next_to_grid(fig, last_ax, mappable, label, cbar_width=cbar_width, cbar_pad=cbar_pad)

    log.debug(f'Added {len(mappables)} row colorbars with width={cbar_width}, pad={cbar_pad}')


def calculate_legend_position(num_items: int, position: str = 'bottom') -> Dict[str, Any]:
    """
    Calculate optimal legend position based on number of items.

    Parameters
    ----------
    num_items : int
        Number of items in the legend
    position : str, default='bottom'
        Desired position ('bottom', 'right', or 'best')

    Returns
    -------
    dict
        Dictionary with legend parameters including 'loc', 'bbox_to_anchor',
        'ncol', and 'bottom_adjust' (amount to adjust figure bottom margin)
    """
    if position == 'bottom':
        ncol = min(4, num_items)
        # Calculate bottom margin based on number of rows needed
        num_legend_rows = int(np.ceil(num_items / ncol))
        bottom_margin = min(0.7, 0.3 + 0.06 * num_legend_rows)

        return {
            'loc': 'upper left',
            'bbox_to_anchor': (-0.15, -0.6, 0, 0),
            'ncol': ncol,
            'bottom_adjust': bottom_margin,
            'frameon': True,
        }
    elif position == 'right':
        return {'loc': 'center left', 'bbox_to_anchor': (1.05, 0.5), 'ncol': 1, 'bottom_adjust': None, 'frameon': True}
    else:  # 'best'
        return {'loc': 'best', 'bbox_to_anchor': None, 'ncol': 1, 'bottom_adjust': None, 'frameon': True}


def apply_layout_with_bottom_legend(
    fig: Figure,
    num_rows: int,
    num_legend_items: int,
    legend_position: str = 'bottom',
    has_suptitle: bool = True,
    h_pad: Optional[float] = None,
    w_pad: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Apply layout for figures with bottom legends.

    This is used for interpretation plots that have legends at the bottom.
    Centralizes the h_pad calculation and legend positioning logic.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to apply layout to
    num_rows : int
        Number of subplot rows
    num_legend_items : int
        Number of items in the legend
    legend_position : str, default='bottom'
        Position for the legend
    has_suptitle : bool, default=True
        Whether the figure has a suptitle

    Returns
    -------
    dict
        Dictionary with legend parameters from calculate_legend_position
    """
    # Calculate dynamic h_pad based on number of rows (same as evaluation plots)
    h_pad = max(3.0, 1.0 + num_rows * 1.0)

    # Get legend position parameters
    legend_pos = calculate_legend_position(num_items=num_legend_items, position=legend_position)

    # Apply tight layout with dynamic padding
    try:
        fig.tight_layout(h_pad=h_pad, w_pad=w_pad if w_pad is not None else DEFAULT_TIGHT_W_PAD)
    except ValueError as e:
        log.warning(f'tight_layout failed: {e}. Continuing with subplots_adjust only.')

    # Adjust margins for suptitle and bottom legend
    top = 0.9 if has_suptitle else 0.98
    bottom = legend_pos.get('bottom_adjust', 0.1)

    fig.subplots_adjust(top=top, bottom=bottom)

    if has_suptitle:
        reserve_suptitle_band(fig)

    log.debug(f'Applied layout with bottom legend: h_pad={h_pad}, top={top}, bottom={bottom}')

    return legend_pos


def apply_layout_with_side_legend(
    fig: Figure,
    num_rows: int,
    has_suptitle: bool = True,
    right_margin: float = DEFAULT_RIGHT_MARGIN,
    h_pad: Optional[float] = None,
    w_pad: Optional[float] = None,
) -> None:
    """
    Apply layout for figures with right-side legends.

    This is used for time series and other plots that have legends on the right side.
    Centralizes the layout application and right margin adjustment.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure to apply layout to
    num_rows : int
        Number of subplot rows
    has_suptitle : bool, default=True
        Whether the figure has a suptitle
    right_margin : float, default=DEFAULT_RIGHT_MARGIN
        Right margin value to make room for legends (0.0-1.0)
    h_pad : float, optional
        Custom vertical padding between rows. If None, calculates automatically.
    """
    # Apply standard layout with dynamic h_pad and optional w_pad
    apply_standard_layout(
        fig, has_suptitle=has_suptitle, has_colorbar=False, num_rows=num_rows, h_pad=h_pad, w_pad=w_pad
    )

    # Adjust right margin to make room for legend
    fig.subplots_adjust(right=right_margin)

    log.debug(f'Applied layout with side legend: num_rows={num_rows}, right={right_margin}, h_pad={h_pad}')


def add_row_subtitles(fig: Figure, axs: np.ndarray, subtitles: List[str], fontsize: int = 16) -> float:
    """
    Add centered subtitles below each row of subplots.

    This function places text centered below each row, positioned in the
    space between rows. It calculates positions after layout is applied
    and returns the lowest y-position to help adjust bottom margin.

    Parameters
    ----------
    fig : Figure
        The matplotlib figure
    axs : np.ndarray
        2D array of axes (rows x cols)
    subtitles : list of str
        List of subtitle strings (one per row)
    fontsize : int, default=16
        Font size for subtitles

    Returns
    -------
    float
        The lowest y-position of any subtitle (for margin adjustment)
    """
    min_y_pos = 1.0  # Track the lowest text position

    for row_idx, (ax_row, subtitle) in enumerate(zip(axs, subtitles)):
        # Get the position of the current row
        if isinstance(ax_row, np.ndarray):
            pos_left = ax_row[0].get_position()
            pos_right = ax_row[-1].get_position()
        else:
            # Single axis per row
            pos_left = pos_right = ax_row.get_position()

        # Center horizontally between left edge of first and right edge of last subplot
        x_center = (pos_left.x0 + pos_right.x1) / 2

        # Calculate y position: if there's a next row, place text halfway between
        # current row bottom and next row top; otherwise place below current row
        if row_idx < len(axs) - 1:
            next_row = axs[row_idx + 1]
            next_row_ax = next_row[0] if isinstance(next_row, np.ndarray) else next_row
            next_row_top = next_row_ax.get_position().y1
            y_pos = (pos_left.y0 + next_row_top) / 2
        else:
            # Last row - place text below with offset proportional to row height
            row_height = pos_left.y1 - pos_left.y0
            y_pos = pos_left.y0 - row_height * 0.15

        min_y_pos = min(min_y_pos, y_pos)
        fig.text(x_center, y_pos, subtitle, ha='center', va='center', fontsize=fontsize, wrap=False)

    log.debug(f'Added {len(subtitles)} row subtitles, min_y_pos={min_y_pos:.3f}')
    return min_y_pos


def format_grid_axes(
    ax: Axes,
    x_length: int,
    y_length: int,
    resolution_x: float,
    resolution_y: float,
    rotation: int = 0,
    remove_labels: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    """
    Apply consistent grid axis formatting.

    Parameters
    ----------
    ax : Axes
        The axes to format
    x_length : int
        Length of grid in x direction (number of cells)
    y_length : int
        Length of grid in y direction (number of cells)
    resolution_x : float
        Grid resolution in x direction
    resolution_y : float
        Grid resolution in y direction
    rotation : int
        Clockwise rotation angle in degrees
    remove_labels : bool, default=False
        Whether to remove tick labels (for cleaner multi-panel plots)
    xlim: tuple, default=None
        Grid x-axes limit
    ylim: tuple, default=None
        Grid y-axes limit
    """

    _, x_max = xlim if xlim is not None else (0, x_length)
    _, y_max = ylim if ylim is not None else (0, y_length)

    # Swap x and y resolution for 90° and 270° rotations
    if rotation in (0, 180):
        display_resolution_x = resolution_x
        display_resolution_y = resolution_y
    else:
        display_resolution_x = resolution_y
        display_resolution_y = resolution_x

    # Calculate appropriate tick intervals (roughly every 2 meters)
    tick_interval_x = max(1, int(2 / display_resolution_x))
    tick_interval_y = max(1, int(2 / display_resolution_y))

    # The (0, 0) origin sits at whichever corner the rotation maps the
    # original (0, 0) corner to -- x_max/y_max for a reversed axis, 0
    # otherwise. Anchoring the ticks there guarantees a tick lands exactly on
    # that corner; anchoring at 0 and reversing the label text instead (as
    # before) leaves the origin tick short of the corner whenever x_max/y_max
    # is not a multiple of the tick interval, so the grid appears to not
    # start at (0, 0).
    x_reversed = rotation in (180, 270)
    y_reversed = rotation in (90, 180)

    if x_reversed:
        x_ticks = np.arange(x_max, -1, -tick_interval_x)[::-1]
        x_labels = [str(int(round((x_max - v) * display_resolution_x))) for v in x_ticks]  # noqa: RUF046
    else:
        x_ticks = np.arange(0, x_max + 1, tick_interval_x)
        x_labels = [str(int(round(v * display_resolution_x))) for v in x_ticks]  # noqa: RUF046

    if y_reversed:
        y_ticks = np.arange(y_max, -1, -tick_interval_y)[::-1]
        y_labels = [str(int(round((y_max - v) * display_resolution_y))) for v in y_ticks]  # noqa: RUF046
    else:
        y_ticks = np.arange(0, y_max + 1, tick_interval_y)
        y_labels = [str(int(round(v * display_resolution_y))) for v in y_ticks]  # noqa: RUF046

    # Set axis limits
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Set ticks and labels
    ax.set_xticks(x_ticks, labels=x_labels)
    ax.set_yticks(y_ticks, labels=y_labels)

    if remove_labels:
        ax.tick_params(labelbottom=False, labelleft=False)

    # Move x-axis to top (common convention for spatial grids)
    ax.xaxis.tick_top()
    ax.tick_params(axis='y', left=True)

    # TODO: check, why this is only done when rotation is activated
    if rotation != 0:
        ax.tick_params(axis='x', pad=8)
        ax.tick_params(axis='y', pad=8)

    log.debug(f'Formatted grid axes: x_length={x_length}, y_length={y_length}')


def add_legend_outside(ax: Axes, position: str = 'right', **legend_kwargs) -> None:
    """
    Add legend outside the plot area for better space utilization.

    Parameters
    ----------
    ax : Axes
        The axes to add legend to
    position : str, default='right'
        Position for legend ('right' or 'bottom')
    **legend_kwargs
        Additional keyword arguments passed to ax.legend()
    """
    if position == 'right':
        # Place legend to the right of the plot
        default_kwargs = {
            'loc': 'center left',
            'bbox_to_anchor': (1.02, 0.5),
            'frameon': True,
            'framealpha': DEFAULT_FRAMEALPHA,
        }
    elif position == 'bottom':
        # Place legend below the plot
        default_kwargs = {
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, -0.15),
            'frameon': True,
            'framealpha': DEFAULT_FRAMEALPHA,
            'ncol': 3,
        }
    else:
        default_kwargs = {}

    # Merge with user-provided kwargs
    default_kwargs.update(legend_kwargs)
    ax.legend(**default_kwargs)

    log.debug(f'Added legend at position={position}')
