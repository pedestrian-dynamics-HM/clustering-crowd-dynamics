"""
Configuration module for visualizations.

This module contains:
- Color schemes and colormaps
- Font sizes and plot styling
- Default parameters for visualizations
- Seaborn style configuration
"""

import logging
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import PillowWriter, FFMpegWriter, HTMLWriter, ImageMagickWriter

# Re-exported, not used here: the visualization modules import these three from
# this module rather than reaching into preprocessing. The noqa keeps `ruff --fix`
# from deleting them as unused.
from ..data import (  # noqa: F401
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_TEMP_RESOLUTION,
    MEASUREMENT_COLUMN_NAMES,
)

log = logging.getLogger(__name__)

#########################
### Color Schemes #######
#########################

COLORS = [(0, 'steelblue'), (0.2, 'aqua'), (0.4, 'teal'), (0.7, 'darkslategray'), (1.0, 'black')]
CMAP_NAME = 'custom_density_map'
CUSTOM_CMAP = LinearSegmentedColormap.from_list(CMAP_NAME, COLORS, N=1000)

CYCLIC_CMAP = LinearSegmentedColormap.from_list(
    'custom_cyclic_map',
    ['#D79A12', '#E55C4B', '#EC2CA3', '#BF40E5', '#8270F1', '#35ADE0', '#3BC4A3', '#71C22F', '#D79A12'],
    N=1000,
)

MODES_CMAP = plt.get_cmap('Spectral')
MODES_ABS_CMAP = plt.get_cmap('gray_r')  # grayscale for absolute-value mode plots

#########################
### Plot Parameters #####
#########################

# Font sizes
TINY_SIZE = 8
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

# Default values
DEFAULT_VMIN = -0.8
DEFAULT_VMAX = 0.8
DEFAULT_GIF_FPS = 6
DEFAULT_MODES_FPS = 6
DEFAULT_VECTOR_SCALE = 0.5
DEFAULT_DPI = 200
DEFAULT_FIG_FILETYPE = 'png'
DEFAULT_GIF_FILETYPE = 'gif'  # Changed from 'mp4' to 'gif' - no external dependencies needed

# Minimum GIF figure size: the timeseries gif figure is sized from grid_shape,
# which for small grids leaves too little room for the suptitle to fit
# without wrapping into the top-positioned axis ticks.
DEFAULT_MIN_GIF_FIG_WIDTH = 4.0
DEFAULT_MIN_GIF_FIG_HEIGHT = 4.0

# Colorbar sizing (for add_row_colorbars and manual positioning)
DEFAULT_CBAR_WIDTH = 0.015
DEFAULT_CBAR_PAD = 0.01

# Colorbar tick labels: at most 3 decimal places, so they stay readable next
# to a mode grid instead of trailing off into the data's full float precision.
COLORBAR_TICK_FORMAT = '%.3f'

# Gap kept between the suptitle and the content below it, as a fraction of the
# figure height, plus the limits on the room reserve_suptitle_band may add
DEFAULT_SUPTITLE_PAD = 0.015
MAX_SUPTITLE_BAND_GROWTH = 0.5
MAX_SUPTITLE_BAND_PASSES = 4

# Layout margins (for subplots_adjust)
DEFAULT_RIGHT_MARGIN = 0.82  # For side legends
DEFAULT_RIGHT_MARGIN_COLORBAR = 0.85  # For colorbars on right
DEFAULT_RIGHT_MARGIN_LEGEND_COLORBAR = 0.75  # For both legend and colorbar on right
DEFAULT_TOP_MARGIN = 0.85  # Standard top margin
DEFAULT_TOP_MARGIN_SUBTITLE = 0.78  # Top margin for figures with per-axes subtitles below the suptitle
DEFAULT_BOTTOM_MARGIN = 0.08  # Standard bottom margin
DEFAULT_LEFT_MARGIN = 0.10  # Standard left margin

# tight_layout padding
DEFAULT_TIGHT_W_PAD = 2.0  # Vertical padding between subplots
DEFAULT_TIGHT_PAD = 1.5  # Overall padding

# Alpha (transparency) values
DEFAULT_GRID_ALPHA = 0.3  # For grid lines
DEFAULT_LINE_ALPHA = 0.8  # For standard lines
DEFAULT_SCATTER_ALPHA = 0.7  # For scatter points
DEFAULT_SHADED_REGION_ALPHA = 0.3  # For axvspan, etc.
DEFAULT_FRAMEALPHA = 0.9  # For legend frames

# Line widths
DEFAULT_THIN_LINEWIDTH = 0.5  # Thin lines (grid, reference lines)
DEFAULT_SHAFT_WIDTH = 0.01  # For quiver arrows

# Other parameters
EV_TOLERANCE = 0.001
T_FOR_FEATURE_PLOT = 40
ERROR_GRID_LABELS = False

#########################
### Animation Settings ##
#########################

animation_writters_by_format = {
    'gif': PillowWriter,
    'apng': PillowWriter,
    'html': HTMLWriter,
    'htm': HTMLWriter,
    'png,': HTMLWriter,
    'mkv': FFMpegWriter,
    'mp4': FFMpegWriter,
    'mjpeg': FFMpegWriter,
    'webp': ImageMagickWriter,
}


def get_animation_writer(gif_filetype: str, fps: int):
    """
    Get animation writer with fallback to GIF if external dependencies are missing.

    This function provides graceful fallback when external tools like ffmpeg or
    ImageMagick are not installed. PillowWriter (for GIF) has no external dependencies.

    Parameters
    ----------
    gif_filetype : str
        Desired file format (e.g., 'gif', 'mp4', 'webp')
    fps : int
        Frames per second

    Returns
    -------
    writer : Writer instance
        Animation writer configured with optimized settings
    actual_filetype : str
        The actual filetype that will be used (may differ from input if fallback occurred)
    """
    if gif_filetype not in animation_writters_by_format:
        log.warning(
            f"Unknown gif_filetype '{gif_filetype}'. Supported formats: "
            f'{list(animation_writters_by_format.keys())}. Falling back to '
            f"'{DEFAULT_GIF_FILETYPE}'."
        )
        gif_filetype = DEFAULT_GIF_FILETYPE

    writer_class = animation_writters_by_format[gif_filetype]

    # Try to create writer, fall back to GIF if external dependency missing
    try:
        if writer_class == FFMpegWriter:
            # Check if ffmpeg is available
            if shutil.which('ffmpeg') is None:
                log.warning(
                    f'ffmpeg not found - cannot create {gif_filetype} file. '
                    f'Falling back to GIF format. '
                    f'To use MP4/MKV, install ffmpeg: brew install ffmpeg'
                )
                writer_class = PillowWriter
                gif_filetype = DEFAULT_GIF_FILETYPE

        elif writer_class == ImageMagickWriter:
            # Check if ImageMagick is available
            if shutil.which('convert') is None:
                log.warning(
                    f'ImageMagick not found - cannot create {gif_filetype} file. '
                    f'Falling back to GIF format. '
                    f'To use WEBP, install ImageMagick: brew install imagemagick'
                )
                writer_class = PillowWriter
                gif_filetype = DEFAULT_GIF_FILETYPE

        # Create writer with optimized settings
        if writer_class == PillowWriter:
            # Optimized GIF settings for smaller file size and better quality
            writer = writer_class(
                fps=fps,
                metadata=dict(artist='Koopman Analysis'),
                bitrate=-1,  # Auto bitrate
            )
        elif writer_class == FFMpegWriter:
            # Optimized MP4 settings
            writer = writer_class(
                fps=fps,
                metadata=dict(artist='Koopman Analysis'),
                codec='libx264',
                extra_args=['-pix_fmt', 'yuv420p', '-crf', '23'],  # Good quality, reasonable size
            )
        else:
            # Default settings for other writers
            writer = writer_class(fps=fps)

        return writer, gif_filetype

    except Exception as e:
        log.warning(f'Failed to create {writer_class.__name__}: {e}. Falling back to GIF.')
        writer = PillowWriter(fps=fps, metadata=dict(artist='Koopman Analysis'), bitrate=-1)
        return writer, DEFAULT_GIF_FILETYPE


#########################
### Seaborn Setup #######
#########################


def setup_seaborn_style():
    """Configure seaborn for consistent styling across all plots."""
    # Set seaborn theme with 'white' style (no grid by default)
    sns.set_theme(style='white', context='paper')

    # Use a palette that complements our custom colormaps
    sns.set_palette('husl')

    # Override with existing matplotlib settings to maintain consistency
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Add subtle grid lines to all plots by default
    plt.rc('axes', grid=True)
    plt.rc('grid', alpha=DEFAULT_GRID_ALPHA)

    log.debug('Seaborn style configured with grid lines enabled')


# Initialize seaborn style when module is imported
setup_seaborn_style()
