"""
The shared context every plotting function needs besides its data.

A run's figures are all drawn the same way: same output folder and file type,
same grid geometry, same rotation and vector scaling, same time axis. Before
this module existed, those values were unpacked from ``VisualizationConfig``
and ``RunMetrics`` at each call site and threaded down by hand, which is what
pushed the plotting signatures to 11-16 parameters.

``PlotStyle`` bundles them into one object built once per run
(:meth:`PlotStyle.from_run`) and passed down unchanged. The rule of thumb:

* *how* a figure is drawn and where it goes -> ``PlotStyle``
* *whether* a figure is produced at all (``plot_supermodes``,
  ``generate_mode_gifs``, ...) -> stays on ``FullConfig``, read by the
  ``visualize_*`` entry points in this package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

from .config import (
    DEFAULT_DPI,
    DEFAULT_FIG_FILETYPE,
    DEFAULT_GIF_FILETYPE,
    DEFAULT_GRID_RESOLUTION,
    DEFAULT_TEMP_RESOLUTION,
    DEFAULT_VECTOR_SCALE,
)

log = logging.getLogger(__name__)

# Only used when a PlotStyle is built by hand (a notebook, a test) without
# saying how big the grid is; every pipeline run gets the real shape from
# RunMetrics.
DEFAULT_GRID_SHAPE = [10, 10]


@dataclass(frozen=True)
class PlotStyle:
    """How this run's figures are drawn, and where they are written."""

    # --- output -----------------------------------------------------------
    path: Optional[Path] = None
    """ Run folder that receives the data/, evaluation/, interpretation/ and
    gifs/ subdirectories. ``None`` means "draw but do not save"."""

    fig_filetype: str = DEFAULT_FIG_FILETYPE
    gif_filetype: str = DEFAULT_GIF_FILETYPE
    dpi: int = DEFAULT_DPI

    # --- rendering --------------------------------------------------------
    rotation: int = 0
    """ Clockwise rotation of every grid plot, in degrees."""

    vector_scaling: float = DEFAULT_VECTOR_SCALE
    """ Inverse scale of quiver arrows: longer for < 1, shorter for > 1."""

    contour: bool = False
    """ For mode visualziation, in case the grid is not wanted"""

    abs_value: bool = False
    """ For mode visualization: plot the magnitude on a grayscale colormap
    (vmin=0) instead of the signed value on the default diverging one."""

    color_levels: Optional[int] = None
    """ Quantize the colormap into this many discrete bands, for a
    contour-plot look. ``None`` keeps a continuous colormap."""

    # --- grid geometry ----------------------------------------------------
    grid_shape: List[int] = field(default_factory=lambda: list(DEFAULT_GRID_SHAPE))
    """ [rows, cols] of the measurement grid."""

    grid_resolution: List[float] = field(default_factory=lambda: list(DEFAULT_GRID_RESOLUTION))
    """ [x, y] cell size in meters."""

    # --- time axis --------------------------------------------------------
    temporal_resolution: float = DEFAULT_TEMP_RESOLUTION
    """ Seconds per timestep, used to label every time axis."""

    embedding_timeshift: int = 0
    """ Timesteps consumed by the temporal embedding before the first prediction."""

    prediction_timeshift: int = 0
    """ Extra timesteps skipped before the initial condition (see PreProcessingConfig)."""

    # --- what to show -----------------------------------------------------
    visualization_timesteps: List[int] = field(default_factory=list)
    """ Timesteps shown as columns in the grid figures."""

    measurement_cells: List[str] = field(default_factory=list)
    """ Cells whose time series get their own line plot, as '(x_y)' keys."""

    koopman_selection: Optional[List[int]] = None
    """ Koopman triples to plot, 1-indexed. ``None`` means "the first 20"."""

    compute_pseudospectrum: bool = False
    """ Colour the eigenvalue plot by spectral residual instead of stability."""

    @property
    def prediction_start(self) -> int:
        """First timestep a prediction exists for -- embedding plus prediction shift.

        Figures that put true and predicted values on one time axis need this
        offset; figures of the raw input data use ``embedding_timeshift``.
        """
        return self.embedding_timeshift + self.prediction_timeshift

    @property
    def run_name(self) -> Optional[str]:
        """Name of the run folder, used to title figures. ``None`` without a path."""
        return Path(self.path).name if self.path is not None else None

    def with_(self, **changes) -> 'PlotStyle':
        """A copy with some fields overridden -- ``dataclasses.replace`` by another name."""
        return replace(self, **changes)

    @classmethod
    def from_run(cls, run_config, run_metrics) -> 'PlotStyle':
        """Build the style for one pipeline run from its config and metrics."""
        return cls(
            path=run_config.meta.unique_run_path,
            fig_filetype=run_config.visualization.fig_filetype,
            gif_filetype=run_config.visualization.gif_filetype,
            rotation=run_config.visualization.rotation,
            vector_scaling=run_config.visualization.vector_scaling,
            grid_shape=run_metrics.grid_shape,
            grid_resolution=run_metrics.grid_resolution,
            temporal_resolution=run_config.preprocessing.temporal_resolution,
            embedding_timeshift=run_metrics.embedding_timeshift,
            prediction_timeshift=run_config.preprocessing.prediction_timeshift,
            visualization_timesteps=list(run_metrics.visualization_timesteps),
            measurement_cells=list(run_metrics.measurement_cells),
            koopman_selection=run_config.visualization.koopman_triple_selection,
            compute_pseudospectrum=run_config.visualization.compute_pseudospectrum,
        )
