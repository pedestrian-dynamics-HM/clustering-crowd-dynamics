"""Visualization utilities for Koopman dynamics analysis."""

from .config import setup_seaborn_style
from .style import PlotStyle
from .grid import (
    plot_features_in_grid,
    visualize_grid_for_series_id,
    visualize_error_grid,
)
from .timeseries import (
    plot_all_series_per_cell,
    plot_predictions_per_cell,
)
from .animations import (
    generate_timeseries_gif,
    generate_modes_gif,
    generate_eigenfunction_modes_gif,
)
from .predictions import (
    plot_mae,
    visualize_ground_truth,
    visualize_model_predictions,
    visualize_mode_decomposition,
)
from .koopman import (
    KoopmanSpectrum,
    plot_eigenvalues,
    plot_modes,
    plot_3d_modes,
    plot_supermodes,
    plot_eigenfunctions,
    plot_mode_decomposition_per_cell,
    plot_iterative_sum_per_cell,
    plot_modes_as_image,
    visualize_koopman_triple,
)
from .diffusion import (
    visualize_diffusion_maps,
    plot_diffusion_map_timeseries,
    plot_diffusion_map_evs,
    scatter3d_with_gif,
)

__all__ = [
    'KoopmanSpectrum',
    'PlotStyle',
    'generate_eigenfunction_modes_gif',
    'generate_modes_gif',
    'generate_timeseries_gif',
    'plot_3d_modes',
    'plot_all_series_per_cell',
    'plot_diffusion_map_evs',
    'plot_diffusion_map_timeseries',
    'plot_eigenfunctions',
    'plot_eigenvalues',
    'plot_features_in_grid',
    'plot_iterative_sum_per_cell',
    'plot_mae',
    'plot_mode_decomposition_per_cell',
    'plot_modes',
    'plot_modes_as_image',
    'plot_predictions_per_cell',
    'plot_supermodes',
    'scatter3d_with_gif',
    'setup_seaborn_style',
    'visualize_diffusion_maps',
    'visualize_error_grid',
    'visualize_grid_for_series_id',
    'visualize_ground_truth',
    'visualize_koopman_triple',
    'visualize_mode_decomposition',
    'visualize_model_predictions',
]
