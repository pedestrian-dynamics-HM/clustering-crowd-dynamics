"""Preprocessing package for trajectory data handling.

This package provides functionality for loading, discretizing, and processing
trajectory data into grid-based observables for analysis.

Only the entry points used by the rest of the package are re-exported here. The
individual steps (ingestion, discretization, grid construction, checkpoint
loading, observable generation) stay module-private; import them from their
module directly if you need them, e.g. in a notebook:

    from dynamics_analysis_with_koopman.data.grid import _kernel
"""

from .checkpoint import read_timeseries_from_checkpoint
from .discretization import DEFAULT_TEMP_RESOLUTION
from .grid import DEFAULT_GRID_RESOLUTION
from .ingestion import MEASUREMENT_COLUMN_NAMES, load_trajectory_data_to_grid

__all__ = [
    'DEFAULT_GRID_RESOLUTION',
    'DEFAULT_TEMP_RESOLUTION',
    'MEASUREMENT_COLUMN_NAMES',
    'load_trajectory_data_to_grid',
    'read_timeseries_from_checkpoint',
]
