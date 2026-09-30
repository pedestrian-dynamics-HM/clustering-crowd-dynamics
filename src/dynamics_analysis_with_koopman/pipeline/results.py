"""Dataclasses for experiment execution and results."""

from dataclasses import dataclass
from typing import Any, Optional

from datafold import TSCDataFrame, EDMD

from ..config.configs import FullConfig


@dataclass
class RunMetrics:
    """Unified runtime metrics model, serialized consistently across train/retrain flows."""

    grid_shape: list[int]
    visualization_timesteps: list[int]
    measurement_cells: list[str]
    embedding_timeshift: int
    prediction_timeshift: int
    grid_resolution: list[float]
    system_observables: list[str]

    # Data split info
    train_parameter: list[int]
    test_parameter: list[int]

    # Derived from data (optional)
    data_column_types: Optional[list[str]] = None
    diff_map_min_eigenvalue: Optional[float] = None
    mae_per_observable: Optional[list[float]] = None
    mae_max: Optional[float] = None
    mae_per_observable_test: Optional[list[float]] = None
    mae_max_test: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to YAML-compatible dict.
        Filters out any field that is `None`"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ExperimentResult:
    """Result of a completed training or retraining experiment."""

    run_id: str
    config: FullConfig
    metrics: RunMetrics
    model: Optional[EDMD] = None
    train_tsc: Optional[TSCDataFrame] = None
    test_tsc: Optional[TSCDataFrame] = None
    train_tsc_predicted: Optional[TSCDataFrame] = None
    test_tsc_predicted: Optional[TSCDataFrame] = None
