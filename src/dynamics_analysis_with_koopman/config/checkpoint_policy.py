from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .configs import FullConfig


@dataclass
class MergeReport:
    applied_paths: list[str]
    ignored_paths: list[str]


# Strict allowlist: only these fields are allowed to override checkpoint values
# when resuming from a checkpoint. preprocessing.tlim is deliberately absent --
# it is merged element-wise below instead, since a single is-None check (as
# _UNSET_DEFERS_TO_CHECKPOINT does for the others) can't express "each of
# tlim[0]/tlim[1] independently inherits the checkpoint's value when unset".
_ALLOWED_OVERRIDE_PATHS = {
    # metadata/runtime, replaced as a whole
    # preprocessing/data selection
    'preprocessing.system_observables',
    'preprocessing.prediction_timeshift',
    'preprocessing.store_tsc_data',
    'preprocessing.store_timeseries_data',
    'preprocessing.preprocess_only',
    'data.load_data_from_checkpoint',
    'data.enrich_timestep_data',
    'data.train_parameter',
    'data.test_parameter',
    # algorithm controls kept mutable for resume/retrain flows
    'algorithm.learning_algo',
    'algorithm.sort_koopman_triplets',
    'algorithm.rank',
    'algorithm.use_temporal_data_scaling',
    'algorithm.use_temporal_embedding',
    'algorithm.has_measurement_noise',
    'algorithm.use_diffusion_maps',
    'algorithm.pseudospectrum_residual_filter',
    'evaluation.store_trained_model',
    'evaluation.extra_prediction_steps',
    'temporal_embedding.num_t_delays',
    'temporal_embedding.frequency',
    'temporal_embedding.lag',
    'temporal_embedding.kappa',
    'diffusion_maps.rank_diffusion_maps',
    'diffusion_maps.n_eigenpairs',
    'diffusion_maps.learn_kernel_params',
    'diffusion_maps.kernel_type',
    'diffusion_maps.kernel_cutoff',
    'diffusion_maps.epsilon',
    'diffusion_maps.k_neighbors',
    'diffusion_maps.delta',
    'diffusion_maps.auto_select_diffusion_maps',
    'diffusion_maps.fix_diffusion_map_coordinates',
    'rbf.use_rbf',
    'rbf.n_samples',
    'polynomial.use_polynomial',
    'polynomial.p_degree',
    # visualization controls
    'visualization.visualization_timesteps',
    'visualization.koopman_triple_selection',
    'visualization.generate_timeseries_gifs',
    'visualization.generate_mode_gifs',
    'visualization.plot_supermodes',
    'visualization.plot_diffusion_maps',
    'visualization.compute_pseudospectrum',
    'visualization.rotation',
    'visualization.vector_scaling',
}

# Paths where an unset (None) current value means "not explicitly requested by
# this run", not "explicitly set to None" -- so it must not itself count as an
# override, unlike e.g. visualization.generate_timeseries_gifs, whose None is a
# meaningful "turn this off" that a learning-stage config can set on purpose.
# Each of these is instead a property fixed once and for all by the checkpoint's
# own preprocessing run (the raw data's orientation; which series it holds), so a
# downstream config that never mentions it should inherit the checkpoint's value
# rather than reset it to the dataclass default.
_UNSET_DEFERS_TO_CHECKPOINT = {
    'visualization.rotation',
    'data.train_parameter',
    'data.test_parameter',
}


def _get_path_value(config: FullConfig, path: str) -> Any:
    """This method first accesses the subconfig, and then the value from the subconfig, for a given dot-separated path."""
    value: Any = config
    for chunk in path.split('.'):
        value = getattr(value, chunk)
    return value


def _set_path_value(config: FullConfig, path: str, value: Any) -> None:
    chunks = path.split('.')
    target: Any = config
    for chunk in chunks[:-1]:
        target = getattr(target, chunk)
    setattr(target, chunks[-1], value)


def merge_checkpoint_config(
    current_config: FullConfig,
    checkpoint_config_dict: dict[str, Any],
) -> tuple[FullConfig, MergeReport]:
    """Merge checkpoint config with current CLI config using strict allowlist overrides."""
    checkpoint_config = FullConfig.from_dict(checkpoint_config_dict)

    # Keep runtime/meta values from current invocation. # this replaces all paths at once with the new paths
    checkpoint_config.meta = current_config.meta

    applied_paths: list[str] = []
    ignored_paths: list[str] = []

    for path in sorted(_ALLOWED_OVERRIDE_PATHS):
        current_value = _get_path_value(current_config, path)
        checkpoint_value = _get_path_value(checkpoint_config, path)

        if path in _UNSET_DEFERS_TO_CHECKPOINT and current_value is None:
            ignored_paths.append(path)
            continue

        if current_value != checkpoint_value:
            _set_path_value(checkpoint_config, path, current_value)
            applied_paths.append(path)
        else:
            ignored_paths.append(path)

    # preprocessing.tlim: merged element-wise rather than through the allowlist
    # loop above (see the comment on _ALLOWED_OVERRIDE_PATHS) -- an unset (None)
    # start or end on the current run's side inherits the checkpoint's value for
    # that same element, exactly like the fields in _UNSET_DEFERS_TO_CHECKPOINT.
    resolved_tlim = [
        current if current is not None else checkpoint
        for current, checkpoint in zip(current_config.preprocessing.tlim, checkpoint_config.preprocessing.tlim)
    ]
    if resolved_tlim != checkpoint_config.preprocessing.tlim:
        checkpoint_config.preprocessing.tlim = resolved_tlim
        applied_paths.append('preprocessing.tlim')
    else:
        ignored_paths.append('preprocessing.tlim')

    # Defensive fallback: only reached if the checkpoint itself never had a
    # rotation set either (e.g. a checkpoint of a checkpoint predating this
    # default, or one loaded from an even older run).
    if checkpoint_config.visualization.rotation is None:
        checkpoint_config.visualization.rotation = 0

    return checkpoint_config, MergeReport(applied_paths=applied_paths, ignored_paths=ignored_paths)
