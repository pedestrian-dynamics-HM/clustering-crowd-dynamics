"""Experiment workflows: concrete train/retrain flows."""

import logging
from pathlib import Path
from typing import Optional
from collections.abc import Callable

import pandas as pd
from datafold import TSCDataFrame, EDMD

from ..config.configs import FullConfig
from ..config.loading import (
    find_checkpoint_path,
    load_checkpoint_model,
    load_diffusion_eigenvalues,
)
from .results import (
    ExperimentResult,
    RunMetrics,
)
from ..data import (
    load_trajectory_data_to_grid,
    read_timeseries_from_checkpoint,
)
from .mode_decompositions import (
    create_train_test_tsc,
    learn,
    get_predictions,
    FD_ACCURACY,
    analyze_diffusion_coordinates,
    refit_model_from_checkpoint_diffusion_maps,
)
from ..plots import (
    visualize_ground_truth,
    visualize_model_predictions,
    visualize_koopman_triple,
    visualize_mode_decomposition,
    visualize_diffusion_maps,
)
from .evaluation import extract_observable_types
from ..utils.persistence import (
    save_config,
    save_metrics,
    save_timeseries_data,
    save_position_timeseries,
    save_trained_model,
    save_predictions,
    save_diffusion_coordinates,
    save_koopman_spectrum,
    save_koopman_modes,
    save_koopman_eigenfunctions,
)

log = logging.getLogger(__name__)


def train_experiment(config: FullConfig) -> ExperimentResult:
    """Train a new Koopman model from raw trajectory data (or a checkpoint's, if configured)."""
    log.info(f'Starting training experiment for run {config.meta.run_id}')

    try:
        if config.data.load_data_from_checkpoint is not None:
            # config already went through resolve_checkpoint_config in bootstrap_config,
            # before this run's own config.yaml was persisted -- only the checkpoint's
            # directory (for its data files) is still needed here.
            checkpoint_path = find_checkpoint_path(config)
            timeseries_data_dict, enrichment_counters, run_metrics = _load_data_and_setup_metrics(
                config,
                from_checkpoint=True,
                checkpoint_path=checkpoint_path,
            )
        else:
            # No checkpoint: load raw trajectory files and preprocess into observable time series.
            timeseries_data_dict, enrichment_counters, run_metrics = _load_data_and_setup_metrics(config)

        # Metrics/config are always saved (see _persist_artifacts below); timeseries data only if
        # store_timeseries_data asks for it, or this isn't a checkpoint-loaded run to begin with
        # (in which case it's the only copy of the loaded data).
        should_save_timeseries = (
            config.data.load_data_from_checkpoint is None or config.preprocessing.store_timeseries_data
        )
        if should_save_timeseries:
            _persist_artifacts(
                config,
                run_metrics,
                timeseries_data_dict,
                enrichment_counters,
                save_timeseries=True,
            )

        result = run_experiment(
            config,
            timeseries_data_dict,
            enrichment_counters,
            run_metrics,
            fit_model_fn=learn,
            visualize_ground_truth_data=True,
            save_timeseries=should_save_timeseries,
        )

        log.info(f'Training experiment completed successfully for run {result.run_id}')
        return result

    except Exception as e:
        log.error(f'Training experiment failed: {e}')
        raise


def retrain_from_checkpoint(config: FullConfig) -> ExperimentResult:
    """Retrain diffusion map coordinates from a checkpoint."""
    log.info(f'Starting retrain-from-checkpoint experiment for run {config.meta.run_id}')

    if config.data.load_data_from_checkpoint is None:
        raise ValueError('retrain_from_checkpoint requires load_data_from_checkpoint to be set')

    if not config.algorithm.use_diffusion_maps:
        raise ValueError('retrain_from_checkpoint only works with diffusion maps enabled')

    try:
        # config already went through resolve_checkpoint_config in bootstrap_config,
        # before this run's own config.yaml was persisted -- only the checkpoint's
        # directory (for its data files and model) is still needed here.
        checkpoint_path = find_checkpoint_path(config)

        timeseries_data_dict, enrichment_counters, run_metrics = _load_data_and_setup_metrics(
            config,
            from_checkpoint=True,
            checkpoint_path=checkpoint_path,
        )

        log.info(f'Loading model from checkpoint {config.data.load_data_from_checkpoint}')
        edmd = load_checkpoint_model(config, checkpoint_path)
        diff_evs_df = load_diffusion_eigenvalues(config, checkpoint_path)

        # Internal invariant: load_checkpoint_model raises if the pickle is missing or unreadable,
        # so it never hands back None.
        assert edmd is not None, 'Model could not be loaded from checkpoint for retraining.'

        def fit_model_fn(train_tsc: TSCDataFrame, config: FullConfig) -> tuple[EDMD, FullConfig]:
            model, _diff_map_min_eigenvalue = refit_model_from_checkpoint_diffusion_maps(
                edmd, diff_evs_df, train_tsc, config
            )
            return model, config

        result = run_experiment(
            config,
            timeseries_data_dict,
            enrichment_counters,
            run_metrics,
            fit_model_fn=fit_model_fn,
            # Unlike train_experiment: no ground-truth plots (checkpoint's own run already made
            # them) and the raw timeseries data isn't re-saved (also already on disk).
            visualize_ground_truth_data=False,
            save_timeseries=False,
            # Preserves this flow's original behaviour: a failed test-set prediction is logged but
            # does not abort the run, so train-set results still get persisted and returned.
            reraise_test_prediction_errors=False,
        )

        log.info(f'Retrain-from-checkpoint experiment completed successfully for run {result.run_id}')
        return result

    except Exception as e:
        log.error(f'Retrain-from-checkpoint experiment failed: {e}')
        raise


def run_experiment(
    config: FullConfig,
    timeseries_data_dict: dict[int, pd.DataFrame],
    enrichment_counters: dict[int, pd.DataFrame],
    run_metrics: RunMetrics,
    fit_model_fn: Callable[[TSCDataFrame, FullConfig], tuple[EDMD, FullConfig]],
    *,
    visualize_ground_truth_data: bool = True,
    save_timeseries: bool = True,
    reraise_test_prediction_errors: bool = True,
) -> ExperimentResult:
    """Shared experiment tail, used by both train_experiment and retrain_from_checkpoint.

    Everything from turning loaded data into TSCs onward is identical between the two flows;
    ``fit_model_fn`` is the one real difference — fitting a new model (``learn``) vs. refitting a
    checkpointed one.
    """
    train_tsc, test_tsc = _create_tsc_collections(timeseries_data_dict, enrichment_counters, config)

    if visualize_ground_truth_data:
        is_checkpoint = config.data.load_data_from_checkpoint is not None
        _visualize_ground_truth(timeseries_data_dict, train_tsc, config, run_metrics, is_checkpoint=is_checkpoint)
        if test_tsc is not None:
            _visualize_ground_truth(timeseries_data_dict, test_tsc, config, run_metrics, is_checkpoint=is_checkpoint)

    if config.preprocessing.preprocess_only:
        # Everything the data-producing half of a run makes is on disk at this point;
        # what follows is the model, so stop here.
        log.info('preprocess_only is set: stopping before the model is fitted.')
        _persist_artifacts(
            config,
            run_metrics,
            timeseries_data_dict,
            enrichment_counters,
            save_timeseries=save_timeseries,
        )
        return ExperimentResult(
            run_id=config.meta.run_id,
            config=config,
            metrics=run_metrics,
            train_tsc=train_tsc,
            test_tsc=test_tsc,
        )

    model, config = fit_model_fn(train_tsc, config)

    if config.evaluation.store_trained_model:
        model_path = config.meta.unique_run_path / 'models' / f'{config.algorithm.learning_algo}_model.pkl'
        save_trained_model(model, model_path)
        log.info(f'Trained model saved to {model_path}...')

    train_tsc_predicted, time_values = _generate_predictions(
        train_tsc,
        model,
        config,
        run_metrics.prediction_timeshift,
        is_test=False,
    )
    mae_train = _visualize_predictions(train_tsc, train_tsc_predicted, config, run_metrics, is_test=False)

    if config.algorithm.use_diffusion_maps:
        diff_maps_df, diff_embedding_df = analyze_diffusion_coordinates(model, config)
        save_diffusion_coordinates(diff_maps_df, diff_embedding_df, config.meta.processed_data_path)
        run_metrics.diff_map_min_eigenvalue = float(diff_maps_df['eigenvalue'].min())

        if config.visualization.plot_diffusion_maps:
            visualize_diffusion_maps(
                run_config=config,
                run_metrics=run_metrics,
                diff_maps_df=diff_maps_df,
                diff_embedding_df=diff_embedding_df,
            )

    _visualize_koopman_triple(train_tsc, model, config, run_metrics, time_values)
    _visualize_mode_decomposition(train_tsc, train_tsc_predicted, model, config, run_metrics)

    run_metrics.mae_per_observable = [float(v) for v in mae_train]
    run_metrics.mae_max = float(max(mae_train)) if mae_train else None

    test_tsc_predicted = None
    if test_tsc is not None:
        try:
            test_tsc_predicted, _ = _generate_predictions(
                test_tsc,
                model,
                config,
                run_metrics.prediction_timeshift,
                is_test=True,
            )

            mae_test = _visualize_predictions(test_tsc, test_tsc_predicted, config, run_metrics, is_test=True)

            run_metrics.mae_per_observable_test = [float(v) for v in mae_test]
            run_metrics.mae_max_test = float(max(mae_test)) if mae_test else None
            log.info(f'Test MAE: max={run_metrics.mae_max_test}')

        except Exception as e:
            log.error(f'Error during test prediction: {e}')
            if reraise_test_prediction_errors:
                raise

    # TODO: Only metrics.
    _persist_artifacts(
        config,
        run_metrics,
        timeseries_data_dict,
        enrichment_counters,
        save_timeseries=save_timeseries,
    )

    # TODO: Only first three parameter are enough for return.
    return ExperimentResult(
        run_id=config.meta.run_id,
        config=config,
        metrics=run_metrics,
        model=model,
        train_tsc=train_tsc,
        test_tsc=test_tsc,
        train_tsc_predicted=train_tsc_predicted,
        test_tsc_predicted=test_tsc_predicted,
    )


def _load_data_and_setup_metrics(
    config: FullConfig,
    from_checkpoint: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], RunMetrics]:
    """Load data and compute initial metrics."""
    log.info('Loading data...')

    if from_checkpoint:
        if checkpoint_path is None:
            raise ValueError('checkpoint_path must be provided when loading from a checkpoint')
        timeseries_data_dict, enrichment_counters = read_timeseries_from_checkpoint(
            config,
            checkpoint_path,
        )
    else:
        # data_files and measurement_grid_size are not needed here; the grid
        # shape is derived from the loaded data below.
        timeseries_data_dict, _, _, enrichment_counters, position_data_dict = load_trajectory_data_to_grid(config)
        if config.preprocessing.store_position_ts:
            save_position_timeseries(position_data_dict, config.meta.processed_data_path)

    if not config.data.train_parameter:
        train_params = [key for key in timeseries_data_dict.keys() if str(key).startswith('100')]
        config.data.train_parameter = list(map(int, train_params))

    if not config.data.test_parameter:
        test_params = [key for key in timeseries_data_dict.keys() if str(key).startswith('200')]
        config.data.test_parameter = list(map(int, test_params))

    max_x = max(int(df['x_idx'].max()) for df in timeseries_data_dict.values())
    max_y = max(int(df['y_idx'].max()) for df in timeseries_data_dict.values())
    grid_shape = [max_y + 1, max_x + 1]

    visualization_timesteps, embedding_timeshift, prediction_timeshift = _finalize_dependent_configs(
        timeseries_data_dict, config
    )

    run_metrics = RunMetrics(
        grid_shape=grid_shape,
        visualization_timesteps=visualization_timesteps,
        measurement_cells=list(config.evaluation.measurement_cells),
        embedding_timeshift=embedding_timeshift,
        prediction_timeshift=prediction_timeshift,
        grid_resolution=list(config.preprocessing.grid_resolution),
        system_observables=list(config.preprocessing.system_observables),
        train_parameter=config.data.train_parameter or [],
        test_parameter=config.data.test_parameter or [],
    )

    log.info(f'Loaded data for {len(timeseries_data_dict)} time series')
    return timeseries_data_dict, enrichment_counters, run_metrics


def _finalize_dependent_configs(
    timeseries_data_dict: dict[int, pd.DataFrame],
    config: FullConfig,
) -> tuple[list[int], int, int]:
    """Calculate derived config values (visualization timesteps, embedding/prediction timeshifts)."""
    embedding_timeshift = 0
    visualization_timeshift = 0
    prediction_timeshift = config.preprocessing.prediction_timeshift

    if config.algorithm.use_temporal_embedding:
        embedding_timeshift = (
            (config.temporal_embedding.num_t_delays - 1) * config.temporal_embedding.frequency
            + config.temporal_embedding.lag
            + 1
        )
        visualization_timeshift += embedding_timeshift
        log.info(
            f'Number of frames needed for temporal embedding in {config.algorithm.learning_algo.upper()}:'
            f' {embedding_timeshift} frames.'
        )
        log.debug(
            f'{config.algorithm.use_temporal_embedding=}, {config.temporal_embedding.num_t_delays=},'
            f' {config.temporal_embedding.frequency=}'
        )

    if config.diffusion_maps.kernel_type == 'temporal_nn':
        visualization_timeshift += FD_ACCURACY
        embedding_timeshift += FD_ACCURACY

    if config.algorithm.learning_algo in ['s_dmd', 's_edmd']:
        visualization_timeshift += embedding_timeshift
        prediction_timeshift += embedding_timeshift

    visualization_timeshift += prediction_timeshift

    if visualization_timeshift > 0:
        first_frame = config.visualization.visualization_timesteps[0]
        visualization_timesteps = [first_frame] + [
            v + visualization_timeshift for v in config.visualization.visualization_timesteps
        ]
        log.info(
            f'Visualization timesteps adjusted to {visualization_timesteps}, but kept the first frame {first_frame}.'
        )
    else:
        visualization_timesteps = config.visualization.visualization_timesteps

    max_timestep_in_data = max([run_data['timeStep'].max() for run_data in timeseries_data_dict.values()])
    if max_timestep_in_data < max(visualization_timesteps):
        raise ValueError(
            f'The requested visualization timestep {max(visualization_timesteps)} exceeds the last timestep '
            f'in the data ({max_timestep_in_data}). Lower visualization_timesteps, or widen tlim so that '
            f'enough frames are loaded.'
        )

    return visualization_timesteps, embedding_timeshift, prediction_timeshift


def _create_tsc_collections(
    timeseries_data_dict: dict[int, pd.DataFrame],
    enrichment_counters: dict[int, pd.DataFrame],
    config: FullConfig,
) -> tuple[TSCDataFrame, Optional[TSCDataFrame]]:
    """Transform raw data into time-series collections."""
    log.info('Creating time-series collections...')

    train_tsc, test_tsc = create_train_test_tsc(
        data=timeseries_data_dict,
        system_observables=config.preprocessing.system_observables,
        train_parameter=config.data.train_parameter or [],
        test_parameter=config.data.test_parameter or [],
        enrichment=enrichment_counters,
    )

    log.info(f'Created train TSC with {train_tsc.n_features} features and {len(train_tsc)} samples')
    if test_tsc is not None:
        log.info(f'Created test TSC with {test_tsc.n_features} features and {len(test_tsc)} samples')

    return train_tsc, test_tsc


def _generate_predictions(
    tsc: TSCDataFrame,
    model: EDMD,
    config: FullConfig,
    prediction_timeshift: int,
    is_test: bool = False,
) -> tuple[TSCDataFrame, list[int]]:
    """Generate predictions on a time-series collection."""
    label = 'test' if is_test else 'train'
    log.info(f'Generating predictions for {label} data...')

    tsc_predicted, time_values = get_predictions(
        tsc=tsc,
        model=model,
        extra_prediction_steps=config.evaluation.extra_prediction_steps,
        prediction_timeshift=prediction_timeshift,
    )

    if config.preprocessing.store_tsc_data:
        save_predictions(tsc, tsc_predicted, config.meta.processed_data_path, is_test=is_test)
        log.info(f'Ground truth and predicted time series saved to {config.meta.processed_data_path}.')

    log.info(f'Generated predictions for {label} data')
    return tsc_predicted, list(time_values)


def _persist_artifacts(
    config: FullConfig,
    metrics: RunMetrics,
    timeseries_data_dict: dict[int, pd.DataFrame],
    enrichment_counters: dict[int, pd.DataFrame],
    save_timeseries: bool = True,
) -> None:
    """Save config, metrics, and optionally timeseries data."""
    log.info('Persisting artifacts...')

    save_config(config, config.meta.unique_run_path / 'config.yaml')
    save_metrics(metrics, config.meta.unique_run_path / 'metrics.yaml')

    if save_timeseries:
        save_timeseries_data(
            timeseries_data_dict,
            enrichment_counters,
            config.meta.processed_data_path,
        )

    log.info('Artifacts persisted')


def _visualize_ground_truth(
    ts_data_dict: dict[int, pd.DataFrame],
    tsc: TSCDataFrame,
    config: FullConfig,
    run_metrics: RunMetrics,
    is_checkpoint: bool = False,
) -> None:
    """Visualize input data without predictions."""
    if not config.visualization.plot_ground_truth:
        return

    log.info('Visualizing ground truth data...')
    observable_types = extract_observable_types(run_metrics.system_observables)

    visualize_ground_truth(
        ts_data_dict=ts_data_dict,
        tsc=tsc,
        run_config=config,
        run_metrics=run_metrics,
        observable_types=observable_types,
        is_checkpoint=is_checkpoint,
    )


def _visualize_predictions(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    config: FullConfig,
    run_metrics: RunMetrics,
    is_test: bool = False,
) -> list[float]:
    """Visualize predictions and compute error metrics."""
    label = 'test' if is_test else 'train'
    log.info(f'Visualizing {label} predictions...')

    requested_timesteps = list(run_metrics.visualization_timesteps)

    def _common_time_values(tsc_data: TSCDataFrame) -> set[int]:
        common: set[int] | None = None
        for series_id in tsc_data.ids:
            series_times = set(tsc_data.loc[series_id].index.tolist())
            common = series_times if common is None else common.intersection(series_times)
        return common or set()

    true_time_values = _common_time_values(tsc)
    pred_time_values = _common_time_values(tsc_predicted)
    valid_timesteps = [t for t in requested_timesteps if t in true_time_values and t in pred_time_values]

    if not valid_timesteps:
        log.warning(
            'Skipping %s prediction visualization: none of the requested timesteps %s are available '
            'in both true and predicted data.',
            label,
            requested_timesteps,
        )
        return []

    if len(valid_timesteps) != len(requested_timesteps):
        log.warning(
            'Adjusted %s visualization timesteps from %s to %s based on available prediction times.',
            label,
            requested_timesteps,
            valid_timesteps,
        )

    original_timesteps = run_metrics.visualization_timesteps
    run_metrics.visualization_timesteps = valid_timesteps
    observable_types = extract_observable_types(run_metrics.system_observables)

    try:
        mae_values = visualize_model_predictions(
            tsc=tsc,
            tsc_predicted=tsc_predicted,
            run_config=config,
            run_metrics=run_metrics,
            observable_types=observable_types,
            is_test=is_test,
        )
    finally:
        run_metrics.visualization_timesteps = original_timesteps

    log.info(f'Completed {label} prediction visualization')
    return mae_values


def _visualize_koopman_triple(
    tsc: TSCDataFrame,
    model,
    config: FullConfig,
    run_metrics: RunMetrics,
    time_values: list[int],
) -> None:
    """Visualize the Koopman triple (eigenvalues, eigenfunctions, modes) and persist its numbers."""
    log.info('Visualizing Koopman triple...')

    observable_types = extract_observable_types(run_metrics.system_observables)

    spectrum = visualize_koopman_triple(
        run_config=config,
        run_metrics=run_metrics,
        tsc=tsc,
        observable_types=observable_types,
        model=model,
        time_values=time_values,
    )

    save_koopman_spectrum(spectrum.eigenvalues, spectrum.residuals, config.meta.processed_data_path)
    if spectrum.modes and spectrum.selection is not None:
        save_koopman_modes(spectrum.modes, spectrum.selection, config.meta.processed_data_path)

    log.info('Koopman triple visualization completed')


def _visualize_mode_decomposition(
    tsc: TSCDataFrame,
    tsc_predicted: TSCDataFrame,
    model,
    config: FullConfig,
    run_metrics: RunMetrics,
) -> None:
    """Visualize the mode decomposition per measurement cell and persist its eigenfunctions."""
    log.info('Visualizing mode decomposition...')

    eigenfunctions, initial_state_idx = visualize_mode_decomposition(
        run_config=config,
        run_metrics=run_metrics,
        tsc=tsc,
        tsc_predicted=tsc_predicted,
        model=model,
    )

    if eigenfunctions is not None and initial_state_idx is not None:
        save_koopman_eigenfunctions(eigenfunctions, initial_state_idx, config.meta.processed_data_path)

    log.info('Mode decomposition visualization completed')
