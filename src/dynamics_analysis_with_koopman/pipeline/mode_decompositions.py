import logging
import time

import numpy as np
import pandas as pd
from datafold import (
    pcfold,
    TSCDataFrame,
    EDMD,
    DMDStandard,
    mpDMD,
    sDMD,
    TSCIdentity,
    TSCFeaturePreprocess,
    TSCTakensEmbedding,
    TSCSampledNetwork,
    DiffusionMaps,
    GaussianKernel,
    MultiquadricKernel,
    InverseMultiquadricKernel,
    ContinuousNNKernel,
    ConeKernel,
    LocalRegressionSelection,
    TSCPolynomialFeatures,
    TSCRadialBasis,
)
from scipy.sparse import csr_matrix
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
from sklearn.pipeline import Pipeline

try:
    from swimnetworks import Dense
except ImportError:
    Dense = None

from ..config.configs import FullConfig

log = logging.getLogger(__name__)

#########################
####### Constants #######
#########################

DEFAULT_SURROGATE_DIM = 3
FD_ACCURACY = 4


#######################
####### Methods #######
#######################


def _data_to_timeseries(data: dict[int, pd.DataFrame], system_observables: list[str]) -> dict[int, pd.DataFrame]:
    """
    Transform the data frame, such that x and y values receive a unique column name.
    In any case, the observable is appended to the column name. This allows to have multiple
    features per grid cell that can be learned simultanously.
    """
    timeseries = {}
    for key, df in data.items():
        df = df.pivot(index='timeStep', columns=['x_idx', 'y_idx'], values=system_observables)
        if df.isnull().values.any():
            log.error(f'The dataframe for series id {key} contains nan values. Please check your timeseries files.')
            log.error(f'This may be caused by an timestep with missing values for some cells.')
        df.columns = [f'({x_idx}_{y_idx})_{feature}' for feature, x_idx, y_idx in df.columns]
        timeseries[key] = df
    return timeseries


def create_train_test_tsc(
    data: dict[int, pd.DataFrame],
    system_observables: list[str],
    train_parameter: list[int],
    test_parameter: list[int],
    enrichment: dict[int, pd.DataFrame] | None = None,
) -> tuple[TSCDataFrame, TSCDataFrame | None]:
    """
    Create a TSCDataFrame from the data dictionary and system observables.
    The data dict keys are used as time series ids (unique integer values).
    """

    log.info(f'Create time series collection for model training.')
    # create timeseries with one feature per grid cell and system observable combination
    timeseries_from_data = _data_to_timeseries(data=data, system_observables=system_observables)

    if enrichment:
        log.info('Enrichment processing for timestep level information...')
        # check that we have the same keys in enrichment and timeseries_from_data
        enrichment_keys, timeseries_keys = set(enrichment.keys()), set(timeseries_from_data.keys())
        if enrichment_keys != timeseries_keys:
            raise ValueError(
                f'Enrichment keys and timeseries keys must match. '
                f'Missing from timeseries: {enrichment_keys - timeseries_keys}. '
                f'Missing from enrichment: {timeseries_keys - enrichment_keys}.'
            )
        for series_id, df in enrichment.items():
            # join with the timeseries data, will drop all time values < 0 and outside of the tlim automatically
            timeseries_from_data[series_id] = timeseries_from_data[series_id].join(
                df.set_index('timeStep'), on='timeStep', how='left'
            )
            if any(timeseries_from_data[series_id].isna().any(axis=0)) or any(
                timeseries_from_data[series_id].isna().any(axis=1)
            ):
                # this means we have nan values either in the column or the rows
                timeseries_from_data[series_id].dropna(inplace=True)
                log.warning(
                    f'Found nan values after joining obsrvables with the enrichment counters, '
                    'dropping these time steps.'
                )
                log.warning('Please check if this is expected behaviour or a data quality issue.')
        log.info(f'Added {df.columns=} columns to data of {series_id} for {len(df)} time steps')

    # must be sorted, otherwise error in index uiqueness check (collections.py, line 903)!
    train_parameter.sort()

    # convert both time series to a single "time series collection" (TSCDataFrame)
    train_tsc = TSCDataFrame.from_frame_list(
        [timeseries_from_data[ts_id] for ts_id in train_parameter], ts_ids=train_parameter
    )
    log.info(f'Created train (and test) data with the following series ids and lengths: ')
    log.info(f'{train_tsc.n_timesteps=}')

    if len(test_parameter) > 0:  # may be empty list
        test_parameter.sort()
        test_tsc = TSCDataFrame.from_frame_list(
            [timeseries_from_data[ts_id] for ts_id in test_parameter], ts_ids=test_parameter
        )
        log.info(f'{test_tsc.n_timesteps=}')
    else:
        test_tsc = None
        log.info('No testing data created.')

    return train_tsc, test_tsc


def _count_entries_in_sparse_matrix(A, threshold=1e-8, is_symmetric=True):
    # Ensure the matrix is in CSC format for fast column-wise operations
    # matrix may be a TSCDataFrame or numpy array
    A = csr_matrix(A)

    # Count all stored (nonzero) values per column
    col_nonzero_counts = A.getnnz(axis=0)

    # Count values greater than the threshold per column
    col_greater_than_threshold_counts = np.array(
        [np.sum(A.data[A.indptr[j] : A.indptr[j + 1]] > threshold) for j in range(A.shape[1])]
    )

    log.debug(f'Number of non-zero entries per column: {col_nonzero_counts}')
    log.debug(f'Number of values greater than {threshold} per column: {col_greater_than_threshold_counts}')

    if not is_symmetric:
        row_nonzero_counts = A.getnnz(axis=1)
        row_greater_than_threshold_counts = np.array(
            [np.sum(A.data[A.indptr[i] : A.indptr[i + 1]] > threshold) for i in range(A.shape[0])]
        )

        log.debug(f'Number of non-zero entries per row: {row_nonzero_counts}')
        log.debug(f'Number of values greater than {threshold} per row: {row_greater_than_threshold_counts}')


def _build_kernel(config: FullConfig):
    """Build the kernel used by diffusion maps and/or the radial basis dictionary step."""
    if config.diffusion_maps.kernel_type == 'gaussian':
        return GaussianKernel(
            epsilon=config.diffusion_maps.epsilon,
            distance=dict(cut_off=config.diffusion_maps.kernel_cutoff),  # 5.0
        )
    elif config.diffusion_maps.kernel_type == 'multiquadric':
        return MultiquadricKernel(
            epsilon=config.diffusion_maps.epsilon,
        )
    elif config.diffusion_maps.kernel_type == 'inv_multiquadric':
        return InverseMultiquadricKernel(
            epsilon=config.diffusion_maps.epsilon,
        )
    elif config.diffusion_maps.kernel_type == 'continuous_nn':
        return ContinuousNNKernel(
            k_neighbor=config.diffusion_maps.k_neighbors,  # determines the length scale,
            delta=config.diffusion_maps.delta,  # unit-less scale parameter.
        )
    elif config.diffusion_maps.kernel_type == 'temporal_nn':
        return ConeKernel(
            zeta=0,
            epsilon=config.diffusion_maps.epsilon,  # 0.1
            fd_accuracy=FD_ACCURACY,  # evaluation accuracy of the finite difference approximation
        )
    else:
        raise NotImplementedError(f'Kernel type {config.diffusion_maps.kernel_type} not implemented.')


def _build_dictionary_steps(
    tsc: TSCDataFrame, config: FullConfig
) -> tuple[list[tuple[str, object]], object | None, object | None, dict]:
    """
    Build the EDMD dictionary transformation steps (temporal rescaling, temporal embedding,
    diffusion maps/rbf, polynomial/hidden layer, auto-selection) for the edmd/mp_edmd/s_edmd
    learning algorithms.
    Returns (dict_steps, kernel, diffusion_maps, fit_kwargs) — kernel and diffusion_maps are None
    unless the diffusion-maps-or-rbf branch actually built them; fit_kwargs holds the extra
    fit-time arguments steps such as temporal embedding or diffusion maps require.
    """
    dict_steps: list[tuple[str, object]] = []
    fit_kwargs: dict = {}
    kernel = None
    diffusion_maps = None

    # add the temporal rescaling if specified
    if config.algorithm.use_temporal_data_scaling is not None:
        log.info(f'Setup temporal rescaling of type {config.algorithm.use_temporal_data_scaling}.')
        dict_steps.append(('scale', TSCFeaturePreprocess.from_name(name=config.algorithm.use_temporal_data_scaling)))

    # add temporal embedding if specified
    if config.algorithm.use_temporal_embedding:
        log.info(
            f'Setup temporal embedding with {config.temporal_embedding.num_t_delays=}, {config.temporal_embedding.kappa=}, {config.temporal_embedding.frequency=} and {config.temporal_embedding.lag=}.'
        )
        # prepare the temporal embedding such that it can be used in edmd
        temporal_embedding = TSCTakensEmbedding(
            delays=config.temporal_embedding.num_t_delays,
            kappa=config.temporal_embedding.kappa,
            frequency=config.temporal_embedding.frequency,
            lag=config.temporal_embedding.lag,
        )
        temporal_embedding.fit(tsc)
        temporal_embedding.transform(tsc)

        # add the time delay to the edmd steps
        dict_steps.append(('timedelay', temporal_embedding))
        embedding_indices = list(-temporal_embedding.delay_indices_)
        embedding_indices.reverse()
        log.info(f'Temporal embedding created, uses temporal embedding indices {embedding_indices + [0]}.')

        if config.algorithm.learning_algo in ['s_edmd']:
            # if there is no temporal embedding, this will just default to 1
            # even when using diffusion maps, to ensure independence of the observable, we need to use the
            # full temporal embedding to shift in the stochastic edmd
            fit_kwargs.update(
                {
                    'dmd__embedding_timeshift': (config.temporal_embedding.num_t_delays - 1)
                    * config.temporal_embedding.frequency
                    + config.temporal_embedding.lag
                    + 1,
                    'dmd__has_measurement_noise': config.algorithm.has_measurement_noise,
                }
            )

    # add diffusion maps if specified
    if config.algorithm.use_diffusion_maps or config.rbf.use_rbf:
        if config.algorithm.use_diffusion_maps:
            log.info(f'Setup diffusion maps with {config.diffusion_maps.n_eigenpairs=}')
            if config.diffusion_maps.learn_kernel_params:
                X_pcm = pcfold.PCManifold(tsc)
                config.diffusion_maps.kernel_cutoff, config.diffusion_maps.epsilon = X_pcm.optimize_parameters()
                log.info(
                    f'Learned kernel parameters from the data: {config.diffusion_maps.epsilon=}, {config.diffusion_maps.kernel_cutoff=}'
                )
            else:
                log.info(
                    f'Use fixed kernel parameters: {config.diffusion_maps.epsilon=}, {config.diffusion_maps.kernel_cutoff=}.'
                )
                if log.isEnabledFor(logging.DEBUG):
                    ks = [3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45]
                    log.debug(f'Compare to parameters for {ks=} nearest neighbors (independent of n_eigenpairs)')
                    X_pcm = pcfold.PCManifold(tsc)
                    for k in ks:
                        cutoff, epsilon = X_pcm.optimize_parameters(k=k)
                        log.debug(f'For {k=} nearest neighbors: {cutoff=} and {epsilon=}')

            kernel = _build_kernel(config)  # needs to be here to use the learned kernel parameters
            log.debug(f'Setup diffusion maps using {kernel=}')
            diffusion_maps = DiffusionMaps(
                kernel=kernel,
                n_eigenpairs=config.diffusion_maps.n_eigenpairs,
            )

            # need to do this here, so the fitting later is done in accordance to number of set target coordinates
            if config.diffusion_maps.fix_diffusion_map_coordinates is not None:
                log.info(
                    f'Select fixed diffusion map coordinates after fitting the model: {config.diffusion_maps.fix_diffusion_map_coordinates}'
                )
                # diffusion_maps.fit(tsc)
                diffusion_maps.set_target_coords(config.diffusion_maps.fix_diffusion_map_coordinates)

            dict_steps.append(('diffusion_maps', diffusion_maps))
            fit_kwargs.update({'diffusion_maps__store_kernel_matrix': True})

        elif config.rbf.use_rbf:
            # creates one additional coordinate per sample / initial condition
            kernel = _build_kernel(config)
            log.debug(f'Setup radial basis using {kernel=}')
            radial_basis = TSCRadialBasis(
                kernel=kernel,
                center_type='random',  # or "initial_condition", "random", ""all_data""
                n_samples=config.rbf.n_samples,  # needed for "random", default at 100
            )
            dict_steps.append(('rbf', radial_basis))

    if config.polynomial.use_polynomial:
        polynomial = TSCPolynomialFeatures(degree=config.polynomial.p_degree)
        dict_steps.append(('polynomial', polynomial))

    # add hidden layer if specified
    elif config.swim.use_sampled_hidden_layer:
        # create pipeline as usual with sklearn
        steps = [
            (
                'hidden',
                Dense(
                    layer_width=config.swim.layer_width,
                    activation=config.swim.activation,
                    parameter_sampler=config.swim.param_sampler,
                    random_seed=config.swim.seed,
                ),
            )
        ]
        network_dictionary = Pipeline(steps=steps)

        # only pipeline needs to be provided, input and output dimensionality will be inferred from the data
        sampled_layer = TSCSampledNetwork(nn=network_dictionary)

        dict_steps.append(('sampled_layer', sampled_layer))

    # Internal invariant: check_config_consistency already rejects an EDMD config with no
    # feature enabled, so reaching here with an empty pipeline would be a bug in this function.
    assert len(dict_steps) > 0, 'At least one transformation step must be specified for EDMD.'

    if config.diffusion_maps.auto_select_diffusion_maps is not None:
        log.info('Setup local regression for diffusion map coordinate selection.')
        if config.diffusion_maps.auto_select_diffusion_maps >= 1:
            log.info(
                f'Select diffusion map coordinates based on intrinsic dimensionality {config.diffusion_maps.auto_select_diffusion_maps}.'
            )
            dict_steps.append(
                (
                    'loc_regress',
                    LocalRegressionSelection(
                        strategy='dim', intrinsic_dim=int(config.diffusion_maps.auto_select_diffusion_maps)
                    ),
                )
            )
        elif config.diffusion_maps.auto_select_diffusion_maps > 0:  # and smaller than 1
            log.info(
                f'Select diffusion map coordinates based on residual threshold {config.diffusion_maps.auto_select_diffusion_maps}.'
            )
            dict_steps.append(
                (
                    'loc_regress',
                    LocalRegressionSelection(
                        strategy='threshold', regress_threshold=config.diffusion_maps.auto_select_diffusion_maps
                    ),
                )
            )

    return dict_steps, kernel, diffusion_maps, fit_kwargs


def _build_dmd(config: FullConfig):
    """Select and construct the DMD estimator that sits at the heart of the EDMD model."""
    # use standard dmd or mpDMD class for the "inner" DMD;
    if config.algorithm.learning_algo in ['mp_edmd', 'mp_dmd']:
        DMD_cls = mpDMD
    elif config.algorithm.learning_algo in ['s_dmd', 's_edmd']:
        DMD_cls = sDMD
    else:  # this is dmd or edmd
        DMD_cls = DMDStandard

    if config.algorithm.pseudospectrum_residual_filter is not None:
        log.info(
            f'Activate filtering to reduce spectral pollution with {config.algorithm.pseudospectrum_residual_filter=}'
        )
        # set the rank and residual filter for the pseudospectrum
        return DMD_cls(
            rank=config.algorithm.rank,
            sys_mode='spectral',  # default, needed for pseudospectrum computation
            residual_filter={**config.algorithm.pseudospectrum_residual_filter, **{'store_residuals': True}},
            # store them on them model (edmd.dmd_model.residuals_)
            compute_pseudospectrum=True,
        )
    elif config.visualization.compute_pseudospectrum:
        log.info(f'Turn on spectral mode and keep data inner product matrixes for residual calculation.')
        # set the rank and turn on pseudospectrum to store relevant DMD matrices
        return DMD_cls(
            rank=config.algorithm.rank,
            sys_mode='spectral',  # default, needed for pseudospectrum computation
            compute_pseudospectrum=True,
        )
    else:
        return DMD_cls(
            rank=config.algorithm.rank,
        )


def learn(tsc: TSCDataFrame, config: FullConfig) -> tuple[EDMD, FullConfig]:
    """
    Execute EDMD and DMD learning on the given time series collection.
    In the case of EDMD, temporal embedding and diffusion maps can be added to the pipeline.
    NOTE: Temporal embedding significantly increases the state space dimensionality.
    Hence, a combination with diffusion maps for dimensionality reduction is recommended.
    """
    log.info('Setup learning algorithm...')
    edmd_kwargs = {
        'include_id_state': False,
        'sort_koopman_triplets': config.algorithm.sort_koopman_triplets,
        'verbose': True,
        'stepwise_transform': config.evaluation.stepwise_transform,
    }

    kernel = None
    diffusion_maps = None

    if config.algorithm.learning_algo in ['edmd', 'mp_edmd', 's_edmd']:  # this means we add dictionary steps
        dict_steps, kernel, diffusion_maps, fit_kwargs = _build_dictionary_steps(tsc, config)

    elif config.algorithm.learning_algo in ['dmd', 'mp_dmd', 's_dmd']:
        # constrcut model pipeline without any additional transformation steps
        log.info(f'Setup DMD of rank {config.algorithm.rank} without temporal embedding or diffusion maps.')
        dict_steps = [('_id', TSCIdentity())]
        fit_kwargs = {}

    else:
        # just for completness, is already checked in config asserts
        raise NotImplementedError(f'Learning algorithm {config.algorithm.learning_algo} not implemented.')

    # configure the DMD which is at the heart of the learning process
    log.info(
        f'Setup {config.algorithm.learning_algo.upper()} with {dict_steps=}, {config.algorithm.rank=}, {config.algorithm.sort_koopman_triplets=}, {config.evaluation.stepwise_transform=}'
    )
    dmd = _build_dmd(config)

    # put everything together in the EDMD model
    model = EDMD(
        dict_steps=dict_steps,  # could be only id function or a transformation
        dmd_model=dmd,
        **edmd_kwargs,
    )

    # execute learning process
    log.info('Execute learning...')
    log.info(f'...using the following pipeline: {dict_steps=}')
    now = time.time()
    # assert isinstance(tsc.n_timesteps, int), f"The time series have different length, this should not be the case."
    try:
        model.fit(tsc, **fit_kwargs)

    except ArpackNoConvergence as e:
        log.error('An error occured during the learning process.')
        log.error(f'ArpackNoConvergence: {e}')
        log.error(
            f'This error means that the kernel matrix cannot be approximated by the allowed number of eigenpairs {config.diffusion_maps.n_eigenpairs=}.'
        )
        log.error(
            f'Try to increase the rank of the kernel matrix by increasing the value for {config.diffusion_maps.epsilon=}.'
        )
        raise e

    except ZeroDivisionError as e:
        log.error('An error occured during the learning process.')
        log.error(f'ZeroDivisionError: {e}')
        log.error(f'Settings used: {config.algorithm.use_diffusion_maps=} with {kernel=}.')
        log.error(f'Try to increase the value for the neighbors in case of ContinousNNKernel.')
        raise e

    log.info(f'Model fitted with {config.algorithm.learning_algo.upper()}.')
    log.info(f'Learning took {time.time() - now:.1f} seconds overall.')
    log.debug((model.feature_names_pred_))  # column names for the state space
    log.debug(model.n_samples_ic_)  # number of samples used for initial condition

    if diffusion_maps is not None and log.isEnabledFor(logging.DEBUG):
        _count_entries_in_sparse_matrix(
            diffusion_maps.kernel_matrix_, threshold=1e-8, is_symmetric=diffusion_maps.kernel.is_symmetric
        )  # type: ignore

    return model, config


def get_predictions(
    tsc: TSCDataFrame,
    model: EDMD,
    extra_prediction_steps: int,
    prediction_timeshift: int,
) -> tuple[TSCDataFrame, np.ndarray]:
    frames_for_initial_condition = model.n_samples_ic_ if model.n_samples_ic_ is not None else 1

    # easier to predict for the full longest system trajectories
    max_time_value = tsc.time_values()[-1]  # gives union of all time values
    time_values = np.arange(
        frames_for_initial_condition + prediction_timeshift, max_time_value + extra_prediction_steps + 1
    )

    # get predictions from the model (all time series at once)
    if frames_for_initial_condition == 1:
        tsc_predicted = model.predict(
            tsc.select_time_values(prediction_timeshift),  # type: ignore
            time_values=time_values,
        )
        log.info(
            f'Predicted time series with initial condition of a single frame, with initial condition at state {prediction_timeshift}'
        )
    else:  # are > 1
        tsc_predicted = model.predict(
            tsc.select_time_values(prediction_timeshift + np.arange(frames_for_initial_condition)),  # type: ignore
            time_values=time_values,
        )
        log.info(
            f'Predicted time series from first state {time_values[0]}, with initial condition reaching across {frames_for_initial_condition} frames.'
        )

    # select only the time values + extra prediction steps per series
    try:
        multi_index = pd.MultiIndex.from_tuples(
            [
                (ts_id, t)
                for ts_id in tsc_predicted.ids
                for t in np.arange(
                    tsc_predicted.loc[ts_id].index[0],  # minimum time
                    tsc.n_timesteps.loc[ts_id] + extra_prediction_steps,
                )  # max time
            ],
            names=['ID', 'time'],
        )
    except AttributeError:  # for tsc.n_timesteps, which is just an int
        multi_index = pd.MultiIndex.from_tuples(
            [
                (ts_id, t)
                for ts_id in tsc_predicted.ids
                for t in np.arange(
                    tsc_predicted.loc[ts_id].index[0],  # minimum time
                    tsc.n_timesteps + extra_prediction_steps,
                )  # max time
            ],
            names=['ID', 'time'],
        )
    tsc_predicted = tsc_predicted.loc[multi_index]

    log.debug(tsc_predicted.initial_states(2))
    return tsc_predicted, time_values


def analyze_diffusion_coordinates(model: EDMD, config: FullConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the diffusion map coordinates from a given EDMD model.
    """
    log.info('Select the diffusion map coordinates.')

    # add diffusion maps if specified
    log.info(
        f'Setup diffusion maps with {config.diffusion_maps.epsilon=}, {config.diffusion_maps.kernel_cutoff=}, {config.diffusion_maps.n_eigenpairs=}'
    )
    log.info(f'Try to get the diffusion maps from the model.')
    diffusion_maps = None
    for _, step in model.steps:
        if isinstance(step, DiffusionMaps):
            diffusion_maps = step
            break

    if diffusion_maps is None:
        raise ValueError(
            'The model does not contain a step utilizing diffusion maps, even though the config asks for '
            'a diffusion map analysis. Please check the model setup and the learning process.'
        )

    # get diffusion map coordinates and eigenvalues, considering the target coordinates (if set)
    diff_evecs, diff_eigenvals = diffusion_maps._select_eigenpairs_target_coords()
    feature_names_out = diffusion_maps.get_feature_names_out()
    evec_names = [name.replace('dmap', 'diff_ev_') for name in feature_names_out]
    diff_map_names = [name.replace('dmap', 'diff_coord_') for name in feature_names_out]

    #### Get infos per diffusion map
    # store the diffusion map specific infos (one row per embedding dimension)
    diff_maps_df = pd.DataFrame(
        {'index': range(len(diff_eigenvals)), 'label': evec_names, 'eigenvalue': diff_eigenvals}
    )

    # execute the ranking for each diffusion map and rank their relevance
    if config.diffusion_maps.rank_diffusion_maps:  # this is very expensive, so turn it on explicitly
        log.info('Try to find the most relevant diffusion map coordinates.')
        start_time = time.time()
        loc_regress = LocalRegressionSelection(
            strategy='dim', intrinsic_dim=DEFAULT_SURROGATE_DIM
        )  # strategy="threshold", regress_threshold=0.9)
        loc_regress.fit(diffusion_maps.eigenvectors_)
        log.info(f'Local regression took {time.time() - start_time:.1f} seconds.')
        log.info(
            f'Identified most informative indices are: {loc_regress.evec_indices_}'
        )  # three indices with largest residual
        # add it to the dataframe for saving
        diff_maps_df['residuals'] = loc_regress.residuals_

    ### Get embedding information
    # extract the diffusion map embedding for each data point / frame
    # one row per frame and time series, one column per diffusion map (embedding dimension)
    diff_embedding_df = pd.DataFrame(diff_evecs.values, index=diff_evecs.index)  # type: ignore
    diff_embedding_df.columns = diff_map_names

    # If an index is present (e.g., time steps), retain it
    if diff_embedding_df.index.name is None:
        diff_embedding_df.index.name = 'timeStep'

    # drop=True: diff_maps_df's row index is just range(len(diff_eigenvals))
    return diff_maps_df.reset_index(drop=True), diff_embedding_df.reset_index()  # do now need multiindex


def refit_model_from_checkpoint_diffusion_maps(
    edmd: EDMD,
    diff_evs_df: pd.DataFrame,
    train_tsc: TSCDataFrame,
    config: FullConfig,
) -> tuple[EDMD, float | None]:
    """Re-fit a checkpointed EDMD model after selecting/fixing its diffusion map coordinates.

    Returns the refitted model and the minimum eigenvalue among the selected coordinates —
    ``None`` if every coordinate from the checkpoint was kept, since none were selected.
    """
    diffusion_maps = None
    dmd_model = None
    for _, step in edmd.steps:
        if isinstance(step, DiffusionMaps):
            diffusion_maps = step
        if isinstance(step, DMDStandard):
            dmd_model = step

    if diffusion_maps is None:
        raise ValueError('Diffusion maps must be part of the checkpoint model')
    if dmd_model is None:
        raise ValueError('DMD model must be part of the checkpoint model')

    num_diffusion_maps = diffusion_maps.n_eigenpairs
    diff_map_min_eigenvalue = None

    if config.diffusion_maps.fix_diffusion_map_coordinates or (
        config.diffusion_maps.auto_select_diffusion_maps is not None
        and config.diffusion_maps.auto_select_diffusion_maps < num_diffusion_maps
    ):
        if config.diffusion_maps.fix_diffusion_map_coordinates:
            log.info(
                f'Fix the diffusion map coordinates to the following indices: {config.diffusion_maps.fix_diffusion_map_coordinates}.'
            )
            diff_map_selection = config.diffusion_maps.fix_diffusion_map_coordinates
            if 0 not in diff_map_selection:
                log.warning('The trivial coordinate 0 is not part of the fixed selection.')
                log.warning('This coordinate is needed to fit the constant components of the dynamical system.')
            diff_subset = diff_evs_df[diff_evs_df['index'].isin(diff_map_selection)]
        else:
            selection_metric = 'eigenvalue'
            if config.diffusion_maps.auto_select_diffusion_maps < num_diffusion_maps:
                diff_evs_df.sort_values(by=selection_metric, ascending=False, inplace=True)
                if config.diffusion_maps.auto_select_diffusion_maps >= 1:
                    diff_subset = diff_evs_df.head(int(config.diffusion_maps.auto_select_diffusion_maps))
                elif 0 <= config.diffusion_maps.auto_select_diffusion_maps < 1:
                    diff_subset = diff_evs_df.loc[
                        diff_evs_df[selection_metric] >= config.diffusion_maps.auto_select_diffusion_maps
                    ]
                elif config.diffusion_maps.auto_select_diffusion_maps < 0:
                    diff_subset = diff_evs_df.head(int(config.diffusion_maps.auto_select_diffusion_maps) - 1)
                diff_map_selection = set(diff_subset.index.unique())
                diff_map_selection.add(0)

        diff_map_min_eigenvalue = float(diff_subset['eigenvalue'].min())
        if len(diff_map_selection) == 0:
            raise ValueError(
                'At least one diffusion map coordinate must be selected -- check '
                'fix_diffusion_map_coordinates / auto_select_diffusion_maps.'
            )
        diffusion_maps.set_target_coords(list(diff_map_selection))
    else:
        log.info('Use all diffusion map coordinates from the checkpoint without adjustments.')

    log.info('Re-fitting model with adjusted diffusion maps...')
    X = edmd._validate_datafold_data(train_tsc, ensure_tsc=False)
    X_dict = edmd.transform(X)
    dmd_model.fit(X=X_dict, y={})
    edmd._inverse_map = edmd._compute_inverse_map(X=X, y=None, X_dict=X_dict, U=None)
    if edmd.dmd_model.is_spectral_mode:
        edmd._koopman_modes = edmd._compute_koopman_modes(inverse_map=edmd._inverse_map)
        if edmd.sort_koopman_triplets:
            edmd._sort_koopman_triplets(X_dict_ic=X_dict.initial_states())
    else:
        edmd._koopman_modes = None

    return edmd, diff_map_min_eigenvalue
