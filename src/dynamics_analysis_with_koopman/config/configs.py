from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any
import logging
import re
import warnings

import yaml

log = logging.getLogger(__name__)

SUPPORTED_TARGET_VALUES = ['density', 'count', 'velocity_x', 'velocity_y']
SUPPORTED_KERNEL_TYPES = ['gaussian', 'continuous_nn', 'temporal_nn', 'multiquadric', 'inv_multiquadric']
SUPPORTED_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
SUPPORTED_LEARNING_ALGOS = ['edmd', 'dmd', 'mp_edmd', 'mp_dmd', 's_dmd', 's_edmd']
SUPPORTED_DATA_SCALING = ['center', 'min-max', 'standard']
SUPPORTED_ROTATION_VALUES = [0, 90, 180, 270]

# Matches the '(x_idx_y_idx)' format that
# plots.grid._parse_measurement_cell expects, e.g. '(4_7)'.
# Kept as a regex here rather than importing that function: this module is
# imported by data.checkpoint and plots.diffusion,
# so importing plots back into it would be circular.
MEASUREMENT_CELL_PATTERN = re.compile(r'^\(\d+_\d+\)$')

FILE_DELIMITER = '_'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = 'configs'
RUN_ARTIFACTS_DIR = 'runs'
DATA_DIR = 'data'
EVALUATION_DIR = 'evaluation'
INTERPRETATION_DIR = 'interpretation'
GIFS_DIR = 'gifs'


# use "meta" for the access, i.e. full_config.meta.scenario (as you did already)
@dataclass
class RunMetaConfig:
    """This class holds all configuration that is needed for handling paths
    and folders for loading, storing and saving.
    """

    # Note None Optional
    scenario: str
    """ Subdirectory of raw_data_dir (or --data-path) this run reads its trajectories
    from, i.e. the experiment identity in the domain sense (2006bottleneck,
    2013crossing90, ...). Purely a data-loading concern -- output grouping is handled
    separately by config_subdir, derived from where the config file itself lives."""

    # TODO: Make this default to main or home/user.
    root_path: Path = PROJECT_ROOT
    """Root directory used to store all artifacts produced by the module."""

    run_id: str | None = None
    """ Unique identifier for the current run, will be set automatically.
    Runtime state rather than configuration: a run_id found in a loaded config file
    is dropped so a new run never writes into a previous run's artifacts folder."""

    run_artifacts_dir: str = RUN_ARTIFACTS_DIR
    """ Run artifacts folder. The folder will be created within the root_dir."""

    config_subdir: str = ''
    """ Groups this run's artifacts under run_artifacts_dir/config_subdir/run_id.
    Normally set automatically by FullConfig.from_yaml from the directory containing
    the config file itself, relative to configs_dir (e.g. 'clustering' for
    configs/clustering/foo.yaml; '' when the config sits directly in configs_dir).
    Persisted like any other meta field, so a dumped run's config.yaml -- reused as a
    new settings file, e.g. via --load-data-from-checkpoint -- keeps grouping with its
    original siblings even though it is itself loaded from inside run_artifacts_dir."""

    configs_dir: str = CONFIGS_DIR
    """ Folder where the configs are stored. The folder will be searched within the root_dir."""

    config_file: str = 'default.yaml'
    """ Specific file to load from the configs_dir"""

    raw_data_dir: str = DATA_DIR
    """ Folder where the raw trajectory data is loaded from.
    The folder will be searched within the root_dir."""

    # NOTE: This could also be separated into its own class
    log_dir: str = run_artifacts_dir
    """ Path to save logs to, defaults to: run_artifacts_dir"""

    log_file_name: str = 'main.log'
    """ Name of the log file. """

    log_level: str = 'INFO'
    """ Defines log level, from DEBUG, INFO, WARNING, ERROR, CRITICAL"""

    git_sha: str = 'unkown'

    # Derived paths. Never pass these in: whatever is given is overwritten by
    # resolve_paths(). They accept a value only so that a dumped config (which
    # contains them) can be loaded back without raising.
    configs_path: Path | None = None
    config_file_path: Path | None = None
    run_artifacts_path: Path | None = None
    raw_data_path: Path | None = None
    unique_run_path: Path | None = None
    log_file_path: Path | None = None
    processed_data_path: Path | None = None
    evaluation_path: Path | None = None
    interpretation_path: Path | None = None
    gifs_path: Path | None = None

    def __post_init__(self):
        self.resolve_paths()

    def resolve_paths(self) -> None:
        """(Re)derive every path field from the current root/dir/run_id settings.

        Idempotent: call again after changing root_path, any *_dir, or run_id.
        The run-specific paths stay None until a run_id is set.

        Updates config in-place, computing:
        - configs_path, config_file_path
        - run_artifacts_path, unique_run_path
        - data_path, log_file_path
        - evaluation_path, interpretation_path, gifs_path

        """
        # Ensure root_path is a Path object (YAML loads as str)
        if isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)

        self.configs_path = self.root_path / self.configs_dir
        self.config_file_path = self.configs_path / self.config_file
        self.run_artifacts_path = self.root_path / self.run_artifacts_dir
        if self.config_subdir:
            self.run_artifacts_path = self.run_artifacts_path / self.config_subdir
        self.raw_data_path = self.root_path / self.raw_data_dir

        if self.run_id is None:
            # we only just loaded the config. Clear the run-specific paths so a
            # value from a loaded config cannot survive as a stale path.
            self.unique_run_path = None
            self.log_file_path = None
            self.processed_data_path = None
            self.evaluation_path = None
            self.interpretation_path = None
            self.gifs_path = None
            return

        # we already have the run_id set
        self.unique_run_path = self.run_artifacts_path / self.run_id
        self.log_file_path = self.unique_run_path / self.log_file_name
        self.processed_data_path = self.unique_run_path / DATA_DIR
        self.evaluation_path = self.unique_run_path / EVALUATION_DIR
        self.interpretation_path = self.unique_run_path / INTERPRETATION_DIR
        self.gifs_path = self.unique_run_path / GIFS_DIR


@dataclass
class PreProcessingConfig:
    """This config holds configuration for the preprocessing."""

    grid_resolution: float | list[float] = 0.1
    """ Grid resolution for the discretization in meters. This setting is used for discretizing the trajectory data during extraction process."""

    system_observables: list[str] = field(default_factory=lambda: ['count'])
    f""" Specifies the input data for Koopman operator learning.
    Possible list values: {SUPPORTED_TARGET_VALUES}"""

    prediction_timeshift: int = 0
    """ Used to mimic the embedding timeshift due to temporal embedding.
    This can be used to improve comparison between edmd and dmd results."""

    obstacles: list[tuple | float | int] = field(default_factory=list)
    """ List of obstacles in the scenario, identical with the settings [(x, y), width, height] from Vadere scenario.
    The x, y coordinates are within the measurement area and are automatically offset by xlim and ylim."""

    obstacle_scaling: float = 1.0
    """ If obstacle_scaling is > 0, the obstacle is added to the grid with the given scaling factor.
    Set obstacle_scaling to 0 to turn of this processing step and to NOT add the obstacle explicitly."""

    temporal_resolution: float = 0.2
    """ Temporal resolution of the data, is set automatically during the data extraction process"""

    interpolate: bool = False
    """ If True, the agent steps in the trajectories is interpolated to the temporal resolution.
    If False, the agent has only two distinct positions, the ones at the beginning and the end of its step."""

    spatial_smoothing_neighborhood: int = 0
    """ Neighborhood for the spatial smoothing of the velocity, i.e. the number of cells to consider for the smoothing within the same timestep.
    If set to 0, no spatial smoothing is applied.
    If set to 1, the neighborhood is the 3x3 grid around the cell (Moore neighborhood).
    NOTE: as of now, this is only considered for the velocity fields."""

    spatial_smoothing_bandwidth: float = 1.0
    """ Bandwidth for the spatial smoothing of the velocity in meters,
    i.e. the kernel width for the smoothing within the same timestep.
    """

    average_across_simulations: bool = False
    """ If True, the data is averaged across all simulations in the directory.
    If False, the data is loaded for each simulation separately.
    NOTE: Use averaging with caution and make sure it makes sense for the application at hand."""

    tlim: list[int | None] = field(default_factory=lambda: [None, None])
    """ Time limit for the data extraction in time steps (starting with t = 0 at 0.0s)
    tlim[0] is the start step, tlim[1] is the end step (including).
    If None is used, no frames are removed from the simulation data.
    NOTE: Keeping tlim[1] to None allows for time series to have different lengths.
    NOTE: the time limit is NOT set in seconds, but in time steps, i.e. it depends on `temporal_resolution`."""

    xlim: list[int | float | None] = field(default_factory=lambda: [None, None])
    """ X-axis limits for the data extraction in meter from Vadere simulation, [x-axis start, x-axis end].
    If None is used, no frames are removed from the simulation data."""

    ylim: list[int | float | None] = field(default_factory=lambda: [None, None])
    """ Y-axis limits for the data extraction in meter from Vadere simulation, [y-axis start, y-axis end].
    If None is used, no frames are removed from the simulation data."""

    store_position_ts: bool = False
    """ If true, the discretized and interpolated position time series is stored.
    This time series is the input to the observable calculation. """

    store_tsc_data: bool = False
    """ If True, the time series data is stored in the run artifacts folder using pickle format.
    This also includes the data predictions of the model."""

    store_timeseries_data: bool = False
    """ If True, this run's loaded timeseries data (freshly ingested, or read from a
    checkpoint) is persisted to the run artifacts' data/ folder as CSV.
    A run that does not load from a checkpoint always stores it regardless of this flag;
    it only gates a checkpoint-loaded run, which otherwise has nothing of its own on disk
    for a further run to resume from -- see load_data_from_checkpoint."""

    preprocess_only: bool = False
    """ If True, the run stops once the data is preprocessed, persisted and the ground
    truth is visualized -- no model is fitted, predicted or evaluated.
    Use it to produce the timeseries a later run resumes from (see
    load_data_from_checkpoint) without paying for a fit whose result is discarded.
    The ExperimentResult of such a run carries no model and no predictions."""


# use "data" for the access, i.e. full_config.data.scenario, unless you have a better idea :)
@dataclass
class DataSelectionConfig:
    train_parameter: list | None = None
    test_parameter: list | None = None
    """ Parameter values are set during data load based on the train-test directories.
    In case of checkpoints, they can be used to select a subset of the data. """

    selection_parameters: list | None = None
    """ Filled with the set given in train_parameter and test_parameter.
    Checks for the pattern \_{parameter}\_ or for equality (i.e. full directory name)
    """

    use_train_test_split: bool = True
    """ If True, data is expected to be separated into train and test subdirectories.
    If False, the separation in directories is ignored."""

    load_data_from_checkpoint: str | None = None
    """ Load data from previous processing run instead of Vadere simulation outputs
    Expects run_id only to load dataset from former run. If None, no preprocessed data is loaded from file."""

    input_file_name: str = 'postvis.traj'
    """ Name of the input file to load the data from, without the file extension.
    If we load experiment data, this property is ignored. """

    enrich_timestep_data: str | None = None
    """ If set, the timeframe data is enriched with the given timestep data at the very end of preprocessing.
    The str points to a subfile of the vadere directory, e.g. counter.csv
    """


@dataclass
class AlgorithmConfig:
    learning_algo: str = 'edmd'
    """ Specifies the learning algorithm for the Koopman operator.
    Options: edmd, dmd"""

    sort_koopman_triplets: bool = True
    """ Sort the Koopman triplets by importantce if True. If False, the triplets are sorted by the absolute value of the eigenvalues."""

    ###############################
    #### DMD Configurations #######
    ###############################

    rank: int | None = None
    """ Rank used for the approximation of the Koopman operator.
    The maximum rank is either the dimension of the system observables or the number of time steps - 1.
    Recall: two shift matices are created from the data points, meaning each has the width = number of time steps - 1."""

    pseudospectrum_residual_filter: float | dict | None = None
    """ Residual filter for the pseudospectrum, i.e. the maximum residual an eigenvalue can have to kept.
    If set to None, no residual filter is applied.
    For a dictionary, the following filter methods are supported:
    - 'keep_n': set a fixed number of Koopman triples to keep.
    - 'threshold': remove all Koopman triples with eigenvalue residuals larger than the value.
    - 'quantile': set the quantile to determine the residual value for which Koopman triples with larger residuals are removed.
    If the value is a float, it will be replaced in datafold by {'threshold': value}.
    """

    #########################################
    #### EDMD-specific Configurations #######
    #########################################

    use_temporal_data_scaling: str | None = None
    """ If set, the data is rescaled to the given value per individual feature.
    Options: 'center', 'min-max', 'standard'
    - 'center' will center the data around 0, i.e. subtract the mean from the data.
    - 'min-max' will rescale the data to the range [0, 1].
    - 'standard' will standardize the data, i.e. subtract the mean and divide by the standard deviation.
    """

    use_temporal_embedding: bool = True
    """ If True, the temporal embedding is used for the Koopman operator learning.
    The configuration is done via t_d, frequency, and kappa."""

    has_measurement_noise: bool = True
    """ Only used for stochastic DMD. If no measurement noise is expected, time delayed observables may have an overlap.
    This can only be used, if temporal embedding is used.
    """

    use_diffusion_maps: bool = True
    """ If True, diffusion maps are used for the Koopman operator learning."""


@dataclass
class SWIMConfig:
    use_sampled_hidden_layer: bool = False
    """ Flag that decides if a sampled hidden layer will be used.
    """

    seed: int = 10
    """ A seed is passed on to random number generators to control reproducibility.
    """

    layer_width: int = 1
    """ Number of nodes in hidden layer.
    """

    activation: str = 'tanh'
    """ Activation function of hidden layer.
    """

    param_sampler: str = 'tanh'
    """ Parameter sampler used by SWIM.
    """

    rcond: float = 1.0
    """ Regularization parameter for least squares solution.
    """


@dataclass
class VisualizationConfig:
    visualization_timesteps: list[int] = field(default_factory=lambda: [0, 10, 20, 30, 40])
    """ List of timesteps to visualize the density grids for"""

    rotation: int | None = None
    f"""Rotates plots clockwise. Valid rotation values: {SUPPORTED_ROTATION_VALUES}. Left unset
    (None), a checkpoint run inherits its checkpoint's rotation; a plain run falls
    back to 0 (see check_config_consistency)."""

    vector_scaling: float = 0.5
    """ Scales the vector in quiver plot inversely,
    i.e. increases length for < 1, and decreses length for > 1."""

    koopman_triple_selection: list[int] | None = None
    """ List of Koopman triplets to highlight or visualize in the modes, eigenvalues and eigenfunctions"""

    generate_timeseries_gifs: list[int] | None = None
    """ Provide a list of selection parameters to generate a gif for the count or density grids"""

    generate_mode_gifs: list[int] | None = None
    """ If True, generates gifs for the modes of the Koopman operator.
    Starts at 1. """

    compute_pseudospectrum: bool = False
    """ If True, the pseudospectrum of the Koopman operator is visualized in the eigenvalue plot.
    This is a very expensive operation and will only be used for the eigenvalue visualization."""

    plot_ground_truth: bool = True
    """ If True, the ground truth is plotted. This setting is ignored when loading from a checkpoint.
    """

    plot_supermodes: bool = False
    """ If True, the supermodes of the Koopman operator are visualized for each time series."""

    plot_diffusion_maps: bool = False
    """ If True, the diffusion maps are visualized for each time series including their gif scatter plot."""

    fig_filetype: str = 'png'
    """ Determines the filetype in which figures are saved, only the available filetypes from matplotlib are supported."""

    gif_filetype: str = 'gif'
    """ Determines the filetype in which gifs are saved, only the available filetypes from matplotlib are supported
    [gif, apng, html, htm, png, mkv, mp4 (recommended), mjpeg, webp]."""


@dataclass
class EvaluationConfig:
    measurement_cells: list[str] = field(default_factory=lambda: ['(90_43)', '(70_43)', '(70_20)'])
    """ List of cells to visualize the prediction error for.
    (y-axis, x-axis) for the measurement area in array indices for the loaded data.
    NOTE: Transposition or flipping only effects the plot, not the axes for measurement cell selection."""

    stepwise_transform: bool = False
    """ If True, the transformation steps are applied stepwise, i.e. the data is transformed in each step."""

    extra_prediction_steps: int = 0
    """ Additional time steps to predict outside of the training series.
    This can be used to show out of sample predictions w.r.t. temporal dimension. """

    store_trained_model: bool = False
    """ If True, the trained model is stored in the run artifacts folder using pickle format."""

    mae_threshold: float = 10.0
    """ Threshold for the mean absolute error for the training set.
    In case the average mae across all time series is above this threshold, the training is stopped.
    If the value is set to a negative value, the threshold is ignored."""


@dataclass
class TemporalEmbeddingConfig:
    num_t_delays: int = 5
    """ Number of time delays ( i.e. frames added to the state space).
    Due to maximum for bottleneck 0.1 is 3 when not using diffusion maps."""

    frequency: int = 10
    """ Frequency of the time delays, i.e. every nth frame is added to the state space.
    NOTE: the first frame to be added is always the previous one, only then the frequency is applied."""

    lag: int = 0
    """ Number of lags for the temporal embedding,
    i.e. the number of additional skipped frames to the temporal embedding.
    NOTE: A lag of 0 means that the first frame before the current state is used
    as the first frame in the temporal embedding."""

    kappa: float = 0.001
    """ Weighting parameter for the temporal embedding.
    Kappa is used as exponent in exponentially decaying weighting factor."""


@dataclass
class DiffusionMapsConfig:
    rank_diffusion_maps: bool = False
    """
    If True, the relevance of each diffusion map is calculated using linear regression.
    NOTE: This is very expensive when turned on, due to a double loop over
    number of eigenpairs AND number of frames across all time series.
    """

    learn_kernel_params: bool = False
    """ If True, the kernel parameters are learned from the data."""

    auto_select_diffusion_maps: float | None = None
    """ If set to > 0, local regression is used for the selection of the diffusion map coordinates.
    If the value is set to a value >= 1, it is assumed to be an integer and provides a fixed number of diffusion maps for selection.
    If the value is between 0 and 1, it is assumed to be a float and gives the residual threshold for diffusion map selection.
    If the value is set to a negative value, a fixed number of diffusion maps is removed from the end when sorted by residuals. """

    fix_diffusion_map_coordinates: list[int] | None = None
    """ If provided, the diffusion maps are reduced to the fixed index of these diffusion maps.
    The indices start at 0, but the first diffusion map is always the trivial one."""

    n_eigenpairs: int = 75
    """ Number of modes used for the diffusion map approximation."""

    kernel_type: str = 'gaussian'
    f""" Type of kernel used for the diffusion maps.
    Options: {SUPPORTED_KERNEL_TYPES}"""

    kernel_cutoff: float | None = None
    """ Cutoff parameter for the Gaussian kernel in the diffusion maps.
    This is the maximum value for kernel distanceset_dependent_settingss considered. """

    epsilon: float = 0.28
    """ Epsilon parameter for the diffusion maps, used for the kernel approximation."""

    k_neighbors: int = 10
    """ Number of neighbors used for the Continuous kNN kernel, used for the diffusion map learning.
    """

    delta: float = 1.0
    """ Delta parameter for the Continuous kNN kernel, used for the diffusion map learning.
    """


@dataclass
class RBFConfig:
    use_rbf: bool = False
    """ Use radial basis functions as the dictionary.
    The radial basis functions use the same kernel configuration parameters as the diffusion maps.
    """

    n_samples: int = 100
    """ Number of samples for fitting the radial basis functions.
    """


@dataclass
class PolynomialConfig:
    use_polynomial: bool = False
    """ Use polynomial feature dictionary.
    The maximum degree of the polynomial basis is configured using `p_degree`.
    """

    p_degree: int = 3
    """ Maximum degree for the polynomial feature dictionary.
    """


def _handle_legacy_data_path(config_data: dict, value: str) -> None:
    """`data_path` -> `raw_data_dir`: not a pure rename, the value needs reshaping."""
    raw_data_dir = value.split('/data')
    if len(raw_data_dir) > 1:
        # only then the default is adjusted to use a subdirectory for a specific scenario
        config_data['raw_data_dir'] = f'data{raw_data_dir[-1]}'
    warnings.warn(
        f"'data_path' is a legacy parameter, replaced by 'raw_data_dir'. Automatic "
        f'replacement performed: data_path={value!r} -> raw_data_dir={config_data.get("raw_data_dir")!r}',
        DeprecationWarning,
        stacklevel=3,
    )


def _handle_legacy_system_observable(config_data: dict, value: Any) -> None:
    """`system_observable` (singular, scalar) -> `system_observables` (list)."""
    config_data['system_observables'] = [value] if not isinstance(value, list) else value
    warnings.warn(
        f"'system_observable' is a legacy parameter name, renamed to 'system_observables' "
        f'and wrapped in a list: system_observables={config_data["system_observables"]!r}',
        DeprecationWarning,
        stacklevel=3,
    )


def _handle_legacy_dropped_key(key_name: str, reason: str):
    """Build a handler for a legacy key that is dropped outright, not renamed.

    Used for keys that predate `_split_legacy_flat_config` raising on unknown keys: they
    were already dead (silently ignored by the old flat `__getattr__` machinery) and are
    registered here only so their removal is documented and old configs still load,
    rather than because the value goes anywhere.
    """

    def handler(config_data: dict, value: Any) -> None:
        warnings.warn(
            f"'{key_name}' is a legacy parameter that no longer has any effect ({reason}). "
            f'Dropped, value ignored: {value!r}.',
            DeprecationWarning,
            stacklevel=3,
        )

    return handler


def _handle_legacy_rotation_flags(config_data: dict) -> None:
    """`flip_ud` + `transpose_data` -> `rotation`.
     Handled separately from `_LEGACY_KEY_HANDLERS` because it is a two-key merge.

    Exact equivalent:
         (False, False) -> rotation=0   (exact)
         (True,  True)  -> rotation=90  (exact)
     Implied rotation (ignoring the mirroring, but raising a warning):
         (True,  False) -> rotation=180 (flip_ud alone)
         (False, True)  -> rotation=90  (transpose_data alone)
    """
    if 'flip_ud' not in config_data and 'transpose_data' not in config_data:
        return

    flip_ud = config_data.pop('flip_ud', False)
    transpose_data = config_data.pop('transpose_data', False)

    if flip_ud and transpose_data:
        rotation = 90
    elif flip_ud:
        rotation = 180
        warnings.warn(
            "'flip_ud' alone has no exact 'rotation' equivalent -- it is a mirror "
            'flip, not a rotation. Approximated as rotation=180 (the mirror is '
            "dropped); check the plot orientation and adjust 'rotation' by hand if "
            'needed.',
            DeprecationWarning,
            stacklevel=3,
        )
    elif transpose_data:
        rotation = 90
        warnings.warn(
            "'transpose_data' alone has no exact 'rotation' equivalent -- it is a "
            'diagonal reflection, not a rotation. Approximated as rotation=90 (the '
            "mirror is dropped); check the plot orientation and adjust 'rotation' by "
            'hand if needed.',
            DeprecationWarning,
            stacklevel=3,
        )
    else:
        rotation = 0

    config_data['rotation'] = rotation


@dataclass
class FullConfig:
    meta: RunMetaConfig
    preprocessing: PreProcessingConfig
    data: DataSelectionConfig
    algorithm: AlgorithmConfig
    swim: SWIMConfig
    visualization: VisualizationConfig
    evaluation: EvaluationConfig
    temporal_embedding: TemporalEmbeddingConfig
    diffusion_maps: DiffusionMapsConfig
    rbf: RBFConfig
    polynomial: PolynomialConfig

    # Every section name FullConfig models, in field-declaration order. The one place
    # this list exists -- everywhere else that needs "all sections" reads from here.
    _SECTION_NAMES = (
        'meta',
        'preprocessing',
        'data',
        'algorithm',
        'swim',
        'visualization',
        'evaluation',
        'temporal_embedding',
        'diffusion_maps',
        'rbf',
        'polynomial',
    )

    # Every legacy/renamed *config-file* key goes here.
    # Two options encoded in dict via (config_data, value):
    # 1. plain string value for a pure rename (the value moves unchanged to the new key)
    # 2. a callable is for anything that also needs to reshape the value
    # dict[str, str | Callable[[dict, Any], None]] -- left unannotated so `@dataclass`
    _LEGACY_KEY_HANDLERS = {
        'workspace_path': 'root_path',
        'settings_file_path': 'config_file',
        'data_path': _handle_legacy_data_path,
        'system_observable': _handle_legacy_system_observable,
        'measurement_area': _handle_legacy_dropped_key(
            'measurement_area', 'removed from RunMetaConfig in 8fdb112, no replacement'
        ),
        'add_obstacle_density': _handle_legacy_dropped_key(
            'add_obstacle_density', 'not a recognized field, no replacement'
        ),
        'generate_gifs': _handle_legacy_dropped_key(
            'generate_gifs', 'superseded by generate_timeseries_gifs / generate_mode_gifs'
        ),
    }

    @classmethod
    def _apply_legacy_key_handlers(cls, config_data: dict) -> dict:
        """Rewrite every known legacy/renamed key in `config_data`, in place.

        Called once, from `_split_legacy_flat_config`, so every loading path -- a
        plain YAML load and a checkpoint's saved config (which also goes through
        `from_dict`) -- gets identical handling.
        """
        for old_key, handler in cls._LEGACY_KEY_HANDLERS.items():
            if old_key not in config_data:
                continue
            value = config_data.pop(old_key)
            if callable(handler):
                handler(config_data, value)
            else:
                config_data[handler] = value
                warnings.warn(
                    f"'{old_key}' is a legacy parameter name, renamed to '{handler}'. Please update your config.",
                    DeprecationWarning,
                    stacklevel=3,
                )
        return config_data

    @classmethod
    def from_defaults(cls) -> 'FullConfig':
        """Creates a configuration entirely from the Python dataclass defaults."""
        # NOTE: If any variables in dataclasses don't have default values
        # (like 'scenario: str'), its necessary to pass a default.

        return cls(
            meta=RunMetaConfig(scenario='default_scenario'),
            preprocessing=PreProcessingConfig(),
            data=DataSelectionConfig(),
            algorithm=AlgorithmConfig(),
            swim=SWIMConfig(),
            visualization=VisualizationConfig(),
            evaluation=EvaluationConfig(),
            temporal_embedding=TemporalEmbeddingConfig(),
            diffusion_maps=DiffusionMapsConfig(),
            rbf=RBFConfig(),
            polynomial=PolynomialConfig(),
        )

    @classmethod
    def from_dict(cls, config_data: dict[str, Any], config_subdir: str = '') -> 'FullConfig':
        """Create a FullConfig from nested or legacy flat dictionaries.

        config_subdir is used for meta.config_subdir only when config_data itself does
        not already set it -- see from_yaml, which derives it from the directory the
        config file itself was loaded from. A config_data that already has
        meta.config_subdir (e.g. a dumped run's config.yaml, reloaded) keeps that
        value instead, so reusing it as a new settings file does not re-derive
        grouping from wherever that dump happens to live (typically run_artifacts_dir).
        """
        if config_data is None:
            config_data = {}

        if cls._looks_like_legacy_flat_config(config_data):
            return cls.from_legacy_flat_dict(config_data, config_subdir=config_subdir)

        # Nested format: every top-level key must be a section this class models
        unknown_sections = set(config_data) - set(cls._SECTION_NAMES)
        if unknown_sections:
            raise ValueError(
                f'Unknown config section(s): {sorted(unknown_sections)}. '
                f'Valid sections are: {sorted(cls._SECTION_NAMES)}.'
            )

        nested_meta = config_data.get('meta', {})
        known_meta_fields = {f.name for f in fields(RunMetaConfig)}
        obsolete_meta_keys = set(nested_meta) - known_meta_fields
        if obsolete_meta_keys:
            # A config dumped by an older version of the code -- typically a
            # checkpoint's config.yaml -- can carry meta fields that have since been
            # removed (e.g. group_runs_by_scenario, experiment). meta.scenario alone
            # is not enough to identify such a config for merge_checkpoint_config's
            # override loop; it always replaces the whole checkpoint meta with the
            # current run's meta anyway, so these are simply dropped rather than
            # raising.
            warnings.warn(
                f'Ignoring obsolete meta field(s) from a loaded config, no longer '
                f'part of RunMetaConfig: {sorted(obsolete_meta_keys)}.',
                DeprecationWarning,
                stacklevel=3,
            )
        nested_prep = config_data.get('preprocessing', {})
        nested_data = config_data.get('data', {})
        nested_algorithm = config_data.get('algorithm', {})
        nested_swim = config_data.get('swim', {})
        nested_visualization = config_data.get('visualization', {})
        nested_evaluation = config_data.get('evaluation', {})
        nested_temporal_embedding = config_data.get('temporal_embedding', {})
        nested_diffusion_maps = config_data.get('diffusion_maps', {})
        nested_rbf = config_data.get('rbf', {})
        nested_polynomial = config_data.get('polynomial', {})

        return cls(
            meta=RunMetaConfig(
                scenario=nested_meta.get('scenario', 'default_scenario'),
                config_subdir=nested_meta.get('config_subdir', config_subdir),
                **{
                    k: v
                    for k, v in nested_meta.items()
                    if k not in ('scenario', 'config_subdir') and k in known_meta_fields
                },
            ),
            preprocessing=PreProcessingConfig(**nested_prep),
            data=DataSelectionConfig(**nested_data),
            algorithm=AlgorithmConfig(**nested_algorithm),
            swim=SWIMConfig(**nested_swim),
            visualization=VisualizationConfig(**nested_visualization),
            evaluation=EvaluationConfig(**nested_evaluation),
            temporal_embedding=TemporalEmbeddingConfig(**nested_temporal_embedding),
            diffusion_maps=DiffusionMapsConfig(**nested_diffusion_maps),
            rbf=RBFConfig(**nested_rbf),
            polynomial=PolynomialConfig(**nested_polynomial),
        )

    @classmethod
    def from_legacy_flat_dict(cls, config_data: dict[str, Any], config_subdir: str = '') -> 'FullConfig':
        """Create a FullConfig from an old flat config dictionary.

        This method exists for backwards compatibility with legacy YAML files that
        stored all fields at the top level. It converts the flat structure to the
        nested config layout used by the current codebase.
        """
        if config_data is None:
            config_data = {}

        warnings.warn(
            'Loading legacy flat config format. Please migrate to nested config sections.',
            DeprecationWarning,
            stacklevel=2,
        )

        nested_config = cls._split_legacy_flat_config(config_data)
        return cls.from_dict(nested_config, config_subdir=config_subdir)

    @classmethod
    def from_yaml(cls, path: Path) -> 'FullConfig':
        """This method loads the config from yaml.

        meta.config_subdir, when not already set in the file itself, is derived from
        the config file's own parent directory name (e.g. 'clustering' for
        configs/clustering/foo.yaml) -- so a run's artifacts are grouped with its
        sibling configs without any of them needing to set this by hand. Derived from
        `path` as given (before it is resolved against CONFIGS_DIR/PROJECT_ROOT in
        _load_yaml), so a bare filename with no directory component groups nowhere
        (config_subdir stays '').
        """
        config_data = cls._load_yaml(path)
        config_subdir = Path(path).parent.name
        if cls._looks_like_legacy_flat_config(config_data):
            return cls.from_legacy_flat_dict(config_data, config_subdir=config_subdir)
        return cls.from_dict(config_data, config_subdir=config_subdir)

    @staticmethod
    def _load_yaml(path: Path | str) -> dict:
        path = Path(path)
        path = path if path.suffix == '.yaml' else path.with_suffix('.yaml')

        candidates = [
            path,
            PROJECT_ROOT / CONFIGS_DIR / path,
            PROJECT_ROOT / path,
        ]

        for candidate_path in candidates:
            clean_path = candidate_path.resolve()

            if clean_path.is_file():
                with clean_path.open('r') as f:
                    # Use UnsafeLoader to handle Python objects like PosixPath stored in checkpoint configs
                    return yaml.load(f, Loader=yaml.UnsafeLoader)

        msg = (
            f"Could not find config file for input: '{path}'.\n"
            f'Checked the following locations:\n'
            f' - {candidates[0].resolve()}\n'
            f' - {candidates[1].resolve()}\n'
            f' - {candidates[2].resolve()}\n'
            f'If not using the default PROJECT_ROOT: {PROJECT_ROOT}\n'
            f'or default CONFIG_DIR: {CONFIGS_DIR}\n'
            f'provide the absolut path to your configuration.'
        )

        raise FileNotFoundError(msg)

    @classmethod
    def _looks_like_legacy_flat_config(cls, config_data: dict[str, Any]) -> bool:
        """Detect legacy flat config files that do not use nested top-level sections."""
        if not isinstance(config_data, dict):
            return False

        return not any(key in config_data for key in cls._SECTION_NAMES)

    @classmethod
    def _split_legacy_flat_config(cls, config_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Convert a legacy flat config dict into nested sections.

        Every key must end up recognized -- either as a current field on one of the
        section dataclasses, or as a legacy name rewritten by
        `_apply_legacy_key_handlers` / `_handle_legacy_rotation_flags`. Anything left
        over is an error, not a silent drop: it is either a typo or a legacy key
        nobody has registered yet.
        """
        config_data = cls._apply_legacy_key_handlers(dict(config_data))
        # separate because it maps two to a single new key
        _handle_legacy_rotation_flags(config_data)

        section_dataclasses = {
            'meta': RunMetaConfig,
            'preprocessing': PreProcessingConfig,
            'data': DataSelectionConfig,
            'algorithm': AlgorithmConfig,
            'swim': SWIMConfig,
            'visualization': VisualizationConfig,
            'evaluation': EvaluationConfig,
            'temporal_embedding': TemporalEmbeddingConfig,
            'diffusion_maps': DiffusionMapsConfig,
            'rbf': RBFConfig,
            'polynomial': PolynomialConfig,
        }
        section_keys = {
            section_name: set(section_cls.__dataclass_fields__.keys())
            for section_name, section_cls in section_dataclasses.items()
        }
        nested: dict[str, dict[str, Any]] = {section_name: {} for section_name in section_dataclasses}

        unknown_keys = []
        for key, value in config_data.items():
            for section_name, known_keys in section_keys.items():
                if key in known_keys:
                    nested[section_name][key] = value
                    break
            else:
                unknown_keys.append(key)

        if unknown_keys:
            raise ValueError(
                f'Unknown config parameter(s): {sorted(unknown_keys)}. If this is a '
                'renamed or removed parameter, add it to FullConfig._LEGACY_KEY_HANDLERS; '
                'otherwise it is likely a typo.'
            )

        return nested

    def apply_cli_overrides(self, cli_args: dict):
        """Dynamically applies flat CLI arguments to the nested config objects."""
        for key, value in cli_args.items():
            if key == 'workspace_path':
                self.meta.root_path = Path(value)
                continue

            if key == 'settings_file_path':
                self.meta.config_file = value.split('/')[-1]
                continue

            if hasattr(self.meta, key):
                setattr(self.meta, key, value)

            elif hasattr(self.preprocessing, key):
                setattr(self.preprocessing, key, value)

            elif hasattr(self.data, key):
                setattr(self.data, key, value)

            elif hasattr(self.algorithm, key):
                setattr(self.algorithm, key, value)

            elif hasattr(self.swim, key):
                setattr(self.swim, key, value)

            elif hasattr(self.visualization, key):
                setattr(self.visualization, key, value)

            elif hasattr(self.evaluation, key):
                setattr(self.evaluation, key, value)

            elif hasattr(self.temporal_embedding, key):
                setattr(self.temporal_embedding, key, value)

            elif hasattr(self.diffusion_maps, key):
                setattr(self.diffusion_maps, key, value)

            elif hasattr(self.rbf, key):
                setattr(self.rbf, key, value)

            elif hasattr(self.polynomial, key):
                setattr(self.polynomial, key, value)

            else:
                log.warning(f"CLI argument '{key}' does not match any config field.")

        # The overrides above can change root_path, config_file, run_id or any *_dir,
        # so the derived paths are stale until they are re-derived. Do it here rather
        # than relying on a later initialize_run_folders, which does not run in every
        # flow (notebooks and scripts that only load a config).
        self.meta.resolve_paths()

    def get_full_config_dict(self) -> dict[str, dict[str, Any]]:
        """Return the config as a nested dict, in the same section/field shape `from_dict` reads."""
        return {section: vars(getattr(self, section)) for section in self._SECTION_NAMES}

    def check_config_consistency(self) -> None:
        """
        Checks for interdependencies and formats settings.
        Translates constraints across the different sub-configurations.

        Raises
        ------
        ValueError
            If the configuration is inconsistent. Every check here guards against a
            value a user can set in a config file or on the command line, so the
            failure is reported as an error rather than an ``assert`` -- asserts
            vanish under ``python -O`` and read as tracebacks rather than as advice.
        """
        # Ensure some settings are initialized
        if self.meta.root_path is None:
            raise ValueError('Workspace (root) path must be set to execute main script.')

        if self.data.load_data_from_checkpoint is None and self.meta.scenario is None:
            raise ValueError('Scenario must be set to load and pre-process trajectory data.')

        # Ensure formatting of some settings
        if self.data.selection_parameters is not None:
            self.data.selection_parameters = list(map(str, self.data.selection_parameters))

        if self.visualization.generate_timeseries_gifs is not None:
            self.visualization.generate_timeseries_gifs = list(map(int, self.visualization.generate_timeseries_gifs))

        if not isinstance(self.preprocessing.grid_resolution, list):
            self.preprocessing.grid_resolution = [
                self.preprocessing.grid_resolution,
                self.preprocessing.grid_resolution,
            ]

        if len(self.preprocessing.grid_resolution) != 2:
            raise ValueError(
                f'Grid resolution must be a single float or a list of two floats for the x and y axis, '
                f'got {self.preprocessing.grid_resolution!r}.'
            )

        # Ensure the value of some settings
        if self.meta.log_level not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f'Log level must be one of {SUPPORTED_LOG_LEVELS}, got {self.meta.log_level!r}.')

        if self.algorithm.learning_algo not in SUPPORTED_LEARNING_ALGOS:
            raise ValueError(
                f'Only {SUPPORTED_LEARNING_ALGOS} are supported for now, got {self.algorithm.learning_algo!r}.'
            )

        for observable in self.preprocessing.system_observables:
            if observable not in SUPPORTED_TARGET_VALUES:
                raise ValueError(
                    f'Only {SUPPORTED_TARGET_VALUES} are supported as system observables for now, got {observable!r}.'
                )

        if self.preprocessing.spatial_smoothing_neighborhood < 0:
            raise ValueError(
                f'Spatial smoothing neighborhood must be a positive integer, '
                f'got {self.preprocessing.spatial_smoothing_neighborhood}.'
            )

        if self.preprocessing.spatial_smoothing_neighborhood > 0 and not self.data.load_data_from_checkpoint:
            if not any(
                observable in self.preprocessing.system_observables for observable in ['velocity_x', 'velocity_y']
            ):
                raise ValueError(
                    'Spatial smoothing is only available for velocity as system observable, '
                    f'but system_observables is {self.preprocessing.system_observables}.'
                )

        if self.algorithm.learning_algo in ['edmd', 'mp_edmd', 's_edmd']:
            if not (
                self.algorithm.use_temporal_embedding
                or self.algorithm.use_diffusion_maps
                or self.rbf.use_rbf
                or self.polynomial.use_polynomial
                or self.swim.use_sampled_hidden_layer
            ):
                raise ValueError(
                    f'At least one feature must be enabled for {self.algorithm.learning_algo!r}: '
                    'temporal_embedding, diffusion_maps, rbf, polynomial, or sampled_hidden_layer.'
                )

            if (
                sum(
                    [
                        self.algorithm.use_diffusion_maps,
                        self.rbf.use_rbf,
                        self.polynomial.use_polynomial,
                        self.swim.use_sampled_hidden_layer,
                    ]
                )
                > 1
            ):
                raise ValueError(
                    'At most one of the following can be enabled: '
                    'diffusion_maps, rbf, polynomial, sampled_hidden_layer.'
                )

        if self.rbf.use_rbf:
            if self.diffusion_maps.kernel_type not in ['gaussian', 'multiquadric', 'inv_multiquadric']:
                raise ValueError(
                    'Radial basis functions are only tested for the gaussian, multiquadric and inv_multiquadric '
                    f'kernels, and cause an error for temporal_nn kernel -- got {self.diffusion_maps.kernel_type!r}.'
                )
            if self.diffusion_maps.kernel_type == 'gaussian' and self.diffusion_maps.kernel_cutoff is not None:
                raise ValueError(
                    'When using radial basis functions, the kernel_cutoff needs to be None, leading to a '
                    'dense distance matrix. Otherwise, a sparse matrix leads to an error during rbf `fit_transform`.'
                )

        if self.algorithm.learning_algo in ['dmd', 'mp_dmd']:
            self.algorithm.use_diffusion_maps = False
            self.algorithm.use_temporal_embedding = False

        if self.algorithm.learning_algo == 'mp_edmd' and self.algorithm.rank is not None:
            raise ValueError('Using mpEDMD is only possible with full rank at the moment, so rank must be None.')

        if (
            self.algorithm.use_temporal_data_scaling is not None
            and self.algorithm.use_temporal_data_scaling not in SUPPORTED_DATA_SCALING
        ):
            raise ValueError(
                f'Only {SUPPORTED_DATA_SCALING} are supported for now, '
                f'got {self.algorithm.use_temporal_data_scaling!r}.'
            )

        if (
            self.algorithm.use_diffusion_maps
            and self.algorithm.rank is not None
            and self.algorithm.rank > self.diffusion_maps.n_eigenpairs
        ):
            raise ValueError(
                f'The DMD rank ({self.algorithm.rank}) must be smaller than or equal to the input dimension of '
                f'the diffusion maps (n_eigenpairs={self.diffusion_maps.n_eigenpairs}).'
            )

        if self.diffusion_maps.kernel_type not in SUPPORTED_KERNEL_TYPES:
            raise ValueError(
                f'Only {SUPPORTED_KERNEL_TYPES} are supported for now, got {self.diffusion_maps.kernel_type!r}.'
            )

        if self.visualization.koopman_triple_selection is not None:
            self.visualization.koopman_triple_selection = list(map(int, self.visualization.koopman_triple_selection))
            if not all(j > 0 for j in self.visualization.koopman_triple_selection):
                raise ValueError(
                    f'0 is not a valid index for a Koopman triplet, the selection starts at 1: '
                    f'{self.visualization.koopman_triple_selection}.'
                )

        if self.visualization.generate_mode_gifs is not None:
            self.visualization.generate_mode_gifs = list(map(int, self.visualization.generate_mode_gifs))
            if not all(j > 0 for j in self.visualization.generate_mode_gifs):
                raise ValueError(
                    f'0 is not a valid index for a Koopman mode, the selection starts at 1: '
                    f'{self.visualization.generate_mode_gifs}.'
                )

        if self.diffusion_maps.auto_select_diffusion_maps is not None:
            if not self.algorithm.use_diffusion_maps:
                raise ValueError('Automatic selection of diffusion maps cannot be used without use_diffusion_maps.')
            if self.diffusion_maps.fix_diffusion_map_coordinates is not None:
                raise ValueError('Automatic selection of diffusion maps cannot be combined with a fixed selection.')

        if self.diffusion_maps.fix_diffusion_map_coordinates is not None:
            if not self.algorithm.use_diffusion_maps:
                raise ValueError('Fixing diffusion map coordinates is only available when use_diffusion_maps is set.')
            if self.diffusion_maps.auto_select_diffusion_maps not in [0, None]:
                raise ValueError('Automatic selection of diffusion maps cannot be combined with a fixed selection.')
            self.diffusion_maps.fix_diffusion_map_coordinates = list(
                map(int, self.diffusion_maps.fix_diffusion_map_coordinates)
            )

        if self.algorithm.pseudospectrum_residual_filter:
            if not self.visualization.compute_pseudospectrum:
                raise ValueError(
                    'Pseudospectrum computation must be turned on in order to use a residual filter on eigenpairs.'
                )
            if isinstance(self.algorithm.pseudospectrum_residual_filter, (float, int)):
                self.algorithm.pseudospectrum_residual_filter = {
                    'threshold': self.algorithm.pseudospectrum_residual_filter
                }

        if (
            not self.algorithm.has_measurement_noise
            and self.algorithm.learning_algo in ['s_dmd', 's_edmd']
            and not self.algorithm.use_temporal_embedding
        ):
            raise ValueError(
                f'{self.algorithm.learning_algo!r} without measurement noise needs use_temporal_embedding: '
                'at least one additional frame is needed for g and f to have an overlapping snapshot.'
            )

        if self.visualization.rotation is None:
            if self.data.load_data_from_checkpoint is None:
                # No checkpoint to inherit a rotation from: fall back to the default
                # now, since nothing downstream will resolve it later.
                self.visualization.rotation = 0
            # else: left unset for merge_checkpoint_config to resolve, from the
            # checkpoint's own rotation or, failing that, the same default.
        elif self.visualization.rotation not in SUPPORTED_ROTATION_VALUES:
            raise ValueError(
                f'Rotation values must be set to one of {SUPPORTED_ROTATION_VALUES}, got {self.visualization.rotation!r}.'
            )

        for cell in self.evaluation.measurement_cells:
            if not MEASUREMENT_CELL_PATTERN.match(cell):
                raise ValueError(f"Measurement cell {cell!r} must be formatted as '(x_idx_y_idx)', e.g. '(4_7)'.")

        # Note: the animation format is validated in
        # plots.config.get_animation_writer, where the writer is created.
        # In case the file type is unknown, the default animation writer is used.
