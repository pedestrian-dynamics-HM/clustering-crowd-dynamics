"""Configuration loading and checkpoint resolution service.

Consolidates CLI parsing, YAML loading, checkpoint resolution, and config operations.
This is the single source of truth for all config-related logic.

Responsibilities:
- Parse CLI arguments and load YAML configuration
- Apply CLI overrides and validate consistency
- Generate run IDs and compute derived paths
- Track git SHA for reproducibility
- Create folder structure for run artifacts
- Resolve and merge checkpoint configurations
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import dill
import git
import pandas as pd
from datafold import EDMD

from .configs import FullConfig, FILE_DELIMITER
from .checkpoint_policy import merge_checkpoint_config
from ..utils.persistence import save_config

log = logging.getLogger(__name__)


# ============================================================================
# CONFIG OPERATIONS: Run ID generation, path computation, git tracking
# ============================================================================


def _generate_run_id(config_file: str) -> str:
    """Generate unique run ID based on config file name and timestamp.

    Format: <config_name>_<YYYYMMDD_HHMMSS>
    """
    ts = datetime.now(tz=ZoneInfo('Europe/Berlin')).strftime('%Y%m%d_%H%M%S')
    config_name = config_file.split('.')[0]
    return config_name + FILE_DELIMITER + ts


def _track_git_sha(config: FullConfig) -> None:
    """Track current git commit SHA for reproducibility.

    Modifies config in-place, setting meta.git_sha.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        config.meta.git_sha = 'not_a_git_repo'
        return

    # A freshly initialized repo has a HEAD that points to a branch without any commit
    if not repo.head.is_valid():
        config.meta.git_sha = 'no_commit_yet'
        return

    config.meta.git_sha = repo.head.commit.hexsha


# ============================================================================
# CONFIG LOADING: Parse CLI args, load YAML, apply overrides
# ============================================================================


def _parser_add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments and default values to parser. If the argument is None its value is later read them from the default.py file."""

    # Get the project root directory (parent of utils folder)
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(utils_dir)
    project_root = os.path.abspath(os.path.join(project_root, '../..'))

    parser.add_argument(
        '--workspace-path',
        type=str,
        default=project_root,
        help='Path to workspace, used as basis ',
    )

    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Run ID for the current run. If not specified, a timestamp will be used.',
    )

    parser.add_argument(
        '--data-path',
        dest='raw_data_dir',
        type=str,
        default=None,
        help=(
            'Directory to load the raw trajectory data from, holding one subdirectory '
            'per scenario. Relative to workspace-path, or absolute. Defaults to the '
            'config file value. Despite the flag name this sets meta.raw_data_dir, not '
            'the derived meta.raw_data_path -- unlike the legacy "data_path" YAML key, '
            'the value is used as given, not reshaped.'
        ),
    )

    parser.add_argument('--log-level', type=str, default=None, help='If specified, it sets the log level.')

    parser.add_argument(
        '-s',
        '--settings-file-path',
        type=str,
        default='default.yaml',
        help='If specified, will load settings from workspace_path + this_path, this will overwrite default configs.',
    )

    parser.add_argument(
        '--load-data-from-checkpoint',
        type=str,
        default=None,
        help='If specified, will load data and config from the checkpoint directory.',
    )

    parser.add_argument(
        '--run-artifacts-path',
        dest='run_artifacts_dir',
        type=str,
        default=None,
        help=(
            'Subdirectory (relative to workspace-path) to save run artifacts to. '
            'Defaults to the config file value (normally "runs"). Despite the flag '
            'name, this sets meta.run_artifacts_dir, not the derived '
            'meta.run_artifacts_path -- see run_artifacts_dir vs run_artifacts_path '
            'in configs.py.'
        ),
    )

    parser.add_argument(
        '--rank',
        type=int,
        default=None,
        help='Rank of the Koopman operator.',
    )

    parser.add_argument(
        '--n-eigenpairs',
        type=int,
        default=None,
        help='Number of diffusion map coordinates.',
    )

    parser.add_argument(
        '-e',
        '--epsilon',
        type=float,
        default=None,
        help='Kernel width for the diffusion map',
    )

    parser.add_argument(
        '-c',
        '--kernel-cutoff',
        type=float,
        default=None,
        help='Cutoff for the kernel',
    )

    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=None,
        help='Number of neighbors for the diffusion map continuous nn kernel',
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=None,
        help='Delta for the diffusion map continuous nn kernel',
    )

    parser.add_argument(
        '-n',
        '--num-t-delays',
        type=int,
        default=None,
        help='Number of time delays for the temporal embedding',
    )

    parser.add_argument(
        '-f',
        '--frequency',
        type=int,
        default=None,
        help='Frequency of the temporal embedding',
    )

    parser.add_argument(
        '-l',
        '--lag',
        type=int,
        default=None,
        help='Number of lags for the temporal embedding',
    )

    parser.add_argument(
        '--mae-threshold',
        '--mae',
        type=float,
        default=None,
        help='Threshold for the mean absolute error',
    )

    parser.add_argument(
        '--auto-select-diffusion-maps',
        type=int,
        default=None,
        help='Number of diffusion maps to select automatically',
    )

    parser.add_argument(
        '--fix-diffusion-map-coordinates',
        nargs='+',
        type=int,
        default=None,
        help='Number of diffusion maps to fix',
    )

    parser.add_argument(
        '--train-parameter',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Series ids to train on, overriding the config file value. Mainly useful '
            'together with --load-data-from-checkpoint, where it selects a subset of '
            "the checkpoint's timeseries -- e.g. a single series -- instead of "
            'training on all of them. Unset, the config file value applies, which '
            'itself falls back to every series the checkpoint holds.'
        ),
    )

    parser.add_argument(
        '--tlim',
        nargs='+',  # at least one value
        type=int,
        default=None,
        help='Time limits for the rolling time frame',
    )

    parser.add_argument(
        '--prediction-timeshift',
        type=int,
        default=None,
        help='Time step index of the initial condition for predictions.',
    )

    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        # None (not False) as the default, so that an absent flag is dropped by the
        # None-filter in parse_and_load_config and leaves the config file's value alone.
        default=None,
        help=(
            'Stop after preprocessing: load and grid the data, persist it and plot the '
            'ground truth, but fit no model. Useful to produce the timeseries a later '
            'run resumes from with --load-data-from-checkpoint.'
        ),
    )

    parser.add_argument(
        '--no-gifs',
        action='store_true',
        help=(
            'Skip gif/animation generation for this run, regardless of the config '
            'file settings (generate_mode_gifs, generate_timeseries_gifs). '
            'Animations are the slowest part of a run and are irrelevant when only '
            'the numeric metrics are needed, e.g. for baseline recording.'
        ),
    )

    return parser


def parse_and_load_config(user_args: list[str]) -> FullConfig:
    """Parse the user arguments and load the configuration from the settings file.
    The settings file is loaded from the workspace path and the path specified in the user arguments."""

    # Use a parser to allow inputs via terminal. These inputs are needed to allow specification of a config yaml file
    parser = argparse.ArgumentParser(description='Command line parser for model training setup.')
    parser = _parser_add_arguments(parser)
    parsed_args = parser.parse_args(user_args)
    # make the arguments a dictionary and remove all the None values, i.e. settings with no user input
    # otherwise the default values would be overwritten by None
    user_config_dict = {key: value for key, value in vars(parsed_args).items() if value is not None}
    log.info(f'Arguments set by the user: {user_config_dict}')

    # no_gifs doesn't map onto a single config field (it clears two), so it is
    # applied separately below rather than through the generic dispatch in apply_cli_overrides.
    no_gifs = user_config_dict.pop('no_gifs', False)

    assert 'settings_file_path' in user_config_dict, (
        'settings_file_path should be set in user_config_dict. Check the default.'
    )  # this should always be the case, because we also set a default there

    # Load config from YAML
    settings_file = user_config_dict['settings_file_path']
    config = FullConfig.from_yaml(settings_file)

    # A dumped run config (e.g. a checkpoint's config.yaml reused as settings file)
    # carries the run_id of the run that produced it. Keeping it would make this run
    # resolve unique_run_path to that previous run's folder and overwrite its
    # artifacts. run_id is runtime state, not configuration, so drop it here and let
    # initialize_run_folders generate a fresh one. An explicit --run-id still wins,
    # since the CLI overrides are applied below.
    if config.meta.run_id is not None:
        log.warning(
            "Ignoring run_id '%s' found in %s: generating a new run_id so the previous "
            "run's artifacts are not overwritten. Pass --run-id to reuse it explicitly.",
            config.meta.run_id,
            settings_file,
        )
        config.meta.run_id = None
        config.meta.resolve_paths()

    # Apply CLI overrides
    config.apply_cli_overrides(user_config_dict)

    # apply the --no-gifs setting
    if no_gifs:
        config.visualization.generate_mode_gifs = None
        config.visualization.generate_timeseries_gifs = None

    # Run consistency checks
    config.check_config_consistency()

    return config


# ============================================================================
# CHECKPOINT: Resolve, load, merge
# ============================================================================


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Checkpoint path plus merged runtime config."""

    path: Path
    config: FullConfig


def find_checkpoint_path(run_config: FullConfig) -> Path:
    """Find checkpoint directory by run ID."""
    checkpoint_run_id = run_config.data.load_data_from_checkpoint
    if checkpoint_run_id is None:
        raise ValueError('load_data_from_checkpoint must be set to load a checkpoint')

    try:
        checkpoint_run_artifacts_path = glob.glob(
            f'**/**/*{checkpoint_run_id}',
            root_dir=str(run_config.meta.root_path),
            recursive=True,
        )[0]
    except IndexError as e:
        raise FileNotFoundError(
            f'Could not find checkpoint directory with run id {checkpoint_run_id} in {run_config.meta.run_artifacts_path}'
        ) from e

    return Path(run_config.meta.root_path) / checkpoint_run_artifacts_path


def resolve_checkpoint_config(run_config: FullConfig) -> ResolvedCheckpoint:
    """Find and merge checkpoint config with current config.

    This method searches for the checkpoint path based on the config,
    loads the checkpoint config, merges it with the current config, and returns the resolved checkpoint.
    Fields left unset on run_config (rotation, tlim, train/test_parameter) are backfilled
    from the checkpoint's own config.yaml as part of that merge -- see
    checkpoint_policy._UNSET_DEFERS_TO_CHECKPOINT and its tlim handling.
    """

    checkpoint_path = find_checkpoint_path(run_config)
    checkpoint_config_dict = FullConfig._load_yaml(checkpoint_path / 'config.yaml')
    resolved_config, merge_report = merge_checkpoint_config(run_config, checkpoint_config_dict)
    if merge_report.applied_paths:
        log.info(f'Applied checkpoint override policy to {len(merge_report.applied_paths)} fields')
    resolved_config.check_config_consistency()

    return ResolvedCheckpoint(path=checkpoint_path, config=resolved_config)


def load_checkpoint_model(run_config: FullConfig, checkpoint_path: Path | None = None) -> EDMD:
    """Load trained EDMD model from checkpoint."""
    checkpoint_path = checkpoint_path or find_checkpoint_path(run_config)
    model_path = checkpoint_path / 'models' / f'{run_config.algorithm.learning_algo}_model.pkl'
    with open(model_path, 'rb') as f:
        return dill.load(f)


def load_diffusion_eigenvalues(run_config: FullConfig, checkpoint_path: Path | None = None) -> pd.DataFrame:
    """Load diffusion map eigenvalues from checkpoint."""
    checkpoint_path = checkpoint_path or find_checkpoint_path(run_config)
    diff_evs_path = checkpoint_path / 'data' / 'diffusion_map_eigenvalues.csv'
    return pd.read_csv(diff_evs_path)


# ============================================================================
# FOLDER STRUCTURE: Create run directories and logging setup
# ============================================================================


def initialize_run_folders(config: FullConfig) -> None:
    """Initialize folder structure and compute paths for a run.

    This function:
    1. Generates run ID (if not already set via CLI)
    2. Computes all derived paths
    3. Tracks git SHA
    4. Creates all required directories

    Args:
        config: FullConfig with base paths already set.
    """
    # Step 1: Generate run ID if not set via CLI
    if config.meta.run_id is None:
        config.meta.run_id = _generate_run_id(config.meta.config_file)

    # Step 2: Compute derived paths with updated run_id
    config.meta.resolve_paths()

    # Step 3: Track git SHA
    _track_git_sha(config)

    # Step 4: Create directories
    log.info(f'Initializing folder structure at {config.meta.unique_run_path}')

    # Create root directories
    config.meta.root_path.mkdir(parents=True, exist_ok=True)
    config.meta.run_artifacts_path.mkdir(parents=True, exist_ok=True)

    # Create run-specific subdirectories (should not exist since they are unique!)
    config.meta.unique_run_path.mkdir(parents=True, exist_ok=False)
    config.meta.processed_data_path.mkdir(parents=True, exist_ok=False)
    config.meta.evaluation_path.mkdir(parents=True, exist_ok=False)
    config.meta.interpretation_path.mkdir(parents=True, exist_ok=False)
    config.meta.gifs_path.mkdir(parents=True, exist_ok=False)

    log.info('Folder structure initialized successfully')


def bootstrap_config(user_args: list[str]) -> FullConfig:
    """Complete config bootstrap: parse, validate, merge, compute paths, create folders.

    This is the main entry point for initializing configuration for a new experiment.
    It performs:
    1. Parse CLI arguments
    2. Load YAML configuration
    3. Apply CLI overrides
    4. Validate config consistency
    5. Merge with the checkpoint config, if loading from one
    6. Generate run ID
    7. Compute paths
    8. Track git SHA
    9. Create folder structure
    10. Persist the fully-resolved config

    Step 5 runs before folders are created and the config is persisted (step 10) --
    not after -- so this run's own config.yaml is written already fully resolved
    (rotation, tlim, train/test_parameter inherited from the checkpoint where this
    run left them unset). A further run chaining off this one as its own checkpoint
    then always finds real values on disk, never the pre-merge placeholders.

    Args:
        user_args: Command-line arguments (typically sys.argv[1:])

    Returns:
        Fully initialized FullConfig with folders created and paths computed.
    """
    # Steps 1-4: Parse and load config
    config = parse_and_load_config(user_args)

    # Step 5: merge with the checkpoint config, if this run loads from one
    if config.data.load_data_from_checkpoint is not None:
        config = resolve_checkpoint_config(config).config

    # Steps 6-9: Initialize filesystem and operations
    initialize_run_folders(config)

    # Step 10: store the fully-resolved config
    save_config(config, config.meta.unique_run_path / 'config.yaml')

    return config
