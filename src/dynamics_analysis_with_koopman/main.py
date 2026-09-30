import logging
import sys
import warnings

# Set before the scientific stack is imported, so its FutureWarnings stay out of
# the run log.
warnings.filterwarnings('ignore', category=FutureWarning)

from dynamics_analysis_with_koopman.pipeline.experiment import retrain_from_checkpoint, train_experiment  # noqa: E402
from dynamics_analysis_with_koopman.config.loading import bootstrap_config  # noqa: E402
from dynamics_analysis_with_koopman.utils.logger import setup_console_logger, setup_custom_logger  # noqa: E402

log = logging.getLogger(__name__)


def _cli_args(user_args: list[str] | None) -> list[str]:
    """Fall back to the process arguments when called as a console script."""
    return sys.argv[1:] if user_args is None else user_args


def _prepare_config_and_logger(user_args: list[str]):
    """Bootstrap config and initialize logger once for both train/retrain flows."""
    # The log file lives in the run folder, which does not exist yet: log to the
    # console for now. setup_custom_logger writes these lines to the file too,
    # once bootstrap_config has created it.
    setup_console_logger()
    log.info('Loading configuration from file, arguments from terminal: %s', user_args)
    config = bootstrap_config(user_args)

    # bootstrap_config created the run folder, so the log file can be written now.
    setup_custom_logger(config.meta.log_file_path, config.meta.log_level)
    log.info('Logging to %s at level %s', config.meta.log_file_path, config.meta.log_level)
    log.debug('Loaded run configuration: %s', config.get_full_config_dict())
    return config, log


def main(user_args=None):
    """Train a new model with complete config bootstrapping.

    Loads config from YAML, applies CLI overrides, initializes folder structure,
    and executes the training pipeline.

    user_args defaults to the command line, so this doubles as the
    `koopman-train` console script; pass a list to call it from a notebook.
    """
    config, log = _prepare_config_and_logger(_cli_args(user_args))
    # Internal invariant: initialize_run_folders always assigns a run_id.
    assert config.meta.run_id is not None, 'Run configuration could not be loaded.'

    result = train_experiment(config)
    log.info('Training finished for run %s', result.run_id)
    return result


def retrain(user_args=None):
    """Retrain from a checkpoint with complete config bootstrapping.

    Loads config from YAML, applies CLI overrides, initializes folder structure,
    and executes the retraining pipeline.

    As with main(), user_args defaults to the command line — this is the
    `koopman-retrain` console script.
    """
    config, log = _prepare_config_and_logger(_cli_args(user_args))

    if config.data.load_data_from_checkpoint is None:
        raise ValueError('retrain requires load_data_from_checkpoint to be set')

    if config.algorithm.use_diffusion_maps:
        result = retrain_from_checkpoint(config)
        log.info('Retraining finished for run %s', result.run_id)
    else:
        raise ValueError(
            'Config has use_diffusion_maps=false; Retraining requires diffusion maps '
            'because it keeps them fix, subselecting them and retraining the DMD only.'
        )

    return result


if __name__ == '__main__':
    main(sys.argv[1:])
