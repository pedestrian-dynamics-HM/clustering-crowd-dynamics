import logging
import os

import colorlog

# Console format, terser the higher the level. The file always gets the full format.
STREAM_FORMATS = {
    logging.WARNING: '%(levelname)s: %(message)s',
    logging.INFO: '%(levelname)s: %(asctime)s: %(message)s',
    logging.DEBUG: '%(levelname)s: %(asctime)s: %(name)s: %(message)s',
}

FILE_FORMAT = '%(levelname)s: %(asctime)s: %(name)s: %(message)s'

# Only the level name is coloured, so the message itself stays readable on both
# light and dark backgrounds. Avoid 'white' here: it renders as the terminal's
# normal foreground and makes the level look uncoloured.
LOG_COLORS = {'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'}

# The top-level package, i.e. "dynamics_analysis_with_koopman". Every module logger
# is a child of it, so configuring this one logger covers the whole pipeline.
PACKAGE_LOGGER_NAME = __name__.split('.')[0]

# Level used before a config has been read and the real level is known.
EARLY_LOG_LEVEL = 'INFO'


class _EarlyRecords(logging.Handler):
    """Holds records logged before the run folder — and its log file — exist."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


# Set by setup_console_logger, consumed and cleared by setup_custom_logger.
_early_records: _EarlyRecords | None = None


def _to_level_number(log_level) -> int:
    """'INFO' -> 20. Config validation already restricts this to the level names."""
    level = logging.getLevelName(str(log_level).upper())
    if not isinstance(level, int):
        raise ValueError(f'Unknown log level {log_level!r}. Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL.')
    return level


def _stream_format(level: int) -> str:
    for threshold in (logging.WARNING, logging.INFO):
        if level >= threshold:
            return STREAM_FORMATS[threshold]
    return STREAM_FORMATS[logging.DEBUG]


def _use_color(stream) -> bool:
    """Colour only when writing to a real terminal that has not opted out.

    Without this, redirecting output to a file writes raw escape codes into it,
    and consoles that do not interpret ANSI (the VS Code debug console, some
    notebook frontends) show the codes as noise.
    """
    if os.environ.get('NO_COLOR'):  # https://no-color.org/
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return hasattr(stream, 'isatty') and stream.isatty()


def _build_stream_handler(level: int) -> logging.StreamHandler:
    """Console handler: stderr, terse format, level name coloured when useful."""
    stream_format = _stream_format(level)
    stream_handler = logging.StreamHandler()  # stderr, the logging default
    stream_handler.setLevel(level)
    if _use_color(stream_handler.stream):
        # Wrap only the level name, so the message keeps the default colour.
        coloured = stream_format.replace('%(levelname)s', '%(log_color)s%(levelname)s%(reset)s', 1)
        stream_handler.setFormatter(colorlog.ColoredFormatter(fmt=coloured, log_colors=LOG_COLORS))
    else:
        stream_handler.setFormatter(logging.Formatter(fmt=stream_format))
    return stream_handler


def _configure(name, level, handlers, keep=None) -> logging.Logger:
    """Attach `handlers` to the `name` logger, replacing whatever was there before.

    The previous handlers are detached and closed, so a second setup call does not
    make every line appear twice. `keep` is detached but left open.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        if handler is not keep:
            handler.close()

    logger.setLevel(level)
    for handler in handlers:
        logger.addHandler(handler)
    # Do not hand records on to the root logger: if anything else calls
    # logging.basicConfig() every line would otherwise be printed twice.
    logger.propagate = False
    return logger


def setup_console_logger(log_level=EARLY_LOG_LEVEL, name=PACKAGE_LOGGER_NAME):
    """Log to the console only, for the phase before the run folder exists.

    `main.log` lives in `runs/<run_id>/`, which `bootstrap_config` creates only
    after it has parsed the CLI arguments and read the YAML — but that work
    already has things to report. Call this first and those messages appear in
    the terminal immediately; `setup_custom_logger` then takes over once the
    path is known.

    Nothing is lost in the meantime: the records are also kept in memory and
    replayed into the log file by `setup_custom_logger`. They are kept down to
    DEBUG even though the console shows `log_level` and above, so a config that
    turns out to ask for DEBUG still gets the early lines in its file.
    """
    global _early_records

    level = _to_level_number(log_level)
    _early_records = _EarlyRecords()

    # DEBUG on the logger with the filtering on the handlers: the console still
    # honours log_level, while the early records are free to keep everything.
    return _configure(name, logging.DEBUG, [_build_stream_handler(level), _early_records])


def setup_custom_logger(log_path, log_level, name=PACKAGE_LOGGER_NAME):
    """Send log output to both `log_path` and the console.

    By default this configures the package logger, so every module that does
    `logging.getLogger(__name__)` writes through these handlers without needing
    to know about this function. Third-party libraries (datafold, matplotlib)
    keep their own logging and are not pulled in. Pass `name=None` to configure
    the root logger instead and capture them too.

    Calling this a second time replaces the handlers instead of adding another
    set, so a process that runs more than one experiment does not log each line
    twice.

    If `setup_console_logger` ran first, everything it kept is written to the
    new log file here — to the file only, since those lines already reached the
    console when they were logged.
    """
    global _early_records

    level = _to_level_number(log_level)

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(fmt=FILE_FORMAT))

    # Set the level on the logger itself, not just the handlers: the pipeline
    # guards expensive debug-only work with `isEnabledFor(logging.DEBUG)`, and
    # that check reads the logger's level. The early records are detached
    # without being closed: they still have to be replayed.
    logger = _configure(
        name,
        level,
        [file_handler, _build_stream_handler(level)],
        keep=_early_records,
    )

    if _early_records is not None:
        for record in _early_records.records:
            if record.levelno >= level:
                file_handler.handle(record)
        _early_records.close()
        _early_records = None

    return logger
