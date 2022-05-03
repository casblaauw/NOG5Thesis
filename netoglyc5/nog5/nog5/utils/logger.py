import yaml
import logging
import logging.config
from pathlib import Path

from .paths import log_path


class LoggingInfo:
    def __init__(self):
        self.LOG_LEVEL = logging.WARNING


logging_info = LoggingInfo()


def setup_logging(run_config: dict, verbose: int = 0, log_config_path: str = None):
    """ Setup ``logging.config``

    Args:
        run_config: path to configuration file for run
        verbose: stdout logging verbosity (0: WARNING, 1: INFO, 2: DEBUG)
        log_config_path : path to configuration file for logging
    """
    if verbose == 0:
        logging_info.LOG_LEVEL = logging.WARNING
    elif verbose == 1:
        logging_info.LOG_LEVEL = logging.INFO
    elif verbose >= 2:
        logging_info.LOG_LEVEL = logging.DEBUG

    if log_config_path is not None:
        log_config_path = Path(log_config_path)
        if log_config_path.exists():
            with open(log_config_path, "rt") as f:
                config = yaml.safe_load(f.read())
    else:
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'simple': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': logging_info.LOG_LEVEL,
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout',
                },
                'debug_file_handler': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': logging.DEBUG,
                    'formatter': 'simple',
                    'filename': 'debug.log',
                    'maxBytes': 10485760,
                    'backupCount': 10,
                    'encoding': 'utf8',
                }
            },
            'root': {
                'level': logging.DEBUG,
                'handlers': ['console', 'debug_file_handler'],
            },
        }

    # modify logging paths based on run config
    run_path = log_path(run_config)
    for handler in config["handlers"]:
        if "filename" in config["handlers"][handler]:
            config["handlers"][handler]["filename"] = str(run_path / config["handlers"][handler]["filename"])
    logging.config.dictConfig(config)
    log = setup_logger(__name__)
    if log_config_path is not None and not log_config_path.exists():
        log.warning(f'"{log_config_path}" not found. Using default logging config.')
    log.info(f"Set logging level to {logging.getLevelName(logging_info.LOG_LEVEL)}")


def setup_logger(name):
    log = logging.getLogger(f'nog5.{name}')
    log.setLevel(logging.DEBUG)
    return log
