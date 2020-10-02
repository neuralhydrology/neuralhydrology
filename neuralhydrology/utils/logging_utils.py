import logging
import sys

LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: str):
    """Initialize logging to `log_file` and stdout.

    Parameters
    ----------
    log_file : str
        Name of the file that will be logged to.
    """
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(handlers=[file_handler, stdout_handler], level=logging.INFO, format="%(asctime)s: %(message)s")

    # Make sure we log uncaught exceptions
    def exception_logging(type, value, tb):
        LOGGER.exception(f"Uncaught exception", exc_info=(type, value, tb))

    sys.excepthook = exception_logging

    LOGGER.info(f"Logging to {log_file} initialized.")
