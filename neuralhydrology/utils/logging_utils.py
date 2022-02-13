import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

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


def get_git_hash() -> Optional[str]:
    """Get git commit hash of the project if it is a git repository.

    Returns
    -------
    Optional[str]
        Git commit hash if project is a git repository, else None.
    """
    # get git commit hash if folder is a git repository
    current_dir = str(Path(__file__).absolute().parent)
    try:
        if subprocess.call(["git", "-C", current_dir, "branch"], stderr=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL) == 0:
            return subprocess.check_output(["git", "-C", current_dir, "describe", "--always"]).strip().decode('ascii')
    except OSError:
        return None  # likely, git is not installed.


def save_git_diff(run_dir: Path):
    """Try to store the git diff to a file.

    Parameters
    ----------
    run_dir : Path
        Directory of the current run.
    """
    base_dir = str(Path(__file__).absolute().parent)
    try:
        # diff should include staged and unstaged changes, hence we use "HEAD"
        out = subprocess.check_output(['git', '-C', base_dir, 'diff', 'HEAD'], stderr=subprocess.DEVNULL)
    except OSError:
        LOGGER.warning('Could not store git diff, likely because git is not installed '
                       'or because your version of git is too old (< 1.8.5)')
        return

    new_diff = out.strip().decode('utf-8')

    if new_diff:
        existing_diffs = list(run_dir.glob('neuralhydrology*.diff'))
        if len(existing_diffs) > 0:
            last_diff_path = run_dir / f'neuralhydrology-{len(existing_diffs) - 1}.diff'
            with last_diff_path.open('r') as last_diff_file:
                last_diff = last_diff_file.read()
            if last_diff == new_diff:
                LOGGER.info(f'Git repository contains uncommitted changes that are stored in {last_diff_path}.')
                return

        file_path = run_dir / f'neuralhydrology-{len(existing_diffs)}.diff'
        LOGGER.warning(f'Git repository contains uncommitted changes. Writing diff to {file_path}.')
        with file_path.open('w') as diff_file:
            diff_file.write(new_diff)
