#!/usr/bin/env python
import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


def _get_args() -> dict:

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "finetune", "continue_training", "continue_training_only"])
    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--gpu-ids', type=int, nargs='+', required=True)
    parser.add_argument('--runs-per-gpu', type=int, required=True)
    # Only used in continue_training / continue_training_only mode. If not provided, the target is read from
    # each run's config.yml.
    parser.add_argument('--target-epochs', type=int, default=None,
                        help="Total epoch target for continue_training. Overrides the value in each run's config.yml.")

    args = vars(parser.parse_args())

    args["directory"] = Path(args["directory"])
    if not args["directory"].is_dir():
        raise ValueError(f"No folder at {args['directory']}")

    return args


def _main():
    args = _get_args()
    schedule_runs(**args)


def _get_last_completed_epoch(run_dir: Path) -> int:
    """Return the epoch number of the most recent saved checkpoint in a run directory.

    Looks in the run directory itself and in any continue_training subfolders,
    since continued runs save weights one level deeper than the original run.
    """
    all_weights = list(run_dir.glob('model_epoch*.pt'))
    all_weights += list(run_dir.glob('continue_training_from_epoch*/model_epoch*.pt'))
    if not all_weights:
        return 0
    return max(int(p.stem[-3:]) for p in all_weights)


def _get_target_epochs_from_config(run_dir: Path) -> int:
    """Read the epoch target from the run's config.yml.

    We do a minimal parse here — just extract the 'epochs' line — to avoid
    importing the full Config class and its dependencies in the scheduler.
    """
    config_path = run_dir / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.yml found in {run_dir}")
    with open(config_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("epochs:"):
                return int(stripped.split(":")[1].strip())
    raise ValueError(f"Could not find 'epochs' key in {config_path}")


def _find_incomplete_runs(directory: Path, target_epochs: Optional[int]) -> List[Path]:
    """Return the list of run directories that have not yet reached their epoch target.

    A run is considered incomplete if its last saved checkpoint epoch is strictly
    less than the target. Runs that have no checkpoint at all (epoch 0) are included.
    """
    incomplete = []
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir():
            continue
        # skip the processed folder if someone put one here by mistake
        if run_dir.name == "processed":
            continue
        last_epoch = _get_last_completed_epoch(run_dir)
        try:
            target = target_epochs if target_epochs is not None else _get_target_epochs_from_config(run_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {run_dir.name}: {e}")
            continue
        if last_epoch < target:
            print(f"  {run_dir.name}: {last_epoch}/{target} epochs done — queuing for continuation")
            incomplete.append(run_dir)
        else:
            print(f"  {run_dir.name}: already at {last_epoch} epochs, skipping")
    return incomplete


def _has_any_started_run(directory: Path) -> bool:
    """Check whether at least one run in `directory` has ever saved a checkpoint.

    Used by 'continue_training_only' to distinguish "nothing to resume because everything
    is already finished" from "nothing to resume because training was never started" (e.g.
    the wrong directory was passed, or the run folders don't exist yet).
    """
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "processed":
            continue
        if _get_last_completed_epoch(run_dir) > 0:
            return True
    return False


def schedule_runs(mode: str, directory: Path, gpu_ids: List[int], runs_per_gpu: int,
                  target_epochs: Optional[int] = None):
    """Schedule multiple runs across one or multiple GPUs.

    Parameters
    ----------
    mode : {'train', 'evaluate', 'finetune', 'continue_training', 'continue_training_only'}
        Use 'train' to schedule fresh training from config files, 'evaluate' to evaluate trained models,
        'finetune' for finetuning, 'continue_training' to resume incomplete runs across multiple GPUs, or
        'continue_training_only' for the same behavior as 'continue_training' except it raises a RuntimeError
        instead of silently doing nothing when `directory` contains no run that has ever saved a checkpoint
        (e.g. wrong path, or training was never actually started).
    directory : Path
        For 'train' and 'finetune': path to a folder of .yml config files.
        For 'evaluate': path to a folder of run directories.
        For 'continue_training' / 'continue_training_only': path to a folder of run directories to resume.
    gpu_ids : List[int]
        List of GPU ids to use.
    runs_per_gpu : int
        Number of runs to start on a single GPU at a time.
    target_epochs : int, optional
        Only for 'continue_training' / 'continue_training_only'. If given, overrides the 'epochs' value from
        each run's config.yml.

    """

    if mode in ["train", "finetune"]:
        processes = list(directory.glob('*.yml'))
        processed_config_directory = directory / "processed"
        if not processed_config_directory.is_dir():
            processed_config_directory.mkdir()
    elif mode == "evaluate":
        processes = [p for p in directory.glob('*') if p.is_dir() and p.name != "processed"]
    elif mode in ("continue_training", "continue_training_only"):
        if mode == "continue_training_only" and not _has_any_started_run(directory):
            raise RuntimeError(
                f"continue_training_only: no run with a saved checkpoint was found in {directory}. "
                "Either the directory is wrong, or training was never started for these runs — "
                "use mode='train' to launch it from the config files first.")
        print(f"Scanning {directory} for incomplete runs...")
        run_dirs = _find_incomplete_runs(directory, target_epochs)
        if not run_dirs:
            print("All runs are already complete. Nothing to do.")
            sys.stdout.flush()
            return
        processes = run_dirs
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    # if used as command line tool, we need full paths to the files/directories
    processes = [str(p.absolute()) if isinstance(p, Path) else p for p in processes]

    # for approximately equal memory usage during hyperparam tuning, randomly shuffle list of processes
    random.shuffle(processes)

    # array to keep track on how many runs are currently running per GPU
    n_parallel_runs = len(gpu_ids) * runs_per_gpu
    gpu_counter = np.zeros((len(gpu_ids)), dtype=int)

    # for command line tool, we need full path to the main.py script
    script_path = str(Path(__file__).absolute().parent / "nh_run.py")

    running_processes = {}
    counter = 0
    while True:

        # start new runs
        for _ in range(n_parallel_runs - len(running_processes)):

            if counter >= len(processes):
                break

            # determine which GPU to use
            node_id = np.argmin(gpu_counter)
            gpu_counter[node_id] += 1
            gpu_id = gpu_ids[node_id]
            process = processes[counter]

            # build the command depending on mode
            if mode in ['train', 'finetune']:
                run_command = f"python {script_path} {mode} --config-file {process} --gpu {gpu_id}"
            elif mode in ('continue_training', 'continue_training_only'):
                run_command = f"python {script_path} continue_training --run-dir {process} --gpu {gpu_id}"
            else:
                run_command = f"python {script_path} evaluate --run-dir {process} --gpu {gpu_id}"

            print(f"Starting run {counter+1}/{len(processes)}: {run_command}")
            running_processes[(run_command, node_id, process)] = subprocess.Popen(run_command,
                                                                                  stdout=subprocess.DEVNULL,
                                                                                  shell=True)

            counter += 1
            time.sleep(2)

        # check for completed runs
        for key, process in running_processes.items():
            if process.poll() is not None:
                print(f"Finished run {key[0]}")
                gpu_counter[key[1]] -= 1
                print("Cleaning up...\n\n")
                try:
                    _ = process.communicate(timeout=5)
                except TimeoutError:
                    print('')
                    print("WARNING: PROCESS {} COULD NOT BE REAPED!".format(key))
                    print('')
                running_processes[key] = None
                if mode in ["train", "finetune"]:
                    dst = processed_config_directory / Path(key[2]).name
                    try:
                        shutil.move(src=key[2], dst=dst)
                        print(f"Moved {key[2]} into directory of processed configs at {dst}.")
                    except Exception as e:
                        # we ignore move errors so the scheduler keeps going for the other runs
                        print(f"Couldn't move {key[2]} because of {e}.")

        # delete possibly finished runs
        running_processes = {key: val for key, val in running_processes.items() if val is not None}
        time.sleep(2)

        if (len(running_processes) == 0) and (counter >= len(processes)):
            break

    print("Done")
    sys.stdout.flush()


if __name__ == "__main__":
    _main()
