#!/usr/bin/env python
import argparse
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


def _get_args() -> dict:

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "finetune"])
    parser.add_argument('--directory', type=str, required=True)
    parser.add_argument('--gpu-ids', type=int, nargs='+', required=True)
    parser.add_argument('--runs-per-gpu', type=int, required=True)

    args = vars(parser.parse_args())

    args["directory"] = Path(args["directory"])
    if not args["directory"].is_dir():
        raise ValueError(f"No folder at {args['directory']}")

    return args


def _main():
    args = _get_args()
    schedule_runs(**args)


def schedule_runs(mode: str, directory: Path, gpu_ids: List[int], runs_per_gpu: int):
    """Schedule multiple runs across one or multiple GPUs.
    
    Parameters
    ----------
    mode : {'train', 'evaluate', 'finetune'}
        Use 'train' if you want to schedule training of multiple models, 'evaluate' if you want to schedule
        evaluation of multiple trained models and 'finetune' if you want to schedule finetuning with multiple configs.
    directory : Path
        If mode is one of {'train', 'finetune'}, this path should point to a folder containing the config files (.yml) 
        to use for model training/finetuning. For each config file, one run is started. If mode is 'evaluate', this path 
        should point to the folder containing the different model run directories.
    gpu_ids : List[int]
        List of GPU ids to use for training/evaluating.
    runs_per_gpu : int
        Number of runs to start on a single GPU.

    """

    if mode in ["train", "finetune"]:
        processes = list(directory.glob('*.yml'))
        processed_config_directory = directory / "processed"
        if not processed_config_directory.is_dir():
            processed_config_directory.mkdir()
    elif mode == "evaluate":
        processes = list(directory.glob('*'))
    else:
        raise ValueError("'mode' must be either 'train' or 'evaluate'")

    # if used as command line tool, we need full path's to the fils/directories
    processes = [str(p.absolute()) for p in processes]

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

            # start run via subprocess call
            if mode in ['train', 'finetune']:
                run_command = f"python {script_path} {mode} --config-file {process} --gpu {gpu_id}"
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
                        # We ignore any error that could happen when moving the file. In the worst case, we don't move 
                        # anything but we don't want the scheduler to stop for that.
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
