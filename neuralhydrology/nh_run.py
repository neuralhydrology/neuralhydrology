#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

# make sure code directory is in path, even if the package is not installed using the setup.py
sys.path.append(str(Path(__file__).parent.parent))
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "continue_training", "finetune", "evaluate"])
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--run-dir', type=str)
    parser.add_argument('--epoch', type=int, help="Epoch, of which the model should be evaluated")
    parser.add_argument('--period', type=str, choices=["train", "validation", "test"], default="test")
    parser.add_argument('--gpu', type=int,
                        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.")
    args = vars(parser.parse_args())

    if (args["mode"] in ["train", "finetune"]) and (args["config_file"] is None):
        raise ValueError("Missing path to config file")

    if (args["mode"] == "continue_training") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory file")

    if (args["mode"] == "evaluate") and (args["run_dir"] is None):
        raise ValueError("Missing path to run directory")

    return args


def _main():
    args = _get_args()
    if (args["run_dir"] is not None) and (args["mode"] == "evaluate"):
        setup_logging(str(Path(args["run_dir"]) / "output.log"))

    if args["mode"] == "train":
        start_run(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["mode"] == "continue_training":
        continue_run(run_dir=Path(args["run_dir"]),
                     config_file=Path(args["config_file"]) if args["config_file"] is not None else None,
                     gpu=args["gpu"])
    elif args["mode"] == "finetune":
        finetune(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["mode"] == "evaluate":
        eval_run(run_dir=Path(args["run_dir"]), period=args["period"], epoch=args["epoch"], gpu=args["gpu"])
    else:
        raise RuntimeError(f"Unknown mode {args['mode']}")


def start_run(config_file: Path, gpu: int = None):
    """Start training a model.
    
    Parameters
    ----------
    config_file : Path
        Path to a configuration file (.yml), defining the settings for the specific run.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """

    config = Config(config_file)

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config)


def continue_run(run_dir: Path, config_file: Path = None, gpu: int = None):
    """Continue model training.
    
    Parameters
    ----------
    run_dir : Path
        Path to the run directory.
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """
    # load config from base run and overwrite all elements with an optional new config
    base_config = Config(run_dir / "config.yml")

    if config_file is not None:
        base_config.update_config(config_file)

    base_config.is_continue_training = True

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        base_config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        base_config.device = "cpu"

    start_training(base_config)


def finetune(config_file: Path = None, gpu: int = None):
    """Finetune a pre-trained model.

    Parameters
    ----------
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
        The config file for finetuning must contain the argument `base_run_dir`, pointing to the folder of the 
        pre-trained model, as well as 'finetune_modules' to indicate which model parts will be trained during
        fine-tuning.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.

    """
    # load finetune config and check for a non-empty list of finetune_modules
    temp_config = Config(config_file)
    if not temp_config.finetune_modules:
        raise ValueError("For finetuning, at least one model part has to be specified by 'finetune_modules'.")

    # extract base run dir, load base run config and combine with the finetune arguments
    config = Config(temp_config.base_run_dir / "config.yml")
    config.update_config({'run_dir': None, 'experiment_name': None})
    config.update_config(config_file)
    config.is_finetuning = True

    # if the base run was a continue_training run, we need to override the continue_training flag from its config.
    config.is_continue_training = False

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config)


def eval_run(run_dir: Path, period: str, epoch: int = None, gpu: int = None):
    """Start evaluating a trained model.
    
    Parameters
    ----------
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    epoch : int, optional
        Define a specific epoch to use. By default, the weights of the last epoch are used.  
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value less than zero indicates CPU.

    """
    config = Config(run_dir / "config.yml")

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_evaluation(cfg=config, run_dir=run_dir, epoch=epoch, period=period)


if __name__ == "__main__":
    _main()
