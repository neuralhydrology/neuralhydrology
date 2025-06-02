#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Optional

# make sure code directory is in path, even if the package is not installed using the setup.py
sys.path.append(str(Path(__file__).parent.parent))
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import setup_logging


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "continue_training", "finetune", "evaluate", "sweep"])
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--run-dir', type=str)
    parser.add_argument('--epoch', type=int, help="Epoch, of which the model should be evaluated")
    parser.add_argument('--period', type=str, choices=["train", "validation", "test"], default="test")
    parser.add_argument('--gpu', type=int,
                        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.")
    parser.add_argument('--sweep-id', type=str,
                        help="Sweep ID for wandb agent. If not provided, will use WANDB_SWEEP_ID environment variable.")
    args = vars(parser.parse_args())

    if (args["mode"] in ["train", "finetune", "sweep"]) and (args["config_file"] is None):
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
    elif args["mode"] == "sweep":
        sweep_run(config_file=Path(args["config_file"]), gpu=args["gpu"], sweep_id=args["sweep_id"])
    elif args["mode"] == "evaluate":
        eval_run(run_dir=Path(args["run_dir"]), period=args["period"], epoch=args["epoch"], gpu=args["gpu"])
    else:
        raise RuntimeError(f"Unknown mode {args['mode']}")


def start_run(config_file: Path, gpu: Optional[int] = None):
    """Start training a model.
    
    Parameters
    ----------
    config_file : Path
        Path to a configuration file (.yml), defining the settings for the specific run.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """

    config = Config(config_file)

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        config.device = "cpu"

    start_training(config)


def continue_run(run_dir: Path, config_file: Optional[Path] = None, gpu: Optional[int] = None):
    """Continue model training.
    
    Parameters
    ----------
    run_dir : Path
        Path to the run directory.
    config_file : Path, optional
        Path to an additional config file. Each config argument in this file will overwrite the original run config.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    # load config from base run and overwrite all elements with an optional new config
    base_config = Config(run_dir / "config.yml")

    if config_file is not None:
        base_config.update_config(config_file)
        base_config.run_dir = run_dir

    base_config.is_continue_training = True

    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        base_config.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        base_config.device = "cpu"

    start_training(base_config)


def finetune(config_file: Optional[Path] = None, gpu: Optional[int] = None):
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
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

    """
    if config_file is None:
        raise ValueError("config_file is required for finetuning")
        
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


def sweep_run(config_file: Path, gpu: Optional[int] = None, sweep_id: Optional[str] = None):
    """Run hyperparameter sweep using wandb.

    This function waits for wandb sweep to send hyperparameter configurations.
    For each configuration, it creates a modified config file with sweep parameters
    and starts training. The function loops continuously until the sweep is complete.

    Parameters
    ----------
    config_file : Path
        Path to the base configuration file (.yml). Sweep parameters will override
        values from this base configuration.
    gpu : int, optional
        GPU id to use. Will override config argument 'device'. A value smaller than zero indicates CPU.
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.
    sweep_id : str, optional
        The sweep ID to connect to. If not provided, it will be read from the WANDB_SWEEP_ID 
        environment variable.

    """
    import wandb
    import os
    
    def train_function():
        """Training function called by wandb.agent for each sweep run."""
        # Load base config
        base_config = Config(config_file)

        # Initialize wandb run - this will receive sweep parameters
        wandb.init(dir=base_config.run_dir)
        
        try:
            # Get sweep parameters from wandb
            sweep_params = dict(wandb.config)
            
            # Convert base config to dict and update with sweep parameters
            base_config.update_config(sweep_params)
            sweep_config = base_config 
            
            # Apply GPU override if specified
            if gpu is not None and gpu >= 0:
                sweep_config.device = f"cuda:{gpu}"
            if gpu is not None and gpu < 0:
                sweep_config.device = "cpu"
            
            # Start training with the sweep configuration
            start_training(sweep_config)
            
            # Save the sweep configuration file instead of deleting it
            if wandb.run is not None:
                # Create sweep directory in the same location as base config
                base_config_dir = config_file.parent
                
                # Get sweep name from wandb run
                sweep_name = "unknown_sweep"
                if hasattr(wandb.run, 'sweep_id') and wandb.run.sweep_id:
                    # Try to get sweep name from wandb API
                    try:
                        api = wandb.Api()
                        sweep = api.sweep(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}")
                        if hasattr(sweep, 'name') and sweep.name:
                            sweep_name = sweep.name
                        else:
                            # Fall back to sweep ID if no name is set
                            sweep_name = wandb.run.sweep_id
                    except:
                        # If API call fails, use sweep ID
                        sweep_name = wandb.run.sweep_id
                
                sweep_dir = base_config_dir / f"sweep_{sweep_name}"
                sweep_dir.mkdir(exist_ok=True)
                
                # Use wandb run name as filename
                run_name = wandb.run.name
                sweep_config_path = sweep_dir / f"{run_name}.yml"
                
                # Copy the temporary config to the sweep directory
                sweep_config.dump_config(sweep_config_path.parent, sweep_config_path.name)
                print(f"Saved sweep configuration to: {sweep_config_path}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            wandb.finish(exit_code=1)
            raise e
        
        finally:
            wandb.finish()
    
    # Get sweep_id from parameter or environment variable
    if sweep_id is None:
        sweep_id = os.getenv('WANDB_SWEEP_ID')
        if sweep_id is None:
            raise ValueError("sweep_id must be provided as parameter or set via WANDB_SWEEP_ID environment variable")
    
    # Use wandb.agent to loop and fetch configurations until sweep is complete
    wandb.agent(sweep_id=sweep_id, project="neuralhydrology", function=train_function)


def eval_run(run_dir: Path, period: str, epoch: Optional[int] = None, gpu: Optional[int] = None):
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
        Don't use this argument if you want to use the device as specified in the config file e.g. MPS.

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
