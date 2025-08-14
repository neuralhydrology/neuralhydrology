from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.figure

from neuralhydrology.training.logging.logger import Logger

try:
    import wandb
except ImportError as e:
    raise ImportError("WandB is not installed. Run pip install wandb") from e

from neuralhydrology.utils.config import Config


class WandBLogger(Logger):
    """Class that logs runs to WandB and saves plots to disk.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.run = None

    def start_logger(self):
        """ Start WandB logging. """
        # Check if wandb is already initialized (e.g., in sweep mode)
        if wandb.run is not None:
            # Use the existing wandb run (for sweeps)
            self.run = wandb.run
            # Update the config with our run configuration
            wandb.config.update(self.cfg.as_dict(), allow_val_change=True)
        else:
            # Initialize a new wandb run (for normal training)
            self.run = wandb.init(
                project=self.cfg.wandb_project,
                dir=self.log_dir / "wandb",
                tags=[self.cfg.experiment_name],
                config=self.cfg.as_dict()
            )

    def stop_logger(self):
        """ Stop WandB logging. """
        self.run.finish()

    def _log_figure(self, figure: matplotlib.figure.Figure, key: str, idx: int):
        self.run.log({f"{key}_{idx + 1}": wandb.Image(figure)}, step=self.update)

    def log_metric(self, metric_name: str, value: float, step: int):
        self.run.log({metric_name: value})

    def log_model(self, weight_path, optimizer_path):
        self.run.log_artifact(
            weight_path,
            name=f"{self.cfg.experiment_name}_model",
            aliases=[f"epoch-{self.epoch}", f"step-{self.update}"],
            type="model",
        )
        if optimizer_path is not None:
            self.run.log_artifact(
                optimizer_path,
                name=f"{self.cfg.experiment_name}_optimizer",
                aliases=[f"epoch-{self.epoch}", f"step-{self.update}"],
                type="optimizer",
            )

    def log_lr(self, learning_rate: float):
        """Log current learning rate.

        Parameters
        ----------
        learning_rate : float
            Current learning rate value.
        """
        self.run.log({"train/learning_rate": learning_rate}, step=self.update)