import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union, List

import matplotlib as mpl
import numpy as np
try:
    import wandb
except ImportError as e:
    raise ImportError("WandB is not installed. Run pip install -e .[wandb]") from e

from neuralhydrology.__about__ import __version__
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.logging_utils import get_git_hash, save_git_diff


class Logger(object):
    """Class that logs runs to WandB and saves plots to disk.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        self._train = True
        self.log_interval = cfg.log_interval
        self.log_dir = cfg.run_dir
        self._img_log_dir = cfg.img_log_dir

        # get git commit hash if folder is a git repository
        cfg.update_config({'commit_hash': get_git_hash()})

        # save git diff to file if branch is dirty
        if cfg.save_git_diff:
            save_git_diff(cfg.run_dir)

        # Additionally, the package version is stored in the config
        cfg.update_config({"package_version": __version__})

        # store a copy of the config into the run folder
        cfg.dump_config(folder=self.log_dir)

        self.epoch = 0
        self.update = 0
        self._metrics = defaultdict(list)
        self.writer = None
        
        self.cfg = cfg

    @property
    def tag(self):
        return "train" if self._train else "valid"

    def train(self) -> 'Logger':
        """Set logging to training period.

        Returns
        -------
        Logger
            The Logger instance, set to training mode.
        """
        self._train = True
        return self

    def valid(self) -> 'Logger':
        """Set logging to validation period.

        Returns
        -------
        Logger
            The Logger instance, set to validation mode.
        """
        self._train = False
        return self

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

    def log_figures(self, figures: List[mpl.figure.Figure], freq: str, preamble: str = ""):
        """Log matplotlib figures as to disk.

        Parameters
        ----------
        figures : List[mpl.figure.Figure]
            List of figures to save.
        freq : str
            Prediction frequency of the figures.
        preamble : str, optional
            Prefix to prepend to the figures' file names.
        """
        for idx, figure in enumerate(figures):
            self.run.log({f"validation/timeseries_{freq}_{idx + 1}": wandb.Image(figure)}, step=self.update)
            figure.savefig(Path(self._img_log_dir, preamble + f'_freq{freq}_epoch{self.epoch}_{idx + 1}'), dpi=300)

    def log_step(self, **kwargs):
        """Log the results of a single step within an epoch.

        Parameters
        ----------
        **kwargs
            Key-value pairs of metric names and values.
        """
        for k, v in kwargs.items():
            self._metrics[k].append(v)

        if not self._train:
            return

        self.update += 1

        if self.log_interval <= 0:
            return

        if self.update % self.log_interval == 0:
            tag = self.tag
            for k, v in kwargs.items():
                self.run.log({f"{tag}/{k}": v}, step=self.update)
                
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
        if self.run is not None:
            self.run.log({"train/learning_rate": learning_rate}, step=self.update)

    def summarise(self) -> Union[float, Dict[str, float]]:
        """"Log the results of the entire training or validation epoch.

        Returns
        -------
        Union[float, Dict[str, float]]
            Average loss if training is summarized, else a dict mapping metric names to median metric values.
        """
        value = {}
        # summarize statistics of training epoch
        if self._train:
            self.epoch += 1

            # summarize training
            for k, v in self._metrics.items():
                mean = np.nanmean(v) if v else np.nan
                value[f'avg_{k}'] = mean

                self.run.log({f"{self.tag}/avg_{k}": mean}, step=self.update)

        # summarize validation
        else:
            for k, v in self._metrics.items():
                if v and isinstance(v[0], tuple):
                    # The only tuple that is passed is the per basin validation loss, which is a list of tuples, where
                    # each element is defined as (basin loss, number of batches). The aggregate across basins is
                    # weighted by the number of batches per basin, to approximate the training loss computation.
                    v_not_nan = [(loss, samples) for loss, samples in v if not np.isnan(loss)]
                    num_samples = sum(samples for _, samples in v_not_nan)
                    if num_samples > 0:
                        weighted_loss = sum(loss * samples / num_samples for loss, samples in v_not_nan)
                    else:
                        weighted_loss = np.nan
                    value[f'avg_{k}'] = weighted_loss
                    self.run.log({f"{self.tag}/avg_{k}": weighted_loss}, step=self.update)
                else:
                    # All other metrics are lists of float values
                    means = np.nanmean(v) if v else np.nan
                    medians = np.nanmedian(v) if v else np.nan
                    value[k] = medians
                    self.run.log(
                        {
                            f"{self.tag}/mean_{k.lower()}": means,
                            f"{self.tag}/median_{k.lower()}": medians
                        },
                        step=self.update
                    )

        # clear buffer
        self._metrics = defaultdict(list)

        return value
