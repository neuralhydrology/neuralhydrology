from pathlib import Path

import matplotlib as mpl
import matplotlib.figure
from torch.utils.tensorboard import SummaryWriter

from neuralhydrology.training.logging.logger import Logger
from neuralhydrology.utils.config import Config


class TensorboardLogger(Logger):
    """Class that logs runs to tensorboard and saves plots to disk.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.writer = None

    def start_logger(self):
        """ Start tensorboard logging. """
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def stop_logger(self):
        """ Stop tensorboard logging. """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def log_figures(self, figures: list[matplotlib.figure.Figure], freq: str, preamble: str = ""):
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
        if self.writer is not None:
            self.writer.add_figure(f'validation/timeseries/{freq}', figures, global_step=self.epoch)

        for idx, figure in enumerate(figures):
            figure.savefig(Path(self._img_log_dir, preamble + f'_freq{freq}_epoch{self.epoch}_{idx + 1}'), dpi=300)

    def log_metric(self, metric: str, value: float, step: int):
        if self.writer is not None:
            self.writer.add_scalar(metric, value, global_step=step)

    def log_model(self, weight_path, optimizer_path):
        pass

    def log_lr(self, learning_rate: float):
        """Log current learning rate.

        Parameters
        ----------
        learning_rate : float
            Current learning rate value.
        """
        if self.writer is not None:
            self.log_metric("train/learning_rate", learning_rate, self.update)