import matplotlib as mpl

from neuralhydrology.training.logging.logger import Logger


class NoOpLogger(Logger):
    """Logger that does nothing."""

    def start_logger(self):
        pass

    def stop_logger(self):
        pass

    def log_figures(self, figures: list[mpl.figure.Figure], freq: str, preamble: str = ""):
        pass

    def log_metric(self, name, value, step):
        pass

    def log_lr(self, learning_rate: float):
        pass
