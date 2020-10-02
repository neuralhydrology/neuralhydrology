import logging

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_head(cfg: Config, n_in: int, n_out: int) -> nn.Module:
    """Get specific head module, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.

    Returns
    -------
    nn.Module
        The model head, as specified in the run configuration.
    """
    if cfg.head.lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=cfg.output_activation)
    else:
        raise NotImplementedError(f"{cfg.head} not implemented or not linked in `get_head()`")

    return head


class Regression(nn.Module):
    """
    Regression head with different output activations.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return {'y_hat': self.net(x)}