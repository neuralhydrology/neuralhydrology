import logging
from typing import Dict

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
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation: str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.
        
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}
