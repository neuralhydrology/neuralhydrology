import numpy as np
import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config


class FC(nn.Module):
    """Auxiliary class to build (multi-layer) fully-connected networks.

    This class is used to build fully-connected embedding networks for static and/or dynamic input data.
    Use the config argument `embedding_hiddens` to specify the hidden neurons of the embedding network. If only one
    number is specified the embedding network consists of a single linear layer that maps the input into the specified
    dimension.
    Use the config argument `embedding_activation` to specify the activation function for intermediate layers. Currently
    supported are 'tanh' and 'sigmoid'.
    Use the config argument `embedding_dropout` to specify the dropout rate in intermediate layers.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    input_size : int, optional
        Number of input features. If not specified, the number of input features is the sum of all static inputs (i.e.,
        catchment attributes, one-hot-encoding, etc.)
    """

    def __init__(self, cfg: Config, input_size: int = None):
        super(FC, self).__init__()

        # If input size is not passed, will use number of all static features as input
        if input_size is None:
            input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
            if cfg.use_basin_id_encoding:
                input_size += cfg.number_of_basins

        if len(cfg.embedding_hiddens) > 1:
            hidden_sizes = cfg.embedding_hiddens[:-1]
        else:
            hidden_sizes = []

        self.output_size = cfg.embedding_hiddens[-1]

        # by default tanh
        activation = self._get_activation(cfg.embedding_activation)

        # create network
        layers = []
        if hidden_sizes:
            for i, hidden_size in enumerate(hidden_sizes):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))

                layers.append(activation)
                # by default 0.0
                layers.append(nn.Dropout(p=cfg.embedding_dropout))

            layers.append(nn.Linear(hidden_size, self.output_size))
        else:
            layers.append(nn.Linear(input_size, self.output_size))

        self.net = nn.Sequential(*layers)
        self._reset_parameters()

    def _get_activation(self, name: str) -> nn.Module:
        if name.lower() == "tanh":
            activation = nn.Tanh()
        elif name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif name.lower() == "linear":
            activation = nn.Identity()
        else:
            raise NotImplementedError(f"{name} currently not supported as activation in this class")
        return activation

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        for layer in self.net:
            if isinstance(layer, nn.modules.linear.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass on the FC model.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape [any, any, input size]

        Returns
        -------
        torch.Tensor
            Embedded inputs of shape [any, any, output_size], where 'output_size' is the last number specified in the
            `embedding_hiddens` config argument.
        """
        return self.net(x)
