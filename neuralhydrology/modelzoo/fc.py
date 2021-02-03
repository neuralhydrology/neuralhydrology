from typing import List

import numpy as np
import torch
import torch.nn as nn


class FC(nn.Module):
    """Auxiliary class to build (multi-layer) fully-connected networks.

    This class is used to build fully-connected embedding networks for static and/or dynamic input data.
    Use the config argument `statics/dynamics_embedding` to specify the architecture of the embedding network. See the
    `InputLayer` class on how to specify the exact embedding architecture.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_sizes : List[int]
        Size of the hidden and output layers.
    activation : str, optional
        Activation function for intermediate layers, default tanh.
    dropout : float, optional
        Dropout rate in intermediate layers.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int], activation: str = 'tanh', dropout: float = 0.0):
        super(FC, self).__init__()

        if len(hidden_sizes) == 0:
            raise ValueError('hidden_sizes must at least have one entry to create a fully-connected net.')

        self.output_size = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]

        activation = self._get_activation(activation)

        # create network
        layers = []
        if hidden_sizes:
            for i, hidden_size in enumerate(hidden_sizes):
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size))

                layers.append(activation)
                layers.append(nn.Dropout(p=dropout))

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
            Embedded inputs of shape [any, any, output_size], where 'output_size' is the size of the last network layer.
        """
        return self.net(x)
