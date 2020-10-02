import numpy as np
import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config


class FC(nn.Module):

    def __init__(self, cfg: Config, input_size: int = None):
        super(FC, self).__init__()

        # If input size is not passed, will use number of all static features as input
        if input_size is None:
            input_size = len(cfg.camels_attributes + cfg.hydroatlas_attributes + cfg.static_inputs)
            if cfg.use_basin_id_encoding:
                input_size += cfg.number_of_basins

        if len(cfg.embedding_hiddens) > 1:
            hidden_sizes = cfg.embedding_hiddens[:-1]
        else:
            hidden_sizes = []

        output_size = cfg.embedding_hiddens[-1]

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

            layers.append(nn.Linear(hidden_size, output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def _get_activation(self, name: str) -> nn.Module:
        if name.lower() == "tanh":
            activation = nn.Tanh()
        elif name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise NotImplementedError(f"{name} currently not supported as activation in this class")
        return activation

    def reset_parameters(self):
        for layer in self.net:
            if isinstance(layer, nn.modules.linear.Linear):
                n_in = layer.weight.shape[1]
                gain = np.sqrt(3 / n_in)
                nn.init.uniform_(layer.weight, -gain, gain)
                nn.init.constant_(layer.bias, val=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
