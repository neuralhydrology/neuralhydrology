import logging
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This module provides optional embedding of dynamic and static inputs. If ``dynamic_embeddings`` or
    ``static_embeddings`` are specified as dictionaries in the config, a fully-connected embedding network will be
    prepended to the timeseries model. The dictionaries have the following keys:

    - ``type`` (default 'fc'): Type of the embedding net. Currently, only 'fc' for fully-connected net is supported.
    - ``hiddens``: List of integers that define the number of neurons per layer in the fully connected network.
      The last number is the number of output neurons. Must have at least length one.
    - ``activation`` (default 'tanh'): activation function of the network. Supported values are 'tanh', 'sigmoid',
      'linear'. The activation function is not applied to the output neurons, which always have a linear activation
      function. An activation function for the output neurons has to be applied in the main model class.
    - ``dropout`` (default 0.0): Dropout rate applied to the embedding network.

    Note that this module does not support multi-frequency runs.

    Parameters
    ----------
    cfg : Config
        The run configuration
    """

    def __init__(self, cfg: Config):
        super(InputLayer, self).__init__()

        dynamics_input_size = len(cfg.dynamic_inputs)
        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.statics_embedding, self.statics_output_size = \
            self._get_embedding_net(cfg.statics_embedding, statics_input_size, 'statics')
        self.dynamics_embedding, self.dynamics_output_size = \
            self._get_embedding_net(cfg.dynamics_embedding, dynamics_input_size, 'dynamics')

        if cfg.statics_embedding is None:
            self.statics_embedding_p_dropout = 0.0  # if net has no statics dropout we tread is as zero
        else:
            self.statics_embedding_p_dropout = cfg.statics_embedding['dropout']
        if cfg.dynamics_embedding is None:
            self.dynamics_embedding_p_dropout = 0.0  # if net has no dynamics dropout we tread is as zero
        else:
            self.dynamics_embedding_p_dropout = cfg.dynamics_embedding['dropout']

        self.output_size = self.dynamics_output_size + self.statics_output_size
        if cfg.head.lower() == "umal":
            self.output_size += 1

    @staticmethod
    def _get_embedding_net(embedding_spec: Optional[dict], input_size: int, purpose: str) -> Tuple[nn.Module, int]:
        """Get an embedding net following the passed specifications.

        If the `embedding_spec` is None, the returned embedding net will be the identity function.

        Parameters
        ----------
        embedding_spec : Optional[dict]
            Specification of the embedding net from the run configuration or None.
        input_size : int
            Size of the inputs into the embedding network.
        purpose : str
            Purpose of the embedding network, used for error messages.

        Returns
        -------
        Tuple[nn.Module, int]
            The embedding net and its output size.
        """
        if embedding_spec is None:
            return nn.Identity(), input_size

        if input_size == 0:
            raise ValueError(f'Cannot create {purpose} embedding layer with input size 0')

        emb_type = embedding_spec['type'].lower()
        if emb_type != 'fc':
            raise ValueError(f'{purpose} embedding type {emb_type} not supported.')

        hiddens = embedding_spec['hiddens']
        if len(hiddens) == 0:
            raise ValueError(f'{purpose} embedding "hiddens" must be a list of hidden sizes with at least one entry')

        dropout = embedding_spec['dropout']
        activation = embedding_spec['activation']

        emb_net = FC(input_size=input_size, hidden_sizes=hiddens, activation=activation, dropout=dropout)
        return emb_net, emb_net.output_size

    def forward(self, data: Dict[str, torch.Tensor], concatenate_output: bool = True) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass on the input layer.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            The input data.
        concatenate_output : bool, optional
            If True (default), the forward method will concatenate the static inputs to each dynamic time step.
            If False, the forward method will return a tuple of (dynamic, static) inputs.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If `concatenate_output` is True, a single tensor is returned. Else, a tuple with one tensor of dynamic
            inputs and one tensor of static inputs.
        """
        # transpose to [seq_length, batch_size, n_features]
        x_d = data['x_d'].transpose(0, 1)

        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            x_s = None

        dynamics_out = self.dynamics_embedding(x_d)
        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)

        if not concatenate_output:
            ret_val = dynamics_out, statics_out
        else:
            if statics_out is not None:
                statics_out = statics_out.unsqueeze(0).repeat(dynamics_out.shape[0], 1, 1)
                ret_val = torch.cat([dynamics_out, statics_out], dim=-1)
            else:
                ret_val = dynamics_out

        return ret_val

    def __getitem__(self, item: str) -> nn.Module:
        # required for dict-like access when freezing submodules' gradients in fine-tuning
        if item == "statics_embedding":
            return self.statics_embedding
        elif item == "dynamics_embedding":
            return self.dynamics_embedding
        else:
            raise KeyError(f"Cannot access {item} on InputLayer")
