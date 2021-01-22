import logging
from typing import Dict, Union, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class InputLayer(nn.Module):
    """Input layer to preprocess static and dynamic inputs.

    This module provides optional embedding of dynamic and static inputs. If ``dynamic_embeddings`` or
    ``static_embeddings`` are set to True in the config, a fully-connected embedding network will be prepended to the
    timeseries model. See the `FC` class for information on how to configure the embedding layers. While the dynamic
    and static embedding layers are always separate (with separate weights), at this point they always share the same
    architecture.

    Note that this module does not support multi-frequency runs.

    Parameters
    ----------
    cfg : Config
        The run configuration
    """

    def __init__(self, cfg: Config):
        super(InputLayer, self).__init__()

        if not cfg.embedding_hiddens and (cfg.statics_embedding or cfg.dynamics_embedding):
            raise ValueError('static or dynamic embeddings are active, but embedding_hiddens is undefined.')
        if cfg.embedding_hiddens and not (cfg.statics_embedding or cfg.dynamics_embedding):
            LOGGER.warning('embedding_hiddens will be ignored since statics_embedding and dynamics_embedding are False')

        self.statics_output_size = 0
        self.dynamics_output_size = 0

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        if cfg.statics_embedding:
            if statics_input_size == 0:
                raise ValueError('Cannot create a static embedding layer with input size 0')
            self.statics_embedding = FC(cfg, statics_input_size)
            self.statics_output_size += self.statics_embedding.output_size
        else:
            self.statics_embedding = nn.Identity()
            self.statics_output_size += statics_input_size

        dynamics_input_size = len(cfg.dynamic_inputs)
        if cfg.dynamics_embedding:
            self.dynamics_embedding = FC(cfg, len(cfg.dynamic_inputs))
            self.dynamics_output_size += self.dynamics_embedding.output_size
        else:
            self.dynamics_embedding = nn.Identity()
            self.dynamics_output_size += dynamics_input_size

        self.output_size = self.dynamics_output_size + self.statics_output_size

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
