import itertools
import logging
from typing import Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.positional_encoding import PositionalEncoding
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)

_EMBEDDING_TYPES = ['full_model', 'hindcast', 'forecast']

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

    def __init__(self, cfg: Config, embedding_type: str = 'full_model'):
        super(InputLayer, self).__init__()

        self.embedding_type = embedding_type
        if embedding_type == 'full_model':
            dynamic_inputs = cfg.dynamic_inputs
            self._x_d_key = 'x_d'
        elif embedding_type == 'forecast':
            dynamic_inputs = cfg.forecast_inputs
            self._x_d_key = 'x_d_forecast'
        elif embedding_type == 'hindcast':
            dynamic_inputs = cfg.hindcast_inputs
            self._x_d_key = 'x_d_hindcast'
        else:
            raise ValueError(
                f'Embedding type {embedding_type} is not recognized. '
                f'Must be one of: {_EMBEDDING_TYPES}.'
            )
        if isinstance(dynamic_inputs, dict):
            if self.nan_handling_method:
                raise ValueError('InputLayer does not support nan handling methods with multiple frequencies.')
            frequencies = list(dynamic_inputs.keys())
            if len(frequencies) > 1:
                raise ValueError('InputLayer only supports single-frequency data')
            dynamics_input_sizes = [len(dynamic_inputs[frequencies[0]])]
            self._dynamic_inputs = {k: [v] for k, v in dynamic_inputs.items()}
        else:
            if isinstance(dynamic_inputs[0], str):
                # Put all features into a single feature group.
                self._dynamic_inputs = [dynamic_inputs]
            else:
                self._dynamic_inputs = dynamic_inputs
            if cfg.timestep_counter:
                # Add timestep counter to each feature group.
                if self.embedding_type == 'hindcast':
                    self._dynamic_inputs = [group + ['hindcast_counter'] for group in self._dynamic_inputs]
                elif self.embedding_type == 'forecast':
                    self._dynamic_inputs += [group + ['forecast_counter'] for group in self._dynamic_inputs]
            self.nan_handling_method = cfg.nan_handling_method
            self.attention = None
            self._nan_fill_value = 0.0
            if self.nan_handling_method == 'input_replacing':
                # +1 for the NaN flag.
                dynamics_input_sizes = [sum(len(group) + 1
                                            for group in self._dynamic_inputs) + cfg.nan_handling_pos_encoding_size]
            elif self.nan_handling_method in ['masked_mean', 'attention']:
                dynamics_input_sizes = [len(group) + (cfg.nan_handling_pos_encoding_size
                                                        if self.nan_handling_method != 'attention' else 0)
                                        for group in self._dynamic_inputs]
            else:
                dynamics_input_sizes = [len(group) for group in self._dynamic_inputs]

        if cfg.head.lower() == "umal":
            dynamics_input_sizes = [size + 1 for size in dynamics_input_sizes]
            if isinstance(self._dynamic_inputs, dict):
                self._dynamic_inputs = {k: v + ['_tau'] for k, v in self._dynamic_inputs.items()}
            else:
                self._dynamic_inputs = [group + ['_tau'] for group in self._dynamic_inputs]

        self._num_autoregression_inputs = 0
        if cfg.autoregressive_inputs:
            self._num_autoregression_inputs = len(cfg.autoregressive_inputs)

        statics_input_size = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)
        if cfg.use_basin_id_encoding:
            statics_input_size += cfg.number_of_basins

        self.statics_embedding, self.statics_output_size = \
            self._get_embedding_net(cfg.statics_embedding, statics_input_size, 'statics')

        self._pos_enc = None
        if cfg.nan_handling_pos_encoding_size > 0:
            if not self.nan_handling_method:
                raise NotImplementedError('Positional encoding is only supported for nan handling methods.')
            self._pos_enc = PositionalEncoding(embedding_dim=cfg.nan_handling_pos_encoding_size,
                                              position_type='concatenate',
                                              dropout=0.0,
                                              max_len=cfg.seq_length)

        dynamics_embeddings = []
        dynamics_output_sizes = []
        for dynamics_input_size in dynamics_input_sizes:
            group_embedding, group_output_size = self._get_embedding_net(cfg.dynamics_embedding,
                                                                         dynamics_input_size,
                                                                         'dynamics')
            dynamics_embeddings.append(group_embedding)
            dynamics_output_sizes.append(group_output_size)
        self.dynamics_embeddings = nn.ModuleList(dynamics_embeddings)
        if not all(size == dynamics_output_sizes[0] for size in dynamics_output_sizes):
            raise ValueError('All dynamics embedding output sizes must be equal.')
        # All output sizes are the same, just use the first one.
        self.dynamics_output_size = dynamics_output_sizes[0]
        if self.nan_handling_method == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=self.dynamics_output_size, num_heads=1)
            self.query_embedding, _ = \
                    self._get_embedding_net(cfg.dynamics_embedding,
                                            (self.statics_output_size + len(self._dynamic_inputs)
                                             + cfg.nan_handling_pos_encoding_size),
                                            'query')

        if cfg.statics_embedding is None:
            self.statics_embedding_p_dropout = 0.0  # if net has no statics dropout we treat is as zero
        else:
            self.statics_embedding_p_dropout = cfg.statics_embedding['dropout']
        if cfg.dynamics_embedding is None:
            self.dynamics_embedding_p_dropout = 0.0  # if net has no dynamics dropout we treat is as zero
        else:
            self.dynamics_embedding_p_dropout = cfg.dynamics_embedding['dropout']

        self.output_size = self.dynamics_output_size + self.statics_output_size + self._num_autoregression_inputs
        self.cfg = cfg

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

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]], concatenate_output: bool = True) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass on the input layer.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
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
        features = self._dynamic_inputs
        if isinstance(features, dict):
            features = features[list(features.keys())[0]]

        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            x_s = None

        statics_out = None
        if x_s is not None:
            statics_out = self.statics_embedding(x_s)

        if self.nan_handling_method == 'masked_mean':
            dynamics_out = self._masked_mean_embedding(data[self._x_d_key])
        elif self.nan_handling_method == 'attention':
            dynamics_out = self._attention(data[self._x_d_key], statics_embedding=statics_out)
        elif self.nan_handling_method == 'input_replacing':
            dynamics_out = self._input_replacing_embedding(data[self._x_d_key])
        else:
            # transpose to [seq_length, batch_size, n_features]
            x_d = torch.cat([data[self._x_d_key][k] for k in itertools.chain(*features)], dim=-1).transpose(0, 1)
            dynamics_out = self.dynamics_embeddings[0](x_d)

        if not concatenate_output:
            ret_val = dynamics_out, statics_out
        else:
            if statics_out is not None:
                statics_out = statics_out.unsqueeze(0).repeat(dynamics_out.shape[0], 1, 1)
                ret_val = torch.cat([dynamics_out, statics_out], dim=-1)
            else:
                ret_val = dynamics_out

            # Append autoregressive inputs to the end of the output.
            # Don't run autoregressive inputs through the embedding layer. This does not work with NaN's.
            if self._num_autoregression_inputs:
                x_autoregressive = torch.cat([data[self._x_d_key][k]
                                              for k in self.cfg.autoregressive_inputs], dim=-1).transpose(0, 1)
                ret_val = torch.cat([ret_val, x_autoregressive], dim=-1)

        return ret_val

    def _attention(self, x_d: dict[str, torch.Tensor], statics_embedding: torch.Tensor | None) -> torch.Tensor:
        """Attention mechanism with statics + positional encoding as query, feature groups as keys and values."""
        if statics_embedding is None:
            raise ValueError('Attention requires static features.')

        dynamics_out = []
        masks = []
        for idx, feature_group in enumerate(self._dynamic_inputs):
            # transpose to [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            # Set NaNs to zero to avoid NaN gradients. The zeros will be ignored by the attention mask.
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)
        dynamics_out = torch.stack(dynamics_out, dim=0)
        # Make sure we have something to return even if all values are NaN.
        dynamics_out = torch.where(torch.isnan(dynamics_out).all(dim=0, keepdim=True),
                                   self._nan_fill_value, dynamics_out)

        n_groups, seq_len, batch_size, embed_dim = dynamics_out.shape
        stacked_masks = torch.stack(masks, dim=0).view(n_groups, seq_len * batch_size, 1)
        # query: (seq_len, batch_size, embed_dim)
        query = statics_embedding.unsqueeze(0).repeat(seq_len, 1, 1)
        if self._pos_enc is not None:
            query = self._pos_enc(query)
        query = torch.cat([query, stacked_masks.squeeze(-1).permute(1, 0).view(seq_len, batch_size, n_groups)],dim=-1)
        query = self.query_embedding(query)
        query = query.unsqueeze(0).view(1, seq_len * batch_size, embed_dim)

        # stacked masks: (n_groups, seq_len * batch_size)
        stacked_masks = stacked_masks.squeeze(-1)
        # key, value: (n_groups, seq_len * batch_size, embed_dim)
        key = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        value = dynamics_out.view(n_groups, seq_len * batch_size, embed_dim)
        key = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, key)
        value = torch.where(stacked_masks.unsqueeze(2), self._nan_fill_value, value)

        # attn_mask: (seq_len * batch_size, 1, n_groups)
        attn_mask = stacked_masks.permute(1, 0).unsqueeze(1)
        # attention_out: (1, seq_len * batch_size, embed_dim)
        attention_out, _ = self.attention(query, key, value, attn_mask=attn_mask, need_weights=False)
        return attention_out.view(1, seq_len, batch_size, embed_dim).squeeze(0)

    def _masked_mean_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs masked mean embedding on the input data."""
        dynamics_out = []
        masks = []
        for idx, feature_group in enumerate(self._dynamic_inputs):
            # transpose to [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            if self._pos_enc is not None:
                x_d_group = self._pos_enc(x_d_group)
            # Set NaNs to zero to avoid NaN gradients.
            x_d_group = torch.where(mask, 0.0, x_d_group)
            group_embedding = self.dynamics_embeddings[idx](x_d_group)
            # Set zeros back to NaN so they are ignored in the mean.
            dynamics_out.append(torch.where(mask, torch.nan, group_embedding))
            masks.append(mask)
        dynamics_out = torch.stack(dynamics_out, dim=0)
        # Make sure the mean works even if all values are NaN.
        dynamics_out = torch.where(torch.isnan(dynamics_out).all(dim=0, keepdim=True),
                                   self._nan_fill_value, dynamics_out)
        return torch.nanmean(dynamics_out, dim=0)

    def _input_replacing_embedding(self, x_d: dict[str, torch.Tensor]) -> torch.Tensor:
        """Adds input masks to the inputs and sets NaNs to zero."""
        dynamics = []
        for feature_group in self._dynamic_inputs:
            # transpose to [seq_length, batch_size, n_features]
            x_d_group = torch.cat([x_d[k] for k in feature_group], dim=-1).transpose(0, 1)
            mask = x_d_group.isnan().any(dim=-1, keepdim=True)
            x_d_group = torch.where(mask, self._nan_fill_value, x_d_group)
            dynamics.append(x_d_group)
            dynamics.append(mask.to(torch.float32))
        dynamics = torch.cat(dynamics, dim=-1)
        if self._pos_enc is not None:
            dynamics = self._pos_enc(dynamics)
        return self.dynamics_embeddings[0](dynamics)

    def __getitem__(self, item: str) -> nn.Module:
        # required for dict-like access when freezing submodules' gradients in fine-tuning
        if item == "statics_embedding":
            return self.statics_embedding
        elif item == "dynamics_embedding":
            return self.dynamics_embedding
        else:
            raise KeyError(f"Cannot access {item} on InputLayer")
