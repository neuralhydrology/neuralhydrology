from __future__ import annotations

import logging
import math
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.positional_encoding import PositionalEncoding
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class Transformer(BaseModel):
    """Transformer model class, which relies on PyTorch's TransformerEncoder class.

    This class implements the encoder of a transformer network which can be used for regression.
    Unless the number of inputs is divisible by the number of transformer heads (``transformer_nheads``), it is
    necessary to use an embedding network that guarantees this. To achieve this, use ``statics/dynamics_embedding``,
    so the static/dynamic inputs will be passed through embedding networks before being concatenated. The embedding
    network will then map the static and dynamic features to size ``statics/dynamics_embedding['hiddens'][-1]``, so the
    total embedding size will be the sum of these values.
    The model configuration is specified in the config file using the following options:

    * ``transformer_positional_encoding_type``: choices to "sum" or "concatenate" positional encoding to other model
      inputs.
    * ``transformer_positional_dropout``: fraction of dropout applied to the positional encoding.
    * ``seq_length``: integer number of timesteps to treat in the input sequence.
    * ``transformer_nheads``: number of self-attention heads.
    * ``transformer_dim_feedforward``: dimension of the feed-forward networks between self-attention heads.
    * ``transformer_dropout``: dropout in the feedforward networks between self-attention heads.
    * ``transformer_nlayers``: number of stacked self-attention + feedforward layers.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, cfg: Config):
        super(Transformer, self).__init__(cfg=cfg)

        # embedding net before transformer
        self.embedding_net = InputLayer(cfg)

        # ensure that the number of inputs into the self-attention layer is divisible by the number of heads
        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads. "
                             "Use statics_embedding/dynamics_embedding and embedding_hiddens to specify the embedding.")

        self._sqrt_embedding_dim = math.sqrt(self.embedding_net.output_size)

        # positional encoder
        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")
        self.positional_encoder = PositionalEncoding(embedding_dim=self.embedding_net.output_size,
                                                      dropout=cfg.transformer_positional_dropout,
                                                      position_type=cfg.transformer_positional_encoding_type,
                                                      max_len=cfg.seq_length)

        # positional mask
        self._mask = None

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=encoder_dim,
                                                    nhead=cfg.transformer_nheads,
                                                    dim_feedforward=cfg.transformer_dim_feedforward,
                                                    dropout=cfg.transformer_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None)

        # head (instead of a decoder)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=encoder_dim, n_out=self.output_size)

        # init weights and biases
        self._reset_parameters()

    def _reset_parameters(self):
        # this initialization strategy was tested empirically but may not be the universally best strategy in all cases.
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on a transformer model without decoder.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        # pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim)

        # mask out future values
        if self._mask is None or self._mask.size(0) != len(x_d):
            self._mask = torch.triu(x_d.new_full((len(x_d), len(x_d)), fill_value=float('-inf')), diagonal=1)

        # encoding
        output = self.encoder(positional_encoding, self._mask)

        # head
        pred = self.head(self.dropout(output.transpose(0, 1)))

        # add embedding and positional encoding to output
        pred['embedding'] = x_d
        pred['positional_encoding'] = positional_encoding

        return pred
