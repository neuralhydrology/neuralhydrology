import logging
from typing import Dict
import numpy as np
import math

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel

LOGGER = logging.getLogger(__name__)

class Transformer(BaseModel):
    """Transformer model class, which relies on PyTorch's TransformerEncoder class.
    This class implements the encoder of a transformer network with a regression or probabilistic head. 
    The model configuration is specified in the config file using the following options:
    -- transformer_embedding_dimension : int representing the dimension of the input embedding space. 
                                         This must be dividible by the number of self-attention heads (transformer_nheads).
    -- transformer_positional_encoding_type : choices to "sum" or "concatenate" positional encoding to other model inputs.
    -- transformer_positional_dropout: fraction of dropout applied to the positional encoding.
    -- seq_length : integer number of timesteps to treat in the input sequence.
    -- transformer_nheads : number of self-attention heads.
    -- transformer_dim_feedforward : dimension of the feed-fowrard networks between self-attention heads.
    -- transformer_dropout: dropout in the feedforward networks between self-attention heads.
    -- transformer_nlayers: number of stacked self-attention + feedforward layers.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Dict):
        super(Transformer, self).__init__(cfg=cfg)

        input_size = len(cfg.dynamic_inputs + cfg.evolving_attributes + cfg.hydroatlas_attributes +
                         cfg.static_attributes)
        if cfg.use_basin_id_encoding:
            input_size += cfg.number_of_basins

        # embedding 
        self._embedding_dim = cfg.transformer_embedding_dimension
        self.embedding = nn.Linear(in_features=input_size, 
                                   out_features=self._embedding_dim)

        # positional encoder
        self.positional_encoding_type = cfg.transformer_positional_encoding_type
        if self.positional_encoding_type.lower() == 'concatenate':
          self._encoder_dim = self._embedding_dim*2
        elif self.positional_encoding_type.lower() == 'sum':
          self._encoder_dim = self._embedding_dim
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self.positional_encoding_type}")
        self.pos_encoder = PositionalEncoding(embedding_dim=self._embedding_dim, 
                                              dropout=cfg.transformer_positional_dropout, 
                                              max_len=cfg.seq_length)

        # positional mask
        self._mask = None

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self._encoder_dim, 
                                                    nhead=cfg.transformer_nheads, 
                                                    dim_feedforward=cfg.transformer_dim_feedforward, 
                                                    dropout=cfg.transformer_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, 
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None) 

        # head (instead of a decoder)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self._encoder_dim, n_out=self.output_size)

        # init weights and biases 
        self._reset_parameters()

    def _reset_parameters(self): 
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on a transformer model without decoder.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. 
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        # transpose to [seq_length, batch_size, n_features]
        x_d = data['x_d'].transpose(0, 1)

        # concat all inputs
        if 'x_s' in data and 'x_one_hot' in data:
            x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        # embedding
        x_d = self.embedding(x_d) * math.sqrt(self._embedding_dim)        
        x_d = self.pos_encoder(x_d, self.positional_encoding_type)

        # mask past values
        if self._mask is None or self._mask.size(0) != len(x_d):
            mask = (torch.tril(torch.ones(len(x_d), len(x_d))) == 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self._mask = mask.to(x_d.device)

        # encoding
        output = self.encoder(x_d, self._mask)

        # head
        pred = self.head(self.dropout(output.transpose(0, 1)))

        return pred

class PositionalEncoding(nn.Module):
    """Class to create a positional encoding vector for time series inputs to a model without an explicit time dimension.
    This class implements a sin/cos type embedding vector with a specified maximum length. Adapted from the PyTorch 
    example here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of the model input, which is typically output of an embedding layer.

    dropout : float
        Dropout rate [0, 1) applied to the embedding vector.

    max_len : int
        Maximum length of positional encoding. Talk about restrctions on max length.
    """

    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim/2)*2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_len*2) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:,:embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos_type):
        """

        Returns
        -------
        Finish this. 
        """
        if pos_type.lower() == 'concatenate':
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2) 
        elif pos_type.lower() == 'sum':
            x = x + self.pe[:x.size(0), :]
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {pos_type}")
        return self.dropout(x)

