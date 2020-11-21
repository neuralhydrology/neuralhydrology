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

    def __init__(self, cfg: Dict):
        super(Transformer, self).__init__(cfg=cfg)

        if cfg.embedding_hiddens:
            LOGGER.warning("## Warning: Embedding settings are ignored. Use EmbCudaLSTM for embeddings")

        input_size = len(cfg.dynamic_inputs + cfg.evolving_attributes + cfg.hydroatlas_attributes +
                         cfg.static_attributes)
        if cfg.use_basin_id_encoding:
            input_size += cfg.number_of_basins

        if cfg.head.lower() == "umal":
            input_size += 1

        # embedding 
        self.embedding_dim = cfg.transformer_embedding_dimension
        self.embedding = nn.Linear(in_features=input_size, 
                                   out_features=self.embedding_dim)

        # positional encoder
        self.posistional_encoding_type = cfg.transformer_positional_encoding_type
        if self.posistional_encoding_type.lower() in ['concat', 'cat', 'concatenate']:
            self.encoder_dim = self.embedding_dim*2
#            self.encoder_dim = input_size*2
        elif self.posistional_encoding_type.lower() in ['sum', 'add', 'addition']:
            self.encoder_dim = self.embedding_dim
#            self.encoder_dim = input_size 
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self.posistional_encoding_type}")
#        self.pos_encoder = PositionalEncoding(d_model=input_size, 
        self.pos_encoder = PositionalEncoding(d_model=self.embedding_dim, 
                                              dropout=cfg.transformer_positional_dropout, 
                                              max_len=cfg.seq_length)

        # positional mask
        self.mask = None

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.encoder_dim, 
                                                    nhead=cfg.transformer_nheads, 
                                                    dim_feedforward=cfg.transformer_dim_feedforward, 
                                                    dropout=cfg.transformer_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, 
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None) # why does normalization matter?

        # head (right now, this is our decoder)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.encoder_dim, n_out=self.output_size)

        # init weights and biases 
        self.reset_parameters()

    def reset_parameters(self): # why are we not resedding the encoder layer weights and biases?
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.bias.data.zero_()

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
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

#        import matplotlib.pyplot as plt
#        import pickle as pkl
#        with open('snow_basin_attention_weights.pkl', 'rb') as f:
#            aow = pkl.load(f)
#        fig, axes = plt.subplots(1+self.cfg['transformer_nheads'], 1)
#        axes[0].plot(x_d[-1,:,[0,2,5,7]])
#        for head in range(self.cfg['transformer_nheads']):
#            axes[head+1].imshow(aow[:,:,head])
#            axes[head+1].set_ylabel('Forcing History Time Step')
#            axes[head+1].set_xlabel('Simulation Time Step')
#            axes[head+1].set_title(f'Attention Head #{head}')
#        fig.show()
#        plt.tight_layout()
#        import pdb; pdb.set_trace()


        # embedding
        x_d = self.embedding(x_d) * math.sqrt(self.embedding_dim)        
        x_d = self.pos_encoder(x_d, self.posistional_encoding_type)

        # mask past values
        if self.mask is None or self.mask.size(0) != len(x_d):
            mask = (torch.triu(torch.ones(len(x_d), len(x_d))) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.mask = mask.to(x_d.device)

        # encoding
        output = self.encoder(x_d, self.mask)

        # head
        pred = self.head(self.dropout(output.transpose(0, 1)))
#        import matplotlib.pyplot as plt
#        import pdb; pdb.set_trace()

        return pred

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(d_model/2)*2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_len*2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:,:d_model].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos_type):
        if pos_type.lower() in ['concat', 'cat', 'concatenate']:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2) 
        elif pos_type.lower() in ['sum', 'add', 'addition']:
            x = x + self.pe[:x.size(0), :]
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {pos_type}")
        return self.dropout(x)

