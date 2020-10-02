from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class EmbCudaLSTM(BaseModel):

    def __init__(self, cfg: Config):
        super(EmbCudaLSTM, self).__init__(cfg=cfg)

        # embedding net before LSTM
        if not cfg.embedding_hiddens:
            raise ValueError('EmbCudaLSTM requires config argument embedding_hiddens.')

        self.embedding_net = FC(cfg=cfg)

        input_size = len(cfg.dynamic_inputs) + cfg.embedding_hiddens[-1]
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self.reset_parameters()

    def reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        x_d = data['x_d'].transpose(0, 1)

        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            raise ValueError('Need x_s or x_one_hot in forward pass.')

        embedding = self.embedding_net(x_s)

        embedding = embedding.unsqueeze(0).repeat(x_d.shape[0], 1, 1)
        x_d = torch.cat([x_d, embedding], dim=-1)

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, seq_length, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output.transpose(0, 1))))

        return pred
