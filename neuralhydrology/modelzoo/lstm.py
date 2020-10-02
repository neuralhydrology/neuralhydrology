import math
from collections import defaultdict

from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class LSTM(BaseModel):
    """Replication of the CuDNN LSTM
    
    The idea of this model is to be able to train an LSTM using the nn.LSTM layer, which uses the
    optimized CuDNN implementation, and later to copy the weights into this model for a more 
    in-depth network analysis.

    Note: Currently only supports one-layer CuDNN LSTMs
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    
    Example
    -------
    >>> # copy weights from PyTorch LSTM to this LSTM implementation
    >>> cudnn_lstm = nn.LSTM(input_size=15, hidden_size=128, num_layers=1)
    >>> lstm = LSTM(cfg=cfg)
    >>> lstm.copy_weights(cudnn_lstm)
    """

    def __init__(self, cfg: Config):
        super(LSTM, self).__init__(cfg=cfg)

        input_size = len(cfg.dynamic_inputs + cfg.static_inputs + cfg.camels_attributes + cfg.hydroatlas_attributes)
        if cfg.use_basin_id_encoding:
            input_size += cfg.number_of_basins

        self._hidden_size = cfg.hidden_size

        self.cell = _LSTMCell(input_size=input_size,
                              hidden_size=self._hidden_size,
                              initial_forget_bias=cfg.initial_forget_bias)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self._hidden_size, n_out=self.output_size)

    def forward(self,
                data: Dict[str, torch.Tensor],
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the LSTM network
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pair.
        h_0 : torch.Tensor, optional
            Initial hidden state, by default 0.
        c_0 : torch.Tensor, optional
            Initial cell state, by default 0.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model output and all intermediate states and gate activations as a dictionary.
        """

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

        seq_len, batch_size, _ = x_d.size()

        # TODO: move hidden and cell state initialization to init and only reset states in forward pass to zero.
        if h_0 is None:
            h_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        if c_0 is None:
            c_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        h_x = (h_0, c_0)

        output = defaultdict(list)
        for x_t in x_d:
            h_0, c_0 = h_x
            cell_output = self.cell(x_t=x_t, h_0=h_0, c_0=c_0)

            h_x = (cell_output['h_n'], cell_output['c_n'])

            for key, cell_out in cell_output.items():
                output[key].append(cell_out.detach())

        # stack to [batch size, sequence length, hidden size]
        pred = {key: torch.stack(val, 1) for key, val in output.items()}
        pred.update(self.head(self.dropout(pred['h_n'])))
        return pred

    def copy_weights(self, cudnnlstm: nn.Module):
        """Copy weights from a PyTorch nn.LSTM into this model class
        
        Parameters
        ----------
        cudnnlstm : nn.Module
            Model instance of a Pytorch nn.LSTM
        """

        assert isinstance(cudnnlstm, nn.modules.rnn.LSTM)
        assert cudnnlstm.num_layers == 1

        self.cell.copy_weights(cudnnlstm, layer=0)


class _LSTMCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, initial_forget_bias: float = 0.0):
        super(_LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias

        self.w_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size, hidden_size))
        self.w_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size, input_size))

        self.b_hh = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self.b_ih = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(3 / self.hidden_size)
        for weight in self.parameters():
            if len(weight.shape) > 1:
                weight.data.uniform_(-stdv, stdv)
            else:
                nn.init.zeros_(weight)

        if self.initial_forget_bias != 0:
            self.b_hh.data[self.hidden_size:2 * self.hidden_size] = self.initial_forget_bias

    def forward(self, x_t: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor) -> Dict[str, torch.Tensor]:
        gates = h_0 @ self.w_hh.T + self.b_hh + x_t @ self.w_ih.T + self.b_ih
        i, f, g, o = gates.chunk(4, 1)

        c_1 = c_0 * torch.sigmoid(f) + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)

        return {'h_n': h_1, 'c_n': c_1, 'i': i, 'f': f, 'g': g, 'o': o}

    def copy_weights(self, cudnnlstm: nn.Module, layer: int):

        assert self.hidden_size == cudnnlstm.hidden_size
        assert self.input_size == cudnnlstm.input_size

        self.w_hh.data = getattr(cudnnlstm, f"weight_hh_l{layer}").data
        self.w_ih.data = getattr(cudnnlstm, f"weight_ih_l{layer}").data
        self.b_hh.data = getattr(cudnnlstm, f"bias_hh_l{layer}").data
        self.b_ih.data = getattr(cudnnlstm, f"bias_ih_l{layer}").data
