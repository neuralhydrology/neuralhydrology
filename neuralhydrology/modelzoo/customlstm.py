import math
from collections import defaultdict

from typing import Dict, Union

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class CustomLSTM(BaseModel):
    """A custom implementation of the LSTM with support for a dedicated embedding network.

    The idea of this model is mainly to be used as an analytical tool, where you can train a model using the optimized
    `CudaLSTM` or `EmbCudaLSTM` classes, and later copy the weights into this model for a more in-depth network 
    analysis (e.g. inspecting model states or gate activations). The advantage of this implementation is that it returns
    the entire time series of state vectors and gate activations.
    However, you can also use this model class for training but note that it will be considerably slower than its 
    optimized counterparts. If your config includes specifications for the `embedding_hiddens` argument, this class
    will mimic the `EmbCudaLSTM` during training. If not, it will default to mimic the `CudaLSTM`, where all static
    inputs are concatenated directly to the time series inputs.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Example
    -------
    >>> # Example for copying the weights of an optimzed `CudaLSTM` or `EmbCudaLSTM` into a `CustomLSTM` instance
    >>> cfg = ... # A config instance corresponding to the original, optimized model
    >>> optimized_lstm = ... # A model instance of `CudaLSTM` or `EmbCudaLSTM`
    >>> 
    >>> # Use the original config to initialize this model to differentiate between `CudaLSTM` and `EmbCudaLSTM`
    >>> custom_lstm = CustomLSTM(cfg=cfg)
    >>>
    >>> # Copy weights into the `LSTM` instance.
    >>> custom_lstm.copy_weights(optimized_lstm)
    """

    def __init__(self, cfg: Config):
        super(CustomLSTM, self).__init__(cfg=cfg)

        # in case this class is used for analysis of an EmbCudaLSTM, we need to initialize the embedding network
        if cfg.model in ["embcudalstm", "lstm"] and cfg.embedding_hiddens:
            self.embedding_net = FC(cfg=cfg)
            self._has_embedding_net = True
            input_size = len(cfg.dynamic_inputs) + cfg.embedding_hiddens[-1]
        else:
            self._has_embedding_net = False
            input_size = len(cfg.dynamic_inputs + cfg.evolving_attributes + cfg.static_attributes +
                             cfg.hydroatlas_attributes)
            if cfg.use_basin_id_encoding:
                input_size += cfg.number_of_basins

        self._hidden_size = cfg.hidden_size

        self.cell = _LSTMCell(input_size=input_size,
                              hidden_size=self._hidden_size,
                              initial_forget_bias=cfg.initial_forget_bias)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self._hidden_size, n_out=self.output_size)

    def _preprocess_embcudalstm(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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

        return x_d

    def _preprocess_lstm(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
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

        return x_d

    def forward(self, data: Dict[str, torch.Tensor], h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the LSTM model.

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
        if self._has_embedding_net:
            x_d = self._preprocess_embcudalstm(data)
        else:
            x_d = self._preprocess_lstm(data)

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
                output[key].append(cell_out)

        # stack to [batch size, sequence length, hidden size]
        pred = {key: torch.stack(val, 1) for key, val in output.items()}
        pred.update(self.head(self.dropout(pred['h_n'])))
        return pred

    def copy_weights(self, optimized_lstm: Union[CudaLSTM, EmbCudaLSTM]):
        """Copy weights from a `CudaLSTM` or `EmbCudaLSTM` into this model class

        Parameters
        ----------
        optimized_lstm : Union[CudaLSTM, EmbCudaLSTM]
            Model instance of a `CudaLSTM` (neuralhydrology.modelzoo.cudalstm) or `EmbCudaLSTM`
            (neuralhydrology.modelzoo.embcudalstm).
            
        Raises
        ------
        RuntimeError
            If `optimized_lstm` is an `EmbCudaLSTM` but this model instance was not created with an embedding network.
        """

        assert isinstance(optimized_lstm, CudaLSTM) or isinstance(optimized_lstm, EmbCudaLSTM)

        if isinstance(optimized_lstm, EmbCudaLSTM):
            if self._has_embedding_net:
                self.embedding_net.load_state_dict(optimized_lstm.embedding_net.state_dict())
            else:
                raise RuntimeError("This model was not initialized with an embedding network.")

        # copy lstm cell weights
        self.cell.copy_weights(optimized_lstm.lstm, layer=0)

        # copy weights of head
        self.head.load_state_dict(optimized_lstm.head.state_dict())


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

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
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
