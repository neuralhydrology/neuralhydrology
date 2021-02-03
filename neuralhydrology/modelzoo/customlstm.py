import math
from collections import defaultdict

from typing import Dict, Union

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class CustomLSTM(BaseModel):
    """A custom implementation of the LSTM with support for a dedicated embedding network.

    The idea of this model is mainly to be used as an analytical tool, where you can train a model using the optimized
    `CudaLSTM` or `EmbCudaLSTM` classes, and later copy the weights into this model for a more in-depth network
    analysis (e.g. inspecting model states or gate activations). The advantage of this implementation is that it returns
    the entire time series of state vectors and gate activations.
    However, you can also use this model class for training but note that it will be considerably slower than its
    optimized counterparts. Depending on the embedding settings, static and/or dynamic features may or may not be fed
    through embedding networks before being concatenated and passed through the LSTM.

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

        self.embedding_net = InputLayer(cfg)

        self._hidden_size = cfg.hidden_size

        self.cell = _LSTMCell(input_size=self.embedding_net.output_size,
                              hidden_size=self._hidden_size,
                              initial_forget_bias=cfg.initial_forget_bias)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self._hidden_size, n_out=self.output_size)

    def forward(self,
                data: Dict[str, torch.Tensor],
                h_0: torch.Tensor = None,
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
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data, concatenate_output=True)

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

        self.embedding_net.load_state_dict(optimized_lstm.embedding_net.state_dict())

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
