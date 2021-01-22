from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class EALSTM(BaseModel):
    """Entity-Aware LSTM (EA-LSTM) model class.

    This model has been proposed by Kratzert et al. [#]_ as a variant of the standard LSTM. The main difference is that
    the input gate of the EA-LSTM is modulated using only the static inputs, while the dynamic (time series) inputs are
    used in all other parts of the model (i.e. forget gate, cell update gate and output gate).
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set 
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow. 
    The `EALSTM` class does only support single timescale predictions. Use `MTSLSTM` to train an LSTM-based model and 
    get predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
        
    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., and Nearing, G.: Towards learning 
        universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets, 
        Hydrol. Earth Syst. Sci., 23, 5089-5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'input_gate', 'dynamic_gates', 'head']

    def __init__(self, cfg: Config):
        super(EALSTM, self).__init__(cfg=cfg)
        self._hidden_size = cfg.hidden_size

        self.embedding_net = InputLayer(cfg)

        self.input_gate = nn.Linear(self.embedding_net.statics_output_size, cfg.hidden_size)

        # create tensors of learnable parameters
        self.dynamic_gates = _DynamicGates(cfg=cfg, input_size=self.embedding_net.dynamics_output_size)
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def _cell(self, x: torch.Tensor, i: torch.Tensor, states: Tuple[torch.Tensor,
                                                                    torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single time step logic of EA-LSTM cell"""
        h_0, c_0 = states

        # calculate gates
        gates = self.dynamic_gates(h_0, x)
        f, o, g = gates.chunk(3, 1)

        c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)

        return h_1, c_1

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the EA-LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. 
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape 
                    [batch size, sequence length, number of target variables].
                - `c_n`: cell state at the last time step of the sequence of shape 
                    [batch size, sequence length, number of target variables].
        """
        # possibly pass dynamic and static inputs through embedding layers
        x_d, x_s = self.embedding_net(data, concatenate_output=False)
        if x_s is None:
            raise ValueError('Need x_s or x_one_hot in forward pass.')

        # TODO: move hidden and cell state initialization to init and only reset states in forward pass to zero.
        h_t = x_d.data.new(x_d.shape[1], self._hidden_size).zero_()
        c_t = x_d.data.new(x_d.shape[1], self._hidden_size).zero_()

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # calculate input gate only once because inputs are static
        i = torch.sigmoid(self.input_gate(x_s))

        # perform forward steps over input sequence
        for x_dt in x_d:

            h_t, c_t = self._cell(x_dt, i, (h_t, c_t))

            # store intermediate hidden/cell state in list
            h_n.append(h_t)
            c_n.append(c_t)

        h_n = torch.stack(h_n, 0).transpose(0, 1)
        c_n = torch.stack(c_n, 0).transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(h_n)))

        return pred


class _DynamicGates(nn.Module):
    """Internal class to wrap the dynamic gate parameters into a dedicated PyTorch Module"""

    def __init__(self, cfg: Config, input_size: int):
        super(_DynamicGates, self).__init__()
        self.cfg = cfg
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * cfg.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(cfg.hidden_size, 3 * cfg.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * cfg.hidden_size))

        # initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.cfg.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)

        if self.cfg.initial_forget_bias is not None:
            self.bias.data[:self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, h: torch.Tensor, x_d: torch.Tensor):
        gates = h @ self.weight_hh + x_d @ self.weight_ih + self.bias
        return gates
