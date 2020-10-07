from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
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
        Hydrol. Earth Syst. Sci., 23, 5089–5110, https://doi.org/10.5194/hess-23-5089-2019, 2019.
    """

    def __init__(self, cfg: Config):
        super(EALSTM, self).__init__(cfg=cfg)
        self._hidden_size = cfg.hidden_size

        input_size_dyn = len(cfg.dynamic_inputs)
        input_size_stat = len(cfg.static_inputs + cfg.camels_attributes + cfg.hydroatlas_attributes)
        if cfg.use_basin_id_encoding:
            input_size_stat += cfg.number_of_basins

        # If hidden units for a embedding network are specified, create FC, otherwise single linear layer
        if cfg.embedding_hiddens:
            self.input_net = FC(cfg=cfg)
        else:
            self.input_net = nn.Linear(input_size_stat, cfg.hidden_size)

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * cfg.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(cfg.hidden_size, 3 * cfg.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * cfg.hidden_size))

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

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
        if 'x_s' in data and 'x_one_hot' in data:
            x_s = torch.cat([data['x_s'], data['x_one_hot']], dim=-1)
        elif 'x_s' in data:
            x_s = data['x_s']
        elif 'x_one_hot' in data:
            x_s = data['x_one_hot']
        else:
            raise ValueError('Need x_s or x_one_hot in forward pass.')

        x_d = data['x_d'].transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        # TODO: move hidden and cell state initialization to init and only reset states in forward pass to zero.
        h_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self._hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        i = torch.sigmoid(self.input_net(x_s))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(h_n)))

        return pred
