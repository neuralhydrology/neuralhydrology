from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class EmbCudaLSTM(BaseModel):
    """EmbCudaLSTM model class, which adds embedding networks for static inputs to the standard LSTM.

    This class extends the standard `CudaLSTM` class to preprocess the static inputs by an embedding network, prior
    to concatenating those values to the dynamic (time series) inputs. Use the config argument `embedding_hiddens` to
    specify the architecture of the fully-connected embedding network. No activation function is applied to the outputs
    of the embedding network.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set 
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow. 
    The `EmbCudaLSTM` class only supports single timescale predictions. Use `MTSLSTM` to train a model and get 
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

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

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the EmbCudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. 
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [1, batch size, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [1, batch size, hidden size].
        """
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
