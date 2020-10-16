import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class CudaLSTM(BaseModel):
    """LSTM model class, which relies on PyTorch's CUDA LSTM class.

    This class implements the standard LSTM combined with a model head, as specified in the config. All features 
    (time series and static) are concatenated and passed to the LSTM directly. If you want to embed the static features
    prior to the concatenation, use the `EmbCudaLSTM` class.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set 
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow. 
    The `CudaLSTM` class only supports single-timescale predictions. Use `MTSLSTM` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['lstm', 'head']

    def __init__(self, cfg: Config):
        super(CudaLSTM, self).__init__(cfg=cfg)

        if cfg.embedding_hiddens:
            LOGGER.warning("## Warning: Embedding settings are ignored. Use EmbCudaLSTM for embeddings")

        input_size = len(cfg.dynamic_inputs + cfg.static_inputs + cfg.hydroatlas_attributes + cfg.camels_attributes)
        if cfg.use_basin_id_encoding:
            input_size += cfg.number_of_basins

        if cfg.head.lower() == "umal":
            input_size += 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary. 
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
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

        lstm_output, (h_n, c_n) = self.lstm(input=x_d)

        # reshape to [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)
        c_n = c_n.transpose(0, 1)

        pred = {'h_n': h_n, 'c_n': c_n}
        pred.update(self.head(self.dropout(lstm_output.transpose(0, 1))))

        return pred
