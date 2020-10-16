from typing import Dict

import torch
from torch import nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class GRU(BaseModel):
    """Gated Recurrent Unit (GRU) class based on the PyTorch GRU implementation.

    This class implements the standard GRU combined with a model head, as specified in the config. All features
    (time series and static) are concatenated and passed to the GRU directly.
    The `GRU` class only supports single-timescale predictions.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['gru', 'head']

    def __init__(self, cfg: Config):

        super(GRU, self).__init__(cfg=cfg)

        input_size = len(cfg.dynamic_inputs + cfg.static_inputs + cfg.hydroatlas_attributes + cfg.camels_attributes)
        if cfg.use_basin_id_encoding:
            input_size += cfg.number_of_basins

        if cfg.head.lower() == "umal":
            input_size += 1

        self.gru = nn.GRU(input_size=input_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the GRU model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pair.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # transpose to [seq_length, batch_size, n_features]
        x_d = data['x_d'].transpose(0, 1)

        # concatenate all inputs
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

        # run the actual GRU
        gru_output, h_n = self.gru(input=x_d)

        # reshape to [batch_size, 1, n_hiddens]
        h_n = h_n.transpose(0, 1)

        pred = {'h_n': h_n}

        # add the final output as it's returned by the head to the prediction dict
        # (this will contain the 'y_hat')
        pred.update(self.head(self.dropout(gru_output.transpose(0, 1))))

        return pred
