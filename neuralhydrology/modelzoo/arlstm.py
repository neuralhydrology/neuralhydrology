import math
from collections import defaultdict

import re
from typing import Dict, List

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class ARLSTM(BaseModel):
    """An autoregressive LSTM.

    This model assumes that the *last* entry in dynamic inpus (x_d) is an observation that is supposed to match
    target data lagged by an integer number of timesteps. If this data is missing (NaN), it is substituted 
    for the model prediction at the same lag. The model adds an extra dynamic input that serves as a binary 
    flag to indicate whether the autoregressive input at a particular timestep is from observation vs. simulation.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(ARLSTM, self).__init__(cfg=cfg)

        self._ar_shift = self._get_ar_shift()
        self._num_ar_inputs = len(self.cfg.autoregressive_inputs)
        if self.output_size != self._num_ar_inputs:
            raise ValueError('The AR-LSTM currently only works if all outputs are used as lagged inputs.')

        self.embedding_net = InputLayer(cfg)

        # increase input size for binary flag
        input_size = self.embedding_net.output_size + self._num_ar_inputs
        self.cell = nn.LSTM(input_size=input_size, hidden_size=cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _get_ar_shift(self) -> int:
        shifts = set()
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Autoregressive inputs must be a shifted variable with form <variable>_shift<lag> ',
                                f'where <lag> is an integer. Instead got: {input}.')
            if not capture[1] in self.cfg.target_variables:
                raise ValueError('Autoregressive inputs must be a shifted target variable. ',
                                f'Instead got a shifted version of: {capture[1]}.')
            shifts.add(int(capture[2]))
        if len(shifts) > 1:
            raise ValueError('Only one AR shift is allowed currently. All autoregressive inputs must use the same shift.')
        shift = shifts.pop()
        if shift <= 0:
            raise ValueError('Autoregressive inputs must be shifted by at least one timestep.')
        return shift    
    
    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.cell.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self,
                data: Dict[str, torch.Tensor],
                h_0: torch.Tensor = None,
                c_0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Autoregressive LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `lstm_output`: full timeseries of the hidden states from the lstm [batch size, sequence length, hidden size].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        _, batch_size, _ = x_d.size()

        if h_0 is None:
            h_0 = x_d.new_zeros((1, batch_size, self.cfg.hidden_size))
        if c_0 is None:
            c_0 = x_d.new_zeros((1, batch_size, self.cfg.hidden_size))

        # initialize flags for autoregressive inputs -- all flags start at 0, indicating obs
        # flags are appended as the last input
        ar_flags = x_d.new_zeros((batch_size, self._num_ar_inputs))
        
        # initialize the 'last' prediction to substitute for any missing AR values in the first timesteps
        last_prediction = x_d.new_zeros((self._ar_shift, batch_size, self.output_size))

        # initialize dictionary to store the output
        lstm_output = []
        y_hat = []

        # manually loop through timesteps
        for x_t in x_d:

            # find locations of missing data (NaN's) and replace with last predictions
            x_embd = x_t[:, :-self._num_ar_inputs]
            x_ar = x_t[:, -self._num_ar_inputs:].clone()
            replace_indexes = torch.isnan(x_ar)
            x_ar[replace_indexes] = last_prediction[-1, replace_indexes]
            ar_flags[:, :] = 0
            ar_flags[replace_indexes] = 1

            # one timestep of lstm
            cell_inputs = torch.unsqueeze(torch.concat([x_embd, x_ar, ar_flags], -1), 0)
            cell_output, (h_0, c_0) = self.cell(cell_inputs, (h_0, c_0))

            # append all timestep output to the output dictionary
            lstm_output.append(cell_output.transpose(0, 1))

            # store the last prediction
            last_prediction[1:] = last_prediction[:-1,].clone()
            prediction = torch.squeeze(self.head(self.dropout(h_0.transpose(0, 1)))['y_hat'], dim=1)
            last_prediction[0] = prediction
            y_hat.append(prediction)

        # stack all ouptuts to sizes in function doc
        pred = {
            'lstm_output': torch.concat(lstm_output, 1), 
            'h_n': h_0.transpose(0, 1), 
            'c_n': c_0.transpose(0, 1),
            'y_hat': torch.stack(y_hat, 1),
        }
        return pred
