import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.customlstm import _LSTMCell
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class ODELSTM(BaseModel):
    """ODE-LSTM from [#]_.

    An ODE-RNN post-processes the hidden state of a normal LSTM.
    Parts of this code are derived from https://github.com/mlech26l/learning-long-term-irregular-ts.
    
    The forward pass in this model works somewhat differently than the other models, because ODE-LSTM relies on
    irregularly timed samples. To simulate such irregularity, we aggregate parts of the input sequence to random
    frequencies. While doing so, we try to take care that we don't aggregate too coarsely right before the model
    should create a high-frequency prediction.

    Since this aggregation means that parts of the input sequence are at random frequencies, we cannot easily
    return predictions for the full input sequence at each frequency. Instead, we only return sequences of length
    predict_last_n for each frequency (we do not apply the random aggregation to these last time steps).

    The following describes the aggregation strategy implemented in the forward method:

    1. slice one: random-frequency steps (cfg.ode_random_freq_lower_bound <= freq <= lowest-freq) until beginning
                  of the second-lowest frequency input sequence.
    2. slice two: random-frequency steps (lowest-freq <= freq <= self._frequencies[1]) until beginning of
                  next-higher frequency input sequence.
    3. repeat step two until beginning of highest-frequency input sequence.
    4. slice three: random-frequency steps (self._frequencies[-2] <= freq <= highest-freq) until predict_last_n
                    of the lowest frequency.
    5. lowest-frequency steps to generate predict_last_n lowest-frequency predictions.
    6. repeat steps four and five for the next-higher frequency (using the same random-frequency bounds but
       generating predictions for the next-higher frequency).
       
    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Lechner, M.; Hasani, R.: Learning Long-Term Dependencies in Irregularly-Sampled Time Series. arXiv, 2020,
        https://arxiv.org/abs/2006.04418.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['lstm_cell', 'ode_cell', 'head']

    def __init__(self, cfg: Config):
        super(ODELSTM, self).__init__(cfg=cfg)
        if len(cfg.use_frequencies) < 2:
            raise ValueError('ODELSTM needs at least two frequencies.')
        if isinstance(cfg.dynamic_inputs, dict) or isinstance(cfg.hidden_size, dict):
            raise ValueError('ODELSTM does not support per-frequency input variables or hidden sizes.')

        # Note: be aware that frequency_factors and slice_timesteps have a slightly different meaning here vs. in
        # MTSLSTM. Here, the frequency_factor is relative to the _lowest_ (not the next-lower) frequency.
        # slice_timesteps[freq] is the input step (counting backwards) in the next-*lower* frequency from where on input
        # data at frequency freq is available.
        self._frequency_factors = {}
        self._slice_timesteps = {}
        self._frequencies = sort_frequencies(cfg.use_frequencies)
        self._init_frequency_factors_and_slice_timesteps()

        # start to count the number of inputs
        self.input_size = len(cfg.dynamic_inputs + cfg.static_attributes + cfg.hydroatlas_attributes +
                              cfg.evolving_attributes)

        if cfg.use_basin_id_encoding:
            self.input_size += cfg.number_of_basins
        if cfg.head.lower() == 'umal':
            self.input_size += 1

        self.lstm_cell = _LSTMCell(self.input_size, self.cfg.hidden_size, cfg.initial_forget_bias)
        self.ode_cell = _ODERNNCell(self.cfg.hidden_size,
                                    self.cfg.hidden_size,
                                    num_unfolds=self.cfg.ode_num_unfolds,
                                    method=self.cfg.ode_method)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.cfg.hidden_size, n_out=self.output_size)

    def _init_frequency_factors_and_slice_timesteps(self):
        for i, freq in enumerate(self._frequencies):
            frequency_factor = get_frequency_factor(self._frequencies[0], freq)
            if frequency_factor != int(frequency_factor):
                raise ValueError('Frequencies must be multiples of the lowest frequency.')
            self._frequency_factors[freq] = int(frequency_factor)

            if i > 0:
                prev_frequency_factor = get_frequency_factor(self._frequencies[i - 1], freq)
                if self.cfg.predict_last_n[freq] % prev_frequency_factor != 0:
                    raise ValueError(
                        'At all frequencies, predict_last_n must align with the steps of the next-lower frequency.')

                if self.cfg.seq_length[freq] > self.cfg.seq_length[self._frequencies[i - 1]] * prev_frequency_factor:
                    raise ValueError('Higher frequencies must have shorter input sequences than lower frequencies.')

                # we want to pass the state of the day _before_ the next higher frequency starts,
                # because e.g. the mean of a day is stored at the same date at 00:00 in the morning.
                slice_timestep = self.cfg.seq_length[freq] / prev_frequency_factor
                if slice_timestep != int(slice_timestep):
                    raise ValueError('At all frequencies, seq_length must align with the next-lower frequency steps.')
                self._slice_timesteps[freq] = int(slice_timestep)

                # in theory, the following conditions would be possible, but they would make the implementation
                # quite complex and are probably hardly ever useful.
                if self.cfg.predict_last_n[self._frequencies[
                        i - 1]] < self.cfg.predict_last_n[freq] / prev_frequency_factor:
                    raise NotImplementedError(
                        'Lower frequencies cannot have smaller predict_last_n values than higher ones.')

        if any(self.cfg.predict_last_n[f] / self._frequency_factors[f] > self._slice_timesteps[self._frequencies[-1]] /
               self._frequency_factors[self._frequencies[-2]] for f in self._frequencies):
            raise NotImplementedError('predict_last_n cannot be larger than sequence length of highest frequency.')

    def _prepare_inputs(self, data: Dict[str, torch.Tensor],
                        freq: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Concat all different inputs to the time series input. """
        suffix = f"_{freq}"
        # transpose to [seq_length, batch_size, n_features]
        x_d = data[f'x_d{suffix}'].transpose(0, 1)

        # concat all inputs
        if f'x_s{suffix}' in data and 'x_one_hot' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
        elif f'x_s{suffix}' in data:
            x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_s], dim=-1)
        elif 'x_one_hot' in data:
            x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
            x_d = torch.cat([x_d, x_one_hot], dim=-1)
        else:
            pass

        # add frequency indicator. This will not be used as normal input, but the ODE-RNN uses it to determine the
        # elapsed time since the last sample.
        frequency_factor = 1 / self._frequency_factors[freq]
        frequency_encoding = torch.ones(x_d.shape[0], x_d.shape[1], 1).to(x_d) * frequency_factor
        return torch.cat([x_d, frequency_encoding], dim=-1)

    def _randomize_freq(self, x_d: torch.Tensor, low_frequency: str, high_frequency: str) -> torch.Tensor:
        """Randomize the frequency of the  input sequence. """
        frequency_factor = int(get_frequency_factor(low_frequency, high_frequency))
        possible_aggregate_steps = list(filter(lambda n: frequency_factor % n == 0, range(1, frequency_factor + 1)))

        t = 0
        max_t = x_d.shape[0] / frequency_factor
        x_d_randomized = []
        while t < max_t:
            highfreq_slice = x_d[t * frequency_factor:(t + 1) * frequency_factor]

            # aggregate to a random frequency between low and high
            random_aggregate_steps = np.random.choice(possible_aggregate_steps)
            if highfreq_slice.shape[0] % random_aggregate_steps == 0:
                randfreq_slice = highfreq_slice.view(-1, random_aggregate_steps, highfreq_slice.shape[1],
                                                     highfreq_slice.shape[2]).mean(dim=1)
                # update the frequency indicators.
                randfreq_slice[:, :, -1] = random_aggregate_steps / self._frequency_factors[high_frequency]
            else:
                # do not randomize last slice if it doesn't align with aggregation steps
                randfreq_slice = highfreq_slice
            x_d_randomized.append(randfreq_slice)

            t += 1

        return torch.cat(x_d_randomized, dim=0)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the ODE-LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Input data for the forward pass. See the documentation overview of all models for details on the dict keys.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model predictions for each target timescale.
        """

        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        slice_one = self._randomize_freq(x_d[self._frequencies[0]][:-self._slice_timesteps[self._frequencies[1]]],
                                         self.cfg.ode_random_freq_lower_bound, self._frequencies[0])

        batch_size = slice_one.shape[1]
        h_0 = slice_one.data.new(batch_size, self.cfg.hidden_size).zero_()
        c_0 = slice_one.data.new(batch_size, self.cfg.hidden_size).zero_()
        h_n, c_n = self._run_odelstm(slice_one, h_0, c_0)

        prev_freq = self._frequencies[0]
        i = 1
        for freq in self._frequencies[1:-1]:
            to_randomize = x_d[freq][:-self._slice_timesteps[self._frequencies[i + 1]]]

            # slice_timestep of this and the next frequency are identical, so we can move on to the next frequency
            if len(to_randomize) == 0:
                continue
            # random-frequency steps until the beginning of the highest-frequency input sequence.
            slice_two = self._randomize_freq(to_randomize, prev_freq, freq)
            h_n, c_n = self._run_odelstm(slice_two, h_n[:, -1], c_n)
            prev_freq = freq
            i += 1

        pred = {}
        prev_freq_end_step = 0
        for freq in self._frequencies:
            # run random-frequency steps until predict_last_n of this frequency
            end_step = -int(self.cfg.predict_last_n[freq] * self._frequency_factors[self._frequencies[-1]] /
                            self._frequency_factors[freq])
            if end_step == 0:
                continue  # this means the current frequency should not be predicted, so we skip it.

            # check that the slice wouldn't be empty. If it wouldn't, this means that the previous frequency's
            # predict_last_n is not aligned with this frequency's predict_last_n, so we can run random-frequency
            # inputs until the beginning of this frequency's predict_last_n.
            if end_step != prev_freq_end_step and end_step > -self.cfg.seq_length[self._frequencies[-1]]:
                slice_three = self._randomize_freq(x_d[self._frequencies[-1]][prev_freq_end_step:end_step], prev_freq,
                                                   self._frequencies[-1])
                h_n, c_n = self._run_odelstm(slice_three, h_n[:, -1], c_n)
                prev_freq_end_step = end_step

            # run predict_last_n steps at the target frequency
            pred_slice = x_d[freq][-self.cfg.predict_last_n[freq]:]
            h_n_out, _ = self._run_odelstm(pred_slice, h_n[:, -1], c_n)
            pred[f'y_hat_{freq}'] = self.head(self.dropout(h_n_out))['y_hat']

        return pred

    def _run_odelstm(self, input_slice: torch.Tensor, h_0: torch.Tensor,
                     c_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ingest `input_slice` into the ODE-LSTM and return hidden states and last cell state. """
        h_x = (h_0, c_0)

        h_n = []
        for x_t in input_slice:
            h_0, c_0 = h_x

            # separate frequency indicator from input variables
            t_input = x_t[:, :-1]
            t_elapsed = x_t[0, -1]

            lstm_out = self.lstm_cell(x_t=t_input, h_0=h_0, c_0=c_0)
            ode_out = self.ode_cell(lstm_out['h_n'], h_0, t_elapsed)

            h_x = (ode_out, lstm_out['c_n'])
            h_n.append(ode_out)

        # stack h to [batch size, sequence length, hidden size]
        return torch.stack(h_n, 1), h_x[1]


class _ODERNNCell(nn.Module):
    """An ODE-RNN cell (Adapted from https://github.com/mlech26l/learning-long-term-irregular-ts) [#]_. 
    
    Parameters
    ----------
    input_size : int
        Input dimension
    hidden_size : int
        Size of the cell's hidden state
    num_unfolds : int
        Number of steps into which each timestep will be broken down to solve the ODE.
    method : {'euler', 'heun', 'rk4'}
        Method to use for ODE solving (Euler's method, Heun's method, or Runge-Kutta 4)
    
    References
    ----------
    .. [#] Lechner, M.; Hasani, R.: Learning Long-Term Dependencies in Irregularly-Sampled Time Series. arXiv, 2020,
        https://arxiv.org/abs/2006.04418.
    """

    def __init__(self, input_size: int, hidden_size: int, num_unfolds: int, method: str):
        super(_ODERNNCell, self).__init__()
        self.method = {
            'euler': self._euler,
            'heun': self._heun,
            'rk4': self._rk4,
        }[method]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_unfolds = num_unfolds

        self.w_ih = nn.Parameter(torch.FloatTensor(hidden_size, input_size))
        self.w_hh = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_size))
        self.scale = nn.Parameter(torch.FloatTensor(hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset the paramters of the ODERNNCell. """
        nn.init.orthogonal_(self.w_hh)
        nn.init.xavier_uniform_(self.w_ih)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.scale, 1.0)

    def forward(self, new_hidden_state: torch.Tensor, old_hidden_state: torch.Tensor, elapsed: float) -> torch.Tensor:
        """Perform a forward pass on the ODERNNCell.
        
        Parameters
        ----------
        new_hidden_state : torch.Tensor
            The current hidden state to be updated by the ODERNNCell.
        old_hidden_state : torch.Tensor
            The previous hidden state.
        elapsed : float
            Time elapsed between new and old hidden state.

        Returns
        -------
        torch.Tensor
            Predicted new hidden state
        """
        delta_t = elapsed / self.num_unfolds

        hidden_state = old_hidden_state
        for i in range(self.num_unfolds):
            hidden_state = self.method(new_hidden_state, hidden_state, delta_t)
        return hidden_state

    def _dfdt(self, inputs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        h_in = torch.matmul(inputs, self.w_ih)
        h_rec = torch.matmul(hidden_state, self.w_hh)
        dh_in = self.scale * torch.tanh(h_in + h_rec + self.bias)
        dh = dh_in - hidden_state
        return dh

    def _euler(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        dy = self._dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def _heun(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        k1 = self._dfdt(inputs, hidden_state)
        k2 = self._dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def _rk4(self, inputs: torch.Tensor, hidden_state: torch.Tensor, delta_t: float) -> torch.Tensor:
        k1 = self._dfdt(inputs, hidden_state)
        k2 = self._dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self._dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self._dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
