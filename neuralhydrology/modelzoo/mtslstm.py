import logging
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class MTSLSTM(BaseModel):
    """Multi-Timescale LSTM (MTS-LSTM) from Gauch et al. [#]_.

    An LSTM architecture that allows simultaneous prediction at multiple timescales within one model.
    There are two flavors of this model: MTS-LTSM and sMTS-LSTM (shared MTS-LSTM). The MTS-LSTM processes inputs at
    low temporal resolutions up to a point in time. Then, the LSTM splits into one branch for each target timescale.
    Each branch processes the inputs at its respective timescale. Finally, one prediction head per timescale generates
    the predictions for that timescale based on the LSTM output.
    Optionally, one can specify:
    - a different hidden size for each LSTM branch (use a dict in the ``hidden_size`` config argument)
    - different dynamic input variables for each timescale (use a dict in the ``dynamic_inputs`` config argument)
    - the strategy to transfer states from the initial shared low-resolution LSTM to the per-timescale
    higher-resolution LSTMs. By default, this is a linear transfer layer, but you can specify 'identity' to use an
    identity operation or 'None' to turn off any transfer (via the ``transfer_mtlstm_states`` config argument).
    - embedding networks for static and dynamic inputs using the ``statics_embedding`` and ``dynamics_embedding`` config arguments

    The sMTS-LSTM variant has the same overall architecture, but the weights of the per-timescale branches (including
    the output heads) are shared.
    Thus, unlike MTS-LSTM, the sMTS-LSTM cannot use per-timescale hidden sizes or dynamic input variables.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Gauch, M., Kratzert, F., Klotz, D., Grey, N., Lin, J., and Hochreiter, S.: Rainfall-Runoff Prediction at
        Multiple Timescales with a Single Long Short-Term Memory Network, arXiv Preprint,
        https://arxiv.org/abs/2010.07921, 2020.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstms', 'transfer_fcs', 'heads']

    def __init__(self, cfg: Config):
        super(MTSLSTM, self).__init__(cfg=cfg)
        self.lstms = None
        self.transfer_fcs = None
        self.heads = None
        self.dropout = None

        self._slice_timestep = {}
        self._frequency_factors = []

        self._seq_lengths = cfg.seq_length
        self._is_shared_mtslstm = self.cfg.shared_mtslstm  # default: a distinct LSTM per timescale
        self._transfer_mtslstm_states = self.cfg.transfer_mtslstm_states  # default: linear transfer layer
        transfer_modes = [None, "None", "identity", "linear"]
        if self._transfer_mtslstm_states["h"] not in transfer_modes \
                or self._transfer_mtslstm_states["c"] not in transfer_modes:
            raise ValueError(f"MTS-LSTM supports state transfer modes {transfer_modes}")

        if len(cfg.use_frequencies) < 2:
            raise ValueError("MTS-LSTM expects more than one input frequency")
        self._frequencies = sort_frequencies(cfg.use_frequencies)

        # initialize embedding networks for each frequency if embeddings are defined
        self.embedding_net = None
        if (cfg.dynamics_embedding is not None) or (cfg.statics_embedding is not None):
            self.embedding_net = nn.ModuleDict()
            for freq in self._frequencies:
                freq_params = {
                    'model': 'mtslstm',
                    'dynamic_inputs': cfg.dynamic_inputs[freq] if isinstance(cfg.dynamic_inputs, dict) else cfg.dynamic_inputs,
                    'dynamics_embedding': cfg.dynamics_embedding,
                    'static_attributes': cfg.static_attributes,
                    'statics_embedding': cfg.statics_embedding,
                    'use_basin_id_encoding': cfg.use_basin_id_encoding,
                    'number_of_basins': cfg.number_of_basins,
                    'head': cfg.head,
                    'seq_length': cfg.seq_length[freq] if isinstance(cfg.seq_length, dict) else cfg.seq_length
                }
                freq_cfg = Config(freq_params)
                self.embedding_net[freq] = InputLayer(freq_cfg, embedding_type='full_model')

        # start to count the number of inputs
        input_sizes = len(cfg.static_attributes + cfg.hydroatlas_attributes + cfg.evolving_attributes)

        # if is_shared_mtslstm, the LSTM gets an additional frequency flag as input.
        if self._is_shared_mtslstm:
            input_sizes += len(self._frequencies)

        if cfg.use_basin_id_encoding:
            input_sizes += cfg.number_of_basins
        if cfg.head.lower() == "umal":
            input_sizes += 1

        if isinstance(cfg.dynamic_inputs, list):
            if isinstance(cfg.dynamic_inputs[0], list):
                raise ValueError("MTS-LSTM does not support input feature groups.")
            self._dynamic_inputs = {freq: cfg.dynamic_inputs for freq in self._frequencies}
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs) for freq in self._frequencies}
        else:
            self._dynamic_inputs = cfg.dynamic_inputs
            if self._is_shared_mtslstm:
                raise ValueError(f'Different inputs not allowed if shared_mtslstm is used.')
            input_sizes = {freq: input_sizes + len(cfg.dynamic_inputs[freq]) for freq in self._frequencies}

        if not isinstance(cfg.hidden_size, dict):
            LOGGER.info("No specific hidden size for frequencies are specified. Same hidden size is used for all.")
            self._hidden_size = {freq: cfg.hidden_size for freq in self._frequencies}
        else:
            self._hidden_size = cfg.hidden_size

        if (self._is_shared_mtslstm
            or self._transfer_mtslstm_states["h"] == "identity"
            or self._transfer_mtslstm_states["c"] == "identity") \
                and any(size != self._hidden_size[self._frequencies[0]] for size in self._hidden_size.values()):
            raise ValueError("All hidden sizes must be equal if shared_mtslstm is used or state transfer=identity.")

        # create layer depending on selected frequencies
        self._init_modules(input_sizes)
        self._reset_parameters()

        # frequency factors are needed to determine the time step of information transfer
        self._init_frequency_factors_and_slice_timesteps()

    def _init_modules(self, input_sizes: Dict[str, int]):
        self.lstms = nn.ModuleDict()
        self.transfer_fcs = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.dropout = nn.Dropout(p=self.cfg.output_dropout)
        for idx, freq in enumerate(self._frequencies):
            if self.embedding_net is not None:
                freq_input_size = self.embedding_net[freq].output_size
                if self._is_shared_mtslstm:
                    freq_input_size += len(self._frequencies)
                if self.cfg.head.lower() == "umal":
                    freq_input_size += 1
            else:
                freq_input_size = input_sizes[freq]

            if self._is_shared_mtslstm and idx > 0:
                self.lstms[freq] = self.lstms[self._frequencies[idx - 1]]  # same LSTM for all frequencies.
                self.heads[freq] = self.heads[self._frequencies[idx - 1]]  # same head for all frequencies.
            else:
                self.lstms[freq] = nn.LSTM(input_size=freq_input_size, hidden_size=self._hidden_size[freq])
                self.heads[freq] = get_head(self.cfg, n_in=self._hidden_size[freq], n_out=self.output_size)

            if idx < len(self._frequencies) - 1:
                for state in ["c", "h"]:
                    if self._transfer_mtslstm_states[state] == "linear":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Linear(self._hidden_size[freq],
                                                                         self._hidden_size[self._frequencies[idx + 1]])
                    elif self._transfer_mtslstm_states[state] == "identity":
                        self.transfer_fcs[f"{state}_{freq}"] = nn.Identity()
                    else:
                        pass

    def _init_frequency_factors_and_slice_timesteps(self):
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                frequency_factor = get_frequency_factor(freq, self._frequencies[idx + 1])
                if frequency_factor != int(frequency_factor):
                    raise ValueError('Adjacent frequencies must be multiples of each other.')
                self._frequency_factors.append(int(frequency_factor))
                # we want to pass the state of the day _before_ the next higher frequency starts,
                # because e.g. the mean of a day is stored at the same date at 00:00 in the morning.
                slice_timestep = int(self._seq_lengths[self._frequencies[idx + 1]] / self._frequency_factors[idx])
                self._slice_timestep[freq] = slice_timestep

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            for freq in self._frequencies:
                hidden_size = self._hidden_size[freq]
                self.lstms[freq].bias_hh_l0.data[hidden_size:2 * hidden_size] = self.cfg.initial_forget_bias

    def _add_frequency_one_hot_encoding(self, x_d: torch.Tensor, freq: str) -> torch.Tensor:
        idx = self._frequencies.index(freq)
        one_hot_freq = torch.zeros(x_d.shape[0], x_d.shape[1], len(self._frequencies), device=x_d.device)
        one_hot_freq[:, :, idx] = 1
        return torch.cat([x_d, one_hot_freq], dim=2)

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], freq: str) -> torch.Tensor:
        """Concat all different inputs to the time series input"""
        if self.embedding_net is not None:
            # use embedding network if available
            input_data = {
                'x_d': data[f'x_d_{freq}'],
                'x_s': data.get('x_s'),
                'x_one_hot': data.get('x_one_hot')
            }
            # if specific x_s for frequency is available, use that
            if f'x_s_{freq}' in data:
                input_data['x_s'] = data[f'x_s_{freq}']
                
            # filter out None values
            input_data = {k: v for k, v in input_data.items() if v is not None}
            x_d = self.embedding_net[freq](input_data, concatenate_output=True)
                       
            # add frequency one-hot encoding if shared_mtslstm is used
            if self._is_shared_mtslstm:
                x_d = self._add_frequency_one_hot_encoding(x_d, freq)

        else:
            # directly use the input data without embedding, following the original implementation
            suffix = f"_{freq}"
            
            # concatenate all dynamic feature tensors from the dictionary
            feature_tensors = []
            for feature_name, feature_tensor in data[f'x_d{suffix}'].items():
                feature_tensors.append(feature_tensor)
            
            if feature_tensors:
                # concatenate all features along the feature dimension
                x_d = torch.cat(feature_tensors, dim=-1).transpose(0, 1)
            else:
                # no dynamic features found, which is invalid
                raise ValueError(f"No dynamic features found for frequency {freq}.")

            # concat all static and one-hot encoded features
            if f'x_s{suffix}' in data and 'x_one_hot' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s, x_one_hot], dim=-1)
            elif f'x_s{suffix}' in data:
                x_s = data[f'x_s{suffix}'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_s' in data:
                x_s = data['x_s'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_s], dim=-1)
            elif 'x_one_hot' in data:
                x_one_hot = data['x_one_hot'].unsqueeze(0).repeat(x_d.shape[0], 1, 1)
                x_d = torch.cat([x_d, x_one_hot], dim=-1)
            else:
                pass

            if self._is_shared_mtslstm:
                x_d = self._add_frequency_one_hot_encoding(x_d, freq)
        
        return x_d

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MTS-LSTM model.
        
        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Input data for the forward pass. See the documentation overview of all models for details on the dict keys.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model predictions for each target timescale.
        """
        x_d = {freq: self._prepare_inputs(data, freq) for freq in self._frequencies}

        # initial states for lowest frequencies are set to zeros
        batch_size = x_d[self._frequencies[0]].shape[1]
        lowest_freq_hidden_size = self._hidden_size[self._frequencies[0]]
        h_0_transfer = x_d[self._frequencies[0]].new_zeros((1, batch_size, lowest_freq_hidden_size))
        c_0_transfer = torch.zeros_like(h_0_transfer)

        outputs = {}
        for idx, freq in enumerate(self._frequencies):
            if idx < len(self._frequencies) - 1:
                # get predictions and state up to the time step of information transfer
                slice_timestep = self._slice_timestep[freq]
                lstm_output_slice1, (h_n_slice1, c_n_slice1) = self.lstms[freq](x_d[freq][:-slice_timestep],
                                                                                (h_0_transfer, c_0_transfer))

                # project the states through a hidden layer to the dimensions of the next LSTM
                if self._transfer_mtslstm_states["h"] is not None:
                    h_0_transfer = self.transfer_fcs[f"h_{freq}"](h_n_slice1)
                if self._transfer_mtslstm_states["c"] is not None:
                    c_0_transfer = self.transfer_fcs[f"c_{freq}"](c_n_slice1)

                # get predictions of remaining part and concat results
                lstm_output_slice2, _ = self.lstms[freq](x_d[freq][-slice_timestep:], (h_n_slice1, c_n_slice1))
                lstm_output = torch.cat([lstm_output_slice1, lstm_output_slice2], dim=0)

            else:
                # for highest frequency, we can pass the entire sequence at once
                lstm_output, _ = self.lstms[freq](x_d[freq], (h_0_transfer, c_0_transfer))

            head_out = self.heads[freq](self.dropout(lstm_output.transpose(0, 1)))
            outputs.update({f'{key}_{freq}': value for key, value in head_out.items()})

        return outputs
