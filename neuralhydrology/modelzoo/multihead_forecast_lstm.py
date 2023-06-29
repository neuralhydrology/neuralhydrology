
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC


class MultiHeadForecastLSTM(BaseModel):
    """A forecasting model that does not roll out over the forecast horizon.
    
    This is a forecasting model that runs a sequential (LSTM) model up to the forecast issue time, 
    and then directly predicts a sequence of forecast timesteps without using a recurrent rollout.
    Prediction is done with a custom ``FC`` (fully connected) layer, which can include depth.

    Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    
    Raises
    ------
    ValueError if forecast_overlap > 0.
    ValueError if a forecast_network is not specified.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = [
        'forecast_mebedding_net'
        'hindcast_embedding_net',
        'hindcast_lstm',
        'forecast_network',
        'hindcast_head',
        'forecast_head'
    ]

    def __init__(self, cfg: Config):
        super(MultiHeadForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Forecast overlap cannot be set for a multi-head forecasting model. '
                             'Please set to None or remove from config file.')

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=cfg.hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        if not cfg.forecast_network:
            raise ValueError('The multihead forecast model requires a forecast network specified in the config file.')

        input_size = self.forecast_embedding_net.output_size*cfg.forecast_seq_length + cfg.hidden_size
        forecast_network_output_size = cfg.forecast_network['hiddens'][-1] * cfg.forecast_seq_length
        self.forecast_network = FC(
            input_size=input_size,
            hidden_sizes=cfg.forecast_network['hiddens'][:-1] + [forecast_network_output_size],
            activation=cfg.forecast_network['activation'],
            dropout=cfg.forecast_network['dropout']
        )

        self.hindcast_head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
        self.forecast_head = get_head(cfg=cfg, n_in=cfg.forecast_network['hiddens'][-1], n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MultiheadForecastLSTM model.
        
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - lstm_output_hindcast: Output sequence from the hindcast LSTM.
                - output_forecast: Predictions (before head layer) from the forecast period.
                - h_n_hindcast: Final hidden state of the hindcast model.
                - c_n_hindcast: Final cell state of the hindcast model.
                - y_hat: Predictions over the sequence from the head layer.
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # run hindcast part of the lstm
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(input=x_h)
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        output_hindcast = self.hindcast_head(self.dropout(lstm_output_hindcast))

        # reshape to [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)

        # run forecast heads
        batch_size = x_f.shape[1]
        x_f = x_f.transpose(0, 1).contiguous()
        x = torch.cat([h_n_hindcast.squeeze(dim=1), x_f.view(batch_size, -1)], dim=-1)
        x = self.forecast_network(x)
        x = x.view(batch_size, self.cfg.forecast_seq_length, -1)
        output_forecast = self.forecast_head(self.dropout(x))

        # start an output dictionary
        pred = {key: torch.cat([output_hindcast[key], output_forecast[key]], dim=1) for key in output_hindcast}

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'output_forecast': output_forecast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,
            }
        )

        return pred
