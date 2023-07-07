from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC


class SequentialForecastLSTM(BaseModel):
    """A forecasting model that uses a single LSTM sequence with multiple embedding layers.

    This is a forecasting model that uses a single sequential (LSTM) model that rolls 
    out through both the hindcast and forecast sequences. The difference between this
    and a standard ``CudaLSTM`` is (1) this model uses both hindcast and forecast
    input features, and (2) it uses a separate embedding network for the hindcast
    period and the forecast period. 
    
    Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError if forecast_overlap > 0
    ValueError if forecast and hindcast embedding nets have different output sizes.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(SequentialForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError('Forecast overlap cannot be set for a sequential forecasting model. '
                             'Please set to None or remove from config file.')

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError('Forecast and hindcast embedding nets must have the same output size when using a sequential forecast LSTM.')

        self.lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=cfg.hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the SequentialForecastLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - lstm_output_hindcast: Output sequence from the hindcast LSTM.
                - lstm_output_forecast: Output sequence from the forecast LSTM.
                - h_n_hindcast: Final hidden state of the hindcast model.
                - c_n_hindcast: Final cell state of the hindcast model.
                - h_n_forecast: Finall hidden state of the forecast model.
                - c_n_forecast: Final cell state of the forecast model.
                - y_hat: Predictions over the sequence from the head layer.
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # run hindcast part of the lstm
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.lstm(input=x_h)
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)

        # run forecast part of the lstm
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.lstm(x_f, (h_n_hindcast, c_n_hindcast))
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)

        # run head
        concatenated_predictions = torch.cat([lstm_output_hindcast, lstm_output_forecast], dim=1)
        pred = self.head(self.dropout(concatenated_predictions))

        # reshape to [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)
        h_n_forecast = h_n_forecast.transpose(0, 1)
        c_n_forecast = c_n_forecast.transpose(0, 1)

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_forecast': lstm_output_forecast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,

                'h_n_forecast': h_n_forecast,
                'c_n_forecast': c_n_forecast,
            }
        )

        return pred
