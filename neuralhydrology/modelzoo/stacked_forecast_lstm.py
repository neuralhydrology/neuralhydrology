from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.fc import FC


class StackedForecastLSTM(BaseModel):
    """A forecasting model using stacked LSTMs for hindcast and forecast.

    This is a forecasting model that uses two stacked sequential (LSTM) models to handle 
    hindcast vs. forecast. The hindcast and forecast sequences must be the same length,
    and the ``forecast_overlap`` config parameter must be set to the correct overlap
    between these two sequences. For example, if we want to use a hindcast sequence
    length of 365 days to make a 7-day forecast, then ``seq_length`` and 
    ``forecast_seq_length`` must both be set to 365, and ``forecast_overlap`` must be
    set to 358 (=365-7). Outputs from the hindcast LSTM are concatenated to the input 
    sequences to the forecast LSTM. This causes a lag of length (``seq_length`` - ``forecast_overlap``)
    timesteps between the latest hindcast data and the newest forecast point, meaning
    that forecasts do not get information from the most recent dynamic inputs. To solve
    this, set the ``bidirectional_stacked_forecast_lstm`` config parameter to True, so
    that the hindcast LSTM runs bidirectional and therefore all outputs from the hindcast
    LSTM receive information from the most recent dynamic input data.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'forecast_lstm', 'hindcast_lstm', 'head']

    def __init__(self, cfg: Config):
        super(StackedForecastLSTM, self).__init__(cfg=cfg)

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_hidden_size = cfg.hindcast_hidden_size
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=self.hindcast_hidden_size,
            bidirectional=cfg.bidirectional_stacked_forecast_lstm
        )
        self.forecast_hidden_size = cfg.forecast_hidden_size
        forecast_input_size = self.forecast_embedding_net.output_size + self.hindcast_hidden_size
        if cfg.bidirectional_stacked_forecast_lstm:
            forecast_input_size += self.hindcast_hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=forecast_input_size,
            hidden_size=self.forecast_hidden_size
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=self.forecast_hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.hindcast_hidden_size:2 * self.hindcast_hidden_size] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.forecast_hidden_size:2 * self.forecast_hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the StackedForecastLSTM model.

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
                - y_hat: Predictions over the sequence from the head layer.

        Raises
        ------
        ValueError if hindcast and forecast sequences are not equal.
        """

        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        if x_h.shape[0] != x_f.shape[0]:
            raise ValueError('Hindcast and forecast sequences must be equal.')

        # run hindcast part of the lstm
        lstm_output_hindcast, _ = self.hindcast_lstm(input=x_h)

        # LSTM here is NOT batch-first, thus why the transpose is necessary below.
        # The LSTM output at this point is (seq, batch, dims).
        forecast_inputs = torch.concat((x_f, lstm_output_hindcast), dim=-1)  # concat along variable dimension

        # run forecast part of the lstm
        lstm_output_forecast, _ = self.forecast_lstm(forecast_inputs)
        
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)

        # run head
        pred = self.head(self.dropout(lstm_output_forecast))

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_forecast': lstm_output_forecast,
            }
        )

        return pred
