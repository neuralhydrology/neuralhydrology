from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer

class HandoffForecastLSTM(BaseModel):
    """An encoder/decoder LSTM model class used for forecasting.

    This is a forecasting model that uses a state-handoff to transition from a hindcast sequence model to a forecast 
    sequence (LSTM) model. The hindcast model is run from the past up to present (the issue time of the forecast) 
    and then passes the cell state and hidden state of the LSTM into a (nonlinear) handoff network, which is then 
    used to initialize the cell state and hidden state of a new LSTM that rolls out over the forecast period. 
    The handoff network is implemented as a custom FC layer, which can have multiple layers. The handoff network is 
    implemented using the ``state_handoff_network`` config parameter. The hindcast and forecast LSTMs have different 
    weights and biases, different heads, and different embedding networks. The hidden size of the hindcast 
    LSTM is set using the ``hindcast_hidden_size`` config parameter and the hidden size of the forecast LSTM is set 
    using the ``forecast_hidden_size`` config parameter.

    The handoff forecast LSTM model can implement a delayed handoff as well, such that the handoff between the hindcast
    and forecast LSTM occurs prior to the forecast issue time. This is controlled by the ``forecast_overlap`` parameter
    in the config file, and the forecast and hindcast LSTMs will run concurrently for the number of timesteps indicated
    by ``forecast_overlap``. We recommend using the ``ForecastOverlapMSERegularization`` regularization option to
    regularize the loss function by (dis)agreement between the overlapping portion of the hindcast and forecast LSTMs.
    This regularization term can be requested by setting the ``regularization`` parameter in the config file to include
    ``forecast_overlap``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    
    Raises
    ------
    ValueError if a state_handoff_network is not specified.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'hindcast_lstm', 'forecast_lstm', 'hindcast_head', 'forecast_head', 'handoff_net']

    def __init__(self, cfg: Config):
        super(HandoffForecastLSTM, self).__init__(cfg=cfg)

        self.initial_hindcast_seq_length = self.cfg.seq_length - self.cfg.forecast_seq_length

        self.forecast_embedding_net = InputLayer(cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg, embedding_type='hindcast')

        self.hindcast_hidden_size = cfg.hindcast_hidden_size
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=self.hindcast_hidden_size
        )
        self.forecast_hidden_size = cfg.forecast_hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=self.forecast_hidden_size
        )

        if not cfg.state_handoff_network:
            raise ValueError('The handoff forecast LSTM requires a state handoff network specified in the config file.')

        self.handoff_net = FC(
            input_size=self.hindcast_hidden_size*2,
            hidden_sizes=cfg.state_handoff_network['hiddens'],
            activation=cfg.state_handoff_network['activation'],
            dropout=cfg.state_handoff_network['dropout']
        )
        self.handoff_linear =  FC(
            input_size=cfg.state_handoff_network['hiddens'][-1],
            hidden_sizes=[self.forecast_hidden_size*2],
            activation='linear',
            dropout=0.0
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.hindcast_head = get_head(cfg=cfg, n_in=self.hindcast_hidden_size, n_out=self.output_size)
        self.forecast_head = get_head(cfg=cfg, n_in=self.forecast_hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.hindcast_hidden_size:2 * self.hindcast_hidden_size] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.forecast_hidden_size:2 * self.forecast_hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the HandoffForecastLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - lstm_output_hindcast: Output sequence from the hindcast LSTM.
                - lstm_output_hindcast_overlap: Output sequence from the hindcast model over the overlap period between forecast and hindcast LSTMs.
                - lstm_output_forecast_overlap: Output sequence from the forecast model over the overlap period between forecast and hindcast LSTMs.
                - lstm_output_forecast: Output sequence from the forecast LSTM.
                - y_forecast: Predictions (after head layer) over the forecast period.
                - y_forecast_overlap: Predictions from the forecast model over the overlap period between forecast and hindcast LSTMs.
                - y_hindcast_overlap: Predictions from the hindcast model over the overlap period between forecast and hindcast LSTMs.
                - y_hindcast: Predictions over the hindcast period.
                - h_n_hindcast: Final hidden state of the hindcast model.
                - c_n_hindcast: Final cell state of the hindcast model.
                - h_n_handoff: Initial hidden state of the forecast model.
                - c_n_handoff: Initial cell state of the forecast model.
                - h_n_forecast: Finall hidden state of the forecast model.
                - c_n_forecast: Final cell state of the forecast model.
                - y_hat: Predictions over the sequence from the head layer. This is a concatenation of hindcast and forecast, and takes from hindcast for the overlap portion.
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_h = self.hindcast_embedding_net(data)
        x_f = self.forecast_embedding_net(data)

        # run the hindcast lstm
        lstm_output_hindcast, (h_n_hindcast, c_n_hindcast) = self.hindcast_lstm(x_h[:self.initial_hindcast_seq_length, ...])
        lstm_output_hindcast = lstm_output_hindcast.transpose(0, 1)
        if x_h.shape[0] > self.initial_hindcast_seq_length:
            lstm_output_hindcast_overlap, _ = self.hindcast_lstm(x_h[self.initial_hindcast_seq_length:, ...], (h_n_hindcast, c_n_hindcast))
            lstm_output_hindcast_overlap = lstm_output_hindcast_overlap.transpose(0, 1)
        else:
            lstm_output_hindcast_overlap = None

        # handoff initial state to forecast lstm
        x = self.handoff_net(torch.cat([h_n_hindcast, c_n_hindcast], -1))
        initial_state = self.handoff_linear(x)
        h_n_handoff, c_n_handoff = initial_state.chunk(2, -1)
        h_n_handoff = h_n_handoff.contiguous()
        c_n_handoff = c_n_handoff.contiguous()

        # run forecast lstm
        lstm_output_forecast, (h_n_forecast, c_n_forecast) = self.forecast_lstm(x_f, (h_n_handoff, c_n_handoff))
        lstm_output_forecast = lstm_output_forecast.transpose(0, 1)
        lstm_output_forecast_overlap = lstm_output_forecast[:, :self.cfg.forecast_overlap, :]
        lstm_output_forecast = lstm_output_forecast[:, self.cfg.forecast_overlap:, :]

        # run heads for hindcast and forecast
        y_hindcast = self.hindcast_head(self.dropout(lstm_output_hindcast))
        y_forecast = self.forecast_head(self.dropout(lstm_output_forecast))
        if x_h.shape[0] > self.initial_hindcast_seq_length:
            y_hindcast_overlap = self.hindcast_head(self.dropout(lstm_output_hindcast_overlap))
            y_forecast_overlap = self.forecast_head(self.dropout(lstm_output_forecast_overlap))
            pred = {key: torch.cat([y_hindcast[key], y_hindcast_overlap[key], y_forecast[key]], dim=1) for key in y_hindcast}
        else:
            pred = {key: torch.cat([y_hindcast[key], y_forecast[key]], dim=1) for key in y_hindcast}
            y_hindcast_overlap, y_forecast_overlap = None, None

        # reshape to [batch_size, seq, n_hiddens]
        h_n_hindcast = h_n_hindcast.transpose(0, 1)
        c_n_hindcast = c_n_hindcast.transpose(0, 1)
        h_n_handoff = h_n_handoff.transpose(0, 1)
        c_n_handoff = c_n_handoff.transpose(0, 1)
        h_n_forecast = h_n_forecast.transpose(0, 1)
        c_n_forecast = c_n_forecast.transpose(0, 1)

        pred.update(
            {
                'lstm_output_hindcast': lstm_output_hindcast,
                'lstm_output_hindcast_overlap': lstm_output_hindcast_overlap,
                'lstm_output_forecast_overlap': lstm_output_forecast_overlap,
                'lstm_output_forecast': lstm_output_forecast,

                'y_forecast': y_forecast,
                'y_forecast_overlap': y_forecast_overlap,
                'y_hindcast_overlap': y_hindcast_overlap,
                'y_hindcast': y_hindcast,

                'h_n_hindcast': h_n_hindcast,
                'c_n_hindcast': c_n_hindcast,

                'h_n_handoff': h_n_handoff,
                'c_n_handoff': c_n_handoff,

                'h_n_forecast': h_n_forecast,
                'c_n_forecast': c_n_forecast,
            }
        )

        return pred
