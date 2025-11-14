import torch
import torch.nn as nn
from neuralhydrology.modelzoo.basemodel import BaseModel


class PersistentLSTM(BaseModel):
    """
    A stateful (persistent) LSTM for NeuralHydrology.
    It carries hidden states across batches inside each epoch,
    and resets the hidden state at the start of every epoch.
    """

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        # NH provides these sizes from the config
        input_size = self.input_size
        hidden_size = cfg.hidden_size
        num_layers = cfg.n_layers
        dropout_p = cfg.dropout

        # Standard PyTorch LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, self.output_size)

        # This stores the persistent state between batches
        self._persistent_hidden = None

    def reset_persistent_state(self):
        """Call at the start of each epoch to clear memory."""
        self._persistent_hidden = None

    def forward(self, data):
        """
        NeuralHydrology gives a dictionary `data`.
        The dynamic input sequence is in data["x_d"].
        Shape: [batch, seq_length, features]
        """
        x = data["x_d"]

        # If this is the first batch in the epoch → start fresh
        if self._persistent_hidden is None:
            B = x.size(0)
            h0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
            c0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
            hidden_states = (h0, c0)
        else:
            hidden_states = self._persistent_hidden

        # Forward through LSTM
        out, new_hidden = self.lstm(x, hidden_states)

        out = self.dropout(out)
        pred = self.fc(out)  # shape: [B, L, output_size]

        # Detach hidden state so no gradient carries over batch-to-batch
        self._persistent_hidden = (
            new_hidden[0].detach(),
            new_hidden[1].detach(),
        )

        # NH expects dict output
        return {"y_hat": pred}
