from typing import Dict, Tuple

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


class MCLSTM(BaseModel):
    """Mass-Conserving LSTM model from Hoedt et al. [1]_.

    This model implements the exact MC-LSTM configuratin that was used by Hoedt et al. [1]_ in the hydrology experiment
    (for a more general and configurable MC-LSTM model class, check the official MC-LSTM
    `GitHub repository <https://github.com/ml-jku/mc-lstm>`_).

    The MC-LSTM is an LSTM-inspired timeseries model that guarantees to conserve the mass of a specified `mass_input` by
    the special design of its architecture. The model consists of three parts:

    - an input/junction gate that distributes the mass input at a specific timestep across the memory cells
    - a redistribution matrix that allows for internal reorganization of the stored mass
    - an output gate that determines the fraction of the stored mass that is subtracted from the memory cells and
      defines the output of the MC-LSTM

    Starting from the general MC-LSTM architecture as presented by Hoedt et al. [1]_ the most notably adaption for the
    hydrology application is the use of a so-called "*trash cell*". The trash cell is one particular cell of the cell
    state vector that is not used for deriving the model prediction, which is defined as the sum of the outgoing mass of
    all memory cells except the trash cell. For more details and different variants that were tested for the application
    to hydrology, see Appendix B.4.2 in Hoedt et al [1]_.

    The config argument `head` is ignored for this model and the model prediction is always computed as the sum over
    the outgoing mass (excluding the trash cell output).

    The config argument `initial_forget_bias` is here used to close (negative values) or to open (positive values) the
    output gate at the beginning of the training. Having the output gate closed means that the MC-LSTM has to actively
    learn when to remove mass from the system, which can be seen as an analogy to an open forget gate in the standard
    LSTM.

    To use this model class, you have to specify the name of the mass input using the `mass_input` config argument.
    Additionally, the mass input and target variable should *not* be normalized. Use the config argument
    `custom_normalization` and set the `centering` and `scaling` key for both to `None` (see
    :doc:`config arguments </usage/config>` for more details on `custom_normalization`).

    Currently, only a single mass input per time step is supported, as well as a single target.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError
        If no or more than one `mass_input` is specified in the config. Additionally, an error is raised if the hidden
        size is smaller than 2, the number of `target_variables` is greater than 1, or `dynamics_embedding` is specified
        in the config, which is (currently) not supported for this model class.

    References
    ----------
    .. [1] Hoedt, P. J., Kratzert, F., Klotz, D., Halmich, C., Holzleitner, M., Nearing, G., Hochreiter, S., and
        Klambauer, G: MC-LSTM: Mass-Conserving LSTM, arXiv Preprint, https://arxiv.org/abs/2101.05186, 2021.
    """
    module_parts = ["embedding_net", "mclstm"]

    def __init__(self, cfg: Config):
        super(MCLSTM, self).__init__(cfg=cfg)

        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars > 1:
            raise ValueError("Currently, MC-LSTM only supports a single mass input")
        elif self._n_mass_vars == 0:
            raise ValueError("No mass input specified. Specify mass input variable using `mass_inputs`")

        if cfg.dynamics_embedding is not None:
            raise ValueError("Embedding for dynamic inputs is not supported with the current version of MC-LSTM")

        if cfg.hidden_size <= 1:
            raise ValueError("At least hidden size 2 is required for one (mandatory) trash cell and a mass cell.")

        if len(cfg.target_variables) > 1:
            raise ValueError("Currently, MC-LSTM only supports single target settings.")

        self.embedding_net = InputLayer(cfg)

        n_aux_inputs = self.embedding_net.statics_output_size + self.embedding_net.dynamics_output_size
        self.mclstm = _MCLSTMCell(mass_input_size=self._n_mass_vars,
                                  aux_input_size=n_aux_inputs,
                                  hidden_size=cfg.hidden_size,
                                  cfg=cfg)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MC-LSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, 1].
                - `m_out`: mass output of the MC-LSTM (including trash cell as index 0) of shape [batch size, sequence
                  length, hidden size].
                - `c`: cell state of the MC-LSTM of shape [batch size, sequence length, hidden size].

        """
        # possibly pass static inputs through embedding layers and concatenate with dynamics
        x_d = self.embedding_net(data, concatenate_output=True)

        # the basedataset stores the mass input at the beginning
        x_m = x_d[:, :, :self._n_mass_vars]
        x_a = x_d[:, :, self._n_mass_vars:]

        # perform forward pass through the MC-LSTM cell
        m_out, c = self.mclstm(x_m, x_a)

        # exclude trash cell from model predictions (see linked publication for details.)
        output = m_out[:, :, 1:].sum(dim=-1, keepdim=True)

        return {'y_hat': output.transpose(0, 1), 'm_out': m_out.transpose(0, 1), 'c': c.transpose(0, 1)}


class _MCLSTMCell(nn.Module):
    """The logic of the MC-LSTM cell"""

    def __init__(self, mass_input_size: int, aux_input_size: int, hidden_size: int, cfg: Config):

        super(_MCLSTMCell, self).__init__()
        self.cfg = cfg
        self._hidden_size = hidden_size

        gate_inputs = aux_input_size + hidden_size + mass_input_size

        # initialize gates
        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.input_gate = _NormalizedGate(in_features=gate_inputs,
                                          out_shape=(mass_input_size, hidden_size),
                                          normalizer="normalized_sigmoid")
        self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                              out_shape=(hidden_size, hidden_size),
                                              normalizer="normalized_relu")

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=self.cfg.initial_forget_bias)

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass on the MC-LSTM cell.

        Parameters
        ----------
        x_m : torch.Tensor
            Mass input that will be conserved by the network.
        x_a : torch.Tensor
            Auxiliary inputs that will be used to modulate the gates but whose information won't be stored internally
            in the MC-LSTM cells.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Outgoing mass and memory cells per time step of shape [sequence length, batch size, hidden size]

        """
        _, batch_size, _ = x_m.size()
        ct = x_m.new_zeros((batch_size, self._hidden_size))

        m_out, c = [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct = self._step(xt_m, xt_a, ct)

            m_out.append(mt_out)
            c.append(ct)

        m_out, c = torch.stack(m_out), torch.stack(c)

        return m_out, c

    def _step(self, xt_m, xt_a, c):
        """ Make a single time step in the MCLSTM. """
        # in this version of the MC-LSTM all available data is used to derive the gate activations. Cell states
        # are L1-normalized so that growing cell states over the sequence don't cause problems in the gates.
        features = torch.cat([xt_m, xt_a, c / (c.norm(1) + 1e-5)], dim=-1)

        # compute gate activations
        i = self.input_gate(features)
        r = self.redistribution(features)
        o = self.output_gate(features)

        # distribute incoming mass over the cell states
        m_in = torch.matmul(xt_m.unsqueeze(-2), i).squeeze(-2)

        # reshuffle the mass in the cell states using the redistribution matrix
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)

        # compute the new mass states
        m_new = m_in + m_sys

        # return the outgoing mass and subtract this value from the cell states.
        return o * m_new, (1 - o) * m_new


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalised gate"""
        return torch.sigmoid(self.fc(x))


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)
