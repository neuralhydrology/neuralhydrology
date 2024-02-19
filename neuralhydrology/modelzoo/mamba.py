from typing import Dict

try:
    from mamba_ssm import Mamba as Mamba_SSM
except ModuleNotFoundError:
    Mamba_SSM = None
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class Mamba(BaseModel):
    """Mamba model class, which relies on https://github.com/state-spaces/mamba/tree/main.

    This class implements the Mamba SSM with a combined model head, as specified in the config file, and a transition
    layer to ensure the input dimensions match the mamba_ssm specifications. Please read the mamba
    documentation to better learn about required hyperparameters.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'transition_layer', 'mamba', 'head']

    def __init__(self, cfg: Config):
        super(Mamba, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        # using a linear layer to move from the emdedded_layer dims to the specified hidden size
        self.transition_layer = nn.Linear(self.embedding_net.output_size, self.cfg.hidden_size)

        if Mamba_SSM is None:
            raise ModuleNotFoundError(
                f"mamba_ssm, and dependencies, required. Please run: pip install mamba_ssm causal-conv1d>=1.1.0"
            )
        else:
            self.mamba = Mamba_SSM(
                d_model=self.cfg.hidden_size,
                d_state=self.cfg.mamba_d_state,
                d_conv=self.cfg.mamba_d_conv,
                expand=self.cfg.mamba_expand,
            )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the CudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [batch size, 1, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [batch size, 1, hidden size].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        # reshaping dimensions to what mamba expects:
        x_d_transition = self.transition_layer(x_d)

        mamba_output = self.mamba(x_d_transition)

        # reshape to [batch_size, seq, n_hiddens]
        mamba_output = mamba_output.transpose(0, 1)

        pred = {'y_hat': mamba_output}
        pred.update(self.head(self.dropout(mamba_output)))
        return pred
