from typing import Dict

from mamba_ssm import Mamba
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class CudaMamba(BaseModel):
    """Mamba model class, which relies on https://github.com/state-spaces/mamba/tree/main.

    This class implements the Mamba model combined with a model head, as specified in the config. Depending on the
    embedding settings, static and/or dynamic features may or may not be fed through embedding networks before being
    concatenated and passed through the Mamba layer.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `CudaMamba` class only supports single-timescale predictions. Use `???` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'mamba', 'head']

    def __init__(self, cfg: Config):
        super(CudaMamba, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.mamba = Mamba(
            d_model=self.cfg.hidden_size,
            d_state=self.cfg.d_state,
            d_conv=self.cfg.d_conv,
            expand=self.cfg.expand,
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
        y_hat = self.mamba(x_d)
        pred = {'y_hat': y_hat}
        pred.update(self.head(self.dropout(y_hat)))

        return pred
