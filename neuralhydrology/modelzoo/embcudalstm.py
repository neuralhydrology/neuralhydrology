import warnings
from builtins import FutureWarning
from typing import Dict

import torch

from neuralhydrology.modelzoo import CudaLSTM
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class EmbCudaLSTM(BaseModel):
    """EmbCudaLSTM model class, which adds embedding networks for static inputs to the standard LSTM.

    .. deprecated:: 0.9.11-beta
       Use :py:class:`neuralhydrology.modelzoo.cudalstm.CudaLSTM` with ``statics_embedding``.

    This class extends the standard `CudaLSTM` class to preprocess the static inputs by an embedding network, prior
    to concatenating those values to the dynamic (time series) inputs. Use the config argument `statics_embedding` to
    specify the architecture of the fully-connected embedding network. No activation function is applied to the outputs
    of the embedding network.
    To control the initial forget gate bias, use the config argument `initial_forget_bias`. Often it is useful to set
    this value to a positive value at the start of the model training, to keep the forget gate closed and to facilitate
    the gradient flow.
    The `EmbCudaLSTM` class only supports single timescale predictions. Use `MTSLSTM` to train a model and get
    predictions on multiple temporal resolutions at the same time.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(EmbCudaLSTM, self).__init__(cfg=cfg)

        warnings.warn('EmbCudaLSTM is deprecated, the functionality is now part of CudaLSTM.', FutureWarning)

        self.cudalstm = CudaLSTM(cfg)

        # duplicate some members so CustomLSTM can access it
        self.embedding_net = self.cudalstm.embedding_net
        self.lstm = self.cudalstm.lstm
        self.head = self.cudalstm.head

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the EmbCudaLSTM model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
                - `h_n`: hidden state at the last time step of the sequence of shape [1, batch size, hidden size].
                - `c_n`: cell state at the last time step of the sequence of shape [1, batch size, hidden size].
        """

        return self.cudalstm.forward(data)
