from typing import Dict, List, Union

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config


class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
    LSTM model or `EA-LSTM` for training an Entity-Aware-LSTM. Refer to
    `Documentation/Modelzoo <https://neuralhydrology.readthedocs.io/en/latest/usage/models.html>`_ for a full list of
    available models and how to integrate a new model. 

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = []

    def __init__(self, cfg: Config):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.output_size = len(cfg.target_variables)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model output and potentially any intermediate states and activations as a dictionary.
        """
        raise NotImplementedError

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        """Sample model predictions, e.g., for MC-Dropout.
        
        This function does `n_samples` forward passes for each sample in the batch. Only useful for models with dropout,
        to perform MC-Dropout sampling. Make sure to set the model to train mode before calling this function 
        (`model.train()`), otherwise dropout won't be active.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.

        Returns
        -------
        torch.Tensor
            Sampled model outputs for the `predict_last_n` (config argument) time steps of each sequence. The shape of 
            the output is ``[batch size, predict_last_n, n_samples]``.
        """
        predict_last_n = self.cfg.predict_last_n
        samples = torch.zeros(data['x_d'].shape[0], predict_last_n, n_samples)
        for i in range(n_samples):
            prediction = self.forward(data)
            samples[:, -predict_last_n:, i] = prediction['y_hat'][:, -predict_last_n:, 0]

        return samples
