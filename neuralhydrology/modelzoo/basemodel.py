from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.evaluation.utils import load_scaler
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.samplingutils import sample_pointpredictions, umal_extend_batch


class BaseModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    Use subclasses of this class for training/evaluating different models, e.g. use `CudaLSTM` for training a standard
    LSTM model or `EA-LSTM` for training an Entity-Aware-LSTM. Refer to  :doc:`Documentation/Modelzoo </usage/models>` 
    for a full list of available models and how to integrate a new model. 

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
        if cfg.head.lower() == 'gmm':
            self.output_size *= 3 * cfg.n_distributions
        elif cfg.head.lower() == 'cmal':
            self.output_size *= 4 * cfg.n_distributions
        elif cfg.head.lower() == 'umal':
            self.output_size *= 2

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Provides point prediction samples from a probabilistic model.
        
        This function wraps the `sample_pointpredictions` function, which provides different point sampling functions
        for the different uncertainty estimation approaches. There are also options to handle negative point prediction 
        samples that arise while sampling from the uncertainty estimates. They can be controlled via the configuration. 
         

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of point predictions that ought ot be sampled form the model. 

        Returns
        -------
        Dict[str, torch.Tensor]
            Sampled point predictions 
        """
        scaler = load_scaler(self.cfg.run_dir)
        return sample_pointpredictions(self, data, n_samples, scaler)

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

    def pre_model_hook(self, data: Dict[str, torch.Tensor], is_train: bool) -> Dict[str, torch.Tensor]:
        """A function to execute before the model in training, validaton and test. 
        The beahvior can be adapted depending on the run configuration and the provided arguments.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs and labels y.
        is_train : bool
            Defines if the hook is executed in train mode or in validation/test mode.

        Returns
        -------
        data : Dict[str, torch.Tensor]
            The modified (or unmodified) data that are used for the training or evaluation.
        """
        if self.cfg.head.lower() == "umal":
            data = umal_extend_batch(data, self.cfg, n_taus=self.cfg.n_taus, extend_y=True)
        else:
            # here one can implement additional pre model hooks
            pass

        return data