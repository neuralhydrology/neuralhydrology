from typing import Dict

import torch

from neuralhydrology.training.basetrainer import BaseTrainer
from neuralhydrology.training.utils import umal_extend_batch


class UMALTrainer(BaseTrainer):
    """Class to train models with the UMAL head.
    
    UMAL can use a batch-extension for training. Thus, this class provides a hook (`_pre_model_hook`) to extend 
    the batch for the training. 

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: dict):
        super(UMALTrainer, self).__init__(cfg=cfg)

    def _pre_model_hook(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return umal_extend_batch(data, self.cfg, n_taus=self.cfg.n_taus, extend_y=True)
