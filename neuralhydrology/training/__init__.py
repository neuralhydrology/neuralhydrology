import logging
import warnings
from typing import List

import torch

import neuralhydrology.training.loss as loss
from neuralhydrology.training import regularization
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Get specific optimizer object, depending on the run configuration.
    
    Currently only 'Adam' is supported.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to be optimized.
    cfg : Config
        The run configuration.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer object that can be used for model training.
    """
    if cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate[0])
    else:
        raise NotImplementedError(f"{cfg.optimizer} not implemented or not linked in `get_optimizer()`")

    return optimizer


def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.
    
    Currently supported are 'MSE', 'NSE', 'RMSE', 'GMMLoss', 'CMALLoss', and 'UMALLoss'.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the 
        head.
    """
    if cfg.loss.lower() == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif cfg.loss.lower() == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "weightednse":
        warnings.warn("'WeightedNSE loss has been removed. Use 'NSE' with 'target_loss_weights'", FutureWarning)
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif cfg.loss.lower() == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif cfg.loss.lower() == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss_obj


def get_regularization_obj(cfg: Config) -> List[regularization.BaseRegularization]:
    """Get list of regularization objects.
    
    Currently, only the 'tie_frequencies' regularization is implemented.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    List[regularization.BaseRegularization]
        List of regularization objects that will be added to the loss during training.

    """
    regularization_modules = []
    for reg_name in cfg.regularization:
        if reg_name == "tie_frequencies":
            regularization_modules.append(regularization.TiedFrequencyMSERegularization(cfg))
        else:
            raise NotImplementedError(f"{reg_name} not implemented or not linked in `get_regularization_obj()`.")

    return regularization_modules
