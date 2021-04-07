from typing import Dict

import pandas as pd
import torch

from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.utils.config import Config


class BaseRegularization(torch.nn.Module):
    """Base class for regularization terms.

    Regularization terms subclass this class by implementing the `forward` method.

    Parameters
    ----------
    cfg: Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(BaseRegularization, self).__init__()
        self.cfg = cfg

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                other_model_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the regularization term.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary of predicted variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary of ground truth variables for each frequency. If more than one frequency is predicted,
            the keys must have suffixes ``_{frequency}``. For the required keys, refer to the documentation
            of the concrete loss.
        other_model_data : Dict[str, torch.Tensor]
            Dictionary of all remaining keys-value pairs in the prediction dictionary that are not directly linked to 
            the model predictions but can be useful for regularization purposes, e.g. network internals, weights etc.
            
        Returns
        -------
        torch.Tensor
            The regularization value.
        """
        raise NotImplementedError


class TiedFrequencyMSERegularization(BaseRegularization):
    """Regularization that penalizes inconsistent predictions across frequencies.

    This regularization can only be used if at least two frequencies are predicted. For each pair of adjacent
    frequencies f and f', where f is a higher frequency than f', it aggregates the f-predictions to f' and calculates
    the mean squared deviation between f' and aggregated f.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError
        If the run configuration only predicts one frequency.
    """

    def __init__(self, cfg: Config):
        super(TiedFrequencyMSERegularization, self).__init__(cfg)
        self._frequencies = sort_frequencies(
            [f for f in cfg.use_frequencies if cfg.predict_last_n[f] > 0 and f not in cfg.no_loss_frequencies])

        if len(self._frequencies) < 2:
            raise ValueError("TiedFrequencyMSERegularization needs at least two frequencies.")

    def forward(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor],
                *args) -> torch.Tensor:
        """Calculate the sum of mean squared deviations between adjacent predicted frequencies.

        Parameters
        ----------
        prediction : Dict[str, torch.Tensor]
            Dictionary containing ``y_hat_{frequency}`` for each frequency.
        ground_truth : Dict[str, torch.Tensor]
            Dictionary continaing ``y_{frequency}`` for each frequency.

        Returns
        -------
        torch.Tensor
            The sum of mean squared deviations for each pair of adjacent frequencies.
        """

        loss = 0
        for idx, freq in enumerate(self._frequencies):
            if idx == 0:
                continue
            frequency_factor = int(get_frequency_factor(self._frequencies[idx - 1], freq))
            freq_pred = prediction[f'y_hat_{freq}']
            mean_freq_pred = freq_pred.view(freq_pred.shape[0], freq_pred.shape[1] // frequency_factor,
                                            frequency_factor, -1).mean(dim=2)
            lower_freq_pred = prediction[f'y_hat_{self._frequencies[idx - 1]}'][:, -mean_freq_pred.shape[1]:]
            loss = loss + torch.mean((lower_freq_pred - mean_freq_pred)**2)

        return loss
