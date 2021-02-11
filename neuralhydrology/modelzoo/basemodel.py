from typing import Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Uniform

from neuralhydrology.utils.config import Config
from neuralhydrology.training.utils import umal_extend_batch


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

    def _handle_negative_values(self, values: torch.Tensor, sample_values: Callable) -> torch.Tensor:
        """Handle negative samples that arise by sampling from the uncertainty estimates.
        
        Currently supports (a) 'clip' for directly clipping values at zero and (b) 'truncate' for resampling values 
        that are below zero.

        Parameters
        ----------
        values : torch.Tensor
            Tensor with the sampled values
        sample_values : Callable
            Sampling function to allow for repeated sampling in the case of truncation-handling. 
            
        Returns
        -------
        torch.Tensor
            Bound values according to user specifications.  
        """
        if self.cfg.negative_sample_handling.lower() == 'clip':
            values = torch.relu(values)
        elif self.cfg.negative_sample_handling.lower() == 'truncate':
            values_smaller_zero = values < 0
            try_count = 0
            while any(values_smaller_zero.flatten()):
                values[values_smaller_zero] = sample_values(values_smaller_zero)
                try_count += 1
                if try_count >= self.cfg.negative_sample_max_retries:
                    break
        elif self.cfg.negative_sample_handling is None or self.cfg.negative_sample_handling.lower() == 'none':
            pass
        else:
            raise NotImplementedError(
                f"The option {self.cfg.negative_sample_handling} is not supported for handling negative samples!")

        return values

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Sample model predictions, e.g., for MC-Dropout.
        
        Naive sampling. This function does `n_samples` forward passes for each sample in the batch. Currently it is 
        only useful for models with dropout, to perform MC-Dropout sampling. 
        Make sure to set the model to train mode before calling this function (`model.train()`), 
        otherwise dropout won't be active.
        
        The negative sample handling currently supports (a) 'clip' for directly clipping values at zero and (b) 
        'truncate' for resampling values that are below zero. The mode can be defined by the config argument 
        'negative_sample_handling'.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
            each frequency.
        """
        # setup:
        if self.cfg.use_frequencies:
            freq_suffixes = [f'_{freq}' for freq in self.cfg.use_frequencies]
        else:
            freq_suffixes = ['']

        # sample for different frequencies:
        samples = {}
        for freq_suffix in freq_suffixes:
            # get predict_last_n for the given the mode:
            if isinstance(self.cfg.predict_last_n, int):
                predict_last_n = self.cfg.predict_last_n
            else:
                predict_last_n = self.cfg.predict_last_n[freq_suffix[1:]]

            # unbound sampling:
            def _sample_values(ids):
                values = torch.zeros(len(ids), predict_last_n, n_samples)
                for i in range(n_samples):
                    # We make a forward-pass of the model for each frequency separately to guarantee independence:
                    prediction = self.forward(data)
                    values[ids, -predict_last_n:, i] = prediction[f'y_hat{freq_suffix}'][:, -predict_last_n:, 0]
                return values

            ids = list(range(data[f'x_d{freq_suffix}'].shape[0]))
            values = _sample_values(ids)

            # bind values according to specifications and add to samples:
            values = self._handle_negative_values(values, _sample_values)
            freq_key = f'y_hat{freq_suffix}'

            samples.update({freq_key: values})

        return samples

    def sample_gmm(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Sample model predictions with the Gaussian Mixture (GMM) head.

        This function generates `n_samples` GMM sample points for each entry in the batch. Concretely, the model is 
        executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
        Good references for learning about GMMs are [#]_ and [#]_. 
        
        The negative sample handling currently supports (a) 'clip' for directly clipping values at zero and 
         (b) 'truncate' for resampling values that are below zero. The mode can be defined by the config argument 
         'negative_sample_handling'.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
            each frequency. 
            
        References
        ----------
        .. [#] C. M. Bishop: Mixture density networks. 1994.
        .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
               URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
        """
        if not self.cfg.head.lower() == "gmm":
            raise NotImplementedError(f"GMM sampling not supported for the {self.cfg.head} head!")

        # setup:
        if self.cfg.use_frequencies:
            freq_suffixes = [f'_{freq}' for freq in self.cfg.use_frequencies]
        else:
            freq_suffixes = ['']

        pred = self.forward(data)

        # sample for different frequencies:
        samples = {}
        for freq_suffix in freq_suffixes:
            # get predict_last_n for the given the mode:
            if isinstance(self.cfg.predict_last_n, int):
                predict_last_n = self.cfg.predict_last_n
            else:
                predict_last_n = self.cfg.predict_last_n[freq_suffix[1:]]

            # GMM has 3 parts: means (m/mu), variances (s/sigma), and weights (p/pi):
            m, s, p = pred[f'mu{freq_suffix}'],\
                      pred[f'sigma{freq_suffix}'], \
                      pred[f'pi{freq_suffix}']
            m_sub = torch.repeat_interleave(m[:, -predict_last_n:, :], n_samples, dim=0)
            s_sub = torch.repeat_interleave(s[:, -predict_last_n:, :], n_samples, dim=0)
            p_sub = torch.repeat_interleave(p[:, -predict_last_n:, :], n_samples, dim=0)

            # unbound sampling:
            categorical = Categorical(p_sub)
            pis = categorical.sample().data
            mask = torch.zeros(s_sub.shape, dtype=torch.bool) \
                .to(device=s_sub.device) \
                .scatter_(2, pis.unsqueeze(2), True)

            # define sampling function:
            def _sample_values(ids):
                values = s_sub \
                    .data.new(s_sub[ids][mask[ids]].shape[0]) \
                    .normal_() \
                    .flatten() \
                    .mul(s_sub[ids][mask[ids]]) \
                    .add(m_sub[ids][mask[ids]])
                return values

            ids = np.ones(s_sub.shape, dtype=bool)
            values = _sample_values(ids=ids)
            # bind values and add to samples:
            values = self._handle_negative_values(values, _sample_values)
            values = values.view(-1, predict_last_n, n_samples)

            freq_key = f'y_hat{freq_suffix}'
            samples.update({freq_key: values})
        return samples

    def sample_cmal(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Sample model predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

        This function generates `n_samples` CMAL sample points for each entry in the batch. Concretely, the model is 
        executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
        General information about CMAL can be found in [#]_.

        The negative sample handling currently supports (a) 'clip' for directly clipping values at zero and (b) 
        'truncate' for resampling values that are below zero. The mode can be defined by the config argument 
        'negative_sample_handling'.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
            each frequency. The shape of the output tensor for each frequency is 
            ``[batch size, predict_last_n, n_samples]``.
            
        References
        ----------
        .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing: 
            Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295,
            2020.
        """
        if not self.cfg.head.lower() == "cmal":
            raise NotImplementedError(f"CMAL sampling not supported for the {self.cfg.head} head!")

        # setup:
        if self.cfg.use_frequencies:
            freq_suffixes = [f'_{freq}' for freq in self.cfg.use_frequencies]
        else:
            freq_suffixes = ['']
        batch_size = data[f'x_d{freq_suffixes[0]}'].shape[0]
        uniform01 = Uniform(low=0, high=1)

        pred = self.forward(data)

        # sample for different frequencies:
        samples = {}
        for freq_suffix in freq_suffixes:
            # get predict_last_n for the given the mode:
            if isinstance(self.cfg.predict_last_n, int):
                predict_last_n = self.cfg.predict_last_n
            else:
                predict_last_n = self.cfg.predict_last_n[freq_suffix[1:]]

            # CMAL has 4 parts: means (m/mu), scales (b), asymmetries (t/) and weights (p/pi):
            m = pred[f'mu{freq_suffix}']
            b = pred[f'b{freq_suffix}']
            t = pred[f'tau{freq_suffix}']
            p = pred[f'pi{freq_suffix}']

            # sampling presets:
            m_sub = torch.repeat_interleave(m[:, -predict_last_n:, :], n_samples, dim=0)
            b_sub = torch.repeat_interleave(b[:, -predict_last_n:, :], n_samples, dim=0)
            t_sub = torch.repeat_interleave(t[:, -predict_last_n:, :], n_samples, dim=0)
            p_sub = torch.repeat_interleave(p[:, -predict_last_n:, :], n_samples, dim=0)

            # unbound sampling:
            values = torch.zeros((batch_size * n_samples, predict_last_n)).to(m.device)
            for step in range(predict_last_n):
                sub_choices = torch.multinomial(p_sub[:, step, :], num_samples=1)
                t_sc = t_sub[:, step, :].gather(1, sub_choices)
                m_sc = m_sub[:, step, :].gather(1, sub_choices)
                b_sc = b_sub[:, step, :].gather(1, sub_choices)

                # define local sampler:
                def _sample_values(ids):
                    prob = uniform01.sample(sample_shape=m_sc[ids].shape).to(m.device)
                    values = torch.where(
                        prob < t_sc[ids],  # needs to be in accordance with the loss
                        m_sc[ids] + ((b_sc[ids] * torch.log(prob / t_sc[ids])) / (1 - t_sc[ids])),
                        m_sc[ids] - ((b_sc[ids] * torch.log((1 - prob) / (1 - t_sc[ids]))) / t_sc[ids]))
                    return values.flatten()

                ids = np.ones(b_sc.shape, dtype=bool)
                value_buffer = _sample_values(ids)
                values[:, step] = self._handle_negative_values(value_buffer, _sample_values)

            # add values to samples:
            freq_key = f'y_hat{freq_suffix}'
            samples.update({freq_key: values.view(-1, predict_last_n, n_samples)})
        return samples

    def sample_umal(self, data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
        """Sample model predictions with the Uncountable Mixture of Asymmetric Laplacians (UMAL) head.

        This function generates `n_samples` UMAL sample points for each entry in the batch. Concretely, the model is 
        executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
        Details about the UMAL approach can be found in [#]_.
        
        The negative sample handling currently supports (a) 'clip' for directly clipping values at zero and (b) 
        'truncate' for resampling values that are below zero. The mode can be defined by the config argument 
        'negative_sample_handling'.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        n_samples : int
            Number of samples to generate for each input sample.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
            each frequency.
            
        References
        ----------
        .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
            with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
            pp. 8838-8848, 2019.
        """
        if not self.cfg.head.lower() == "umal":
            raise NotImplementedError(f"UMAL sampling not supported for the {self.cfg.head} head")

        # setup:
        if self.cfg.use_frequencies:
            freq_suffixes = [f'_{freq}' for freq in self.cfg.use_frequencies]
        else:
            freq_suffixes = ['']
        batch_size_original = data[f'x_d{freq_suffixes[0]}'].shape[0]
        # n_taus expands the batch by itself and adds a sampled tau as input (new_batch_size = n_taus*batch_size):
        data = umal_extend_batch(data, self.cfg, n_taus=self.cfg.n_taus)

        pred = self.forward(data)

        # sample for different frequencies:
        samples = {}
        for freq_suffix in freq_suffixes:

            # get predict_last_n for the given the mode:
            if isinstance(self.cfg.predict_last_n, int):
                predict_last_n = self.cfg.predict_last_n
            else:
                predict_last_n = self.cfg.predict_last_n[freq_suffix[1:]]

            # UMAL has 2 parts: means (m/mu), scales (b); the tau is randomly chosen:
            m = pred[f'mu{freq_suffix}']
            b = pred[f'b{freq_suffix}']
            t = data[f'tau{freq_suffix}']

            # sampling presets:
            m_sub = torch.cat(m[:, -predict_last_n:, :].split(batch_size_original, 0), 2)
            b_sub = torch.cat(b[:, -predict_last_n:, :].split(batch_size_original, 0), 2)
            t_sub = torch.cat(t[:, -predict_last_n:, :].split(batch_size_original, 0), 2)

            # sample and bind values according to specification:
            uniform01 = Uniform(low=0, high=1)
            values = torch.zeros((batch_size_original, predict_last_n, n_samples)).to(m.device)
            for n in range(n_samples):
                sub_choice = np.random.randint(0, self.cfg.n_taus)
                m_sc = m_sub[:, :, sub_choice]
                b_sc = b_sub[:, :, sub_choice]
                t_sc = t_sub[:, :, sub_choice]

                def _sample_values(ids):
                    prob = uniform01.sample(sample_shape=m_sc[ids].shape).to(m.device)
                    values = torch.where(
                        prob < t_sc[ids],  # needs to be in accordance with the loss
                        m_sc[ids] + ((b_sc[ids] * torch.log(prob / t_sc[ids])) / (1 - t_sc[ids])),
                        m_sc[ids] - ((b_sc[ids] * torch.log((1 - prob) / (1 - t_sc[ids]))) / t_sc[ids]))
                    return values.flatten()

                ids = np.ones(b_sc.shape, dtype=bool)
                values_buffer = _sample_values(ids)
                values[:, -predict_last_n:, n] = self._handle_negative_values(values_buffer, _sample_values).\
                    reshape(m_sc.shape[0], predict_last_n)

            # add values to samples:
            freq_key = f'y_hat{freq_suffix}'
            samples.update({freq_key: values.view(-1, predict_last_n, n_samples)})
        return samples
