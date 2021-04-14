from typing import List, Dict, Callable

import numpy as np
import torch
from torch.distributions import Categorical, Uniform

from neuralhydrology.utils.config import Config
from neuralhydrology.training.utils import umal_extend_batch


def sample_pointpredictions(model: 'BaseModel', data: Dict[str, torch.Tensor],
                            n_samples: int) -> Dict[str, torch.Tensor]:
    """Point prediction samplers for the different uncertainty estimation approaches.
    
    This function provides different point sampling functions for the different uncertainty estimation approaches 
    (i.e. Gaussian Mixture Models (GMM), Countable Mixtures of Asymmetric Laplacians (CMAL), Uncountable Mixtures of 
    Asymmetric Laplacians (UMAL), and Monte-Carlo Dropout (MCD); note: MCD can be combined with the others, by setting 
    `mc_dropout` to `True` in the configuration file). 
    
    There are also options to handle negative point prediction samples that arise while sampling from the uncertainty 
    estimates. This functionality currently supports (a) 'clip' for directly clipping values at zero and 
    (b) 'truncate' for resampling values that are below zero. 
    
    Parameters
    ----------
    model : BaseModel
        The neuralhydrology model from which to sample from.
    data : Dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        The number of point prediction samples that should be created.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of 
        each frequency.
    """

    if model.cfg.head.lower() == "gmm":
        samples = sample_gmm(model, data, n_samples)
    elif model.cfg.head.lower() == "cmal":
        samples = sample_cmal(model, data, n_samples)
    elif model.cfg.head.lower() == "umal":
        samples = sample_umal(model, data, n_samples)
    elif model.cfg.head.lower() == "regression":
        samples = sample_mcd(model, data, n_samples)  # regression head assumes mcd
    else:
        raise NotImplementedError(f"Sampling mode not supported for head {model.cfg.head.lower()}!")

    return samples


def _subset_target(parameter: Dict[str, torch.Tensor], n_target: int, steps: int) -> Dict[str, torch.Tensor]:
    # determine which output neurons correspond to the n_target target variable
    start = n_target * steps
    end = (n_target + 1) * steps
    parameter_sub = parameter[:, :, start:end]

    return parameter_sub


def _handle_negative_values(cfg: Config, values: torch.Tensor, sample_values: Callable) -> torch.Tensor:
    """Handle negative samples that arise while sampling from the uncertainty estimates.

    Currently supports (a) 'clip' for directly clipping values at zero and (b) 'truncate' for resampling values 
    that are below zero. 

    Parameters
    ----------
    cfg : Config
        The run configuration.
    values : torch.Tensor
        Tensor with the sampled values.
    sample_values : Callable
        Sampling function to allow for repeated sampling in the case of truncation-handling. 

    Returns
    -------
    torch.Tensor
        Bound values according to user specifications.  
    """
    if cfg.negative_sample_handling.lower() == 'clip':
        values = torch.relu(values)
    elif cfg.negative_sample_handling.lower() == 'truncate':
        values_smaller_zero = values < 0
        try_count = 0
        while any(values_smaller_zero.flatten()):
            values[values_smaller_zero] = sample_values(values_smaller_zero)
            values_smaller_zero = values < 0
            try_count += 1
            if try_count >= cfg.negative_sample_max_retries:
                break
    elif cfg.negative_sample_handling is None or cfg.negative_sample_handling.lower() == 'none':
        pass
    else:
        raise NotImplementedError(
            f"The option {cfg.negative_sample_handling} is not supported for handling negative samples!")

    return values


def _sample_gaussian_mixtures(ids: List[int], m_sub: torch.Tensor, s_sub: torch.Tensor,
                              p_sub: torch.Tensor) -> torch.Tensor:
    # unbound sampling:
    categorical = Categorical(p_sub)
    pis = categorical.sample().data
    mask_gmm = torch.zeros(p_sub.shape, dtype=torch.bool) \
        .to(device=p_sub.device) \
        .scatter_(2, pis.unsqueeze(2), True)

    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    values = s_sub \
        .data.new(s_sub[ids][mask_gmm[ids]].shape[0]) \
        .normal_() \
        .flatten() \
        .mul(s_sub[ids][mask_gmm[ids]]) \
        .add(m_sub[ids][mask_gmm[ids]])
    return values


def _sample_asymmetric_laplacians(ids: List[int], m_sub: torch.Tensor, b_sub: torch.Tensor,
                                  t_sub: torch.Tensor) -> torch.Tensor:
    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    prob = torch.FloatTensor(m_sub[ids].shape) \
        .uniform_(0, 1) \
        .to(m_sub.device)  # sample uniformly between zero and 1
    values = torch.where(
        prob < t_sub[ids],  # needs to be in accordance with the loss
        m_sub[ids] + ((b_sub[ids] * torch.log(prob / t_sub[ids])) / (1 - t_sub[ids])),
        m_sub[ids] - ((b_sub[ids] * torch.log((1 - prob) / (1 - t_sub[ids]))) / t_sub[ids]))
    return values.flatten()


class _SamplingSetup():

    def __init__(self, model: 'BaseModel', data: Dict[str, torch.Tensor], head: str):
        # make model checks:
        cfg = model.cfg
        if not cfg.head.lower() == head.lower():
            raise NotImplementedError(f"{head} sampling not supported for the {cfg.head} head!")

        dropout_modules = [model.dropout.p]

        # Multi-Timescale models don't have an embedding_net
        implied_statics_embedding, implied_dynamics_embedding = None, None
        if cfg.model.lower() not in ['mtslstm', 'odelstm']:
            implied_statics_embedding = model.embedding_net.statics_embedding_p_dropout
            implied_dynamics_embedding = model.embedding_net.dynamics_embedding_p_dropout
            dropout_modules += [implied_statics_embedding, implied_dynamics_embedding]
        # account for transformer
        implied_transformer_dropout = None
        if cfg.model.lower() == 'transfomer':
            implied_transformer_dropout = cfg.transformer_dropout
            dropout_modules.append(implied_transformer_dropout)

        max_implied_dropout = max(dropout_modules)
        # check lower bound dropout:
        if cfg.mc_dropout and max_implied_dropout <= 0.0:
            raise RuntimeError(f"""{cfg.model} with `mc_dropout` activated requires a dropout rate larger than 0.0
                               The current implied dropout-rates are:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")
        # check upper bound dropout:
        if cfg.mc_dropout and max_implied_dropout >= 1.0:
            raise RuntimeError(f"""The maximal dropout-rate is 1. Please check your dropout-settings:
                               The current implied dropout-rates are:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")

        # assign setup properties:
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.number_of_targets = len(cfg.target_variables)
        self.mc_dropout = cfg.mc_dropout
        self.predict_last_n = cfg.predict_last_n

        # determine appropriate frequency suffix:
        if self.cfg.use_frequencies:
            self.freq_suffixes = [f'_{freq}' for freq in cfg.use_frequencies]
        else:
            self.freq_suffixes = ['']

        self.batch_size_data = data[f'x_d{self.freq_suffixes[0]}'].shape[0]

    def _get_frequency_last_n(self, freq_suffix: str):
        if isinstance(self.predict_last_n, int):
            frequency_last_n = self.predict_last_n
        else:
            frequency_last_n = self.predict_last_n[freq_suffix[1:]]
        return frequency_last_n


def sample_mcd(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
    """MC-Dropout based point predictions sampling.

    Naive sampling. This function does `n_samples` forward passes for each sample in the batch. Currently it is 
    only useful for models with dropout, to perform MC-Dropout sampling. 
    Note: Calling this function will force the model to train mode (`model.train()`) and not set it back to its original
    state. 

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b) 
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument 
    'negative_sample_handling'.

    Parameters
    ----------
    model : BaseModel
        A model with a non-probabilistic head.
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
    setup = _SamplingSetup(model, data, model.cfg.head)

    # force model into train mode for mc_dropout:
    if setup.mc_dropout:
        model.train()

    # sample for different frequencies and targets:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        sample_points = []
        frequency_last_n = setup._get_frequency_last_n(freq_suffix=freq_suffix)

        for nth_target in range(setup.number_of_targets):
            # unbound sampling:
            def _sample_values(ids: List[int]) -> torch.Tensor:
                # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
                target_values = torch.zeros(len(ids), frequency_last_n, n_samples)
                for i in range(n_samples):  # forward-pass for each frequency separately to guarantee independence
                    prediction = model(data)
                    value_buffer = prediction[f'y_hat{freq_suffix}'][:, -frequency_last_n:, 0]
                    target_values[ids, -frequency_last_n:, i] = value_buffer.detach().cpu()
                return target_values

            ids = list(range(data[f'x_d{freq_suffix}'].shape[0]))
            values = _sample_values(ids)

            # bind values and add to sample_points:
            values = _handle_negative_values(setup.cfg, values, _sample_values)
            sample_points.append(values)

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: torch.stack(sample_points, 2)})

    return samples


def sample_gmm(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
    """Sample point predictions with the Gaussian Mixture (GMM) head.

    This function generates `n_samples` GMM sample points for each entry in the batch. Concretely, the model is 
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
    Good references for learning about GMMs are [#]_ and [#]_. 

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and 
     (b) 'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument 
     'negative_sample_handling'.
     
    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`) 
    and not set it back to its original state. 

    Parameters
    ----------
    model : BaseModel
        A model with a GMM head.
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
    setup = _SamplingSetup(model, data, "gmm")

    # force model into train mode if mc_dropout:
    if setup.mc_dropout:
        model.train()

    # make predictions:
    pred = model(data)

    # sample for different frequencies:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # get predict_last_n for the given the mode:
        frequency_last_n = setup._get_frequency_last_n(freq_suffix=freq_suffix)

        # initialize sample_points tensor for sampling:
        sample_points = torch.zeros((setup.batch_size_data, frequency_last_n, setup.number_of_targets, n_samples))
        sample_points *= torch.tensor(float('nan'))  # set initial sample_points to nan

        # GMM has 3 parts: means (m/mu), variances (s/sigma), and weights (p/pi):
        m, s, p = pred[f'mu{freq_suffix}'], \
                  pred[f'sigma{freq_suffix}'], \
                  pred[f'pi{freq_suffix}']

        for nth_target in range(setup.number_of_targets):
            m_target = _subset_target(m[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            s_target = _subset_target(s[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            p_target = _subset_target(p[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)

            mask_nan = ~torch.isnan(m_target[:, -1, 0])
            if any(mask_nan):  # skip if the complete mini-batch is invalid
                m_sub = torch.repeat_interleave(m_target[mask_nan, :, :], n_samples, dim=0)
                s_sub = torch.repeat_interleave(s_target[mask_nan, :, :], n_samples, dim=0)
                p_sub = torch.repeat_interleave(p_target[mask_nan, :, :], n_samples, dim=0)

                # sample values, handle negatives and add to sample points:
                values = _sample_gaussian_mixtures(np.ones(s_sub.shape, dtype=bool), m_sub, s_sub, p_sub)
                values = _handle_negative_values(
                    setup.cfg, values, sample_values=lambda ids: _sample_gaussian_mixtures(ids, m_sub, s_sub, p_sub))
                values = values.view(-1, frequency_last_n, n_samples)

                sample_points[mask_nan, :, nth_target, :] = values.detach().cpu()

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: sample_points})
    return samples


def sample_cmal(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
    """Sample point predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

    This function generates `n_samples` CMAL sample points for each entry in the batch. Concretely, the model is 
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
    General information about CMAL can be found in [#]_.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b) 
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument 
    'negative_sample_handling'.

    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`) 
    and not set it back to its original state. 

    Parameters
    ----------
    model : BaseModel
        A model with a CMAL head.
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
    setup = _SamplingSetup(model, data, "cmal")

    # force model into train mode if mc_dropout
    if setup.mc_dropout:
        model.train()

    # make predictions:
    pred = model(data)

    # sample for different frequencies:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # get predict_last_n for the given the mode:
        frequency_last_n = setup._get_frequency_last_n(freq_suffix=freq_suffix)

        # CMAL has 4 parts: means (m/mu), scales (b), asymmetries (t/) and weights (p/pi):
        m = pred[f'mu{freq_suffix}']
        b = pred[f'b{freq_suffix}']
        t = pred[f'tau{freq_suffix}']
        p = pred[f'pi{freq_suffix}']

        sample_points = []
        for nth_target in range(setup.number_of_targets):
            # sampling presets:
            m_target = _subset_target(m[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            b_target = _subset_target(b[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            t_target = _subset_target(t[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)
            p_target = _subset_target(p[:, -frequency_last_n:, :], nth_target, setup.cfg.n_distributions)

            m_target = torch.repeat_interleave(m_target, n_samples, dim=0)
            b_target = torch.repeat_interleave(b_target, n_samples, dim=0)
            t_target = torch.repeat_interleave(t_target, n_samples, dim=0)
            p_target = torch.repeat_interleave(p_target, n_samples, dim=0)

            # sampling procedure:
            values = torch.zeros((setup.batch_size_data * n_samples, frequency_last_n)).to(setup.device)
            values *= torch.tensor(float('nan'))  # set target sample_points to nan
            for nth_timestep in range(frequency_last_n):

                mask_nan = ~torch.isnan(p_target[:, nth_timestep, 0])
                if any(mask_nan):  # skip if the complete mini-batch is invalid
                    sub_choices = torch.multinomial(p_target[mask_nan, nth_timestep, :], num_samples=1)
                    t_sub = t_target[mask_nan, nth_timestep, :].gather(1, sub_choices)
                    m_sub = m_target[mask_nan, nth_timestep, :].gather(1, sub_choices)
                    b_sub = b_target[mask_nan, nth_timestep, :].gather(1, sub_choices)

                    ids = np.ones(b_sub.shape, dtype=bool)
                    values_unbound = _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)
                    values[mask_nan, nth_timestep] = _handle_negative_values(
                        setup.cfg,
                        values_unbound,
                        sample_values=lambda ids: _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub))

            # add the values to the sample_points:
            values = values.reshape(-1, frequency_last_n, n_samples)
            values = values.detach().cpu()
            sample_points.append(values)

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: torch.stack(sample_points, 2)})
    return samples


def sample_umal(model: 'BaseModel', data: Dict[str, torch.Tensor], n_samples: int) -> Dict[str, torch.Tensor]:
    """Sample point predictions with the Uncountable Mixture of Asymmetric Laplacians (UMAL) head.

    This function generates `n_samples` UMAL sample points for each entry in the batch. Concretely, the model is 
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures. 
    Details about the UMAL approach can be found in [#]_.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b) 
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument 
    'negative_sample_handling'.
    
    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`) 
    and not set it back to its original state. 

    Parameters
    ----------
    model : BaseModel
        A model with an UMAL head.
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
    setup = _SamplingSetup(model, data, "umal")

    # force model into train mode if mc_dropout:
    if setup.mc_dropout:
        model.train()

    # n_taus expands the batch by itself and adds a sampled tau as input (new_batch_size = n_taus*batch_size):
    data = umal_extend_batch(data, setup.cfg, n_taus=setup.cfg.n_taus)

    # make predictions:
    pred = model(data)

    # sample:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        # get predict_last_n for the given the mode:
        frequency_last_n = setup._get_frequency_last_n(freq_suffix=freq_suffix)

        # UMAL has 2 parts: means (m/mu), scales (b); the tau is randomly chosen:
        m = pred[f'mu{freq_suffix}']
        b = pred[f'b{freq_suffix}']
        t = data[f'tau{freq_suffix}']

        # sampling presets:
        m_wide = torch.cat(m[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)
        b_wide = torch.cat(b[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)

        # for now we just use a single tau for all targets:
        t_target = torch.cat(t[:, -frequency_last_n:, :].split(setup.batch_size_data, 0), 2)

        # sample over targets:
        sample_points = torch.zeros((setup.batch_size_data, frequency_last_n, setup.number_of_targets, n_samples))
        sample_points *= torch.tensor(float('nan'))  # set initial sample_points to nan
        for nth_target in range(setup.number_of_targets):
            # sampling presets:
            m_target = _subset_target(m_wide[:, -frequency_last_n:, :], nth_target, setup.cfg.n_taus)
            b_target = _subset_target(b_wide[:, -frequency_last_n:, :], nth_target, setup.cfg.n_taus)

            # sample over n_samples:
            for nth_sample in range(n_samples):
                sub_choice = np.random.randint(0, setup.cfg.n_taus)

                mask_nan = ~torch.isnan(m_target[:, 0, 0])
                if any(mask_nan):  # skip computation if entire mini-batch is invalid
                    m_sub = m_target[mask_nan, :, sub_choice]
                    b_sub = b_target[mask_nan, :, sub_choice]
                    t_sub = t_target[mask_nan, :, sub_choice]

                    ids = np.ones(b_sub.shape, dtype=bool)
                    values_unbound = _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)
                    values = _handle_negative_values(
                        setup.cfg,
                        values_unbound,
                        sample_values=lambda ids: _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub))

                    # add values to sample_points:
                    values = values.detach().cpu().unsqueeze(1)
                    sample_points[mask_nan, -frequency_last_n:, nth_target, nth_sample] = values

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: sample_points})
    return samples
