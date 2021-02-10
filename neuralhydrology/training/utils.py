from typing import Dict

import torch

from neuralhydrology.utils.config import Config


def umal_extend_batch(data: Dict[str, torch.Tensor], cfg: Config, n_taus: int = 1, extend_y: bool = False) \
        -> Dict[str, torch.Tensor]:
    """This function extends the batch for the usage in UMAL (see: [#]_). 
    
    UMAl makes a MC approximation to a mixture integral by sampling random asymmetry parameters (tau). This can be 
    parallelized by expanding the batch for each tau.  
    
    Parameters
    ----------
    data : Dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    cfg : Config
        The run configuration.
    n_taus : int
        Number of taus to expand the batch.
    extend_y : bool
        Option to also extend the labels/y. 
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary, containing expanded input features and tau samples as key-value pairs.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """
    # setup:
    if cfg.use_frequencies:
        freq_suffixes = [f'_{freq}' for freq in cfg.use_frequencies]
    else:
        freq_suffixes = ['']

    for freq_suffix in freq_suffixes:
        batch_size, seq_length, input_size = data[f'x_d{freq_suffix}'].shape

        if isinstance(cfg.predict_last_n, int):
            predict_last_n = cfg.predict_last_n
        else:
            predict_last_n = cfg.predict_last_n[freq_suffix[1:]]

        # sample tau within [tau_down, tau_up] and add to data:
        tau = (cfg.tau_up - cfg.tau_down) * torch.rand(batch_size * n_taus, 1, 1) + cfg.tau_down
        tau = tau.repeat(1, seq_length, 1)  # in our convention tau remains the same over all inputs
        tau = tau.to(data[f'x_d{freq_suffix}'].device)
        data[f'tau{freq_suffix}'] = tau[:, -predict_last_n:, :]

        # extend dynamic inputs with tau and expand batch:
        x_d = data[f'x_d{freq_suffix}'].repeat(n_taus, 1, 1)
        data[f'x_d{freq_suffix}'] = torch.cat([x_d, tau], dim=-1)
        if f'x_s{freq_suffix}' in data:
            data[f'x_s{freq_suffix}'] = data[f'x_s{freq_suffix}'].repeat(n_taus, 1)
        if f'x_one_hot{freq_suffix}' in data:
            data[f'x_one_hot{freq_suffix}'] = data[f'x_one_hot{freq_suffix}'].repeat(n_taus, 1)
        if extend_y:
            data[f'y{freq_suffix}'] = data[f'y{freq_suffix}'].repeat(n_taus, 1, 1)

    return data
