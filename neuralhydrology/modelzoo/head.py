import logging
from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


def get_head(cfg: Config, n_in: int, n_out: int) -> nn.Module:
    """Get specific head module, depending on the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.

    Returns
    -------
    nn.Module
        The model head, as specified in the run configuration.
    """
    if cfg.head.lower() == "regression":
        head = Regression(n_in=n_in, n_out=n_out, activation=cfg.output_activation)
    elif cfg.head.lower() == "gmm":
        head = GMM(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "umal":
        head = UMAL(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "cmal":
        head = CMAL(n_in=n_in, n_out=n_out)
    elif cfg.head.lower() == "":
        raise ValueError(f"No 'head' specified in the config but is required for {cfg.model}")
    else:
        raise NotImplementedError(f"{cfg.head} not implemented or not linked in `get_head()`")

    return head


class Regression(nn.Module):
    """Single-layer regression head with different output activations.
    
    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons.
    activation : str, optional
        Output activation function. Can be specified in the config using the `output_activation` argument. Supported
        are {'linear', 'relu', 'softplus'}. If not specified (or an unsupported activation function is specified), will
        default to 'linear' activation.
    """

    def __init__(self, n_in: int, n_out: int, activation: str = "linear"):
        super(Regression, self).__init__()

        # TODO: Add multi-layer support
        layers = [nn.Linear(n_in, n_out)]
        if activation != "linear":
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "softplus":
                layers.append(nn.Softplus())
            else:
                LOGGER.warning(f"## WARNING: Ignored output activation {activation} and used 'linear' instead.")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the Regression head.
        
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the model predictions in the 'y_hat' key.
        """
        return {'y_hat': self.net(x)}


class GMM(nn.Module):
    """Gaussian Mixture Density Network

    A mixture density network with Gaussian distribution as components. Good references are [#]_ and [#]_. The latter 
    one forms the basis for our implementation. As such, we also use two layers in the head to provide it with 
    additional flexibility, and exponential activation for the variance estimates and a softmax for weights.  

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 3 times the number of components.
    n_hidden : int
        Size of the hidden layer.
    
    References
    ----------
    .. [#] C. M. Bishop: Mixture density networks. 1994.
    .. [#] D. Ha: Mixture density networks with tensorflow. blog.otoro.net, 
           URL: http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow, 2015.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(GMM, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a GMM head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the GMM components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing mixture parameters and weights; where the key 'mu' stores the means, the key
            'sigma' the variances, and the key 'pi' the weights.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        # split output into mu, sigma and weights
        mu, sigma, pi = h.chunk(3, dim=-1)

        return {'mu': mu, 'sigma': torch.exp(sigma) + self._eps, 'pi': torch.softmax(pi, dim=-1)}


class CMAL(nn.Module):
    """Countable Mixture of Asymmetric Laplacians.

    An mixture density network with Laplace distributions as components.

    The CMAL-head uses an additional hidden layer to give it more expressiveness (same as the GMM-head).
    CMAL is better suited for many hydrological settings as it handles asymmetries with more ease. However, it is also
    more brittle than GMM and can more often throw exceptions. Details for CMAL can be found in [#]_.

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 4 times the number of components.
    n_hidden : int
        Size of the hidden layer.
        
    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing: 
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295, 2020.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(CMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        self._softplus = torch.nn.Softplus(2)
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a CMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the CMAL components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary, containing the mixture component parameters and weights; where the key 'mu'stores the means,
            the key 'b' the scale parameters, the key 'tau' the skewness parameters, and the key 'pi' the weights).
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent, t_latent, p_latent = h.chunk(4, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._softplus(b_latent) + self._eps  # scale > 0 (softplus was working good in tests)
        t = (1 - self._eps) * torch.sigmoid(t_latent) + self._eps  # 0 > tau > 1
        p = (1 - self._eps) * torch.softmax(p_latent, dim=-1) + self._eps  # sum(pi) = 1 & pi > 0

        return {'mu': m, 'b': b, 'tau': t, 'pi': p}


class UMAL(nn.Module):
    """Uncountable Mixture of Asymmetric Laplacians.

    An implicit approximation to the mixture density network with Laplace distributions which does not require to
    pre-specify the number of components. An additional hidden layer is used to provide the head more expressiveness.
    General details about UMAL can be found in [#]_. A major difference between their implementation 
    and ours is the binding-function for the scale-parameter (b). The scale needs to be lower-bound. The original UMAL 
    implementation uses an elu-based binding. In our experiment however, this produced under-confident predictions
    (too large variances). We therefore opted for a tailor-made binding-function that limits the scale from below and 
    above using a sigmoid. It is very likely that this needs to be adapted for non-normalized outputs.   

    Parameters
    ----------
    n_in : int
        Number of input neurons.
    n_out : int
        Number of output neurons. Corresponds to 2 times the output-size, since the scale parameters are also predicted.
    n_hidden : int
        Size of the hidden layer.

    References
    ----------
    .. [#] A. Brando, J. A. Rodriguez, J. Vitria, and A. R. Munoz: Modelling heterogeneous distributions 
        with an Uncountable Mixture of Asymmetric Laplacians. Advances in Neural Information Processing Systems, 
        pp. 8838-8848, 2019.
    """

    def __init__(self, n_in: int, n_out: int, n_hidden: int = 100):
        super(UMAL, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self._upper_bound_scale = 0.5  # this parameter found empirical by testing UMAL for a limited set of basins
        self._eps = 1e-5

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform a UMAL head forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous model part. It provides the basic latent variables to compute the UMAL components.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the means ('mu') and scale parameters ('b') to parametrize the asymmetric Laplacians.
        """
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)

        m_latent, b_latent = h.chunk(2, dim=-1)

        # enforce properties on component parameters and weights:
        m = m_latent  # no restrictions (depending on setting m>0 might be useful)
        b = self._upper_bound_scale * torch.sigmoid(b_latent) + self._eps  # bind scale from two sides.
        return {'mu': m, 'b': b}
