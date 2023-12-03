from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
from neuralhydrology.utils.config import Config


class BaseConceptualModel(nn.Module):
    """Abstract base model class, don't use this class for model training.

    The purpose is to have some common operations that all conceptual models will need. Use subclasses of this class
    for training/evaluating different conceptual models, e.g. 'SHM'.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    """

    def __init__(self, cfg: Config):
        super(BaseConceptualModel, self).__init__()
        self.cfg = cfg
        # Check if the dynamic_conceptual_inputs and the target_variables are in the custom normalization. This is
        # necessary as conceptual models are mass conservative.
        if any(item not in cfg.custom_normalization for item in cfg.dynamic_conceptual_inputs + cfg.target_variables):
            raise RuntimeError("dynamic_conceptual_inputs and target_variables require custom_normalization")

    def forward(self, x_conceptual: torch.Tensor, lstm_out: torch.Tensor) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        raise NotImplementedError

    def _get_dynamic_parameters_conceptual(self, lstm_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Map the output of the data-driven part of the predefined ranges of the conceptual model that is being used.

        Parameters
        ----------
        lstm_out: torch.Tensor
            Tensor of size [batch_size, time_steps, n_param] that will be mapped to the predefined ranges of the
            conceptual model parameters to act as the dynamic parameterization.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dynamic parameterization of the conceptual model.
        """
        dynamic_parameters = {}
        for index, (parameter_name, parameter_range) in enumerate(self.parameter_ranges.items()):
            range_t = torch.tensor(parameter_range, dtype=torch.float32, device=lstm_out.device)
            range_t = range_t.repeat(lstm_out.shape[0], 1)  # To run all the elements of the batch in parallel
            dynamic_parameters[parameter_name] = range_t[:, :1] + torch.sigmoid(lstm_out[:, :, index]) * (range_t[:, 1:] - range_t[:, :1])

        return dynamic_parameters

    def _initialize_information(self, conceptual_inputs: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Initialize the structures to store the time evolution of the internal states and the outflow of the conceptual
        model

        Parameters
        ----------
        conceptual_inputs: torch.Tensor
            Inputs of the conceptual model (dynamic forcings)

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            - states: Dict[str, torch.Tensor]
                Dictionary to store the time evolution of the internal states (buckets) of the conceptual model
            - q_out: torch.Tensor
                Tensor to store the outputs of the conceptual model
        """

        states = {}
        # initialize dictionary to store the evolution of the states
        for name, value in self.initial_states.items():
            states[name] = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1]), dtype=torch.float32,
                                       device=conceptual_inputs.device)

        # initialize vectors to store the evolution of the outputs
        out = torch.zeros((conceptual_inputs.shape[0], conceptual_inputs.shape[1], len(self.cfg.target_variables)),
                          dtype=torch.float32, device=conceptual_inputs.device)

        return states, out


    @property
    def initial_states(self):
        raise NotImplementedError

    @property
    def parameter_ranges(self):
        raise NotImplementedError

