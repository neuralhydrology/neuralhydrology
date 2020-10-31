from typing import Dict

import torch

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config


class TemplateModel(BaseModel):
    """Initialize the model

    Each model constructor receives the config instance as its only input. From this, the entire model has to be
    created (with potential use of other modules, such as FC from fc.py). During training and inference, the forward
    method of this class will be called with the model inputs and has to return the predictions.

    Each Model inherits from the BaseModel, which implements some universal functionality. The basemodel also 
    defines the output_size, which can be used here as a given attribute (self.output_size).

    To be generally useable within this codebase, the output layer should not be implemented in this Module,
    but rather using the get_head() function from neuralhydrology.modelzoo.head.

    Parameters
    ----------
    cfg : Config
        Configuration of the run, read from the config file with some additional keys (such as number of basins).
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = []

    def __init__(self, cfg: Config):

        super(TemplateModel, self).__init__(cfg=cfg)

        ###########################
        # Create model parts here #
        ###########################

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model

                By convention, each forward pass has to accept a dict of input tensors. Usually, this dict contains
                'x_d' and, possibly, x_s and x_one_hot. If x_d and x_s are available at multiple frequencies,
                the keys 'x_d' and 'x_s' have frequency suffixes such as 'x_d_1H' for hourly data.
                Furthermore, by definition, each model has to return a dict containing the network predictions in
                'y_hat', potentially in addition to other dictionary keys. LSTM-based models should stick to the
                convention to return (at least) the following three tensors: y_hat, h_n, c_n (or, in the multi-
                frequency case, y_hat_1H, y_hat_1D, etc.).

                Parameters
                ----------
                data : Dict[str, torch.Tensor]
                     Dictionary with tensors
                        - x_d of shape [batch size, sequence length, features] containing the dynamic input data.
                        - x_s of shape [batch size, features] containing static input features. These are the
                            concatenation of what is defined in the config under static_attributes and evolving_attributes.
                            In case not a single camels attribute or static input feature is defined in the config,
                            x_s will not be present.
                        - x_one_hot of shape [batch size, number of basins] containing the one hot encoding of the
                            basins. In case 'use_basin_id_encoding' is set to False in the config, x_one_hot will
                            not be present.
                            
                        Note: If the input data are available at multiple frequencies (via use_frequencies), each input
                            tensor will have a suffix "_{freq}" indicating the tensor's frequency.

                Returns
                -------
                The network prediction has to be returned under the dictionary key 'y_hat' (or, if multiple frequencies
                are predicted, 'y_hat_{freq}'. Furthermore, make sure to return predictions for each time step, even if
                you want to train sequence-to-one. Which predictions are used for training the network is controlled in
                the train_epoch() function in neuralhydrology/training/basetrainer.py. Other return values should be the
                hidden states as 'h_n' and cell states 'c_n'. Further return values are possible.
                """
        ###############################
        # Implement forward pass here #
        ###############################
        pass
