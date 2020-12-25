Modelzoo
========

The following section gives an overview of all implemented models. See `Implementing a new model`_ for details
on how to add your own model to the neuralHydrology package.

BaseModel
---------
Abstract base class from which all models derive. Do not use this class for model training.

CudaLSTM
--------
:py:class:`neuralhydrology.modelzoo.cudalstm.CudaLSTM` is a network using the standard PyTorch LSTM implementation.
All features (``x_d``, ``x_s``, ``x_one_hot``) are concatenated and passed to the network at each time step.
The initial forget gate bias can be defined in config.yml (``initial_forget_bias``) and will be set accordingly during
model initialization.

CustomLSTM
----------
:py:class:`neuralhydrology.modelzoo.customlstm.CustomLSTM` is a variant of the ``CudaLSTM`` and ``EmbCudaLSTM``
that returns all gate and state activations for all time steps. This class is mainly implemented for exploratory
reasons. You can use the method ``model.copy_weights()`` to copy the weights of a ``CudaLSTM`` or ``EmbCudaLSTM`` model
into a ``CustomLSTM`` model. This allows to use the fast CUDA implementations for training, and only use this class for
inference with more detailed outputs. You can however also use this model during training (``model: customlstm`` in the
config.yml) or as a starter for your own modifications to the LSTM cell. Note, however, that the runtime of this model
is considerably slower than its optimized counterparts.

EA-LSTM
-------
:py:class:`neuralhydrology.modelzoo.ealstm.EALSTM` is an implementation of the Entity-Aware LSTM, as introduced in
`Kratzert et al. "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets" <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`__.
The static features (``x_s`` and/or ``x_one_hot``) are used to compute the input gate activations, while the dynamic
inputs ``x_d`` are used in all other gates of the network.
The initial forget gate bias can be defined in config.yml (``initial_forget_bias``). If ``embedding_hiddens`` is passed, the input gate consists of the so-defined
FC network and not a single linear layer.

EmbCudaLSTM
-----------
:py:class:`neuralhydrology.modelzoo.embcudalstm.EmbCudaLSTM` is similar to `CudaLSTM`_,
with the only difference that static inputs (``x_s`` and/or ``x_one_hot``) are passed through an embedding network
(defined, for instance, by ``embedding_hiddens``) before being concatenated to the dynamic inputs ``x_d``
at each time step.

MTS-LSTM
--------
:py:class:`neuralhydrology.modelzoo.mtslstm.MTSLSTM` is a newly proposed model by `Gauch et al. "Rainfall--Runoff Prediction at Multiple Timescales with a Single Long Short-Term Memory Network" <https://arxiv.org/abs/2010.07921>`__.
This model allows the training on more than temporal resolution (e.g., daily and hourly inputs) and
returns multi-timescale model predictions accordingly. A more detailed tutorial will follow shortly.

ODE-LSTM
--------
:py:class:`neuralhydrology.modelzoo.odelstm.ODELSTM` is a PyTorch implementation of the ODE-LSTM proposed by
`Lechner and Hasani <https://arxiv.org/abs/2006.04418>`_. This model can be used with unevenly sampled inputs and can
be queried to return predictions for any arbitrary time step.

Transformer
-----------
:py:class:`neuralhydrology.modelzoo.transformer.Transformer` is the encoding portion of a standard transformer network with self-attention. 
This uses the standard PyTorch TransformerEncoder implementation. All features (``x_d``, ``x_s``, ``x_one_hot``) are concatenated and passed 
to the network at each time step. Instead of a decoder, this model uses a standard head (e.g., linear). 
The model requires the following hyperparamters specified in the config file: 
``transformer_embedding_dimension`` : int representing the dimension of the input embedding space. 
This must be dividible by the number of self-attention heads (transformer_nheads).
``transformer_positional_encoding_type``: choices to "add" or "concatenate" positional encoding to other model inputs.
``transformer_positional_dropout``: fraction of dropout applied to the positional encoding.
``transformer_nheads``: number of self-attention heads.
``transformer_dim_feedforward``: dimension of the feed-fowrard networks between self-attention heads.
``transformer_dropout``: dropout in the feedforward networks between self-attention heads.
``transformer_nlayers``: number of stacked self-attention + feedforward layers.


Implementing a new model
------------------------
The listing below shows the skeleton of a template model you can use to start implementing your own model.
Once you have implemented your model, make sure to modify :py:func:`neuralhydrology.modelzoo.__init__.get_model`.
Furthermore, make sure to select a *unique* model abbreviation that will be used to specify the model in the config.yml
files.

.. code-block:: python

    from typing import Dict

    import torch

    from neuralhydrology.modelzoo.basemodel import BaseModel


    class TemplateModel(BaseModel):

        def __init__(self, cfg: dict):
            """Initialize the model

            Each model receives as only input the config dictionary. From this, the entire model has to be implemented in
            this class (with potential use of other modules, such as FC from fc.py). So this class will get the model inputs
            and has to return the predictions.

            Each Model inherits from the BaseModel, which implements some universal functionality. The basemodel also
            defines the output_size, which can be used here as a given attribute (self.output_size)

            Parameters
            ----------
            cfg : dict
                Configuration of the run, read from the config file with some additional keys (such as number of basins).
            """
            super(TemplateModel, self).__init__(cfg=cfg)

            ###########################
            # Create model parts here #
            ###########################

        def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """Forward pass through the model

            By convention, each forward pass has to accept a dict of input tensors. Usually, this dict contains 'x_d' and,
            possibly, x_s and x_one_hot. If x_d and x_s are available at multiple frequencies, the keys 'x_d' and 'x_s'
            have frequency suffixes such as 'x_d_1H' for hourly data.
            Furthermore, by definition, each model has to return a dict containing the network predictions in 'y_hat',
            potentially in addition to other keys. LSTM-based models should stick to the convention to return (at least)
            the following three tensors: y_hat, h_n, c_n (or, in the multi-frequency case, y_hat_1H, y_hat_1D, etc.).

            Parameters
            ----------
            data : Dict[str, torch.Tensor]
                 Dictionary with tensors
                    - x_d of shape [batch size, sequence length, features] containing the dynamic input data.
                    - x_s of shape [batch size, features] containing static input features. These are the concatenation
                        of what is defined in the config under static_attributes and evolving_attributes. In case not a single
                        camels attribute or static input feature is defined in the config, x_s will not be present.
                    - x_one_hot of shape [batch size, number of basins] containing the one hot encoding of the basins.
                        In case 'use_basin_id_encoding' is set to False in the config, x_one_hot will not be present.
                    Note: If the input data are available at multiple frequencies (via use_frequencies), each input tensor
                        will have a suffix "_{freq}" indicating the tensor's frequency.

            Returns
            -------
            The network prediction has to be returned under the dictionary key 'y_hat' (or, if multiple frequencies are
            predicted, 'y_hat_{freq}'. Furthermore, make sure to return predictions for each time step, even if you want
            to train sequence-to-one. Which predictions are used for training the network is controlled in the train_epoch()
            function in neuralhydrology/training/basetrainer.py. Other return values should be the hidden states as 'h_n' and cell
            states 'c_n'. Further return values are possible.
            """
            ###############################
            # Implement forward pass here #
            ###############################
            pass
