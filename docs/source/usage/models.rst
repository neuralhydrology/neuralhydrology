Modelzoo
========

The following section gives an overview of all implemented models in NeuralHydrology. Conceptually, all models in our package consist of two parts, the model class (which constitutes the core of the model as such) and the model heads (which relate the outputs of the model class to the predicted variables). The section `Model Heads`_ provides a list of all implemented model heads, and the section `Model Classes`_ a list of all model classes. If you want to implement your own model within the package you best start at the section `Implementing a new model`_, which provides the necessary details to do so. 


Model Heads
-----------
The head of the model is used on top of the model class and relates the outputs of the `Model Classes`_ to the predicted variable. Currently four model heads are available: `Regression`_, `GMM`_, `CMAL`_ and `UMAL`_. The latter three heads provide options for probabilistic modelling. A detailed overview can be found in `Klotz et al. "Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling" <https://arxiv.org/abs/2012.14295>`__. 

Regression
^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.head.Regression` provides a single layer *regression* head, that includes different activation options for the output. (namely a linear, relu and softplus). 

It is possible to obtain probabilistic predictions with the regression head by using Monte-Carlo Dropout. Its usage is defined in the config.yml by setting ``mc_dropout``. The sampling behavior is governed by picking the number of samples (``n_samples``) and the approach for handling negative samples (``negative_sample_handling``).   

GMM
^^^
:py:class:`neuralhydrology.modelzoo.head.GMM` implements a *Gaussian Mixture Model* head. That is, a mixture density network with Gaussian distributions as components. Each Gaussian component is defined by two parameters (the mean, the variance) and by a set of weights. The current implementation of the GMM head uses two layers. Specific output activations are used for the variances (:py:func:`torch.exp`) and the weights (:py:func:`torch.softmax`).

The number of components can be set in the config.yml using ``n_distributions``. Additionally, the sampling behavior (for the inference) is defined with config.yml by setting the number of samples (``n_samples``), and the approach for handling negative samples (``negative_sample_handling``).  

CMAL
^^^^
:py:class:`neuralhydrology.modelzoo.head.CMAL` implements a *Countable Mixture of Asymmetric Laplacians* head. That is, a mixture density network with asymmetric Laplace distributions as components. The name is a homage to `UMAL`_, which provides an uncountable extension. The CMAL components are defined by three parameters (location, scale, and asymmetry) and linked by a set of weights. The current implementation of the CMAL head uses two layers. Specific output activations are used for the component scales (:py:class:`torch.nn.Softplus(2)`), the asymmetries (:py:func:`torch.sigmoid`), and the weights (:py:func:`torch.softmax`). In our preliminary experiments this heuristic achieved better results. 

The number of components can be set in the config.yml using ``n_distributions``. Additionally, one can sample from CMAL. The behavior of which is defined by setting the number of samples (``n_samples``), and the approach for handling negative samples (``negative_sample_handling``).  

UMAL
^^^^
:py:class:`neuralhydrology.modelzoo.head.UMAL` implements an *Uncountable Mixture of Asymmetric Laplacians* head. That is, a mixture density network that uses an uncountable amount of asymmetric Laplace distributions as components. The *uncountable property* is achieved by implicitly learning the conditional density and approximating it, when needed, with a Monte-Carlo integration, using sampled asymmetry parameters. The UMAL components are defined by two parameters (the location and the scale) and linked by a set of weights. The current implementation uses two hidden layers. The output activation for the scale has some major differences to the original implementation, since it is upper bounded (using :py:func:`0.5*torch.sigmoid`).

During inference the number of components and weights used for the Monte-Carlo approximation are defined in the config.yml by ``n_taus``. The additional argument ``umal_extend_batch`` allows to explicitly account for this integration step during training by repeatedly sampling the asymmetry parameter and extending the batch by ``n_taus``. Furthermore, depending on the used output activation the sampling of the asymmetry parameters can yield unwarranted model behavior. Therefore the lower- and upper-bounds of the sampling can be adjusted using the ``tau_down`` and ``tau_up`` options in the config yml. 
The sampling for UMAL is defined by choosing the number of samples (``n_samples``), and the approach for handling negative samples (``negative_sample_handling``).  


Model Classes
-------------

BaseModel
^^^^^^^^^
Abstract base class from which all models derive. Do not use this class for model training.

ARLSTM
^^^^^^
:py:class:`neuralhydrology.modelzoo.arlstm.ARLSTM` is an autoregressive long short term memory network (LSTM)
that assumes one input is a time-lagged version of the output. All features (``x_d``, ``x_s``, ``x_one_hot``) 
are concatenated and passed to the timeseries network at each time step, along with a binary flag that indicates 
whether the autoregressive input (i.e., lagged target data) is missing (False) or present (True). The length of
the autoregressive lag can be specified in the config file by specifying the lag on the autoregressive input.
Any missing data in the autoregressive inputs is imputed with appropriately lagged model output, and gradients are 
calculated through this imputation during backpropagation. Only one autoregressive input is currently supported, 
and it is assumed that this is the last variable in the ``x_d`` vector. This model uses a standard pytorch LSTM 
cell, but only runs the optimized LSTM one timestep at a time, and is therefore significantly slower than the CudaLSTM.  

CudaLSTM
^^^^^^^^
:py:class:`neuralhydrology.modelzoo.cudalstm.CudaLSTM` is a network using the standard PyTorch LSTM implementation.
All features (``x_d``, ``x_s``, ``x_one_hot``) are concatenated and passed to the network at each time step.
If ``statics/dynamics_embedding`` are used, the static/dynamic inputs will be passed through embedding networks before
being concatenated.
The initial forget gate bias can be defined in config.yml (``initial_forget_bias``) and will be set accordingly during
model initialization.

CustomLSTM
^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.customlstm.CustomLSTM` is a variant of the ``CudaLSTM``
that returns all gate and state activations for all time steps. This class is mainly implemented for exploratory
reasons. You can use the method ``model.copy_weights()`` to copy the weights of a ``CudaLSTM`` model
into a ``CustomLSTM`` model. This allows to use the fast CUDA implementations for training, and only use this class for
inference with more detailed outputs. You can however also use this model during training (``model: customlstm`` in the
config.yml) or as a starter for your own modifications to the LSTM cell. Note, however, that the runtime of this model
is considerably slower than its optimized counterparts.

EA-LSTM
^^^^^^^
:py:class:`neuralhydrology.modelzoo.ealstm.EALSTM` is an implementation of the Entity-Aware LSTM, as introduced in
`Kratzert et al. "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets" <https://hess.copernicus.org/articles/23/5089/2019/hess-23-5089-2019.html>`__.
The static features (``x_s`` and/or ``x_one_hot``) are used to compute the input gate activations, while the dynamic
inputs ``x_d`` are used in all other gates of the network.
The initial forget gate bias can be defined in config.yml (``initial_forget_bias``).
If ``statics/dynamics_embedding`` are used, the static/dynamic inputs will first be passed through embedding networks.
The output of the static embedding network will then be passed through the input gate, which consists of a single linear
layer.

EmbCudaLSTM
^^^^^^^^^^^
.. deprecated:: 0.9.11-beta
   Use `CudaLSTM`_ with ``statics_embedding``.

:py:class:`neuralhydrology.modelzoo.embcudalstm.EmbCudaLSTM` is similar to `CudaLSTM`_,
with the only difference that static inputs (``x_s`` and/or ``x_one_hot``) are passed through an embedding network
before being concatenated to the dynamic inputs ``x_d`` at each time step.

GRU
^^^
:py:class:`neuralhydrology.modelzoo.gru.GRU` is a network using the standard PyTorch GRU implementation.
All features (``x_d``, ``x_s``, ``x_one_hot``) are concatenated and passed to the network at each time step.
If ``statics/dynamics_embedding`` are used, the static/dynamic inputs will be passed through embedding networks before
being concatenated.

.. _MC-LSTM:

Mamba
^^^^^
:py:class:`neuralhydrology.modelzoo.mamba.Mamba` is a state space model SSM using the PyTorch implementation
https://github.com/state-spaces/mamba/tree/main from `Gu and Dao (2023) <https://arxiv.org/abs/2312.00752>`_.

There are two required dependencies for Mamba: ``mamba_ssm`` and ``causal-conv1d``, which are the mamba ssm layer and
implementation of a simple causal Conv1d layer used inside the Mamba block, respectively.

There are three hyperparameters which can be set in the config file:
- ``mamba_d_conv``: Local convolution width (Default is set to 4)
- ``mamba_d_state``: SSM state expansion factor (Default is set to 16)
- ``mamba_expand``: Block expansion factor (Default is set to 2)

MC-LSTM
^^^^^^^
:py:class:`neuralhydrology.modelzoo.mclstm.MCLSTM` is a concept for a mass-conserving model architecture inspired by the
LSTM that was recently proposed by `Hoedt et al. (2021) <https://arxiv.org/abs/2101.05186>`_. The implementation included
in this library is the exact model configuration that was used for the hydrology experiments in the linked publication 
(for details, see Appendix B.4.2).
The inputs for the model are split into two groups: i) the mass input, whose values are stored in the memory cells of 
the model and from which the target is calculated and ii) auxiliary inputs, which are used to control the gates 
within the model. In this implementation, only a single mass input per timestep (e.g. precipitation) is allowed, which
has to be specified with the config argument ``mass_inputs``. Make sure to exclude the mass input feature, as well as
the target variable from the standard feature normalization. This can be done using the ``custom_normalization`` config argument
and by setting the ``centering`` and ``scaling`` key to ``None``. For example, if the mass input is named "precipitation"
and the target feature is named "discharge", this would look like this:

.. code-block:: yaml

    custom_normalization:
        precipitation:
            centering: None
            scaling: None
        discharge:
            centering: None
            scaling: None

All inputs specified by the ``dynamic_inputs`` config argument are used as auxiliary inputs, as are (possibly embedded)
static inputs (e.g. catchment attributes).
The config argument ``head`` is ignored for this model and the model prediction is always computed as the sum over the 
outgoing mass (excluding the trash cell output).

MTS-LSTM
^^^^^^^^
:py:class:`neuralhydrology.modelzoo.mtslstm.MTSLSTM` is a newly proposed model by `Gauch et al. "Rainfall--Runoff Prediction at Multiple Timescales with a Single Long Short-Term Memory Network" <https://arxiv.org/abs/2010.07921>`__.
This model allows the training on more than temporal resolution (e.g., daily and hourly inputs) and
returns multi-timescale model predictions accordingly. A more detailed tutorial will follow shortly.

ODE-LSTM
^^^^^^^^
:py:class:`neuralhydrology.modelzoo.odelstm.ODELSTM` is a PyTorch implementation of the ODE-LSTM proposed by
`Lechner and Hasani <https://arxiv.org/abs/2006.04418>`_. This model can be used with unevenly sampled inputs and can
be queried to return predictions for any arbitrary time step.

Transformer
^^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.transformer.Transformer` is the encoding portion of a standard transformer network with self-attention. 
This uses the standard PyTorch TransformerEncoder implementation. All features (``x_d``, ``x_s``, ``x_one_hot``) are concatenated and passed 
to the network at each time step. Unless the number of inputs is divisible by the number of transformer heads (``transformer_nheads``), it is
necessary to use an embedding network that guarantees this. To achieve this, use ``statics/dynamics_embedding``, so the static/dynamic
inputs will be passed through embedding networks before being concatenated. The embedding network will then map the static and dynamic features
to size ``statics/dynamics_embedding['hiddens'][-1]``, so the total embedding size will be the sum of these values.
Instead of a decoder, this model uses a standard head (e.g., linear). 
The model requires the following hyperparameters specified in the config file: 

* ``transformer_positional_encoding_type``: choices to "sum" or "concatenate" positional encoding to other model inputs.
* ``transformer_positional_dropout``: fraction of dropout applied to the positional encoding.
* ``transformer_nheads``: number of self-attention heads.
* ``transformer_dim_feedforward``: dimension of the feedforward networks between self-attention heads.
* ``transformer_dropout``: dropout in the feedforward networks between self-attention heads.
* ``transformer_nlayers``: number of stacked self-attention + feedforward layers.

Handoff-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.handoff_forecast_lstm.HandoffForecastLSTM` is a forecasting model that uses a state-handoff to transition 
from a hindcast sequence model to a forecast sequence (LSTM) model. The hindcast model is run from the past up to present (the issue time of the forecast) 
and then passes the cell state and hidden state of the LSTM into a (nonlinear) handoff network, which is then used to initialize the cell state and 
hidden state of a new LSTM that rolls out over the forecast period. The handoff network is implemented as a custom FC layer, which can have multiple layers.
The handoff network is implemented using the ``state_handoff_network`` config parameter.
The hindcast and forecast LSTMs have different weights and biases, different heads, and can have different embedding networks. The hidden size of the 
hindcast LSTM is set using the ``hindcast_hidden_size`` config parameter and the hidden size of the forecast LSTM is set using the ``forecast_hidden_size`` 
config parameter, which both default to ``hidden_size`` if not set explicitly.

The handoff forecast LSTM model can implement a delayed handoff as well, such that the handoff between the hindcast and forecast LSTM occurs prior to the 
forecast issue time. This is controlled by the ``forecast_overlap`` parameter in the config file, and the forecast and hindcast LSTMs will run concurrently 
for the number of timesteps indicated by ``forecast_overlap``. We recommend using the ``ForecastOverlapMSERegularization`` regularization option to regularize 
the loss function by (dis)agreement between the overlapping portion of the hindcast and forecast LSTMs. This regularization term can be requested by setting 
the ``regularization`` parameter in the config file to include ``forecast_overlap``.

Multihead-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.multihead_forecast_lstm.MultiheadForecastLSTM` is a forecasting model that runs a sequential (LSTM) model up to the forecast 
issue time, and then directly predicts a sequence of forecast timesteps without using a recurrent rollout. Prediction is done with a custom ``FC`` (fully connected) 
layer, which can have multiple layers. Do not use this model with ``forecast_overlap`` > 0.


Sequential-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.sequential_forecast_lstm.SequentialForecastLSTM` is a forecasting model that uses a single sequential (LSTM) model that rolls 
out through both the hindcast and forecast sequences. The difference between this and a standard `CudaLSTM`_ is (1) this model uses both hindcast and forecast 
input features, and (2) it uses a separate embedding network for the hindcast period and the forecast period. Do not use this model with ``forecast_overlap`` > 0.

Stacked-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^
:py:class:`neuralhydrology.modelzoo.stacked_forecast_lstm.StackedForecastLSTM` is a forecasting model that uses two stacked sequential (LSTM) models to handle 
hindcast vs. forecast. The hindcast and forecast sequences must be the same length, and the ``forecast_overlap`` config parameter must be set to the correct overlap
between these two sequences. For example, if we want to use a hindcast sequence length of 365 days to make a 7-day forecast, then ``seq_length`` and 
``forecast_seq_length`` must both be set to 365, and ``forecast_overlap`` must be set to 358 (=365-7). Outputs from the hindcast LSTM are concatenated to the input 
sequences to the forecast LSTM. This causes a lag of length (``seq_length`` - ``forecast_overlap``) timesteps between the latest hindcast data and the newest 
forecast point, meaning that forecasts do not get information from the most recent dynamic inputs. To solve this, set the ``bidirectional_stacked_forecast_lstm``
config parameter to True, so that the hindcast LSTM runs bidirectional and therefore all outputs from the hindcast LSTM receive information from the most recent 
dynamic input data, however be aware that this can potentially result in hairy forecasts.


Implementing a new model
^^^^^^^^^^^^^^^^^^^^^^^^
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
