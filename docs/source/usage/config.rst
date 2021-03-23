Configuration Arguments
=======================

This page provides a list of possible configuration arguments.
Check out the file `examples/config.yml.example <https://github.com/neuralhydrology/neuralhydrology/blob/master/examples/config.yml.example>`__ for an example of how a config file could look like.

General experiment configurations
---------------------------------

-  ``experiment_name``: Defines the name of your experiment that will be
   used as a folder name (+ date-time string), as well as the name in
   TensorBoard.

-  ``run_dir``: Full or relative path to where the run directory is
   stored (if empty runs are stored in ${current\_working\_dir}/runs/)

-  ``train_basin_file``: Full or relative path to a text file containing
   the training basins (use data set basin id, one id per line).
-  ``validation_basin_file``: Full or relative path to a text file
   containing the validation basins (use data set basin id, one id per
   line).
-  ``test_basin_file``: Full or relative path to a text file containing
   the training basins (use data set basin id, one id per line).

-  ``train_start_date``: Start date of the training period (first day of
   discharge) in the format ``DD/MM/YYYY``. Can also be a list of dates
   to specify multiple training periods. If a list is specified, ``train_end_date``
   must also be a list of equal length. Corresponding pairs of start and
   end date denote the different periods.
-  ``train_end_date``: End date of the training period (last day of
   discharge) in the format ``DD/MM/YYYY``. Can also be a list of dates.
   If a list is specified, also ``train_start_date`` must be a list with
   an equal length.
-  ``validation_start_date``: Start date of the validation period (first
   day of discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``validation_end_date``: End date of the validation period (last day
   of discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``test_start_date``: Start date of the test period (first day of
   discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``test_end_date``: End date of the validation period (last day of
   discharge) in the format ``DD/MM/YYYY``. Can also be 
   a list of dates (similar to train period specifications).
-  ``per_basin_train_periods_file``: Alternatively to specifying a global
   train period for all basins (using ``train_start_date`` and ``train_end_date``)
   it is also possible to use individual periods for each basin. For that to work
   you have to create a dictionary with one key per basin id. For each basin id,
   the dictionary contains two keys ``start_dates`` and ``end_dates``, which
   contain a list of pandas TimeStamps that indicate the start and end dates
   of the train periods. ``start_dates`` and ``end_dates`` have to be a list,
   even in case of a single period per basin (see example below). Then use the
   pickle library, to store this dictionary to disk and use the path to this
   pickle file as the value for this config argument.

.. code-block::

   import pandas as pd

   dates = {
        'basin_a': {
            'start_dates': [pd.to_datetime('01/01/1980')],
            'end_dates': [pd.to_datetime('31/12/1999')]
        },
        'basin_b': {
            'start_dates': [pd.to_datetime('01/01/1980'), pd.to_datetime('01/01/2000')],
            'end_dates': [pd.to_datetime('31/12/1990'), pd.to_datetime('01/01/2005')]
        }
    }

-  ``per_basin_validation_periods_file``: Same as ``per_basin_train_periods_file``
   but indicating individual periods that are used as validation periods.
-  ``per_basin_test_periods_file``: Same as ``per_basin_train_periods_file``
   but indicating individual periods that are used as test periods.

-  ``seed``: Fixed random seed. If empty, a random seed is generated for
   this run.

-  ``device``: Which device to use in format of ``cuda:0``, ``cuda:1``,
   etc, for GPUs or ``cpu``

Validation settings
-------------------

-  ``validate_every``: Integer that specifies in which interval a
   validation is performed. If empty, no validation is done during
   training.

-  ``validate_n_random_basins``: Integer that specifies how many random
   basins to use per validation. Values larger *n_basins* are clipped
   to *n_basins*.

-  ``metrics``: List of metrics to calculate during validation/testing.
   See
   :py:mod:`neuralhydrology.evaluation.metrics`
   for a list of available metrics.

-  ``save_validation_results``: True/False, if True, stores the
   validation results to disk as a pickle file. Otherwise they are only
   used for TensorBoard

General model configuration
---------------------------

-  ``model``: Defines the model class, i.e. the core of the model, that will be used. Names
   have to match the values in `this
   function <https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/__init__.py#L17>`__,
   e.g., [``cudalstm``, ``ealstm``, ``mtslstm``]

-  ``head``: The prediction head that is used on top of the output of
   the model class. Currently supported are ``regression``, ``gmm``, ``cmal``, and ``umal``.
   Make sure to pass the necessary options depending on your
   choice of the head (see below).

-  ``hidden_size``: Hidden size of the model class. In the case of an
   LSTM, this reflects the number of LSTM states.

-  ``initial_forget_bias``: Initial value of the forget gate bias.

-  ``output_dropout``: Dropout applied to the output of the LSTM

Regression head
~~~~~~~~~~~~~~~
Can be ignored if ``head != 'regression'``

-  ``output_activation``: Which activation to use on the output
   neuron(s) of the linear layer. Currently supported are ``linear``,
   ``relu``, ``softplus``. If empty, ``linear`` is used.
-  ``mc_dropout``: True/False. Wheter Monte-Carlo dropout is used to 
   sample during inference. 
   
GMM head
~~~~~~~~
Can be ignored if ``head != 'gmm'``

-  ``n_distributions``: The number of distributions used for the GMM head. 
-  ``n_samples``: Number of samples generated  (per time-step) from GMM. 
-  ``negative_sample_handling``: How to account for negative samples. 
   Possible values are ``none`` for doing nothing, ``clip`` for clipping 
   the values at zero, and ``truncate`` for resampling values that
   were drawn below zero. If the last option is chosen, the additional 
   argument ``negative_sample_max_retries`` controls how often the values 
   are resampled. 
-  ``negative_sample_max_retries``: The number of repeated samples for the 
   ``truncate`` option of the ``negative_sample_max_retries`` argument.
-  ``mc_dropout``: True/False. Whether Monte-Carlo dropout is used to 
   sample during inference. 

CMAL head
~~~~~~~~~
Can be ignored if ``head != 'cmal'``

-  ``n_distributions``: The number of distributions used for the CMAL head. 
-  ``n_samples``: Number of samples generated  (per time-step) from CMAL. 
-  ``negative_sample_handling``: Approach for handling negative sampling. 
   Possible values are ``none`` for doing nothing, ``clip`` for clipping 
   the values at zero, and ``truncate`` for resampling values that
   were drawn below zero. If the last option is chosen, the additional 
   argument ``negative_sample_max_retries`` controls how often the values 
   are resampled. 
-  ``negative_sample_max_retries``: The number of repeated samples for the 
   ``truncate`` option of the ``negative_sample_max_retries`` argument.
-  ``mc_dropout``: True/False. Whether Monte-Carlo dropout is used to 
   sample during inference.    


UMAL head
~~~~~~~~~
Can be ignored if ``head != 'umal'``

-  ``n_taus``: The number of taus sampled to approximate the 
   uncountable distributions.
-  ``umal_extend_batch``: True/False. Whether the batches should be 
   extended ``n_taus`` times, to account for a specific approximation 
   density already during the training.
-  ``tau_down`` The lower sampling bound of asymmetry parameter (should be 
   above 0, below 1 and smaller than ``tau_up``).
-  ``tau_up`` The upper sampling bound of asymmetry parameter (should be 
   above 0, below 1 and larger than ``tau_down``).   
-  ``n_samples``: Number of samples generated  (per time-step) from UMAL. 
-  ``negative_sample_handling``: Approach for handling negative sampling. 
   Possible values are ``none`` for doing nothing, ``clip`` for clipping 
   the values at zero, and ``truncate`` for resampling values that
   were drawn below zero. If the last option is chosen, the additional 
   argument ``negative_sample_max_retries`` controls how often the values 
   are resampled. 
-  ``negative_sample_max_retries``: The number of repeated samples for the 
   ``truncate`` option of the ``negative_sample_max_retries`` argument.
-  ``mc_dropout``: True/False. Whether Monte-Carlo dropout is used to 
   sample during inference. 

Multi-timescale training settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are used if ``model == mtslstm``.

-  ``transfer_mtslstm_states``: Specifies if and how hidden and cell
   states are transferred from lower to higher frequencies. This
   configuration should be a dictionary with keys ``h`` (hidden state)
   and ``c`` (cell state). Possible values are
   ``[None, linear, identity]``. If ``transfer_mtslstm_states`` is not
   provided or empty, the default is linear transfer.

-  ``shared_mtslstm``: If False, will use a distinct LSTM with
   individual weights for each timescale. If True, will use a single
   LSTM for all timescales and use one-hot-encoding to identify the
   current input timescale. In both cases, ``transfer_mtslstm_states``
   can be used to configure hidden and cell state transfer.

Transformer settings
~~~~~~~~~~~~~~~~~~~~

These are used if ``model == transformer``.

-  ``transformer_nlayers``: Number of multi-head self-attention layers in the 
   transformer encoder.
-  ``transformer_positional_encoding_type``: Choices are ``[sum, concatenate]``.
   Used to change the way that the positional encoding is used in transformer
   embedding layer. `sum` means that the positional encoding is added to the values
   of the inputs for that layer, while `concatenate` means that the embedding is concatenated
   as additional input features.
-  ``transformer_dim_feedforward``: Dimension of dense layers used between
   self-attention layers in transformer encoder.
-  ``transformer_positional_dropout``: Dropout applied only to the positional
   encoding before using in transformer encoder.
-  ``transformer_dropout``: Dropout used in transformer encoder layers.
-  ``transformer_nhead``: Number of parallel transformer heads.

ODE-LSTM settings
~~~~~~~~~~~~~~~~~

These are used if ``model == odelstm``.

-  ``ode_method``: Method to use to solve the ODE. One of
   ``[euler, rk4, heun]``.

-  ``ode_num_unfolds``: Number of iterations to break each ODE solving
   step into.

-  ``ode_random_freq_lower_bound``: Lowest frequency that will be used
   to randomly aggregate the first slice of the input sequence. See the
   documentation of the ODELSTM class for more details on the frequency
   randomization.

MC-LSTM settings
~~~~~~~~~~~~~~~~

These are used if ``model == mclstm``.

-  ``mass_inputs``: List of features that are used as mass input in the MC-LSTM model, i.e. whose quantity is conserved
   over time. Currently, the MC-LSTM configuration implemented here only supports a single mass input. Make sure to
   exclude this feature from the default normalization (see :ref:`MC-LSTM <MC-LSTM>` description).

Embedding network settings
--------------------------

These settings define fully connected networks that are used in various places, such as the embedding network
for static or dynamic features in the single-frequency models or as an optional extended input gate network in
the EA-LSTM model. For multi-timescale models, these settings can be ignored.

- ``statics_embedding``: None (default) or a dict that defines the embedding network for static inputs.
   The dictionary can have the following keys:

   - ``type`` (default 'fc'): Type of the embedding net. Currently, only 'fc' for fully-connected net is supported.
   - ``hiddens``: List of integers that define the number of neurons per layer in the fully connected network.
     The last number is the number of output neurons. Must have at least length one.
   - ``activation`` (default 'tanh'): activation function of the network. Supported values are 'tanh', 'sigmoid', 'linear'.
     The activation function is not applied to the output neurons, which always have a linear activation function.
     An activation function for the output neurons has to be applied in the main model class.
   - ``dropout`` (default 0.0): Dropout rate applied to the embedding network.

  Note that for EA-LSTM, there will always be an additional linear layer that maps to the EA-LSTM's hidden size. This
  means that the the embedding layer output size does not have to be equal to ``hidden_size``.

- ``dynamics_embedding``: None (default) or a dict that defines the embedding network for dynamic inputs. See ``statics_embedding``
  for a description of the dictionary structure.

Training settings
-----------------

-  ``optimizer``: Specify which optimizer to use. Currently supported
   is Adam (standard). New optimizers can be added
   :py:func:`here <neuralhydrology.training.get_optimizer>`.

-  ``loss``: Which loss to use. Currently supported are ``MSE``,
   ``NSE``, ``RMSE``, ``GMMLoss``, ``CMALLoss``, and ``UMALLoss``. New 
   losses can be added :py:mod:`here <neuralhydrology.training.loss>`.

-  ``target_loss_weights``: A list of float values specifying the 
   per-target loss weight, when training on multiple targets at once. 
   Can be combined with any loss. By default, the weight of each target
   is ``1/n`` with ``n`` being the number of target variables. The order 
   of the weights corresponds to the order of the ``target_variables``.

-  ``regularization``: List of optional regularization terms. Currently
   supported is ``tie_frequencies``, which couples the predictions of
   all frequencies via an MSE term. New regularizations can be added
   :py:mod:`here <neuralhydrology.training.regularization>`.

-  ``learning_rate``: Learning rate. Can be either a single number (for
   a constant learning rate) or a dictionary. If it is a dictionary, the
   keys must be integer that reflect the epochs at which the learning
   rate is changed to the corresponding value. The key ``0`` defines the
   initial learning rate.

-  ``batch_size``: Mini-batch size used for training.

-  ``epochs``: Number of training epochs

-  ``use_frequencies``: Defines the time step frequencies to use (daily,
   hourly, ...). Use `pandas frequency
   strings <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`__
   to define frequencies. Note: The strings need to include values,
   e.g., '1D' instead of 'D'. If used, ``predict_last_n`` and
   ``seq_length`` must be dictionaries.

-  ``no_loss_frequencies``: Subset of frequencies from
   ``use_frequencies`` that are "evaluation-only", i.e., the model will
   get input and produce output in the frequencies listed here, but they
   will not be considered in the calculation of loss and regularization
   terms.

-  ``seq_length``: Length of the input sequence. If ``use_frequencies``
   is used, this needs to be a dictionary mapping each frequency to a
   sequence length, else an int.

-  ``predict_last_n``: Defines which time steps are used to calculate
   the loss, counted backwards. Can't be larger than ``seq_length``.
   Sequence-to-one would be ``predict_last_n: 1`` and
   sequence-to-sequence (with e.g. a sequence length of 365)
   ``predict_last_n: 365``. If ``use_frequencies`` is used, this needs
   to be a dictionary mapping each frequency to a
   predict\_last\_n-value, else an int.

-  ``target_noise_std``: Defines the standard deviation of gaussian
   noise which is added to the labels during training. Set to zero or
   leave empty to *not* add noise.

-  ``clip_gradient_norm``: If a value, clips norm of gradients to that
   specific value during training. Leave empty for not clipping.

-  ``num_workers``: Number of (parallel) threads used in the data
   loader.

-  ``save_weights_every``: Interval, in which the weights of the model
   are stored to disk. ``1`` means to store the weights after each
   epoch, which is the default if not otherwise specified.
   
Finetune settings
-----------------

Ignored if ``mode != finetune``

-  ``finetune_modules``: List of model parts that will be trained
   during fine-tuning. All parts *not* listed here will not be
   updated. Check the documentation of each model to see a list
   of available module parts.

Logger settings
---------------

-  ``log_interval``: Interval at which the training loss is logged, 
   by default 10.
-  ``log_tensorboard``: True/False. If True, writes logging results into
   TensorBoard file. The default, if not specified, is True.

-  ``log_n_figures``: If a (integer) value greater than 0, saves the
   predictions as plots of that n specific (random) basins during
   validations.

Data settings
-------------

-  ``dataset``: Defines which data set will be used. Currently supported
   are ``camels_us`` (CAMELS data set by Newman et al.), ``CAMELS_GB``
   (the GB version of CAMELS by Coxon et al.), ``CAMELS_CL`` (the CL
   version of CAMELS by Alvarez-Garreton et al.), and 
   ``hourly_camels_us`` (hourly data for 516 CAMELS basins).

-  ``data_dir``: Full or relative path to the root directory of the data set.

-  ``train_data_file``: If not empty, uses the pickled file at this path
   as the training data. Can be used to not create the same data set
   multiple times, which saves disk space and time. If empty, creates
   new data set and optionally stores the data in the run directory (if
   ``save_train_data`` is True).

-  ``cache_validation_data``: True/False. If True, caches validation data 
   in memory for the time of training, which does speed up the overall
   training time. By default True, since even larger datasets are usually
   just a few GB in memory, which most modern machines can handle.

-  ``dynamic_inputs``: List of variables to use as time series inputs.
   Names must match the exact names as defined in the data set. Note: In
   case of multiple input forcing products, you have to append the
   forcing product behind each variable. E.g., 'prcp(mm/day)' of the
   daymet product is 'prcp(mm/day)_daymet'. When training on multiple
   frequencies (cf. ``use_frequencies``), it is possible to define
   dynamic inputs for each frequency individually. To do so,
   ``dynamic_inputs`` must be a dict mapping each frequency to a list of
   variables. E.g., to use precipitation from daymet for daily and from
   nldas-hourly for hourly predictions:

   ::

       dynamic_inputs:
         1D:
           - prcp(mm/day)_daymet
         1H:
           - total_precipitation_nldas_hourly

-  ``target_variables``: List of the target variable(s). Names must match
   the exact names as defined in the data set.

-  ``clip_targets_to_zero``: Optional list of target variables to clip to
   zero during evaluation.

-  ``duplicate_features``: Can be used to duplicate time series features
   (e.g., for different normalizations). Can be either a str, list or dictionary
   (mapping from strings to ints). If string, duplicates the corresponding
   feature once. If list, duplicates all features in that list once. Use
   a dictionary to specify the exact number of duplicates you like.
   To each duplicated feature, we append ``_copyN``, where `N` is counter
   starting at 1.

-  ``lagged_features``: Can be used to add a lagged copy of another
   feature to the list of available input/output features. Has to be a
   dictionary mapping from strings to int or a list of ints, where the string 
   specifies the feature name and the int(s) the number of lagged time steps. Those values
   can be positive or negative (see
   `pandas shift <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html>`__
   for details). If a list of integers is provided, only unique values are considered.
   We append ``_shiftN`` to each lagged feature, where `N` is the shift count.

-  ``custom_normalization``: Has to be a dictionary, mapping from
   time series feature names to ``centering`` and/or ``scaling``. Using
   this argument allows to overwrite the default zero mean, unit
   variance normalization per feature. Supported options for
   ``centering`` are 'None' or 'none', 'mean', 'median' and min.
   None/none sets the centering parameter to 0.0, mean to the feature
   mean, median to the feature median, and min to the feature
   minimum, respectively. Supported options for `scaling` are
   'None' or 'none', 'std', 'minmax'. None/none sets the scaling
   parameter to 1.0, std to the feature standard deviation and
   minmax to the feature max minus the feature min. The combination
   of centering: min and scaling: minmax results in min/max
   feature scaling to the range [0,1].

-  ``additional_feature_files``: Path to a pickle file (or list of paths
   for multiple files), containing a dictionary with each key
   corresponding to one basin id and the value is a date-time indexed
   pandas DataFrame. Allows the option to add any arbitrary data that is
   not included in the standard data sets. **Convention**: If a column
   is used as static input, the value to use for specific sample should
   be in same row (datetime) as the target discharge value.

-  ``evolving_attributes``: Columns of the DataFrame loaded with the
   ``additional_feature_files`` that should be used as "static" features.
   These values will be used as static inputs, but they can evolve over time.
   Convention: The value to use for a specific input sequence should be in the
   same row (datetime) as the last time step of that sequence.
   Names must match the column names in the DataFrame. Leave empty to
   not use any additional static feature.

-  ``use_basin_id_encoding``: True/False. If True, creates a
   basin-one-hot encoding as a(n) (additional) static feature vector for
   each sample.

-  ``static_attributes``: Which static attributes to use (e.g., from the static camels attributes for the CAMELS
   dataset). Leave empty if none should be used. For hydroatlas attributes, use ``hydroatlas_attributes`` instead.
   Names must match the exact names as defined in the data set.

-  ``hydroatlas_attributes``: Which HydroATLAS attributes to use. Leave
   empty if none should be used. Names must match the exact names as
   defined in the data set.

CAMELS US specific
~~~~~~~~~~~~~~~~~~

Can be ignored if ``dataset not in ['camels_us', 'hourly_camels_us']``

-  ``forcings``: Can be either a string or a list of strings that
   correspond to forcing products in the camels data set. Also supports
   ``maurer_extended``, ``nldas_extended``, and (for
   ``hourly_camels_us``) ``nldas_hourly``.
