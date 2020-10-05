Quick Start
============

Installation
------------
For now, download or clone the repository to your local machine and install a local, editable copy.
This is a good idea if you want to edit the ``neuralhydrology`` code (e.g., adding new models or datasets)

.. code-block::

    git clone https://github.com/neuralhydrology/neuralhydrology.git
    cd neuralhydrology
    pip install -e .

Besides adding the package to your Python environment, it will also add three bash scripts:
`nh-run`, `nh-run-scheduler` and `nh-results-ensemble`. For details, see below.

Data
----
Training and evaluating models requires a dataset.
If you're unsure where to start, a common dataset is CAMELS US, available at
`CAMELS US (NCAR) <https://ral.ucar.edu/solutions/products/camels>`_.
Download the "CAMELS time series meteorology, observed flow, meta data" and place the actual data folder
(``basin_dataset_public_v1p2``) in a directory.
This directory will be referred to as the "data directory", or ``data_dir``.


Training a model
----------------
To train a model, prepare a configuration file, then run::

    nh-run train --config-file /path/to/config.yml

If you want to train multiple models, you can make use of the ``nh-run-scheduler`` command.
Place all configs in a folder, then run::

    nh-run-scheduler train --config-dir /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y

With X, you can specify how many models should be trained on parallel on a single GPU.
With Y, you can specify which GPUs to use for training (use the id as specified in ``nvidia-smi``).


Evaluating a model
------------------
To evaluate a trained model on the test set, run::

    nh-run evaluate --run-dir /path/to/run_dir/

If the optional argument ``--epoch N`` (where N is the epoch to evaluate) is not specified,
the weights of the last epoch are used.

To evaluate all runs in a specific directory you can, similarly to training, run::

    nh-run-scheduler evaluate --run-dir /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y


To merge the predictons of a number of runs (stored in ``$DIR1``, ...) into one averaged ensemble,
use the ``nh-results-ensemble`` script::

    nh-results-ensemble --run-dirs $DIR1 $DIR2 ... --save-file /path/to/target/file.p --metrics NSE MSE ...

``--metrics`` specifies which metrics will be calculated for the averaged predictions.
