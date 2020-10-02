![#](docs/source/_static/img/neural-hyd-logo-black.png)

Python library to train neural networks with a strong focus on hydrological applications.

This package has been used extensively in research over the last year and was used in various academic publications. 
The core idea of this package is modularity in all places to allow easy integration of new datasets, new model 
architectures or any training related aspects (e.g. loss functions, optimizer, regularization). 
One of the core concepts of this code base are configuration files, which lets anyone train neural networks without
touching the code itself. The `NeuralHydrology` package is build on top of the deep learning framework 
[Pytorch](https://pytorch.org/), since it has proven to be the most flexible and useful for research purposes.

We (AI for Earth Science group at Institute for Machine Learning, Johannes Kepler University, Linz, Austria) are using
this code in our day-to-day research and will continue to integrate our new research findings into this public repository.

**Note:** We will gradually add more examples/documentation over the next couple of days/weeks.

- Documentation: [neuralhydrology.readthedocs.io](https://neuralhydrology.readthedocs.io)
- Research Blog: [neuralhydrology.github.io](https://neuralhydrology.github.io)
- Bug reports/Feature requests [https://github.com/neuralhydrology/neuralhydrology/issues](https://github.com/neuralhydrology/neuralhydrology/issues)

# Getting started

## Requirements

We recommend to use Anaconda/Miniconda. With one of the two installed, a dedicated environment with all requirements 
installed can be set up from the environment files provided in 
[environments](https://github.com/neuralhydrology/neuralhydrology/environments). 

If you have no CUDA capable GPU available run

```bash
conda env create -f environments/environment_cpu.yml
```

With a CUDA capable GPU available, check which CUDA version your GPU supports and then run e.g. (for CUDA 10.2)

```bash
conda env create -f environments/environment_cuda10_2.yml
```

If neither Minicoda/Anaconda are available, make sure to Python environment with all packages installed that are listed 
in one of the environment files.

## Installation

For now download or clone the repository to your local machine and install a local, editable copy. 
This is a good idea if you want to edit the ``neuralhydrology`` code (e.g., adding new models or datasets).::

```bash
    git clone https://github.com/neuralhydrology/neuralhydrology.git
    cd neuralhydrology
    pip install -e .
```
Besides adding the package to your Python environment, it will also add three bash scripts: 
`nh-run`, `nh-run-scheduler` and `nh-results-ensemble`. For details, see below.


## Data

Training and evaluating models requires a dataset.
If you're unsure where to start, a common dataset is CAMELS US, available at
[CAMELS US (NCAR)](https://ral.ucar.edu/solutions/products/camels).
Download the "CAMELS time series meteorology, observed flow, meta data" and place the actual data folder
(`basin_dataset_public_v1p2`) in a directory.
This directory will be referred to as the "data directory", or `data_dir`.

## Configuration file

One of the core concepts of this package is the usage of configuration files (`.yml`). Basically, all configurations 
required to train a neural network can be specified via these configuration files and no code has to be touched.
Training a model does require a `.yml` file that specifies the run configuration. We will add a detailed explanation
for within the next weeks that explains the config files and arguments in more detail. For now refer to the 
[example config](https://github.com/neuralhydrology/neuralhydrology/blob/master/examples/config.yml.example) for a full
list of all available arguments (with inline documentation). For an example of a configuration file that can be used to 
train a standard LSTM for a single CAMELS US basin, check 
[1_basin_config.yml](https://github.com/neuralhydrology/neuralhydrology/blob/master/examples/1_basin_config.yml.example).

## Train a model

To train a model, prepare a configuration file, then run::

```bash
    nh-run train --config-file /path/to/config.yml
```
If you want to train multiple models, you can make use of the ``nh-run-scheduler`` command.
Place all configs in a folder, then run::
```bash
    nh-run-scheduler train --config-dir /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y
```
With X, you can specify how many models should be trained on parallel on a single GPU.
With Y, you can specify which GPUs to use for training (use the id as specified in ``nvidia-smi``).

## Evaluate a model

To evaluate a trained model on the test set, run::

    nh-run evaluate --run-dir /path/to/run_dir/

If the optional argument ``--epoch N`` (where N is the epoch to evaluate) is not specified,
the weights of the last epoch are used. You can also use ``--period `` if you want to evaluate the model on the 
train period ``--period train``) or validation period (``--period validation``) 

To evaluate all runs in a specific directory you can, similarly to training, run::

    nh-run-scheduler evaluate --run-dir /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y


To merge the predictons of a number of runs (stored in ``$DIR1``, ...) into one averaged ensemble,
use the ``nh-results-ensemble`` script::

    nh-results-ensemble --run-dirs $DIR1 $DIR2 ... --save-file /path/to/target/file.p --metrics NSE MSE ...

``--metrics`` specifies which metrics will be calculated for the averaged predictions.

# Contact

If you have any questions regarding the usage of this repository, feature requests or comments, please open an issue.
You can also reach out to Frederik Kratzert (kratzert(at)ml.jku.at) by email.
