Quick Start
============

Prerequisites
-------------
As a first step you need a Python environment with all required dependencies. We recommend using `uv <https://github.com/astral-sh/uv>`_ for environment management.

First, install ``uv``:

.. code-block::

    # On Linux and macOS:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # On Windows, use PowerShell:
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

If you already have a Python environment, you can install the package directly. However, we recommend creating a virtual environment.

Installation
------------
There are two ways how you can install NeuralHydrology: Editable or non-editable.
If all you want to do is run experiments with existing datasets and existing models, you can use the non-editable
installation. To install the latest release from PyPI:

.. code-block::

    uv pip install neuralhydrology

To install the package directly from the current master branch of this repository, including any changes that are not yet part of a release, run:

.. code-block::

    uv pip install git+https://github.com/neuralhydrology/neuralhydrology.git

If you want to try implementing your own models or datasets, you'll need an editable installation.
For this, start by downloading or cloning the repository to your local machine.
If you use git, you can run:

.. code-block::

    git clone https://github.com/neuralhydrology/neuralhydrology.git

If you don't know git, you can also download the code from `here <https://github.com/neuralhydrology/neuralhydrology/zipball/master>`__ and extract the zip-file.

After you cloned or downloaded the zip-file, you'll end up with a directory called "neuralhydrology" (or "neuralhydrology-master").
Next, we'll go to that directory and install a local, editable copy of the package.
This will also install all required dependencies (including PyTorch). ``uv`` will automatically select the appropriate PyTorch version for your system (CPU or CUDA).

.. code-block::

    cd neuralhydrology
    uv sync

This command creates a virtual environment in `.venv`. To use the installed scripts (`nh-run`, `nh-schedule-runs` and `nh-results-ensemble`), you can either activate the environment:

.. code-block::

    source .venv/bin/activate
    nh-run ...

Or run them directly with `uv run`:

.. code-block::

    uv run nh-run ...

For development, you might want to install additional dependencies (like jupyter, pytest, etc.):

.. code-block::

    uv sync --all-groups

Data
----
Training and evaluating models requires a dataset.
If you're unsure where to start, a common dataset is CAMELS US, available at
`CAMELS US (NCAR) <https://ral.ucar.edu/solutions/products/camels>`_.
This dataset is used in all of our tutorials and we have a `dedicated tutorial <../tutorials/data-prerequisites.nblink>`_ with download instructions that you might want to look at.


Training a model
----------------
To train a model, prepare a configuration file, then run::

    uv run nh-run train --config-file /path/to/config.yml

If you want to train multiple models, you can make use of the ``nh-schedule-runs`` command.
Place all configs in a folder, then run::

    uv run nh-schedule-runs train --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y

With X, you can specify how many models should be trained on parallel on a single GPU.
With Y, you can specify which GPUs to use for training (use the id as specified in ``nvidia-smi``).


Continuing training
--------------------
Training doesn't always finish in one sitting, for example when you're on a cluster with a walltime
limit. To resume a single run from its last saved checkpoint, run::

    uv run nh-run continue_training --run-dir /path/to/run_dir/

``run_dir`` must point to the directory of the run you want to resume (the one created by the
original ``train`` run), not to a config file. The ``epochs`` value in that run's ``config.yml``
is read as the *total* number of epochs you want in the end, not as "how many more epochs to run" —
so you never need to edit it between sessions, unless you want to push the target further than
originally planned.

To resume every unfinished run in a folder at once, across multiple GPUs::

    uv run nh-schedule-runs continue_training --directory /path/to/runs_dir/ --runs-per-gpu X --gpu-ids Y

This scans ``runs_dir`` for run directories, checks each one's last saved epoch against its
``epochs`` target, and resumes only the ones that haven't reached it yet; runs that are already
complete are skipped silently. If you'd rather get an explicit error when the given directory
doesn't contain any run that has ever saved a checkpoint (for example because of a wrong path),
use ``continue_training_only`` instead of ``continue_training``.

.. note::
    If ``compile_model`` is enabled (the default, see the Configuration Arguments page), make sure
    you resume with the same NeuralHydrology version (or at least
    the same ``compile_model`` setting) that produced the checkpoint. Compiling wraps the model and
    prefixes every key in its saved weights with ``_orig_mod.``; checkpoints are always saved and
    loaded through the unwrapped model, so a checkpoint written by one version/setting loads
    correctly under another. But if you ever see an error like
    ``Missing key(s) ... "_orig_mod.lstm..."`` or ``Unexpected key(s) ... "lstm..."`` when resuming
    or evaluating, it means the checkpoint was produced by a version of the code that did not yet
    unwrap the model before saving — upgrade and the checkpoint will load normally.


Evaluating a model
------------------
To evaluate a trained model on the test set, run::

    uv run nh-run evaluate --run-dir /path/to/run_dir/

If the optional argument ``--epoch N`` (where N is the epoch to evaluate) is not specified,
the weights of the last epoch are used.

To evaluate all runs in a specific directory you can, similarly to training, run::

    uv run nh-schedule-runs evaluate --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y


To merge the predictons of a number of runs (stored in ``$DIR1``, ...) into one averaged ensemble,
use the ``nh-results-ensemble`` script::

    uv run nh-results-ensemble --run-dirs $DIR1 $DIR2 ... --output-dir /path/to/output/directory --metrics NSE MSE ...

``--metrics`` specifies which metrics will be calculated for the averaged predictions.
