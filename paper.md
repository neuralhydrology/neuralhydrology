EXAMPLES: https://joss.theoj.org/

---
title: 'NeuralHydrology --- A Python library for Deep Learning research in hydrology'
tags:
  - python
  - hydrology
  - neural networks
  - deep learning
  - machine learning
  - rainfall-runoff modeling
authors:
  - name: Frederik Kratzert
    orcid: 0000-0002-8897-7689
    affiliation: 1
  - name: Martin Gauch
    orcid: 0000-0002-4587-898X
    affiliation: 2
  - name: Grey Nearing
    orcid: 0000-0001-7031-6770
    affiliation: 1
- name: Daniel Klotz
    orcid: 0000-0002-9843-6798
    affiliation: 2
affiliations:
 - name: Google Research
   index: 1
 - name: Institute for Machine Learning, Johannes Kepler University Linz, Linz, Austria
   index: 2

date: 14 July 2021
bibliography: paper.bib
---

# Summary and statement of need

Since ancient times humans have strived to describe environmental processes related to water [@biswas1970history; @angelakis2012evolution].
Throughout this history, hydrologists built various process-based prediction models that simulate processes from soil moisture to streamflow generation (a collection of historical references can be found in @loague2010rainfall). More recently, Deep Learning models have emerged as extremely powerful and more accurate alternatives to these traditional modeling approaches [@kratzert@regional; @kratzert2019pub; @gauch2021mts; @klotz2021uncertainty; @lees2021benchmarking]. For hydrologists, embracing this new data-driven paradigm is challenging [@nearing2021role; @keith2020deep]: Not only are the models conceptually different, but they are also built, optimized, and evaluated with different strategies and toolsets.

NeuralHydrology is a Python library based on PyTorch [@paszke2019pytorch] that is designed to build, apply, and experiment with Deep Learning models with a strong focus on hydrological applications. Originally designed for our internal research needs, the library was generalized and open-sourced to allow anyone to experiment with Deep Learning models as easily as possible: Pre-built models and data loaders allow for quick experiments, yet the framework is also easily extensible to new models, data sets, loss functions, or metrics to suit more advanced use-cases.

NeuralHydrology is targeted towards students and researchers who want to experiment with Deep Learning models for rainfall-runoff modeling or other problems related to hydrology. As such, the library was designed to be picked up by beginners with little programming experience. For example, NeuralHydrology allows to train state-of-the-art rainfall-runoff models by only editing a config file and without touching a single line of code.


# Functionality

NeuralHydrology is available on GitHub and can be installed via the pip package manager or by cloning the repository. The documentation provides a detailed explanation of the installation steps.

## Basic Concepts

At the core of each experiment with NeuralHydrology is a YAML configuration file that specifies input and target data, model architecture, metrics to calculate, as well as training, validation, and test periods, along with some more technical settings that are described in the documentation. The configuration file also specifies whether the experiment is conducted on a CPU or a GPU --- for Deep Learning experiments, the latter can drastically improve runtimes.

The library can be used in two ways: First, via the command line interface (CLI). After installation via pip, the package registers several commands that can be executed via the command line, most importantly `nh_run`. This is the main entrypoint to start training and validation of models. Second, via the Python API. This option allows for more sophisticated use-cases, as users can define more precisely which parts of the framework they intend to use. There exist methods to conduct a full training or evaluation just like it is possible through the CLI, but it is equally possible to only use specific models, metrics, or data sets in the context of one’s own code. NeuralHydrology also provides scripts for job scheduling to run multiple experiments in parallel on one or more GPUs.

Lastly, the modular design of the library allows contributors not only to extend the functionality by adding new models, data sets, or metrics, but also to run comparative experiments: for instance, one can compare different models in the same setting or one model on different data sets.

## Example use-cases

The most basic use-case for the NeuralHydrology library is basic rainfall--runoff modeling as described in Kratzert et al. 20xx, where we train a machine learning model on meteorological data and discharge observations from a set of basins (a process also known as calibration) and subsequently apply the model to a different time period, either to evaluate its goodness-of-fit, or to actually generate predictions for practical use. Since the experiment configuration is highly flexible, users can define which inputs they want to ingest into the model (e.g., whether discharge from previous time steps should be an input variable) and which variable(s) they want to predict. Users can easily train an ensemble of multiple models and average their predictions, which often greatly enhances the overall accuracy.

Beyond this most basic application, NeuralHydrology also supports prediction in ungauged basins, where a model is trained on one set of basins and subsequently applied to another set of basins for which no discharge observations exist (see e.g. @kratzert2019pub). Lastly, NeuralHydrology also includes different options to train neural networks for uncertainty estimation, where the model does not only regress a single target value (e.g. discharge) per time step, but rather outputs probability distributions for each target variable (for details, see @klotz2021uncertainty).

# References
