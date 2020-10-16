Tutorials
=========

| **Introduction to neuralHydrology**
| If you're new to the neuralHydrology package, `this tutorial <https://neuralhydrology.readthedocs.io/tutorials/introduction.html>`__ is the place to get started. It walks you through the basic command-line and API usage patterns, and you get to train and evaluate your first model.

| **Adding a New Model: Gated Recurrent Unit (GRU)**
| Once you know the basics, you might want to add your own model. Using the `GRU <https://en.wikipedia.org/wiki/Gated_recurrent_unit>`__ model as an example, `this tutorial <https://neuralhydrology.readthedocs.io/tutorials/adding-gru.html>`__ shows how and where to add models in the neuralHydrology codebase.

| **Adding a New Dataset: CAMELS-CL**
| Always using the United States CAMELS dataset is getting boring? `This tutorial <https://neuralhydrology.readthedocs.io/tutorials/add-dataset.html>`__ shows you how to add a new dataset: The Chilean version of CAMELS.

| **Multi-Timescale Prediction**
| In one of our `papers <https://arxiv.org/abs/2010.07921>`__, we introduced Multi-Timescale LSTMs that can predict at multiple timescales simultaneously. If you need predictions at sub-daily granularity or you want to generate daily and hourly predictions (or any other timescale), `this tutorial <https://neuralhydrology.readthedocs.io/tutorials/multi-timescale.html>`__ explains how to get there.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   introduction
   adding-gru
   add-dataset
   multi-timescale
