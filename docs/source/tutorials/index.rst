Tutorials
=========

| **Introduction to neuralHydrology**
| If you're new to the neuralHydrology package, :doc:`this tutorial <introduction>` is the place to get started. It walks you through the basic command-line and API usage patterns, and you get to train and evaluate your first model.

| **Adding a New Model: Gated Recurrent Unit (GRU)**
| Once you know the basics, you might want to add your own model. Using the `GRU <https://en.wikipedia.org/wiki/Gated_recurrent_unit>`__ model as an example, :doc:`this tutorial <adding-gru>` shows how and where to add models in the neuralHydrology codebase.

| **Adding a New Dataset: CAMELS-CL**
| Always using the United States CAMELS dataset is getting boring? :doc:`This tutorial <add-dataset>` shows you how to add a new dataset: The Chilean version of CAMELS.

| **Multi-Timescale Prediction**
| In one of our `papers <https://arxiv.org/abs/2010.07921>`__, we introduced Multi-Timescale LSTMs that can predict at multiple timescales simultaneously. If you need predictions at sub-daily granularity or you want to generate daily and hourly predictions (or any other timescale), :doc:`this tutorial <multi-timescale>` explains how to get there.

| **Inspecting the internals of LSTMs**
| Model interpretability is an ongoing research topic. We showed in previous publications (e.g. `this one <https://arxiv.org/abs/1903.07903>`__) that LSTM internals can be linked to physical processes. In :doc:`this tutorial <inspect-lstm>`, we show how to extract those model internals with our library.

| **Finetuning models**
| A common way to increase model performance with deep learning models is called finetuning. Here, first a model is trained on a large and diverse dataset, before second, the model is finetuned to the actual problem of interest. In :doc:`this tutorial <finetuning>`, we show how you can perform finetuning with our library.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   introduction
   adding-gru
   add-dataset
   multi-timescale
   inspect-lstm
   finetuning
