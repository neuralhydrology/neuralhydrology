Tutorials
=========

All tutorials are based on Jupyter notebooks that are hosted on GitHub. 
If you want to run the code yourself, you can find the notebooks in the `examples folder <https://github.com/neuralhydrology/neuralhydrology/tree/master/examples>`__ of the NeuralHydrology GitHub repository.

| **Data Prerequisites**
| For most of our tutorials you will need some data to train and evaluate models. In all of these examples we use the publicly available CAMELS US dataset. :doc:`This tutorial <data-prerequisites>` will guide you through the download process of the different dataset pieces and explain how the code expects the local folder structure.

| **Introduction to NeuralHydrology**
| If you're new to the NeuralHydrology package, :doc:`this tutorial <introduction>` is the place to get started. It walks you through the basic command-line and API usage patterns, and you get to train and evaluate your first model.

| **Adding a New Model: Gated Recurrent Unit (GRU)**
| Once you know the basics, you might want to add your own model. Using the `GRU <https://en.wikipedia.org/wiki/Gated_recurrent_unit>`__ model as an example, :doc:`this tutorial <adding-gru>` shows how and where to add models in the NeuralHydrology codebase.

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

   data-prerequisites
   introduction
   adding-gru
   add-dataset
   multi-timescale
   inspect-lstm
   finetuning
