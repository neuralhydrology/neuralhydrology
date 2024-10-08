'''
purpose: 
- abstract away:
    - editing config (yaml) file (currently need to modify yaml file with hyperparamers)
    - whether or not we use physical model as data inputs
    - ensemble runs (currently need to train models in a loop, then manually collect paths, 
        then eval_run in a loop, then create results ensemble, then retrieve data, then plot)
    - model preformance metrics + visualizations (currently no good graphs (percentiles for ensemble run, 
        comparing preformance with physical model), need to write code to get metrics)

'''

from pathlib import Path
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class UCB_trainer:
    def __init__(self, hyperparams: dict, num_ensemble_members: int = 1, physics_informed: bool = True):
        """
        Initializes the UCB_trainer object.

        Args:
            hyperparams (dict): A dictionary of hyperparameters for the model.
            num_ensemble_members (int): The number of ensemble members.
            physics_informed (bool): Whether to use internal states of physical model as features.
        """
        self.hyperparams = hyperparams
        self.ensemble_members = num_ensemble_members
        self.physics_informed = physics_informed

    def train(self):
        """
        Public method to handle the training process for individual models or ensembles.
        """
        pass

    def results(self):
        """
        Public method to return metrics and data visualizations of model preformance.
        """
        pass

    def _train_model(self):
        """
        Private method to train an individual model.
        """
        pass

    def _eval_model(self, model_id):
        """
        Private method to evaluate an individual model after training.
        """
        pass

    def _train_ensemble(self):
        """
        Private method to train an ensemble of models.
        """
        pass

    def _create_config(self) -> Config:
        """
        Private method to create Configuration object for training from user specifications.
        """
        pass

    def _generate_plot1(self):
        """
        Private method to generate a plot for the results.
        """
        pass

    def _generate_plot2(self):
        """
        Private method to generate a plot for the results.
        """
        pass

    def _get_metrics(self):
        """
        Private method to get and return metrics after training and evaluation.
        """
        pass