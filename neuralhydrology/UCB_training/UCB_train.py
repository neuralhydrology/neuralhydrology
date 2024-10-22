'''
purpose: 
- abstract away:
    - editing config (yaml) file (currently need to modify yaml file with hyperparamers)
    - whether or not we use physical model as data inputs
    - ensemble runs (currently need to train models in a loop, then manually collect paths, 
        then eval_run in a loop, then create results ensemble, then retrieve data, then plot)
    - model preformance metrics + visualizations (currently no good graphs (percentiles for ensemble run, 
        comparing preformance with physical model), need to write code to get metrics)
TODO:
    - add logging
    - add support for turning on and off physics based inputs
    - add more visualizations
    - make default args better
    - add percentiles to ensemble runs
'''

from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.nh_results_ensemble import create_results_ensemble
from neuralhydrology.evaluation.metrics import calculate_all_metrics

class UCB_trainer:
    def __init__(self, path_to_csv_folder: Path, hyperparams: dict, num_ensemble_members: int = 1, physics_informed: bool = True, gpu: int = -1):
        """
        Initializes the UCB_trainer object.

        Args:
            hyperparams (dict): A dictionary of hyperparameters for the model.
            num_ensemble_members (int): The number of ensemble members.
            physics_informed (bool): Whether to use internal states of physical model as features.
        """
        self._hyperparams = hyperparams
        self._num_ensemble_members = num_ensemble_members
        self._physics_informed = physics_informed
        self._gpu = gpu
        self._data_dir = path_to_csv_folder
        
        self._config = None
        self._model = None
        self._test_predictions = None
        self._test_observed = None
        self._metrics = None

        self._create_config()

        if self._config.epochs % self._config.save_weights_every != 0:
            raise ValueError(
                "The 'save_weights_every' parameter must divide the 'epochs' parameter evenly. Ensure 'epochs' is a multiple of "
                "'save_weights_every' to use the most recent weights for the final model."
                )


    def train(self):
        """
        Public method to handle the training and evaluating process for individual models or ensembles. Sets self.model.
        """
        if self._num_ensemble_members == 1:
            self._model = self._train_model() # returns run directory of single model
            self._eval_model(self._model)
        else:
            self.model = self._train_ensemble() # returns dict with predictions on test set and metrics
        return

    def results(self) -> dict:
        """
        Public method to return metrics and plot data visualizations of model preformance.
        """
        self._get_predictions()
        self._metrics = calculate_all_metrics(self._test_observed, self._test_predictions)
        
        self._generate_obs_sim_plt()
        return self._metrics

    def _train_model(self) -> Path:
        """
        Private method to train an individual model. Returns the path to the model results.
        """
    
        # check if a GPU has been specified. If yes, overwrite config
        if self._gpu is not None and self._gpu >= 0:
            self._config.device = f"cuda:{self._gpu}"
        if self._gpu is not None and self._gpu < 0:
            self._config.device = "cpu"

        start_training(self._config)
        path = self._config.run_dir
        return path

    def _eval_model(self, run_directory, period="test"):
        """
        Private method to evaluate an individual model after training. 
        """
        eval_run(run_dir=run_directory, period=period)
        return

    def _train_ensemble(self, period="test") -> dict:
        """
        Private method to train and evaluate an ensemble of models.
        """
        paths = [] #store the path of the results of the model
        for _ in range(self._num_ensemble_members):
            path = self._train_model()
            paths.append(path)

        #for each path evaluate the model    
        for p in paths:
            self._eval_model(run_directory=p, period=period)
            #self._eval_model(run_dir=p, period="validation") 

        ensemble_run = create_results_ensemble(paths, period=period)
        return ensemble_run

    def _get_predictions(self) -> dict:
        """
        Private method to get and return predicted values and metrics after training and evaluation.
        """
        if self._num_ensemble_members == 1:
            with open(self._model / "test" / f"model_epoch{str(self._config.epochs).zfill(3)}" / "test_results.p", "rb") as fp:
                results = pickle.load(fp)
                self._test_observed = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs'].sel(time_step=0)
                self._test_predictions = results['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim'].sel(time_step=0)

        else:
            self._test_observed = self._model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_obs']
            self._test_predictions = self._model['Tuler']['1D']['xr']['ReservoirInflowFLOW-OBSERVED_sim']
        
        return

    def _create_config(self) -> Config:
        """
        Private method to create Configuration object for training from user specifications.
        """
        config = Config(Path('./template_config.yaml'))
        config.update_config(self._hyperparams)
        config.update_config({'data_dir': self._data_dir})
        config.update_config({'physics_informed': self._physics_informed})
        self._config = config
        return

    def _generate_obs_sim_plt(self):
        """
        Private method to plot observed and simulated values over time.
        """
        date_indexer = "date" if self._num_ensemble_members == 1 else "datetime"
        fig, ax = plt.subplots(figsize=(16,10))
        ax.plot(self._test_observed[date_indexer], self._test_observed, label="Obs")
        ax.plot(self._test_predictions[date_indexer], self._test_predictions, label="Sim")
        ax.set_ylabel("ReservoirInflowFLOW-OBSERVED")
        ax.legend()
        plt.show()