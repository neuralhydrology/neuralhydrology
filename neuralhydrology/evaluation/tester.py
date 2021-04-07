import logging
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils.utils import get_frequency_factor, load_basin_file, sort_frequencies
from neuralhydrology.evaluation import plots
from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.training.logger import Logger
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import AllNaNError, NoTrainDataError

LOGGER = logging.getLogger(__name__)


class BaseTester(object):
    """Base class to run inference on a model.

    Use subclasses of this class to evaluate a trained model on its train, test, or validation period.
    For regression settings, `RegressionTester` is used; for uncertainty prediction, `UncertaintyTester`.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}, optional
        The period to evaluate, by default 'test'.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        self.cfg = cfg
        self.run_dir = run_dir
        self.init_model = init_model
        if period in ["train", "validation", "test"]:
            self.period = period
        else:
            raise ValueError(f'Invalid period {period}. Must be one of ["train", "validation", "test"]')

        # determine device
        self._set_device()

        if self.init_model:
            self.model = get_model(cfg).to(self.device)

        # pre-initialize variables, defined in class methods
        self.basins = None
        self.scaler = None
        self.id_to_int = {}
        self.additional_features = []

        # placeholder to store cached validation data
        self.cached_datasets = {}

        self._load_run_data()

    def _set_device(self):
        if self.cfg.device is not None:
            if "cuda" in self.cfg.device:
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id > torch.cuda.device_count():
                    raise RuntimeError(f"This machine does not have GPU #{gpu_id} ")
                else:
                    self.device = torch.device(self.cfg.device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_run_data(self):
        """Load run specific data from run directory"""

        # get list of basins
        self.basins = load_basin_file(getattr(self.cfg, f"{self.period}_basin_file"))

        # load feature scaler
        scaler_file = self.run_dir / "train_data" / "train_data_scaler.p"
        with scaler_file.open('rb') as fp:
            self.scaler = pickle.load(fp)

        # check for old scaler files, where the center/scale parameters had still old names
        if "xarray_means" in self.scaler.keys():
            self.scaler["xarray_feature_center"] = self.scaler.pop("xarray_means")
        if "xarray_stds" in self.scaler.keys():
            self.scaler["xarray_feature_scale"] = self.scaler.pop("xarray_stds")

        # load basin_id to integer dictionary for one-hot-encoding
        if self.cfg.use_basin_id_encoding:
            file_path = self.run_dir / "train_data" / "id_to_int.p"
            with file_path.open("rb") as fp:
                self.id_to_int = pickle.load(fp)

        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _get_weight_file(self, epoch: int):
        """Get file path to weight file"""
        if epoch is None:
            weight_file = sorted(list(self.run_dir.glob('model_epoch*.pt')))[-1]
        else:
            weight_file = self.run_dir / f"model_epoch{str(epoch).zfill(3)}.pt"

        return weight_file

    def _load_weights(self, epoch: int = None):
        """Load weights of a certain (or the last) epoch into the model."""
        weight_file = self._get_weight_file(epoch)

        LOGGER.info(f"Using the model weights from {weight_file}")
        self.model.load_state_dict(torch.load(weight_file, map_location=self.device))

    def _get_dataset(self, basin: str) -> BaseDataset:
        """Get dataset for a single basin."""
        ds = get_dataset(cfg=self.cfg,
                         is_train=False,
                         period=self.period,
                         basin=basin,
                         additional_features=self.additional_features,
                         id_to_int=self.id_to_int,
                         scaler=self.scaler)
        return ds

    def evaluate(self,
                 epoch: int = None,
                 save_results: bool = True,
                 metrics: Union[list, dict] = [],
                 model: torch.nn.Module = None,
                 experiment_logger: Logger = None) -> dict:
        """Evaluate the model.
        
        Parameters
        ----------
        epoch : int, optional
            Define a specific epoch to evaluate. By default, the weights of the last epoch are used.
        save_results : bool, optional
            If True, stores the evaluation results in the run directory. By default, True.
        metrics : Union[list, dict], optional
            List of metrics to compute during evaluation. Can also be a dict that specifies per-target metrics
        model : torch.nn.Module, optional
            If a model is passed, this is used for validation.
        experiment_logger : Logger, optional
            Logger can be passed during training to log metrics

        Returns
        -------
        dict
            A dictionary containing one xarray per basin with the evaluation results.
        """
        if model is None:
            if self.init_model:
                self._load_weights(epoch=epoch)
                model = self.model
            else:
                raise RuntimeError("No model was initialized for the evaluation")

        # during validation, depending on settings, only evaluate on a random subset of basins
        basins = self.basins
        if self.period == "validation":
            if len(basins) > self.cfg.validate_n_random_basins:
                random.shuffle(basins)
                basins = basins[:self.cfg.validate_n_random_basins]

        # force model to train-mode when doing mc-dropout evaluation
        if self.cfg.mc_dropout:
            model.train()
        else:
            model.eval()

        results = defaultdict(dict)

        pbar = tqdm(basins, file=sys.stdout)
        pbar.set_description('# Validation' if self.period == "validation" else "# Evaluation")

        for basin in pbar:

            if self.cfg.cache_validation_data and basin in self.cached_datasets.keys():
                ds = self.cached_datasets[basin]
            else:
                try:
                    ds = self._get_dataset(basin)
                except NoTrainDataError as error:
                    # skip basin
                    continue
                if self.cfg.cache_validation_data and self.period == "validation":
                    self.cached_datasets[basin] = ds

            loader = DataLoader(ds, batch_size=self.cfg.batch_size, num_workers=0)

            y_hat, y = self._evaluate(model, loader, ds.frequencies)

            predict_last_n = self.cfg.predict_last_n
            seq_length = self.cfg.seq_length
            # if predict_last_n/seq_length are int, there's only one frequency
            if isinstance(predict_last_n, int):
                predict_last_n = {ds.frequencies[0]: predict_last_n}
            if isinstance(seq_length, int):
                seq_length = {ds.frequencies[0]: seq_length}
            lowest_freq = sort_frequencies(ds.frequencies)[0]
            for freq in ds.frequencies:
                if predict_last_n[freq] == 0:
                    continue  # this frequency is not being predicted
                results[basin][freq] = {}

                # rescale observations
                feature_scaler = self.scaler["xarray_feature_scale"][self.cfg.target_variables].to_array().values
                feature_center = self.scaler["xarray_feature_center"][self.cfg.target_variables].to_array().values
                y_freq = y[freq] * feature_scaler + feature_center
                # rescale predictions
                if y_hat[freq].ndim == 3 or (len(feature_scaler) == 1):
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                elif y_hat[freq].ndim == 4:
                    # if y_hat has 4 dim and we have multiple features we expand the dimensions for scaling
                    feature_scaler = np.expand_dims(feature_scaler, (0, 1, 3))
                    feature_center = np.expand_dims(feature_center, (0, 1, 3))
                    y_hat_freq = y_hat[freq] * feature_scaler + feature_center
                else:
                    raise RuntimeError(f"Simulations have {y_hat[freq].ndim} dimension. Only 3 and 4 are supported.")

                # create xarray
                data = self._create_xarray(y_hat_freq, y_freq)

                # get warmup-offsets across all frequencies
                offsets = {
                    freq: ds.get_period_start(basin) + (seq_length[freq] - 1) * to_offset(freq)
                    for freq in ds.frequencies
                }
                max_offset_freq = max(offsets, key=offsets.get)
                start_date = offsets[max_offset_freq]

                # determine the end of the first sequence (first target in sequence-to-one)
                # we use the end_date stored in the dataset, which also covers issues with per-basin different periods
                end_date = ds.dates[basin]["end_dates"][0] + pd.Timedelta(days=1, seconds=-1)

                # date range at the lowest frequency
                date_range = pd.date_range(start=start_date, end=end_date, freq=lowest_freq)
                if len(date_range) != data[f"{self.cfg.target_variables[0]}_obs"][1].shape[0]:
                    raise ValueError("Evaluation date range does not match generated predictions.")

                # freq_range are the steps of the current frequency at each lowest-frequency step
                frequency_factor = int(get_frequency_factor(lowest_freq, freq))
                freq_range = list(range(frequency_factor - predict_last_n[freq], frequency_factor))

                # create datetime range at the current frequency
                freq_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
                # remove datetime steps that are not being predicted from the datetime range
                mask = np.ones(frequency_factor).astype(bool)
                mask[:-predict_last_n[freq]] = False
                freq_date_range = freq_date_range[np.tile(mask, len(date_range))]

                xr = xarray.Dataset(data_vars=data, coords={'date': date_range, 'time_step': freq_range})
                results[basin][freq]['xr'] = xr

                # only warn once per freq
                if frequency_factor < predict_last_n[freq] and basin == basins[0]:
                    tqdm.write(f'Metrics for {freq} are calculated over last {frequency_factor} elements only. '
                               f'Ignoring {predict_last_n[freq] - frequency_factor} predictions per sequence.')

                if metrics:
                    for target_variable in self.cfg.target_variables:
                        # stack dates and time_steps so we don't just evaluate every 24H when use_frequencies=[1D, 1H]
                        obs = xr.isel(time_step=slice(-frequency_factor, None)) \
                            .stack(datetime=['date', 'time_step'])[f"{target_variable}_obs"]
                        obs['datetime'] = freq_date_range
                        # check if there are observations for this period
                        if not all(obs.isnull()):
                            sim = xr.isel(time_step=slice(-frequency_factor, None)) \
                                .stack(datetime=['date', 'time_step'])[f"{target_variable}_sim"]
                            sim['datetime'] = freq_date_range

                            # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                            if target_variable in self.cfg.clip_targets_to_zero:
                                sim = xarray.where(sim < 0, 0, sim)

                            if 'samples' in sim.dims:
                                sim = sim.mean(dim='samples')

                            var_metrics = metrics if isinstance(metrics, list) else metrics[target_variable]
                            if 'all' in var_metrics:
                                var_metrics = get_available_metrics()
                            try:
                                values = calculate_metrics(obs, sim, metrics=var_metrics, resolution=freq)
                            except AllNaNError as err:
                                msg = f'Basin {basin} ' \
                                    + (f'{target_variable} ' if len(self.cfg.target_variables) > 1 else '') \
                                    + (f'{freq} ' if len(ds.frequencies) > 1 else '') \
                                    + str(err)
                                LOGGER.warning(msg)
                                values = {metric: np.nan for metric in var_metrics}

                            # add variable identifier to metrics if needed
                            if len(self.cfg.target_variables) > 1:
                                values = {f"{target_variable}_{key}": val for key, val in values.items()}
                            # add frequency identifier to metrics if needed
                            if len(ds.frequencies) > 1:
                                values = {f"{key}_{freq}": val for key, val in values.items()}
                            if experiment_logger is not None:
                                experiment_logger.log_step(**values)
                            for k, v in values.items():
                                results[basin][freq][k] = v

        if (self.period == "validation") and (self.cfg.log_n_figures > 0) and (experiment_logger is not None):
            self._create_and_log_figures(results, experiment_logger, epoch)

        if save_results:
            self._save_results(results, epoch)

        return results

    def _create_and_log_figures(self, results: dict, experiment_logger: Logger, epoch: int):
        basins = list(results.keys())
        random.shuffle(basins)
        for target_var in self.cfg.target_variables:
            max_figures = min(self.cfg.validate_n_random_basins, self.cfg.log_n_figures, len(basins))
            for freq in results[basins[0]].keys():
                figures = []
                for i in range(max_figures):
                    xr = results[basins[i]][freq]['xr']
                    obs = xr[f"{target_var}_obs"].values
                    sim = xr[f"{target_var}_sim"].values
                    # clip negative predictions to zero, if variable is listed in config 'clip_target_to_zero'
                    if target_var in self.cfg.clip_targets_to_zero:
                        sim = xarray.where(sim < 0, 0, sim)
                    figures.append(
                        self._get_plots(
                            obs, sim, title=f"{target_var} - Basin {basins[i]} - Epoch {epoch} - Frequency {freq}")[0])
                # make sure the preamble is a valid file name
                experiment_logger.log_figures(figures, freq, preamble=re.sub(r"[^A-Za-z0-9\._\-]+", "", target_var))

    def _save_results(self, results: dict, epoch: int = None):
        # use name of weight file as part of the result folder name
        weight_file = self._get_weight_file(epoch=epoch)

        result_file = self.run_dir / self.period / weight_file.stem / f"{self.period}_results.p"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with result_file.open("wb") as fp:
            pickle.dump(results, fp)

        LOGGER.info(f"Stored results at {result_file}")

    def _evaluate(self, model: BaseModel, loader: DataLoader, frequencies: List[str]):
        """Evaluate model"""
        predict_last_n = self.cfg.predict_last_n
        if isinstance(predict_last_n, int):
            predict_last_n = {frequencies[0]: predict_last_n}  # if predict_last_n is int, there's only one frequency

        preds, obs = {}, {}
        with torch.no_grad():
            for data in loader:

                for key in data:
                    data[key] = data[key].to(self.device)
                predictions = self._generate_predictions(model, data)

                for freq in frequencies:
                    if predict_last_n[freq] == 0:
                        continue  # no predictions for this frequency
                    freq_key = '' if len(frequencies) == 1 else f'_{freq}'
                    y_hat_sub, y_sub = self._subset_targets(model, data, predictions, predict_last_n[freq], freq_key)

                    if freq not in preds:
                        preds[freq] = y_hat_sub.detach().cpu()
                        obs[freq] = y_sub.cpu()
                    else:
                        preds[freq] = torch.cat((preds[freq], y_hat_sub.detach().cpu()), 0)
                        obs[freq] = torch.cat((obs[freq], y_sub.detach().cpu()), 0)

            for freq in preds.keys():
                preds[freq] = preds[freq].numpy()
                obs[freq] = obs[freq].numpy()

        return preds, obs

    def _generate_predictions(self, model: BaseModel, data: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def _subset_targets(self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: np.ndarray,
                        predict_last_n: int, freq: str):
        raise NotImplementedError

    def _create_xarray(self, y_hat: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        raise NotImplementedError


class RegressionTester(BaseTester):
    """Tester class to run inference on a regression model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(RegressionTester, self).__init__(cfg, run_dir, period, init_model)

    def _generate_predictions(self, model: BaseModel, data: Dict[str, torch.Tensor]):
        return model(data)

    def _subset_targets(self, model: BaseModel, data: Dict[str, torch.Tensor], predictions: np.ndarray,
                        predict_last_n: np.ndarray, freq: str):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (('date', 'time_step'), y[:, :, i])
            data[f"{var}_sim"] = (('date', 'time_step'), y_hat[:, :, i])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.regression_plot(qobs, qsim, title)


class UncertaintyTester(BaseTester):
    """Tester class to run inference on an uncertainty model.

    Use the `evaluate` method of this class to evaluate a trained model on its train, test, or validation period.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    run_dir : Path
        Path to the run directory.
    period : {'train', 'validation', 'test'}
        The period to evaluate.
    init_model : bool, optional
        If True, the model weights will be initialized with the checkpoint from the last available epoch in `run_dir`.
    """

    def __init__(self, cfg: Config, run_dir: Path, period: str = "test", init_model: bool = True):
        super(UncertaintyTester, self).__init__(cfg, run_dir, period, init_model)

    def _generate_predictions(self, model: BaseModel, data: Dict[str, torch.Tensor]):
        samples = model.sample(data, self.cfg.n_samples)
        model.eval()
        return samples

    def _subset_targets(self,
                        model: BaseModel,
                        data: Dict[str, torch.Tensor],
                        predictions: np.ndarray,
                        predict_last_n: int,
                        freq: str = None):
        y_hat_sub = predictions[f'y_hat{freq}'][:, -predict_last_n:, :]
        y_sub = data[f'y{freq}'][:, -predict_last_n:, :]
        return y_hat_sub, y_sub

    def _create_xarray(self, y_hat: np.ndarray, y: np.ndarray):
        data = {}
        for i, var in enumerate(self.cfg.target_variables):
            data[f"{var}_obs"] = (('date', 'time_step'), y[:, :, i])
            data[f"{var}_sim"] = (('date', 'time_step', 'samples'), y_hat[:, :, i, :])
        return data

    def _get_plots(self, qobs: np.ndarray, qsim: np.ndarray, title: str):
        return plots.uncertainty_plot(qobs, qsim, title)
