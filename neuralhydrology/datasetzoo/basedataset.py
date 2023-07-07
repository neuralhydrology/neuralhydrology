import logging
import pickle
import re
import sys
import warnings
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from pandas.tseries import frequencies
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
from ruamel.yaml import YAML
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError, NoEvaluationDataError
from neuralhydrology.utils import samplingutils

LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Base data set class to load and preprocess data.
    
    Use subclasses of this class for training/evaluating a model on a specific data set. E.g. use `CamelsUS` for the US
    CAMELS data set and `CamelsGB` for the CAMELS GB data set.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise, the basin(s) is(are) read from the
        appropriate basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train

        if period not in ["train", "validation", "test"]:
            raise ValueError("'period' must be one of 'train', 'validation' or 'test' ")
        else:
            self.period = period

        if period in ["validation", "test"]:
            if not scaler:
                raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")

            if cfg.use_basin_id_encoding and not id_to_int:
                raise ValueError("For basin id embedding, the id_to_int dictionary has to be passed anything but train")

        if self.cfg.timestep_counter:
            if not self.cfg.forecast_inputs:
                raise ValueError('Timestep counter only works for forecast data.')
            if cfg.forecast_overlap:
                overlap_zeros = torch.zeros((cfg.forecast_overlap, 1))
                forecast_counter = torch.Tensor(range(1, cfg.forecast_seq_length - cfg.forecast_overlap + 1)).unsqueeze(-1)
                self.forecast_counter = torch.concatenate([overlap_zeros, forecast_counter], dim=0)
                self.hindcast_counter = torch.zeros((cfg.seq_length - cfg.forecast_seq_length + cfg.forecast_overlap, 1))
            else:
                self.forecast_counter = torch.Tensor(range(1, cfg.forecast_seq_length + 1)).unsqueeze(-1)
                self.hindcast_counter = torch.zeros((cfg.seq_length - cfg.forecast_seq_length, 1))
            
        if basin is None:
            self.basins = utils.load_basin_file(getattr(cfg, f"{period}_basin_file"))
        else:
            self.basins = [basin]
        self.additional_features = additional_features
        self.id_to_int = id_to_int
        self.scaler = scaler
        # don't compute scale when finetuning
        if is_train and not scaler:
            self._compute_scaler = True
        else:
            self._compute_scaler = False

        # check and extract frequency information from config
        self.frequencies = []
        self.seq_len = None
        self._predict_last_n = None
        self._initialize_frequency_configuration()

        # during training we log data processing with progress bars, but not during validation/testing
        self._disable_pbar = cfg.verbose == 0 or not self.is_train

        # initialize class attributes that are filled in the data loading functions
        self._x_d = {}
        self._x_h = {}
        self._x_f = {}
        self._x_s = {}
        self._attributes = {}
        self._y = {}
        self._per_basin_target_stds = {}
        self._dates = {}
        self.start_and_end_dates = {}
        self.num_samples = 0
        self.period_starts = {}  # needed for restoring date index during evaluation

        # get the start and end date periods for each basin
        self._get_start_and_end_dates()

        # if additional features files are passed in the config, load those files
        if (not additional_features) and cfg.additional_feature_files:
            self._load_additional_features()

        if cfg.use_basin_id_encoding:
            if self.is_train:
                # creates lookup table for the number of basins in the training set
                self._create_id_to_int()

        # load and preprocess data
        self._load_data()

        if self.is_train:
            self._dump_scaler()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        sample = {}
        for freq, seq_len, idx in zip(self.frequencies, self.seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            # slice until idx + 1 because slice-end is excluding
            hindcast_start_idx = idx + 1 - seq_len
            global_end_idx = idx + 1
            if self._x_d:
                sample[f'x_d{freq_suffix}'] = self._x_d[basin][freq][hindcast_start_idx:global_end_idx]
            elif self._x_h:
                hindcast_end_idx = idx + 1 - self.cfg.forecast_seq_length
                forecast_start_idx = idx + 1 - self.cfg.forecast_seq_length
                if self.cfg.forecast_overlap and self.cfg.forecast_overlap > 0:
                    hindcast_end_idx += self.cfg.forecast_overlap
                sample[f'x_h{freq_suffix}'] = self._x_h[basin][freq][hindcast_start_idx:hindcast_end_idx]
                sample[f'x_f{freq_suffix}'] = self._x_f[basin][freq][forecast_start_idx:global_end_idx]
            else:
                raise ValueError('Data must include x_d or x_h.')

            sample[f'y{freq_suffix}'] = self._y[basin][freq][hindcast_start_idx:global_end_idx]
            sample[f'date{freq_suffix}'] = self._dates[basin][freq][hindcast_start_idx:global_end_idx]

            # check for static inputs
            static_inputs = []
            if self._attributes:
                static_inputs.append(self._attributes[basin])
            if self._x_s:
                static_inputs.append(self._x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

            if self.cfg.timestep_counter:
                if self._x_d:
                    torch.concatenate([sample[f'x_d{freq_suffix}'], self.hindcast_counter], dim=-1)
                else:
                    torch.concatenate([sample[f'x_h{freq_suffix}'], self.hindcast_counter], dim=-1)
                    torch.concatenate([sample[f'x_f{freq_suffix}'], self.forecast_counter], dim=-1)

        if self._per_basin_target_stds:
            sample['per_basin_target_stds'] = self._per_basin_target_stds[basin]
        if self.id_to_int:
            sample['x_one_hot'] = torch.nn.functional.one_hot(torch.tensor(self.id_to_int[basin]),
                                                              num_classes=len(self.id_to_int)).to(torch.float32)

        return sample

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """This function has to return the attributes in a basin-indexed DataFrame."""
        raise NotImplementedError

    def _create_id_to_int(self):
        self.id_to_int = {str(b): i for i, b in enumerate(np.random.permutation(self.basins))}

        # dump id_to_int dictionary into run directory for validation
        file_path = self.cfg.train_dir / "id_to_int.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(self.id_to_int, fp)

    def _dump_scaler(self):
        # dump scaler dictionary into run directory for inference
        scaler = defaultdict(dict)
        for key, value in self.scaler.items():
            if isinstance(value, pd.Series) or isinstance(value, xarray.Dataset):
                scaler[key] = value.to_dict()
            else:
                raise RuntimeError(f"Unknown datatype for scaler: {key}. Supported are pd.Series and xarray.Dataset")
        file_path = self.cfg.train_dir / "train_data_scaler.yml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w") as fp:
            yaml = YAML()
            yaml.dump(dict(scaler), fp)

    def _get_start_and_end_dates(self):

        # if no per-basin periods file exist, same periods are taken for all basins from the config
        if getattr(self.cfg, f"per_basin_{self.period}_periods_file") is None:

            # even if single dates, everything is mapped to lists, so we can iterate over them
            if isinstance(getattr(self.cfg, f'{self.period}_start_date'), list):
                if self.period != "train":
                    raise ValueError("Evaluation on split periods currently not supported")
                start_dates = getattr(self.cfg, f'{self.period}_start_date')
            else:
                start_dates = [getattr(self.cfg, f'{self.period}_start_date')]
            if isinstance(getattr(self.cfg, f'{self.period}_end_date'), list):
                end_dates = getattr(self.cfg, f'{self.period}_end_date')
            else:
                end_dates = [getattr(self.cfg, f'{self.period}_end_date')]

            self.start_and_end_dates = {b: {'start_dates': start_dates, 'end_dates': end_dates} for b in self.basins}

        # read periods from file
        else:
            with open(getattr(self.cfg, f"per_basin_{self.period}_periods_file"), 'rb') as fp:
                self.start_and_end_dates = pickle.load(fp)

    def _load_additional_features(self):
        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, n_duplicates in self.cfg.duplicate_features.items():
            for n in range(1, n_duplicates + 1):
                df[f"{feature}_copy{n}"] = df[feature]

        return df
    
    def _add_missing_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.cfg.target_variables:
            if var not in df.columns:
                df[var] = np.nan

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:

        # check that all autoregressive inputs are contained in the list of shifted variables
        self._check_autoregressive_inputs()

        # create the shifted varaibles, as requested
        for feature, shift in self.cfg.lagged_features.items():
            if isinstance(shift, list):
                # only consider unique shift values, otherwise we have columns with identical names
                for s in set(shift):
                    df[f"{feature}_shift{s}"] = df[feature].shift(periods=s, freq="infer")
            elif isinstance(shift, int):
                df[f"{feature}_shift{shift}"] = df[feature].shift(periods=shift, freq="infer")
            else:
                raise ValueError("The value of the 'lagged_features' arg must be either an int or a list of ints")

        return df

    def _check_autoregressive_inputs(self):
        # The dataset requires that AR inputs be lagged features, however in general when constructing the dataset
        # we do not care whether these are lagged targets, specifically. The requirement that AR inputs be lagged
        # targets, although typical for AR models, is not strictly required and depends on how these features are
        # used in any particular model.
        for input in self.cfg.autoregressive_inputs:
            capture = re.compile(r'^(.*)_shift(\d+)$').search(input)
            if not capture:
                raise ValueError('Autoregressive inputs must be a shifted variable with form <variable>_shift<lag> ',
                                 f'where <lag> is an integer. Instead got: {input}.')
            if capture[1] not in self.cfg.lagged_features or int(
                    capture[2]) not in self.cfg.lagged_features[capture[1]]:
                raise ValueError('Autoregressive inputs must be in the list of "lagged_inputs".')
        return

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        # if no netCDF file is passed, data set is created from raw basin files
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []

            # list of columns to keep, everything else will be removed to reduce memory footprint
            keep_cols = self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.mass_inputs + self.cfg.autoregressive_inputs

            if isinstance(self.cfg.dynamic_inputs, list):
                keep_cols += self.cfg.dynamic_inputs
            else:
                # keep all frequencies' dynamic inputs
                keep_cols += [i for inputs in self.cfg.dynamic_inputs.values() for i in inputs]
            # make sure that even inputs that are used in multiple frequencies occur only once in the df

            keep_cols = list(sorted(set(keep_cols)))

            if not self._disable_pbar:
                LOGGER.info("Loading basin data into xarray data set.")
            for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):
                df = self._load_basin_data(basin)

                # add columns from dataframes passed as additional data files
                df = pd.concat([df, *[d[basin] for d in self.additional_features]], axis=1)

                # if target variables are missing for basin, add empty column to still allow predictions to be made
                if not self.is_train:
                    df = self._add_missing_targets(df)

                # check if any feature should be duplicated
                df = self._duplicate_features(df)

                # check if a shifted copy of a feature should be added
                df = self._add_lagged_features(df)

                # remove unnecessary columns
                try:
                    df = df[keep_cols]
                except KeyError:
                    not_available_columns = [x for x in keep_cols if x not in df.columns]
                    msg = [
                        f"The following features are not available in the data: {not_available_columns}. ",
                        f"These are the available features: {df.columns.tolist()}"
                    ]
                    raise KeyError("".join(msg))

                # remove random portions of the timeseries of dynamic features
                for holdout_variable, holdout_dict in self.cfg.random_holdout_from_dynamic_features.items():
                    df[holdout_variable] = samplingutils.bernoulli_subseries_sampler(
                        data=df[holdout_variable].values,
                        missing_fraction=holdout_dict['missing_fraction'],
                        mean_missing_length=holdout_dict['mean_missing_length'],
                    )

                # Make end_date the last second of the specified day, such that the
                # dataset will include all hours of the last day, not just 00:00.
                start_dates = self.start_and_end_dates[basin]["start_dates"]
                end_dates = [
                    date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
                ]

                native_frequency = utils.infer_frequency(df.index)
                if not self.frequencies:
                    self.frequencies = [native_frequency]  # use df's native resolution by default

                # Assert that the used frequencies are lower or equal than the native frequency. There may be cases
                # where our logic cannot determine whether this is the case, because pandas might return an exotic
                # native frequency. In this case, all we can do is print a warning and let the user check themselves.
                try:
                    freq_vs_native = [utils.compare_frequencies(freq, native_frequency) for freq in self.frequencies]
                except ValueError:
                    LOGGER.warning('Cannot compare provided frequencies with native frequency. '
                                   'Make sure the frequencies are not higher than the native frequency.')
                    freq_vs_native = []
                if any(comparison > 1 for comparison in freq_vs_native):
                    raise ValueError(f'Frequency is higher than native data frequency {native_frequency}.')

                # used to get the maximum warmup-offset across all frequencies. We don't use to_timedelta because it
                # does not support all frequency strings. We can't calculate the maximum offset here, because to
                # compare offsets, they need to be anchored to a specific date (here, the start date).
                offsets = [(self.seq_len[i] - self._predict_last_n[i]) * to_offset(freq)
                           for i, freq in enumerate(self.frequencies)]

                basin_data_list = []
                # create xarray data set for each period slice of the specific basin
                for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
                    # if the start date is not aligned with the frequency, the resulting datetime indices will be off
                    if not all(to_offset(freq).is_on_offset(start_date) for freq in self.frequencies):
                        misaligned = [freq for freq in self.frequencies if not to_offset(freq).is_on_offset(start_date)]
                        raise ValueError(f'start date {start_date} is not aligned with frequencies {misaligned}.')
                    # add warmup period, so that we can make prediction at the first time step specified by period.
                    # offsets has the warmup offset needed for each frequency; the overall warmup starts with the
                    # earliest date, i.e., the largest offset across all frequencies.
                    warmup_start_date = min(start_date - offset for offset in offsets)
                    df_sub = df[warmup_start_date:end_date]

                    # make sure the df covers the full date range from warmup_start_date to end_date, filling any gaps
                    # with NaNs. This may increase runtime, but is a very robust way to make sure dates and predictions
                    # keep in sync. In training, the introduced NaNs will be discarded, so this only affects evaluation.
                    full_range = pd.date_range(start=warmup_start_date, end=end_date, freq=native_frequency)
                    df_sub = df_sub.reindex(pd.DatetimeIndex(full_range, name=df_sub.index.name))

                    # as double check, set all targets before period start to NaN
                    df_sub.loc[df_sub.index < start_date, self.cfg.target_variables] = np.nan

                    basin_data_list.append(df_sub)

                if not basin_data_list:
                    # Skip basin in case no start and end dates where defined.
                    continue

                # In case of multiple time slices per basin, stack the time slices in the time dimension.
                df = pd.concat(basin_data_list, axis=0)

                # Because of overlaps between warmup period of one slice and training period of another slice, there can
                # be duplicated indices. The next block of code creates two subset dataframes. First, a subset with all
                # non-duplicated indices. Second, a subset with duplicated indices, of which we keep the rows, where the
                # target value is not NaN (because we remove the target variable during warmup periods but want to keep
                # them if they are target in another temporal slice).
                df_non_duplicated = df[~df.index.duplicated(keep=False)]
                df_duplicated = df[df.index.duplicated(keep=False)]

                filtered_duplicates = []
                for _, grp in df_duplicated.groupby('date'):
                    mask = ~grp[self.cfg.target_variables].isna().any(1)
                    if not mask.any():
                        # In case all duplicates have a NaN value for the targets, pick the first. This can happen, if
                        # the day itself has a missing observation.
                        filtered_duplicates.append(grp.head(1))
                    else:
                        # If at least one duplicate has values in the target columns, take the first of these rows.
                        filtered_duplicates.append(grp[mask].head(1))

                if filtered_duplicates:
                    # Combine the filtered duplicates with the non-duplicates.
                    df_filtered_duplicates = pd.concat(filtered_duplicates, axis=0)
                    df = pd.concat([df_non_duplicated, df_filtered_duplicates], axis=0)
                else:
                    # Else, if no duplicates existed, continue with only the non-duplicate df.
                    df = df_non_duplicated

                # Sort by DatetimeIndex and reindex to fill gaps with NaNs.
                df = df.sort_index(axis=0, ascending=True)
                df = df.reindex(
                    pd.DatetimeIndex(data=pd.date_range(df.index[0], df.index[-1], freq=native_frequency),
                                     name=df.index.name))

                # Convert to xarray Dataset and add basin string as additional coordinate
                xr = xarray.Dataset.from_dataframe(df.astype(np.float32))
                xr = xr.assign_coords({'basin': basin})
                data_list.append(xr)

            if not data_list:
                # If no period for no basin has defined timeslices, raise error.
                if self.is_train:
                    raise NoTrainDataError
                else:
                    raise NoEvaluationDataError

            # create one large dataset that has two coordinates: datetime and basin
            xr = xarray.concat(data_list, dim="basin")

            if self.is_train and self.cfg.save_train_data:
                self._save_xarray_dataset(xr)

        else:
            with self.cfg.train_data_file.open("rb") as fp:
                d = pickle.load(fp)
            xr = xarray.Dataset.from_dict(d)
            if not self.frequencies:
                native_frequency = utils.infer_frequency(xr["date"].values)
                self.frequencies = [native_frequency]

        return xr

    def _save_xarray_dataset(self, xr: xarray.Dataset):
        """Store newly created train data set to disk"""
        file_path = self.cfg.train_dir / "train_data.p"

        file_path.parent.mkdir(parents=True, exist_ok=True)

        # netCDF has issues with storing variables with '/' in the var names. Solution: convert to dict, then pickle
        with file_path.open("wb") as fp:
            pickle.dump(xr.to_dict(), fp)

    def _calculate_per_basin_std(self, xr: xarray.Dataset):
        basin_coordinates = xr["basin"].values.tolist()
        if not self._disable_pbar:
            LOGGER.info("Calculating target variable stds per basin")
        nan_basins = []
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):

            obs = xr.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                # calculate std for each target
                per_basin_target_stds = torch.tensor(np.expand_dims(np.nanstd(obs, axis=1), 0), dtype=torch.float32)
            else:
                nan_basins.append(basin)
                per_basin_target_stds = torch.full((1, obs.shape[0]), np.nan, dtype=torch.float32)

            self._per_basin_target_stds[basin] = per_basin_target_stds

        if len(nan_basins) > 0:
            LOGGER.warning("The following basins had not enough valid target values to calculate a standard deviation: "
                           f"{', '.join(nan_basins)}. NSE loss values for this basin will be NaN.")

    def _create_lookup_table(self, xr: xarray.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        # list to collect basins ids of basins without a single training sample
        basins_without_samples = []
        basin_coordinates = xr["basin"].values.tolist()
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):

            # store data of each frequency as numpy array of shape [time steps, features] and dates as numpy array of
            # shape (time steps,)
            x_d, x_s, y, dates = {}, {}, {}, {}
            x_d_column_names = []

            # keys: frequencies, values: array mapping each lowest-frequency
            # sample to its corresponding sample in this frequency
            frequency_maps = {}
            lowest_freq = utils.sort_frequencies(self.frequencies)[0]

            # converting from xarray to pandas DataFrame because resampling is much faster in pandas.
            df_native = xr.sel(basin=basin).to_dataframe()
            for freq in self.frequencies:
                # make sure that possible mass inputs are sorted to the beginning of the dynamic feature list
                if isinstance(self.cfg.dynamic_inputs, list):
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs
                else:
                    dynamic_cols = self.cfg.mass_inputs + self.cfg.dynamic_inputs[freq]

                df_resampled = df_native[dynamic_cols + self.cfg.target_variables + self.cfg.evolving_attributes +
                                         self.cfg.autoregressive_inputs].resample(freq).mean()

                # pull all of the data that needs to be validated
                x_d[freq] = df_resampled[dynamic_cols].values
                x_d_column_names = dynamic_cols
                y[freq] = df_resampled[self.cfg.target_variables].values
                if self.cfg.evolving_attributes:
                    x_s[freq] = df_resampled[self.cfg.evolving_attributes].values

                # Add dates of the (resampled) data to the dates dict
                dates[freq] = df_resampled.index.to_numpy()

                # number of frequency steps in one lowest-frequency step
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                # array position i is the last entry of this frequency that belongs to the lowest-frequency sample i.
                if len(df_resampled) % frequency_factor != 0:
                    raise ValueError(f'The length of the dataframe at frequency {freq} is {len(df_resampled)} '
                                     f'(including warmup), which is not a multiple of {frequency_factor} (i.e., the '
                                     f'factor between the lowest frequency {lowest_freq} and the frequency {freq}. '
                                     f'To fix this, adjust the {self.period} start or end date such that the period '
                                     f'(including warmup) has a length that is divisible by {frequency_factor}.')
                frequency_maps[freq] = np.arange(len(df_resampled) // frequency_factor) \
                                       * frequency_factor + (frequency_factor - 1)

            # store first date of sequence to be able to restore dates during inference
            if not self.is_train:
                self.period_starts[basin] = pd.to_datetime(xr.sel(basin=basin)["date"].values[0])

            # we can ignore the deprecation warning about lists because we don't use the passed lists
            # after the validate_samples call. The alternative numba.typed.Lists is still experimental.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

                # checks inputs and outputs for each sequence. valid: flag = 1, invalid: flag = 0
                # manually unroll the dicts into lists to make sure the order of frequencies is consistent.
                # during inference, we want all samples with sufficient history (even if input is NaN), so
                # we pass x_d, x_s, y as None.
                flag = validate_samples(x_d=[x_d[freq] for freq in self.frequencies] if self.is_train else None,
                                        x_s=[x_s[freq] for freq in self.frequencies] if self.is_train and x_s else None,
                                        y=[y[freq] for freq in self.frequencies] if self.is_train else None,
                                        frequency_maps=[frequency_maps[freq] for freq in self.frequencies],
                                        seq_length=self.seq_len,
                                        predict_last_n=self._predict_last_n)

            # Concatenate autoregressive columns to dynamic inputs *after* validation, so as to not remove
            # samples with missing autoregressive inputs.
            # AR inputs must go at the end of the df/array (this is assumed by the AR model).
            if self.cfg.autoregressive_inputs:
                for freq in self.frequencies:
                    x_d[freq] = np.concatenate([x_d[freq], df_resampled[self.cfg.autoregressive_inputs].values], axis=1)
                x_d_column_names += self.cfg.autoregressive_inputs

            valid_samples = np.argwhere(flag == 1)
            for f in valid_samples:
                # store pointer to basin and the sample's index in each frequency
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            # only store data if this basin has at least one valid sample in the given period
            if valid_samples.size > 0:
                if self.cfg.forecast_inputs:
                    if not self.cfg.hindcast_inputs:
                        raise ValueError('Hindcast inputs must be provided if forecast inputs are provided.')
                    hindcast_indexes = [idx for idx, variable in enumerate(x_d_column_names) if variable in self.cfg.hindcast_inputs]
                    forecast_indexes = [idx for idx, variable in enumerate(x_d_column_names) if variable in self.cfg.forecast_inputs]
                    self._x_h[basin] = {freq: torch.from_numpy(_x_d[:, hindcast_indexes].astype(np.float32)) for freq, _x_d in x_d.items()}
                    self._x_f[basin] = {freq: torch.from_numpy(_x_d[:, forecast_indexes].astype(np.float32)) for freq, _x_d in x_d.items()}
                else:
                    self._x_d[basin] = {freq: torch.from_numpy(_x_d.astype(np.float32)) for freq, _x_d in x_d.items()}
                self._y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                if x_s:
                    self._x_s[basin] = {freq: torch.from_numpy(_x_s.astype(np.float32)) for freq, _x_s in x_s.items()}
                self._dates[basin] = dates
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(
                f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

        if self.num_samples == 0:
            if self.is_train:
                raise NoTrainDataError
            else:
                raise NoEvaluationDataError

    def _load_hydroatlas_attributes(self):
        df = utils.load_hydroatlas_attributes(self.cfg.data_dir, basins=self.basins)

        # remove all attributes not defined in the config
        drop_cols = [c for c in df.columns if c not in self.cfg.hydroatlas_attributes]
        df = df.drop(drop_cols, axis=1)

        if self.is_train:
            # sanity check attributes for NaN in per-feature standard deviation
            utils.attributes_sanity_check(df=df)

        return df

    def _load_combined_attributes(self):
        """This function loads data set specific attributes and combines them with hydroatlas attributes"""
        dfs = []

        # load dataset specific attributes from the subclass
        if self.cfg.static_attributes:
            df = self._load_attributes()

            # remove all attributes not defined in the config
            missing_attrs = [attr for attr in self.cfg.static_attributes if attr not in df.columns]
            if len(missing_attrs) > 0:
                raise ValueError(f'Static attributes {missing_attrs} are missing.')
            df = df[self.cfg.static_attributes]

            # in case of training (not finetuning) check for NaNs in feature std.
            if self._compute_scaler:
                utils.attributes_sanity_check(df=df)

            dfs.append(df)

        # Hydroatlas attributes can be used everywhere
        if self.cfg.hydroatlas_attributes:
            dfs.append(self._load_hydroatlas_attributes())

        if dfs:
            # combine all attributes into a single dataframe
            df = pd.concat(dfs, axis=1)

            # check if any attribute specified in the config is not available in the dataframes
            combined_attributes = self.cfg.static_attributes + self.cfg.hydroatlas_attributes
            missing_columns = [attr for attr in combined_attributes if attr not in df.columns]
            if missing_columns:
                raise ValueError(f"The following attributes are not available in the dataset: {missing_columns}")

            # fix the order of the columns to be alphabetically
            df = df.sort_index(axis=1)

            # calculate statistics and normalize features
            if self._compute_scaler:
                self.scaler["attribute_means"] = df.mean()
                self.scaler["attribute_stds"] = df.std()

            if any([k.startswith("camels_attr") for k in self.scaler.keys()]):
                LOGGER.warning(
                    "Deprecation warning: Using old scaler files won't be supported in the upcoming release.")

                # Here we assume that only camels attributes are used
                df = (df - self.scaler['camels_attr_means']) / self.scaler["camels_attr_stds"]
            else:
                df = (df - self.scaler['attribute_means']) / self.scaler["attribute_stds"]

            # preprocess each basin feature vector as pytorch tensor
            for basin in self.basins:
                attributes = df.loc[df.index == basin].values.flatten()
                self._attributes[basin] = torch.from_numpy(attributes.astype(np.float32))

    def _load_data(self):
        # load attributes first to sanity-check those features before doing the compute expensive time series loading
        self._load_combined_attributes()

        xr = self._load_or_create_xarray_dataset()

        if self.cfg.loss.lower() in ['nse', 'weightednse']:
            # get the std of the discharge for each basin, which is needed for the (weighted) NSE loss.
            self._calculate_per_basin_std(xr)

        if self._compute_scaler:
            # get feature-wise center and scale values for the feature normalization
            self._setup_normalization(xr)

        # performs normalization
        xr = (xr - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"]

        self._create_lookup_table(xr)

    def _setup_normalization(self, xr: xarray.Dataset):
        # default center and scale values are feature mean and std
        self.scaler["xarray_feature_scale"] = xr.std(skipna=True)
        self.scaler["xarray_feature_center"] = xr.mean(skipna=True)

        # check for feature-wise custom normalization
        for feature, feature_specs in self.cfg.custom_normalization.items():
            for key, val in feature_specs.items():
                # check for custom treatment of the feature center
                if key == "centering":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["xarray_feature_center"][feature] = np.float32(0.0)
                    elif val.lower() == "median":
                        self.scaler["xarray_feature_center"][feature] = xr[feature].median(skipna=True)
                    elif val.lower() == "min":
                        self.scaler["xarray_feature_center"][feature] = xr[feature].min(skipna=True)
                    elif val.lower() == "mean":
                        # Do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown centering method {val}")

                # check for custom treatment of the feature scale
                elif key == "scaling":
                    if (val is None) or (val.lower() == "none"):
                        self.scaler["xarray_feature_scale"][feature] = np.float32(1.0)
                    elif val == "minmax":
                        self.scaler["xarray_feature_scale"][feature] = xr[feature].max(skipna=True) - \
                                                                       xr[feature].min(skipna=True)
                    elif val == "std":
                        # Do nothing, since this is the default
                        pass
                    else:
                        raise ValueError(f"Unknown scaling method {val}")
                else:
                    # raise ValueError to point to the correct argument names
                    raise ValueError("Unknown dict key. Use 'centering' and/or 'scaling' for each feature.")

    def get_period_start(self, basin: str) -> pd.Timestamp:
        """Return the first date in the period for a given basin
        
        Parameters
        ----------
        basin : str
            The basin id

        Returns
        -------
        pd.Timestamp
            First date in the period for the specific basin. Necessary during evaluation to restore the dates.
        """
        return self.period_starts[basin]

    def _initialize_frequency_configuration(self):
        """Checks and extracts configuration values for 'use_frequency', 'seq_length', and 'predict_last_n'"""

        # If use_frequencies is not supplied, we'll fill it with the native frequency while loading the df.
        self.frequencies = self.cfg.use_frequencies

        self.seq_len = self.cfg.seq_length
        self._predict_last_n = self.cfg.predict_last_n
        if not self.frequencies:
            if not isinstance(self.seq_len, int) or not isinstance(self._predict_last_n, int):
                raise ValueError('seq_length and predict_last_n must be integers if use_frequencies is not provided.')
            self.seq_len = [self.seq_len]
            self._predict_last_n = [self._predict_last_n]
        else:
            # flatten per-frequency dictionaries into lists that are ordered as use_frequencies
            if not isinstance(self.seq_len, dict) \
                    or not isinstance(self._predict_last_n, dict) \
                    or any([freq not in self.seq_len for freq in self.frequencies]) \
                    or any([freq not in self._predict_last_n for freq in self.frequencies]):
                raise ValueError('seq_length and predict_last_n must be dictionaries with one key per frequency.')
            self.seq_len = [self.seq_len[freq] for freq in self.frequencies]
            self._predict_last_n = [self._predict_last_n[freq] for freq in self.frequencies]

    @staticmethod
    def collate_fn(
            samples: List[Dict[str, Union[torch.Tensor, np.ndarray]]]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith('date'):
                # Dates are stored as a numpy array of datetime64, which we maintain as numpy array.
                batch[feature] = np.stack([sample[feature] for sample in samples], axis=0)
            else:
                # Everything else is a torch.Tensor
                batch[feature] = torch.stack([sample[feature] for sample in samples], dim=0)
        return batch


@njit()
def validate_samples(x_d: List[np.ndarray], x_s: List[np.ndarray], y: List[np.ndarray], seq_length: List[int],
                     predict_last_n: List[int], frequency_maps: List[np.ndarray]) -> np.ndarray:
    """Checks for invalid samples due to NaN or insufficient sequence length.

    Parameters
    ----------
    x_d : List[np.ndarray]
        List of dynamic input data; one entry per frequency
    x_s : List[np.ndarray]
        List of additional static input data; one entry per frequency
    y : List[np.ndarray]
        List of target values; one entry per frequency
    seq_length : List[int]
        List of sequence lengths; one entry per frequency
    predict_last_n: List[int]
        List of predict_last_n; one entry per frequency
    frequency_maps : List[np.ndarray]
        List of arrays mapping lowest-frequency samples to their corresponding last sample in each frequency;
         one list entry per frequency.

    Returns
    -------
    np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.
    """
    # number of samples is number of lowest-frequency samples (all maps have this length)
    n_samples = len(frequency_maps[0])

    # 1 denote valid sample, 0 denote invalid sample
    flag = np.ones(n_samples)
    for i in range(len(frequency_maps)):  # iterate through frequencies
        for j in prange(n_samples):  # iterate through lowest-frequency samples
            # find the last sample in this frequency that belongs to the lowest-frequency step j
            last_sample_of_freq = frequency_maps[i][j]
            if last_sample_of_freq < seq_length[i] - 1:
                flag[j] = 0  # too early for this frequency's seq_length (not enough history)
                continue

            # any NaN in the dynamic inputs makes the sample invalid
            if x_d is not None:
                _x_d = x_d[i][last_sample_of_freq - seq_length[i] + 1:last_sample_of_freq + 1]
                if np.any(np.isnan(_x_d)):
                    flag[j] = 0
                    continue

            # all-NaN in the targets makes the sample invalid
            if y is not None:
                _y = y[i][last_sample_of_freq - predict_last_n[i] + 1:last_sample_of_freq + 1]
                if np.prod(np.array(_y.shape)) > 0 and np.all(np.isnan(_y)):
                    flag[j] = 0
                    continue

            # any NaN in the static features makes the sample invalid
            if x_s is not None:
                _x_s = x_s[i][last_sample_of_freq]
                if np.any(np.isnan(_x_s)):
                    flag[j] = 0

    return flag
