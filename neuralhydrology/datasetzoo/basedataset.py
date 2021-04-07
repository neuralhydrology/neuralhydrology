import logging
import pickle
import sys
import warnings
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from pandas.tseries import frequencies
from pandas.tseries.frequencies import to_offset
import torch
import xarray
from numba import NumbaPendingDeprecationWarning
from numba import njit, prange
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config

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
        If period is either 'validation' or 'test', this input is required. It contains the means and standard 
        deviations for each feature and is stored to the run directory during training (train_data/train_data_scaler.p)
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
        self.x_d = {}
        self.x_s = {}
        self.attributes = {}
        self.y = {}
        self.per_basin_target_stds = {}
        self.dates = {}
        self.num_samples = 0
        self.one_hot = None
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

            # create empty tensor of the same length as basins in id to int lookup table
            self.one_hot = torch.zeros(len(self.id_to_int), dtype=torch.float32)

        # load and preprocess data
        self._load_data()

        if self.is_train:
            self._dump_scaler()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        basin, indices = self.lookup_table[item]

        # This check is for multiple periods per basin, where we add '_periodX' to the basin name
        # For catchment attributes, one-hot-encoding we need the raw basin_id.
        if basin.split('_')[-1].startswith('period'):
            basin_id = "_".join(basin.split('_')[:-1])
        else:
            basin_id = basin

        sample = {}
        for freq, seq_len, idx in zip(self.frequencies, self.seq_len, indices):
            # if there's just one frequency, don't use suffixes.
            freq_suffix = '' if len(self.frequencies) == 1 else f'_{freq}'
            # slice until idx + 1 because slice-end is excluding
            sample[f'x_d{freq_suffix}'] = self.x_d[basin][freq][idx - seq_len + 1:idx + 1]
            sample[f'y{freq_suffix}'] = self.y[basin][freq][idx - seq_len + 1:idx + 1]

            # check for static inputs
            static_inputs = []
            if self.attributes:
                static_inputs.append(self.attributes[basin_id])
            if self.x_s:
                static_inputs.append(self.x_s[basin][freq][idx])
            if static_inputs:
                sample[f'x_s{freq_suffix}'] = torch.cat(static_inputs, dim=-1)

        if self.per_basin_target_stds:
            sample['per_basin_target_stds'] = self.per_basin_target_stds[basin]
        if self.one_hot is not None:
            x_one_hot = self.one_hot.zero_()
            x_one_hot[self.id_to_int[basin_id]] = 1
            sample['x_one_hot'] = x_one_hot

        return sample

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError

    def _load_attributes(self) -> pd.DataFrame:
        """This function has to return the attributes in a basin-indexed DataFrame."""
        raise NotImplementedError

    def _create_id_to_int(self):
        self.id_to_int = {b: i for i, b in enumerate(np.random.permutation(self.basins))}

        # dump id_to_int dictionary into run directory for validation
        file_path = self.cfg.train_dir / "id_to_int.p"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as fp:
            pickle.dump(self.id_to_int, fp)

    def _dump_scaler(self):
        # dump scaler dictionary into run directory for validation
        file_path = self.cfg.train_dir / "train_data_scaler.p"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as fp:
            pickle.dump(self.scaler, fp)

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

            self.dates = {b: {'start_dates': start_dates, 'end_dates': end_dates} for b in self.basins}

        # read periods from file
        else:
            with open(getattr(self.cfg, f"per_basin_{self.period}_periods_file"), 'rb') as fp:
                self.dates = pickle.load(fp)

    def _load_additional_features(self):
        for file in self.cfg.additional_feature_files:
            with open(file, "rb") as fp:
                self.additional_features.append(pickle.load(fp))

    def _duplicate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature, n_duplicates in self.cfg.duplicate_features.items():
            for n in range(1, n_duplicates + 1):
                df[f"{feature}_copy{n}"] = df[feature]

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        # if no netCDF file is passed, data set is created from raw basin files
        if (self.cfg.train_data_file is None) or (not self.is_train):
            data_list = []

            # list of columns to keep, everything else will be removed to reduce memory footprint
            keep_cols = self.cfg.target_variables + self.cfg.evolving_attributes + self.cfg.mass_inputs

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

                # check if any feature should be duplicated
                df = self._duplicate_features(df)

                # check if a shifted copy of a feature should be added
                df = self._add_lagged_features(df)

                # remove unnecessary columns
                df = df[keep_cols]

                # make end_date the last second of the specified day, such that the
                # dataset will include all hours of the last day, not just 00:00.
                start_dates = self.dates[basin]["start_dates"]
                end_dates = [date + pd.Timedelta(days=1, seconds=-1) for date in self.dates[basin]["end_dates"]]

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

                    # For multiple slices per basin, a number is added to the basin string starting from the 2nd slice
                    xr = xarray.Dataset.from_dataframe(df_sub)
                    basin_str = basin if i == 0 else f"{basin}_period{i}"
                    xr = xr.assign_coords({'basin': basin_str})
                    data_list.append(xr.astype(np.float32))

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
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):
            coords = [b for b in basin_coordinates if b == basin or b.startswith(f"{basin}_period")]
            if len(coords) > 0:
                # gather all observations from different split periods and concatenate them into one array but
                # take care of multi targets
                obs = []
                for coord in coords:
                    # add arrays of shape [targets, time steps, 1]. Note, even with split periods of different length
                    # the xarray data arrays are of same length (filled with NaNs), so concatenation later works.
                    obs.append(xr.sel(basin=coord)[self.cfg.target_variables].to_array().values[:, :, np.newaxis])
                # concat to shape [targets, time steps, periods], then reshape to [targets, time steps * periods]
                obs = np.concatenate(obs, axis=-1).reshape(len(self.cfg.target_variables), -1)
                if np.sum(~np.isnan(obs)) > 2:
                    # calculate std for each target
                    per_basin_target_stds = torch.tensor([np.nanstd(obs, axis=1)], dtype=torch.float32)
                    # we store duplicates of the std for each coordinate of the same basin, so we are faster in getitem
                    for coord in coords:
                        self.per_basin_target_stds[coord] = per_basin_target_stds

    def _create_lookup_table(self, xr: xarray.Dataset):
        lookup = []
        if not self._disable_pbar:
            LOGGER.info("Create lookup table and convert to pytorch tensor")

        # list to collect basins ids of basins without a single training sample
        basins_without_samples = []
        basin_coordinates = xr["basin"].values.tolist()
        for basin in tqdm(basin_coordinates, file=sys.stdout, disable=self._disable_pbar):

            # store data of each frequency as numpy array of shape [time steps, features]
            x_d, x_s, y = {}, {}, {}

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

                df_resampled = df_native[dynamic_cols + self.cfg.target_variables +
                                         self.cfg.evolving_attributes].resample(freq).mean()
                x_d[freq] = df_resampled[dynamic_cols].values
                y[freq] = df_resampled[self.cfg.target_variables].values
                if self.cfg.evolving_attributes:
                    x_s[freq] = df_resampled[self.cfg.evolving_attributes].values

                # number of frequency steps in one lowest-frequency step
                frequency_factor = int(utils.get_frequency_factor(lowest_freq, freq))
                # array position i is the last entry of this frequency that belongs to the lowest-frequency sample i.
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
            valid_samples = np.argwhere(flag == 1)
            for f in valid_samples:
                # store pointer to basin and the sample's index in each frequency
                lookup.append((basin, [frequency_maps[freq][int(f)] for freq in self.frequencies]))

            # only store data if this basin has at least one valid sample in the given period
            if valid_samples.size > 0:
                self.x_d[basin] = {freq: torch.from_numpy(_x_d.astype(np.float32)) for freq, _x_d in x_d.items()}
                self.y[basin] = {freq: torch.from_numpy(_y.astype(np.float32)) for freq, _y in y.items()}
                if x_s:
                    self.x_s[basin] = {freq: torch.from_numpy(_x_s.astype(np.float32)) for freq, _x_s in x_s.items()}
            else:
                basins_without_samples.append(basin)

        if basins_without_samples:
            LOGGER.info(
                f"These basins do not have a single valid sample in the {self.period} period: {basins_without_samples}")
        self.lookup_table = {i: elem for i, elem in enumerate(lookup)}
        self.num_samples = len(self.lookup_table)

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
                self.attributes[basin] = torch.from_numpy(attributes.astype(np.float32))

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
                                                                       xr[feature].m(skipna=True)
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
