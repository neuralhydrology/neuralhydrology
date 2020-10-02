import logging
import pickle

import numpy as np
import pandas as pd

from neuralhydrology.data import utils, CamelsUS
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class HourlyCamelsUS(CamelsUS):
    """Data set class providing hourly data for CAMELS US basins.
    
    This class extends the `CamelsUS` dataset class by hourly in- and output data. Currently, only NLDAS forcings are
    available at an hourly resolution.

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
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'static_inputs' and
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
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._netcdf_dataset = None  # if available, we remember the dataset to load faster
        self._warn_slow_loading = True
        super(HourlyCamelsUS, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        if not any(f.endswith('_hourly') for f in self.cfg.forcings):
            raise ValueError('Forcings include no hourly forcings set.')
        for forcing in self.cfg.forcings:
            if forcing[-7:] == '_hourly':
                df = self.load_hourly_data(basin, forcing)
            else:
                # load daily CAMELS forcings and upsample to hourly
                df, _ = utils.load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
                df = df.resample('1H').ffill()
            if len(self.cfg.forcings) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        # add stage, if requested
        if 'gauge_height_m' in self.cfg.target_variables:
            df = df.join(utils.load_hourly_us_stage(self.cfg.data_dir, basin))
            df.loc[df['gauge_height_m'] < 0, 'gauge_height_m'] = np.nan

        # convert discharge to 'synthetic' stage, if requested
        if 'synthetic_qobs_stage_meters' in self.cfg.target_variables:
            attributes = utils.load_camels_us_attributes(data_dir=self.cfg.data_dir, basins=[basin])
            with open(self.cfg.rating_curve_file, 'rb') as f:
                rating_curves = pickle.load(f)
            df['synthetic_qobs_stage_meters'] = np.nan
            if basin in rating_curves.keys():
                discharge_m3s = df['qobs_mm_per_hour'].values / 1000 * attributes.area_gages2[basin] * 1e6 / 60**2
                df['synthetic_qobs_stage_meters'] = rating_curves[basin].discharge_to_stage(discharge_m3s)

        return df

    def load_hourly_data(self, basin: str, forcings: str) -> pd.DataFrame:
        """Load a single set of hourly forcings and discharge. If available, loads from NetCDF, else from csv.
        
        Parameters
        ----------
        basin : str
            Identifier of the basin for which to load data.
        forcings : str
            Name of the forcings set to load.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with forcings and discharge values for the specified basin.
        """
        fallback_csv = False
        try:
            if self._netcdf_dataset is None:
                self._netcdf_dataset = utils.load_hourly_us_netcdf(self.cfg.data_dir, forcings)
            df = self._netcdf_dataset.sel(basin=basin).to_dataframe()
        except FileNotFoundError:
            fallback_csv = True
            if self._warn_slow_loading:
                LOGGER.warning('## Warning: Hourly NetCDF file not found. Falling back to slower csv files.')
                self._warn_slow_loading = False  # only warn once
        except KeyError:
            fallback_csv = True
            LOGGER.warning(f'## Warning: NetCDF file does not contain data for {basin}. Trying slower csv files.')
        if fallback_csv:
            df = utils.load_hourly_us_forcings(self.cfg.data_dir, basin, forcings)

            # add discharge
            df = df.join(utils.load_hourly_us_discharge(self.cfg.data_dir, basin))

        return df
