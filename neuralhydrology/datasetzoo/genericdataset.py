from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class GenericDataset(BaseDataset):
    """Data set class for the generic dataset that reads data for any region based on common file layout conventions.

    To use this dataset, the data_dir must contain a folder 'time_series' and (if static attributes are used) a folder
    'attributes'. The folder 'time_series' contains one netcdf file (.nc or .nc4) per basin, named '<basin_id>.nc/nc4'.
    The netcdf file has to have one coordinate called `date`, containing the datetime index. The folder 'attributes' 
    contains one or more comma-separated file (.csv) with static attributes, indexed by basin id. Attributes files can 
    be divided into groups of basins or groups of features (but not both).

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
        super(GenericDataset, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data. """
        df = load_timeseries(data_dir=self.cfg.data_dir, basin=basin)

        return df

    def _load_attributes(self):
        return load_attributes(self.cfg.data_dir, basins=self.basins)


def load_attributes(data_dir: Path, basins: List[str] = None) -> pd.DataFrame:
    """Load static attributes.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory. This folder must contain an 'attributes' folder with one or multiple csv files.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of
        all basins are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns. If the attributes folder contains multiple
        files, they will be concatenated along the mismatching axis. I.e., attribute files may be split by basin
        groups XOR split by attributes (but not both).
    """
    attributes_path = data_dir / 'attributes'
    if not attributes_path.exists():
        raise RuntimeError(f"Attributes folder not found at {attributes_path}")

    files = list(attributes_path.glob('*.csv'))
    if len(files) == 0:
        raise ValueError('No attributes files found')

    # Read-in attributes into one big dataframe. Sort by both axes so we can check for identical axes.
    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={0: str})  # make sure we read the basin id as str
        df = df.set_index(df.columns[0]).sort_index(axis=0).sort_index(axis=1)
        if df.index.has_duplicates or df.index.has_duplicates:
            raise ValueError(f'Attributes file {f} contains duplicate basin ids or features.')
        dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        if np.all([len(dfs[i]) == len(dfs[i + 1]) and dfs[i].index == dfs[i + 1].index for i in range(len(dfs) - 1)]):
            # same basins -> concatenate attributes
            if np.any(np.unique(np.concatenate([df.columns for df in dfs]), return_counts=True)[1] > 1):
                raise ValueError('If attributes dataframes refer to the same basins, no attribute name may occur '
                                 'multiple times across the different attributes files.')
            concat_axis = 1
        elif np.all([
                dfs[i].shape[1] == dfs[i + 1].shape[1] and dfs[i].columns == dfs[i + 1].columns
                for i in range(len(dfs) - 1)
        ]):
            # same attributes -> concatenate basins
            if np.any(np.unique(np.concatenate([df.index for df in dfs]), return_counts=True)[1] > 1):
                raise ValueError('If attributes dataframes refer to different basins, no basin id name may occur '
                                 'multiple times across the different attributes files.')
            concat_axis = 0
        else:
            raise ValueError('Attribute files must have either same index or same columns.')

        df = pd.concat(dfs, axis=concat_axis)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def load_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load time series data from netCDF files into pandas DataFrame.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory. This folder must contain a folder called 'time_series' containing the time series
        data for each basin as a single time-indexed netCDF file called '<basin_id>.nc/nc4'.
    basin : str
        The basin identifier.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame containing the time series data as stored in the netCDF file.
    """
    files_dir = data_dir / "time_series"
    netcdf_files = list(files_dir.glob("*.nc4"))
    netcdf_files.extend(files_dir.glob("*.nc"))
    netcdf_file = [f for f in netcdf_files if f.stem == basin]
    if len(netcdf_file) == 0:
        raise FileNotFoundError(f"No netCDF file found for basin {basin} in {files_dir}")
    if len(netcdf_file) > 1:
        raise ValueError(f"Multiple netCDF files found for basin {basin} in {files_dir}")

    xr = xarray.open_dataset(netcdf_file[0])
    return xr.to_dataframe()
