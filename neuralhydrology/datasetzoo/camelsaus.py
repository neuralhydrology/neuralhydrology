from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CamelsAUS(BaseDataset):
    """Data set class for the CAMELS-AUS dataset by [#]_.

    For more efficient data loading during model training/evaluating, this dataset class expects the CAMELS-AUS dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined. Use the :func:`preprocess_camels_aus_dataset` function to process the original dataset 
    layout into this format.

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
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    References
    ----------
    .. [#] Fowler, K. J. A., Acharya, S. C., Addor, N., Chou, C., and Peel, M. C.: CAMELS-AUS: hydrometeorological time
        series and landscape attributes for 222 catchments in Australia, Earth Syst. Sci. Data, 13, 3847-3867, 
        https://doi.org/10.5194/essd-13-3847-2021, 2021. 
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsAUS, self).__init__(cfg=cfg,
                                        is_train=is_train,
                                        period=period,
                                        basin=basin,
                                        additional_features=additional_features,
                                        id_to_int=id_to_int,
                                        scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        return load_camels_aus_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_camels_aus_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_aus_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS-AUS data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-AUS directory. This folder must contain a folder called 'preprocessed' containing the 
        per-basin csv files created by :func:`preprocess_camels_aus_dataset`.
    basin : str
        Basin identifier as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
        
    Raises
    ------
    FileNotFoundError
        If no sub-folder called 'preprocessed' exists within the root directory of the CAMELS-AUS dataset.
    """
    preprocessed_dir = data_dir / "preprocessed"
    if not preprocessed_dir.is_dir():
        msg = [
            f"No preprocessed data directory found at {preprocessed_dir}. Use preprocessed_camels_aus_dataset in ",
            "neuralhydrology.datasetzoo.camelsaus to preprocess the CAMELS-AUS data set once into per-basin files."
        ]
        raise FileNotFoundError("".join(msg))
    basin_file = preprocessed_dir / f"{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    return df


def load_camels_aus_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS-AUS attributes.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-AUS directory. Assumes that CAMELS_AUS_Attributes&Indices_MasterTable.csv is located in the
        data directory root folder.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_file = data_dir / 'CAMELS_AUS_Attributes&Indices_MasterTable.csv'

    df = pd.read_csv(attributes_file, index_col="station_id")

    # convert all columns, where possible, to numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    # convert the two columns specifying record period start and end to datetime format
    df["start_date"] = pd.to_datetime(df["start_date"], format="%Y%m%d")
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d")

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def preprocess_camels_aus_dataset(data_dir: Path):
    """Preprocess CAMELS-AUS data set and create per-basin files for more flexible and faster data loading.
    
    This function will read-in all time series text files and create per-basin csv files in a new subfolder called
    "preprocessed".
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-AUS data set. Expects different subfolders with the original names, specifically
        '05_hydrometeorology' and '03_streamflow'.

    Raises
    ------
    FileExistsError
        If a sub-folder called 'preprocessed' already exists in `data_dir`.
    """
    # check if data has already been pre-processed other-wise create dst folder
    dst_dir = data_dir / "preprocessed"
    if dst_dir.is_dir():
        raise FileExistsError(
            "Subdirectory 'preprocessed' already exists. Delete this folder if you want to reprocess the data.")
    dst_dir.mkdir()

    # Load all different forcing files into memory
    forcing_dir = data_dir / "05_hydrometeorology"
    files = [f for f in forcing_dir.glob('**/*.csv') if f.name != 'ClimaticIndices.csv']
    dfs = {}
    for f in tqdm(files, desc="Read meteorological forcing data into memory"):
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df.year.map(str) + "/" + df.month.map(str) + "/" + df.day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index('date')
        dfs[f.stem] = df

    # Load streamflow data into memory and replace invalid measurements (-99) with NaNs
    print("Read streamflow data into memory.")
    df = pd.read_csv(data_dir / "03_streamflow" / "streamflow_mmd.csv")
    df["date"] = pd.to_datetime(df.year.map(str) + "/" + df.month.map(str) + "/" + df.day.map(str), format="%Y/%m/%d")
    df = df.set_index('date')
    df[df < 0] = np.nan
    dfs["streamflow_mmd"] = df

    # Extract list of basins from column names
    basins = [c for c in dfs["streamflow_mmd"].columns if c not in ['year', 'month', 'day']]

    # Create per-basin dataframes by combining all forcing variables + streamflow and save to disk.
    for basin in tqdm(basins, desc="Create per-basin dataframes and save data to disk."):
        data = {}
        for key, df in dfs.items():
            data[key] = df[basin]
        df = pd.DataFrame(data)
        df.to_csv(dst_dir / f"{basin}.csv")

    print(f"Finished processing the CAMELS-AUS data set. Resulting per-basin csv files have been stored at {dst_dir}")
