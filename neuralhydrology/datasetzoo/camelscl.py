import sys
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray
from tqdm import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CamelsCL(BaseDataset):
    """Data set class for the CAMELS CL dataset by [#]_.

    For more efficient data loading during model training/evaluating, this dataset class expects the CAMELS-CL dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined. Use the :func:`preprocess_camels_cl_dataset` function to process the original dataset 
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
    .. [#] Alvarez-Garreton, C., Mendoza, P. A., Boisier, J. P., Addor, N., Galleguillos, M., Zambrano-Bigiarini, M.,
        Lara, A., Puelma, C., Cortes, G., Garreaud, R., McPhee, J., and Ayala, A.: The CAMELS-CL dataset: catchment
        attributes and meteorology for large sample studies - Chile dataset, Hydrol. Earth Syst. Sci., 22, 5817-5846,
        https://doi.org/10.5194/hess-22-5817-2018, 2018.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsCL, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        return load_camels_cl_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_camels_cl_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_cl_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS CL data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS CL directory. This folder must contain a folder called 'preprocessed' containing the 
        per-basin csv files created by :func:`preprocess_camels_cl_dataset`.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
        
    Raises
    ------
    FileNotFoundError
        If no sub-folder called 'preprocessed' exists within the root directory of the CAMELS CL dataset.
    """
    preprocessed_dir = data_dir / "preprocessed"
    if not preprocessed_dir.is_dir():
        msg = [
            f"No preprocessed data directory found at {preprocessed_dir}. Use preprocessed_camels_cl_dataset in ",
            "neuralhydrology.datasetzoo.camelscl to preprocess the CAMELS CL data set once into per-basin files."
        ]
        raise FileNotFoundError("".join(msg))
    basin_file = preprocessed_dir / f"{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    return df


def load_camels_cl_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS CL attributes

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS CL directory. Assumes that a file called '1_CAMELScl_attributes.txt' exists.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_file = data_dir / '1_CAMELScl_attributes.txt'

    df = pd.read_csv(attributes_file, sep="\t", index_col="gauge_id").transpose()

    # convert all columns, where possible, to numeric
    df = df.apply(pd.to_numeric, errors='ignore')

    # convert the two columns specifying record period start and end to datetime format
    df["record_period_start"] = pd.to_datetime(df["record_period_start"])
    df["record_period_end"] = pd.to_datetime(df["record_period_end"])

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def preprocess_camels_cl_dataset(data_dir: Path):
    """Preprocess CAMELS-CL data set and create per-basin files for more flexible and faster data loading.
    
    This function will read-in all daily time series csv files and create per-basin csv files in a new subfolder called
    "preprocessed". This code is specifically designed for the "CAMELS-CL versión 2022 enero" version that can be 
    downloaded from `here <https://www.cr2.cl/camels-cl/>`__.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-CL data set. All csv-files from the original dataset should be present in this folder. 

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

    # list of available time series features included in CAMELS-CL
    available_features = ['q_mm', 'precip', 'tmin', 'tmax', 'tmean', 'pet']

    # get list of text files for those features
    files = [f for f in list(data_dir.glob('*day.csv')) if any([x in f.name for x in available_features])]

    # read-in all text files as pandas dataframe
    dfs = {}
    for file in tqdm(files, file=sys.stdout, desc="Loading txt files into memory"):
        df = pd.read_csv(file, index_col="date", parse_dates=['date'])
        feature_name = file.stem.rsplit('_', maxsplit=1)[0]
        dfs[feature_name] = df

    # create one dataframe per basin with all features. Shorten record to period of valid entries
    basins = list(df.columns)
    for basin in tqdm(basins, file=sys.stdout, desc="Creating per-basin dataframes and saving to disk"):
        # collect basin columns from all feature dataframes.
        col_data, col_names = [], []
        for feature, feature_df in dfs.items():
            col_names.append(feature)
            col_data.append(feature_df[basin])
        df = pd.DataFrame({name: data for name, data in zip(col_names, col_data)})

        # remove all rows with NaNs, then reindex to have continuous data frames from first to last record
        df = df.dropna(axis=0, how="all")
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1]), fill_value=np.nan)

        # correct index column name to 'date' and save resulting dataframe to disk.
        df.index.name = "date"
        df.to_csv(dst_dir / f"{basin}.csv")

    print(f"Finished processing the CAMELS CL data set. Resulting per-basin csv files have been stored at {dst_dir}")
