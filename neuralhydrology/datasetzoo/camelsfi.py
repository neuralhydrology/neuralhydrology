# This code has been created by slightly modifying camelsgb.py, since CAMELS-FI uses similar directory structure and file naming convention as CAMELS-GB

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class CamelsFI(BaseDataset):
    """
    Dataset class for the CAMELS-FI dataset
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
    Seppä, I., Gonzales Inca, C., Uusikivi, J., and Alho, P.: CAMELS-FI: hydrometeorological time series and landscape properties for 320 catchments in Finland, Earth System Science Data, 18, 4745–4769, https://doi.org/10.5194/essd-18-4745-2026, 2026.
    """
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsFI, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        df = load_camels_fi_timeseries(data_dir=self.cfg.data_dir, basin=basin)

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_camels_fi_attributes(self.cfg.data_dir, basins=self.basins)

def load_camels_fi_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS-FI attributes from the dataset

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS FI directory. This folder must contain the corresponding 
        csv files for each attribute group (ending with _attributes.csv).
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
        
    Raises
    ------
    FileNotFoundError
        If no root directory exists.

    References
    ----------
    Seppä, I., Gonzales Inca, C., Uusikivi, J., and Alho, P.: CAMELS-FI: hydrometeorological time series and landscape properties for 320 catchments in Finland, Earth System Science Data, 18, 4745–4769, https://doi.org/10.5194/essd-18-4745-2026, 2026.
    """
    attributes_path = data_dir 

    if not attributes_path.exists():
        raise FileNotFoundError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('*_attributes.csv')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=',', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        
        dfs.append(df_temp)
        

    df = pd.concat(dfs, axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df

def load_camels_fi_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS FI data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS FI directory. This folder must contain a folder called 'timeseries' containing the forcing
        files for each basin as .csv file. The file names have to start with 'CAMELS_FI_hydromet_timeseries' and contain _{basin_id}_ after that.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    
    References
    ----------
    Seppä, I., Gonzales Inca, C., Uusikivi, J., and Alho, P.: CAMELS-FI: hydrometeorological time series and landscape properties for 320 catchments in Finland, Earth System Science Data, 18, 4745–4769, https://doi.org/10.5194/essd-18-4745-2026, 2026.
    """
    forcing_path = data_dir / 'timeseries'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('**/CAMELS_FI_hydromet_timeseries*.csv'))
    file_path = [f for f in files if f"_{basin}_" in f.name]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    df = pd.read_csv(file_path, sep=',', header=0, dtype={'date': str})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index("date")

    return df
