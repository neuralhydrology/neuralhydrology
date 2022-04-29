from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import xarray
from tqdm import tqdm

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

_CAMELS_BR_TIMESERIES_SUBDIRS = [
    '03_CAMELS_BR_streamflow_mm_selected_catchments',
    '04_CAMELS_BR_streamflow_simulated',
    '05_CAMELS_BR_precipitation_chirps',
    '06_CAMELS_BR_precipitation_mswep',
    '07_CAMELS_BR_precipitation_cpc',
    '08_CAMELS_BR_evapotransp_gleam',
    '09_CAMELS_BR_evapotransp_mgb',
    '10_CAMELS_BR_potential_evapotransp_gleam',
    '11_CAMELS_BR_temperature_min_cpc',
    '12_CAMELS_BR_temperature_mean_cpc',
    '13_CAMELS_BR_temperature_max_cpc'
]

class CamelsBR(BaseDataset):
    """Data set class for the CAMELS-BR dataset by [#]_.

    For more efficient data loading during model training/evaluating, this dataset class expects the CAMELS-BR dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined. Use the :func:`preprocess_camels_br_dataset` function 
    to process the original dataset layout into this format.

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
    .. [#] Chagas, V. B. P., Chaffe, P. L. B., Addor, N., Fan, F. M., Fleischmann, A. S., Paiva, R. C. D., and Siqueira,
        V. A.: CAMELS-BR: hydrometeorological time series and landscape attributes for 897 catchments in Brazil, Earth 
        Syst. Sci. Data, 12, 2075-2096, https://doi.org/10.5194/essd-12-2075-2020, 2020.  
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CamelsBR, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        return load_camels_br_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_camels_br_attributes(self.cfg.data_dir, basins=self.basins)


def load_camels_br_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS-BR data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-BR directory. This folder must contain a folder called 'preprocessed' containing the 
        per-basin csv files created by :func:`preprocess_camels_br_dataset`.
    basin : str
        Basin identifier number as string.

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
            f"No preprocessed data directory found at {preprocessed_dir}. Use preprocessed_camels_br_dataset in ",
            "neuralhydrology.datasetzoo.camelsbr to preprocess the CAMELS-BR data set once into per-basin files."
        ]
        raise FileNotFoundError("".join(msg))
    basin_file = preprocessed_dir / f"{basin}.csv"
    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])
    return df


def load_camels_br_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS-BR attributes.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-BR directory. Assumes that the subdirectory 01_CAMELS_BR_attributes is located in the
        data directory root folder.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_path = Path(data_dir) / '01_CAMELS_BR_attributes'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_br_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=' ', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def preprocess_camels_br_dataset(data_dir: Path):
    """Preprocess CAMELS-BR data set and create per-basin files for more flexible and faster data loading.
    
    This function will read-in all time series text files and create per-basin csv files containing all timeseries 
    features at once in a new subfolder called "preprocessed". Will only consider the 897 basin for which streamflow and
    forcings exist. Note that simulated streamflow only exists for 593 out of 897 basins.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS-BR data set containing the different subdirectories that can be downloaded as individual zip
        archives.

    Raises
    ------
    FileExistsError
        If a sub-folder called 'preprocessed' already exists in `data_dir`.
    FileNotFoundError
        If any of the subdirectories of CAMELS-BR is not found in `data_dir`, specifically the folders starting with 
        `03_*` up to `13_*`.
    """
    # check if data has already been pre-processed other-wise create dst folder
    dst_dir = data_dir / "preprocessed"
    if dst_dir.is_dir():
        raise FileExistsError(
            "Subdirectory 'preprocessed' already exists. Delete this folder if you want to reprocess the data.")
    dst_dir.mkdir()

    # Streamflow and forcing data are stored in different subdirectories that start with a numeric value each. The first
    # one is streamflow mm/d starting with 03 and the last is max temp starting with 13.
    timeseries_folders = [data_dir / subdirectory for subdirectory in _CAMELS_BR_TIMESERIES_SUBDIRS]
    if any([not p.is_dir() for p in timeseries_folders]):
        missing_subdirectories = [p.name for p in timeseries_folders if not p.is_dir()]
        raise FileNotFoundError(
            f"The following directories were expected in {data_dir} but do not exist: {missing_subdirectories}")

    # Since files is sorted, we can pick the first one, streamflow, and extract the basins names from there
    basins = [x.stem.split('_')[0] for x in timeseries_folders[0].glob('*.txt')]
    print(f"Found {len(basins)} basin files under {timeseries_folders[0].name}")

    for basin in tqdm(basins, desc="Combining timeseries data from different subdirectories into one file per basin"):
        data = {}
        for timeseries_folder in timeseries_folders:
            basin_file = list(timeseries_folder.glob(f'{basin}_*'))
            if basin_file:
                df = pd.read_csv(basin_file[0], sep=' ')
                df["date"] = pd.to_datetime(df.year.map(str) + "/" + df.month.map(str) + "/" + df.day.map(str),
                                            format="%Y/%m/%d")
                df = df.set_index('date')
                feat_col = [c for c in df.columns if c not in ['year', 'month', 'day']][0]
                data[feat_col] = df[feat_col]
        df = pd.DataFrame(data)
        df.to_csv(dst_dir / f"{basin}.csv")

    print(f"Finished processing the CAMELS-BR data set. Resulting per-basin csv files have been stored at {dst_dir}")
