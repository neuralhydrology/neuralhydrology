from pathlib import Path
from typing import List, Dict, Union, Optional

import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class Caravan(BaseDataset):
    """Data set class for the Caravan data set by [#]_.

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
    .. [#] Kratzert, F., Nearing, G., Addor, N., Erickson, T., Gauch, M., Gilon, O., Gudmundsson, L., Hassidim, A., 
        Klotz, D., Nevo, S., Shalev, G., & Matias, Y.. (2022). Caravan - A global community dataset for large-sample 
        hydrology. Accepted for publication in Nature Scientific Data. Preprint: 
        https://eartharxiv.org/repository/view/3345/, 2023
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(Caravan, self).__init__(cfg=cfg,
                                      is_train=is_train,
                                      period=period,
                                      basin=basin,
                                      additional_features=additional_features,
                                      id_to_int=id_to_int,
                                      scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load timeseries data from netcdf files."""
        return load_caravan_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load input and output data from text files."""
        return load_caravan_attributes(data_dir=self.cfg.data_dir)


def load_caravan_attributes(data_dir: Path,
                            basins: Optional[List[str]] = None,
                            subdataset: Optional[str] = None) -> pd.DataFrame:
    """Load the attributes of the Caravan dataset.

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of Caravan that has to include a sub-directory called 'attributes' which contain the 
        attributes of all sub-datasets in separate folders.
    basins : List[str], optional
        If passed, returns only attributes for the basins specified in this list. Otherwise, the attributes of all 
        basins are returned.
    subdataset : str, optional
        If passed, returns only the attributes of one sub-dataset. Otherwise, the attributes of all sub-datasets are 
        loaded.

    Raises
    ------
    FileNotFoundError
        If the requested sub-dataset does not exist or any sub-dataset for the requested basins is missing.
    ValueError
        If any of the requested basins does not exist in the attribute files or if both, basins and sub-dataset are 
        passed but at least one of the basins is not part of the corresponding sub-dataset.

    Returns
    -------
    pd.DataFrame
        A basin indexed DataFrame with all attributes as columns.
    """
    if subdataset:
        subdataset_dir = data_dir / "attributes" / subdataset
        if not subdataset_dir.is_dir():
            raise FileNotFoundError(f"No subdataset {subdataset} found at {subdataset_dir}.")
        subdataset_dirs = [subdataset_dir]

    else:
        subdataset_dirs = [d for d in (data_dir / "attributes").glob('*') if d.is_dir()]

    if basins:
        # Get list of unique sub datasets from the basin strings.
        subdataset_names = list(set(x.split('_')[0] for x in basins))

        # Make sure the subdatasets exist and are not in conflict with the subdataset argument, if passed.
        if subdataset:
            # subdataset_names is only allowed to be size 1 in this case.
            if len(subdataset_names) > 1 or subdataset_names[0] != subdataset:
                raise ValueError("At least one of the passed basins is not part of the passed subdataset.")
        else:
            # Check if all subdatasets exist.
            missing_subdatasets = [s for s in subdataset_names if not (data_dir / "attributes" / s).is_dir()]

            if missing_subdatasets:
                raise FileNotFoundError(f"Could not find subdataset directories for {missing_subdatasets}.")

        # Subset subdataset_dirs to only the required subsets.
        subdataset_dirs = [s for s in subdataset_dirs if s.name in subdataset_names]

    # Load all required attribute files.
    dfs = []
    for subdataset_dir in subdataset_dirs:
        dfs.append(_load_attribute_files_of_subdataset(subdataset_dir))

    # Merge all DataFrames along the basin index.
    df = pd.concat(dfs, axis=0)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        # Subset to only the requested basins.
        df = df.loc[basins]

    return df


def load_caravan_timeseries(data_dir: Path, basin: str, filetype: str = "netcdf") -> pd.DataFrame:
    """Loads the timeseries data of one basin from the Caravan dataset.
    
    Parameters
    ----------
    data_dir : Path
        Path to the root directory of Caravan that has to include a sub-directory called 'timeseries'. This 
        sub-directory has to contain another sub-directory called either 'csv' or 'netcdf', depending on the choice 
        of the filetype argument. By default, netCDF files are loaded from the 'netcdf' subdirectory.
    basin : str
        The Caravan gauge id string in the form of {subdataset_name}_{gauge_id}.
    filetype : str, optional
        Can be either 'csv' or 'netcdf'. Depending on this value, this function will load the timeseries data from the
        netcdf files (default) or csv files.

    Raises
    ------
    ValueError
        If filetype is not in ['csv', 'netcdf'].
    FileNotFoundError
        If no timeseries file exists for the basin.
    """
    # Get the subdataset name from the basin string.
    subdataset_name = basin.split('_')[0]

    if filetype == "netcdf":
        filepath = data_dir / "timeseries" / "netcdf" / subdataset_name / f"{basin}.nc"
    elif filetype == "csv":
        filepath = data_dir / "timeseries" / "csv" / subdataset_name / f"{basin}.csv"
    else:
        raise ValueError("filetype has to be either 'csv' or 'netcdf'.")

    if not filepath.is_file():
        raise FileNotFoundError(f"No basin file found at {filepath}.")

    # Load timeseries data.
    if filetype == "netcdf":
        df = xarray.open_dataset(filepath).to_dataframe()
    else:
        df = pd.read_csv(filepath, parse_dates=['date'])
        df = df.set_index('date')

    return df


def _load_attribute_files_of_subdataset(subdataset_dir: Path) -> pd.DataFrame:
    """Loads all attribute files for one subdataset and merges them into one DataFrame."""
    dfs = []
    for csv_file in subdataset_dir.glob("*.csv"):
        dfs.append(pd.read_csv(csv_file, index_col="gauge_id"))
    return pd.concat(dfs, axis=1)
