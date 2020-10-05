from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xarray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset

########################################################################################################################
#                                           CAMELS US utility functions                                                #
########################################################################################################################


def load_camels_us_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS US attributes from the dataset provided by [#]_
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'camels_attributes_v2.0' folder (the original 
        data set) containing the corresponding txt files for each attribute group.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pandas.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
        
    References
    ----------
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and 
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """
    attributes_path = Path(data_dir) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if basins:
        # drop rows of basins not contained in the passed list
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df


def load_camels_us_forcings(data_dir: Path, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
    """Load the forcing data for a basin of the CAMELS US data set.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'basin_mean_forcing' folder containing one 
        subdirectory for each forcing. The forcing directories have to contain 18 subdirectories (for the 18 HUCS) as in
        the original CAMELS data set. In each HUC folder are the forcing files (.txt), starting with the 8-digit basin 
        id.
    basin : str
        8-digit USGS identifier of the basin.
    forcings : str
        Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory. 
        
    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    int
        Catchment area (m2), specified in the header of the forcing file.
    """
    forcing_path = data_dir / 'basin_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    df = pd.read_csv(file_path, sep='\s+', header=3)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_camels_us_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
    """Load the discharge data for a basin of the CAMELS US data set.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
        subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge files 
        (.txt), starting with the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.
    area : int
        Catchment area (m2), used to normalize the discharge.

    Returns
    -------
    pd.Series
        Time-index pandas.Series of the discharge values (mm/day)
    """

    discharge_path = data_dir / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs


########################################################################################################################
#                                    HOURLY CAMELS US utility functions                                                #
########################################################################################################################


def load_hourly_us_forcings(data_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load the hourly forcing data for a basin of the CAMELS US data set.
    
    The hourly forcings are not included in the original data set by Newman et al. (2017).

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain an 'hourly' folder containing one subdirectory
        for each forcing, which contains the forcing files (.csv) for each basin. Files have to contain the 8-digit 
        basin id.
    basin : str
        8-digit USGS identifier of the basin.
    forcings : str
        Must match the folder names in the 'hourly' directory. E.g. 'nldas_hourly'

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcing data.
    """
    forcing_path = data_dir / 'hourly' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('*.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')

    return pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly' with a subdirectory 
        'usgs_streamflow' which contains the discharge files (.csv) for each basin. File names must contain the 8-digit 
        basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index Series of the discharge values (mm/hour)
    """
    discharge_path = data_dir / 'hourly' / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*usgs-hourly.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {discharge_path}')

    return pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])


def load_hourly_us_stage(data_dir: Path, basin: str) -> pd.Series:
    """Load the hourly stage data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly' with a subdirectory 
        'usgs_stage' which contains the stage files (.csv) for each basin. File names must contain the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.

    Returns
    -------
    pd.Series
        Time-index Series of the stage values (m)
    """
    stage_path = data_dir / 'hourly' / 'usgs_stage'
    files = list(stage_path.glob('**/*_utc.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {stage_path}')

    df = pd.read_csv(file_path,
                     sep=',',
                     index_col=['datetime'],
                     parse_dates=['datetime'],
                     usecols=['datetime', 'gauge_height_ft'])
    df = df.resample('H').mean()
    df["gauge_height_m"] = df["gauge_height_ft"] * 0.3048

    return df["gauge_height_m"]


def load_hourly_us_netcdf(data_dir: Path, forcings: str) -> xarray.Dataset:
    """Load hourly forcing and discharge data from preprocessed netCDF file.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a folder called 'hourly', containing the netCDF file.
    forcings : str
        Name of the forcing product. Must match the ending of the netCDF file. E.g. 'nldas_hourly' for 
        'usgs-streamflow-nldas_hourly.nc'

    Returns
    -------
    xarray.Dataset
        Dataset containing the combined discharge and forcing data of all basins (as stored in the netCDF)  
    """
    netcdf_path = data_dir / 'hourly' / f'usgs-streamflow-{forcings}.nc'
    if not netcdf_path.is_file():
        raise FileNotFoundError(f'No NetCDF file for hourly streamflow and {forcings} at {netcdf_path}.')

    return xarray.open_dataset(netcdf_path)


########################################################################################################################
#                                           CAMELS GB utility functions                                                #
########################################################################################################################


def load_camels_gb_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS GB attributes from the dataset provided by [#]_

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS GB directory. This folder must contain an 'attributes' folder containing the corresponding 
        csv files for each attribute group (ending with _attributes.csv).
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
        
    References
    ----------
    .. [#] Coxon, G., Addor, N., Bloomfield, J. P., Freer, J., Fry, M., Hannaford, J., Howden, N. J. K., Lane, R., 
        Lewis, M., Robinson, E. L., Wagener, T., and Woods, R.: CAMELS-GB: Hydrometeorological time series and landscape 
        attributes for 671 catchments in Great Britain, Earth Syst. Sci. Data Discuss., 
        https://doi.org/10.5194/essd-2020-49,  in review, 2020. 
    """
    attributes_path = Path(data_dir) / 'attributes'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('*_attributes.csv')

    # Read-in attributes into one big dataframe
    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=',', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    if basins:
        # drop rows of basins not contained in the passed list
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df


def load_camels_gb_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS GB data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS GB directory. This folder must contain a folder called 'timeseries' containing the forcing
        files for each basin as .csv file. The file names have to start with 'CAMELS_GB_hydromet_timeseries'.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    """
    forcing_path = data_dir / 'timeseries'
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('**/CAMELS_GB_hydromet_timeseries*.csv'))
    file_path = [f for f in files if f"_{basin}_" in f.name]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    df = pd.read_csv(file_path, sep=',', header=0, dtype={'date': str})
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.set_index("date")

    return df


########################################################################################################################
#                                         General utility functions                                                    #
########################################################################################################################


def load_hydroatlas_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load HydroATLAS attributes into a pandas DataFrame

    Parameters
    ----------
    data_dir : Path
        Path to the root directory of the dataset. Must contain a folder called 'hydroatlas_attributes' with a file
        called `attributes.csv`. The attributes file is expected to have one column called `basin_id`.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame containing the HydroATLAS attributes.
    """
    attribute_file = data_dir / "hydroatlas_attributes" / "attributes.csv"
    if not attribute_file.is_file():
        raise FileNotFoundError(attribute_file)

    df = pd.read_csv(attribute_file, dtype={'basin_id': str})
    df = df.set_index('basin_id')

    if basins:
        drop_basins = [b for b in df.index if b not in basins]
        df = df.drop(drop_basins, axis=0)

    return df


def load_basin_file(basin_file: Path) -> List[str]:
    """Load list of basins from text file.
    
    Parameters
    ----------
    basin_file : Path
        Path to a basin txt file. File has to contain one basin id per row.

    Returns
    -------
    List[str]
        List of basin ids as strings.
    """
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = sorted(basin.strip() for basin in basins)
    return basins


def attributes_sanity_check(data_dir: Path, attribute_set: str, basins: List[str], attribute_list: List[str]):
    """Utility function to check if the standard deviation of one (or more) attributes is zero.
    
    This utility function can be used to check if any attribute has a standard deviation of zero. This would lead to 
    NaN's, when normalizing the features and thus would lead to NaN's when training the model. The function will raise
    a `RuntimeError` if one or more zeros have been detected and will print the list of corresponding attribute names
    to the console.
    
    Parameters
    ----------
    data_dir : Path
        Path to the root directory of the data set
    attribute_set : {'hydroatlas', 'camels_us', 'hourly_camels_us', 'camels_gb'}
        Name of the attribute set to check.
    basins : 
        List of basins to consider in the check.
    attribute_list : 
        List of attribute names to consider in the check.

    Raises
    ------
    ValueError
        For an unknown 'attribute_set'
    RuntimeError
        If one or more attributes have a standard deviation of zero.
    """
    if attribute_set == "hydroatlas":
        df = load_hydroatlas_attributes(data_dir, basins)
    elif attribute_set in ["camels_us", "hourly_camels_us"]:
        df = load_camels_us_attributes(data_dir, basins)
    elif attribute_set == "camels_gb":
        df = load_camels_gb_attributes(data_dir, basins)
    else:
        raise ValueError(f"Unknown 'attribute_set' {attribute_set}")
    drop_cols = [c for c in df.columns if c not in attribute_list]
    df = df.drop(drop_cols, axis=1)
    attributes = []
    if any(df.std() == 0.0) or any(df.std().isnull()):
        for k, v in df.std().iteritems():
            if (v == 0) or (np.isnan(v)):
                attributes.append(k)
    if attributes:
        msg = [
            "The following attributes have a std of zero or NaN, which results in NaN's ",
            "when normalizing the features. Remove the attributes from the attribute feature list ",
            "and restart the run. \n", f"Attributes: {attributes}"
        ]
        raise RuntimeError("".join(msg))


def sort_frequencies(frequencies: List[str]) -> List[str]:
    """Sort the passed frequencies from low to high frequencies.

    Use `pandas frequency strings
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_
    to define frequencies. Note: The strings need to include values, e.g., '1D' instead of 'D'.

    Parameters
    ----------
    frequencies : List[str]
        List of pandas frequency identifiers to be sorted.

    Returns
    -------
    List[str]
        Sorted list of pandas frequency identifiers.
    """
    deltas = {freq: pd.to_timedelta(freq) for freq in frequencies}
    return sorted(deltas, key=deltas.get)[::-1]


def infer_frequency(index: Union[pd.DatetimeIndex, np.ndarray]) -> str:
    """Infer the frequency of an index of a pandas DataFrame/Series or xarray DataArray.

    Parameters
    ----------
    index : Union[pd.DatetimeIndex, np.ndarray]
        DatetimeIndex of a DataFrame/Series or array of datetime values.

    Returns
    -------
    str
        Frequency of the index as a `pandas frequency string
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases>`_

    Raises
    ------
    ValueError
        If the frequency cannot be inferred from the index or is zero.
    """
    native_frequency = pd.infer_freq(index)
    if native_frequency is None:
        raise ValueError(f'Cannot infer a legal frequency from dataset: {native_frequency}.')
    if native_frequency[0] not in '0123456789':  # add a value to the unit so to_timedelta works
        native_frequency = f'1{native_frequency}'
    if pd.to_timedelta(native_frequency) == pd.to_timedelta(0):
        raise ValueError('Inferred dataset frequency is zero.')
    return native_frequency


def infer_datetime_coord(xr: Union[DataArray, Dataset]) -> str:
    """Checks for coordinate with 'date' in its name and returns the name.
    
    Parameters
    ----------
    xr : Union[DataArray, Dataset]
        Array to infer coordinate name of.
        
    Returns
    -------
    str
        Name of datetime coordinate name.
        
    Raises
    ------
    RuntimeError
        If none or multiple coordinates with 'date' in its name are found.
    """
    candidates = [c for c in list(xr.coords) if "date" in c]
    if len(candidates) > 1:
        raise RuntimeError("Found multiple coordinates with 'date' in its name.")
    if not candidates:
        raise RuntimeError("Did not find any coordinate with 'date' in its name")

    return candidates[0]
