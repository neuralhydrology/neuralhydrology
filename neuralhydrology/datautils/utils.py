from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset


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


def attributes_sanity_check(df: pd.DataFrame):
    """Utility function to check if the standard deviation of one (or more) attributes is zero.
    
    This utility function can be used to check if any attribute has a standard deviation of zero. This would lead to 
    NaN's, when normalizing the features and thus would lead to NaN's when training the model. The function will raise
    a `RuntimeError` if one or more zeros have been detected and will print the list of corresponding attribute names
    to the console.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of catchment attributes as columns.

    Raises
    ------
    RuntimeError
        If one or more attributes have a standard deviation of zero.
    """
    # Iterate over attributes and check for NaNs
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
