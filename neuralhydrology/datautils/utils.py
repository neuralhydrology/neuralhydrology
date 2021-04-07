import functools
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
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
    
    Note: Basins names are not allowed to end with '_period*'
    
    Parameters
    ----------
    basin_file : Path
        Path to a basin txt file. File has to contain one basin id per row, while empty rows are ignored.

    Returns
    -------
    List[str]
        List of basin ids as strings.
        
    Raises
    ------
    ValueError
        In case of invalid basin names that would cause problems internally.
    """
    with basin_file.open('r') as fp:
        basins = sorted(basin.strip() for basin in fp if basin.strip())

    # sanity check basin names
    problematic_basins = [basin for basin in basins if basin.split('_')[-1].startswith('period')]
    if problematic_basins:
        msg = [
            f"The following basin names are invalid {problematic_basins}. Check documentation of the ",
            "'load_basin_file()' functions for details."
        ]
        raise ValueError(" ".join(msg))

    return basins


def attributes_sanity_check(df: pd.DataFrame):
    """Utility function to check the suitability of the attributes for model training.
    
    This utility function can be used to check if any attribute has a standard deviation of zero. This would lead to 
    NaN's when normalizing the features and thus would lead to NaN's when training the model. It also checks if any
    attribute for any basin contains a NaN, which would also cause NaNs during model training.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of catchment attributes as columns.

    Raises
    ------
    RuntimeError
        If one or more attributes have a standard deviation of zero or any attribute for any basin is NaN.
    """
    # Check for NaNs in standard deviation of attributes.
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

    # Check for NaNs in any attribute of any basin
    nan_df = df[df.isnull().any(axis=1)]
    if len(nan_df) > 0:
        failure_cases = defaultdict(list)
        for basin, row in nan_df.iterrows():
            for feature, value in row.iteritems():
                if np.isnan(value):
                    failure_cases[basin].append(feature)
        # create verbose error message
        msg = ["The following basins/attributes are NaN, which can't be used as input:"]
        for basin, features in failure_cases.items():
            msg.append(f"{basin}: {features}")
        raise RuntimeError("\n".join(msg))


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

    Raises
    ------
    ValueError
        If a pair of frequencies in `frequencies` is not comparable via `compare_frequencies`.
    """
    return sorted(frequencies, key=functools.cmp_to_key(compare_frequencies))


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

    # Assert that the frequency corresponds to a positive time delta. We first add one offset to the base datetime
    # to make sure it's aligned with the frequency. Otherwise, adding an offset of e.g. 0Y would round up to the
    # nearest year-end, so we'd incorrectly miss a frequency of zero.
    base_datetime = pd.to_datetime('2001-01-01 00:00:00') + to_offset(native_frequency)
    if base_datetime >= base_datetime + to_offset(native_frequency):
        raise ValueError('Inferred dataset frequency is zero or negative.')
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


def compare_frequencies(freq_one: str, freq_two: str) -> int:
    """Compare two frequencies.

    Note that only frequencies that work with `get_frequency_factor` can be compared.

    Parameters
    ----------
    freq_one : str
        First frequency.
    freq_two : str
        Second frequency.

    Returns
    -------
    int
        -1 if `freq_one` is lower than `freq_two`, +1 if it is larger, 0 if they are equal.

    Raises
    ------
    ValueError
        If the two frequencies are not comparable via `get_frequency_factor`.
    """
    freq_factor = get_frequency_factor(freq_one, freq_two)
    if freq_factor < 1:
        return 1
    if freq_factor > 1:
        return -1
    return 0


def get_frequency_factor(freq_one: str, freq_two: str) -> float:
    """Get relative factor between the two frequencies.

    Parameters
    ----------
    freq_one : str
        String representation of the first frequency.
    freq_two : str
        String representation of the second frequency.

    Returns
    -------
    float
        Ratio of `freq_one` to `freq_two`.

    Raises
    ------
    ValueError
        If the frequency factor cannot be determined. This can be the case if the frequencies do not represent a fixed
        time delta and are not directly comparable (e.g., because they have the same unit)
        E.g., a month does not represent a fixed time delta. Thus, 1D and 1M are not comparable. However, 1M and 2M are
        comparable since they have the same unit.
    """
    if freq_one == freq_two:
        return 1

    offset_one = to_offset(freq_one)
    offset_two = to_offset(freq_two)
    if offset_one.n < 0 or offset_two.n < 0:
        # Would be possible to implement, but we should never need negative frequencies, so it seems reasonable to
        # fail gracefully rather than to open ourselves to potential unexpected corner cases.
        raise NotImplementedError('Cannot compare negative frequencies.')
    # avoid division by zero errors
    if offset_one.n == offset_two.n == 0:
        return 1
    if offset_two.n == 0:
        return np.inf
    if offset_one.name == offset_two.name:
        return offset_one.n / offset_two.n

    # some simple hard-coded cases
    factor = None
    regex_month_or_day = '-(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|MON|TUE|WED|THU|FRI|SAT|SUN)$'
    for i, (one, two) in enumerate([(offset_one, offset_two), (offset_two, offset_one)]):
        # the offset anchor is irrelevant for the ratio between the frequencies, so we remove it from the string
        name_one = re.sub(regex_month_or_day, '', one.name)
        name_two = re.sub(regex_month_or_day, '', two.name)
        if (name_one in ['A', 'Y'] and name_two == 'M') or (name_one in ['AS', 'YS'] and name_two == 'MS'):
            factor = 12 * one.n / two.n
        if (name_one in ['A', 'Y'] and name_two == 'Q') or (name_one in ['AS', 'YS'] and name_two == 'QS'):
            factor = 4 * one.n / two.n
        if (name_one == 'Q' and name_two == 'M') or (name_one == 'QS' and name_two == 'MS'):
            factor = 3 * one.n / two.n
        if name_one == 'W' and name_two == 'D':
            factor = 7 * one.n / two.n

        if factor is not None:
            if i == 1:
                return 1 / factor  # `one` was `offset_two`, `two` was `offset_one`
            return factor

    # If all other checks didn't match, we try to convert the frequencies to timedeltas. However, we first need to avoid
    # two cases: (1) pd.to_timedelta currently interprets 'M' as minutes, while it means months in to_offset.
    # (2) Using 'M', 'Y', and 'y' in pd.to_timedelta is deprecated and won't work in the future, so we don't allow it.
    if any(re.sub(regex_month_or_day, '', offset.name) in ['M', 'Y', 'A', 'y'] for offset in [offset_one, offset_two]):
        raise ValueError(f'Frequencies {freq_one} and/or {freq_two} are not comparable.')
    try:
        factor = pd.to_timedelta(freq_one) / pd.to_timedelta(freq_two)
    except ValueError as err:
        raise ValueError(f'Frequencies {freq_one} and/or {freq_two} are not comparable.') from err
    return factor
