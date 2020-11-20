import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

from neuralhydrology.datasetzoo.camelsus import load_camels_us_forcings, load_camels_us_attributes
from neuralhydrology.datautils import pet

LOGGER = logging.getLogger(__name__)


def calculate_camels_us_dyn_climate_indices(data_dir: Path,
                                         basins: List[str],
                                         window_length: int,
                                         forcings: str,
                                         variable_names: Dict[str, str] = None,
                                         output_file: Path = None) -> Dict[str, pd.DataFrame]:
    """Calculate dynamic climate indices for the CAMELS US dataset.
    
    Compared to the long-term static climate indices included in the CAMELS US data set, this function computes the same
    climate indices by a moving window approach over the entire data set. That is, for each time step, the climate 
    indices are re-computed from the last `window_length` time steps. The resulting dictionary of DataFrames can be
    used with the `additional_feature_files` argument.
    Unlike in CAMELS, the '_freq' indices will be fractions, not number of days. To compare the values to the ones in
    CAMELS, they need to be multiplied by 365.25.
    
    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory.
    basins : List[str]
        List of basin ids.
    window_length : int
        Look-back period to use to compute the climate indices.
    forcings : str
        Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory.
    variable_names : Dict[str, str], optional
        Mapping of the forcings' variable names, needed if forcings other than DayMet, Maurer, or NLDAS are used.
        If provided, this must be a dictionary that maps the keys 'prcp', 'tmin', 'tmax', 'srad' to the forcings'
        respective variable names.
    output_file : Path, optional
        If specified, stores the resulting dictionary of DataFrames to this location as a pickle dump.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with one time-indexed DataFrame per basin. By definition, the climate indices for a given day in the
        DataFrame are computed from the `window_length` previous time steps (including the given day).
    """
    camels_attributes = load_camels_us_attributes(data_dir=data_dir, basins=basins)
    additional_features = {}

    if variable_names is None:
        if forcings.startswith('nldas'):
            variable_names = {'prcp': 'PRCP(mm/day)', 'tmin': 'Tmin(C)', 'tmax': 'Tmax(C)', 'srad': 'SRAD(W/m2)'}
        elif forcings.startswith('daymet') or forcings.startswith('maurer'):
            variable_names = {'prcp': 'prcp(mm/day)', 'tmin': 'tmin(C)', 'tmax': 'tmax(C)', 'srad': 'srad(W/m2)'}
        else:
            raise ValueError(f'No predefined variable mapping for {forcings} forcings. Provide one in variable_names.')

    for basin in tqdm(basins, file=sys.stdout):
        df, _ = load_camels_us_forcings(data_dir=data_dir, basin=basin, forcings=forcings)
        lat = camels_attributes.loc[camels_attributes.index == basin, 'gauge_lat'].values
        elev = camels_attributes.loc[camels_attributes.index == basin, 'elev_mean'].values
        df["PET(mm/d)"] = pet.get_priestley_taylor_pet(t_min=df[variable_names['tmin']].values,
                                                       t_max=df[variable_names['tmax']].values,
                                                       s_rad=df[variable_names['srad']].values,
                                                       lat=lat,
                                                       elev=elev,
                                                       doy=df.index.dayofyear.values)

        clim_indices = calculate_dyn_climate_indices(df[variable_names['prcp']],
                                                     df[variable_names['tmax']],
                                                     df[variable_names['tmin']],
                                                     df['PET(mm/d)'],
                                                     window_length=window_length)

        if np.any(clim_indices.isna()):
            raise ValueError(f"NaN in new features of basin {basin}")

        clim_indices = clim_indices.reindex(df.index)  # add NaN rows for the first window_length - 1 entries
        additional_features[basin] = clim_indices

    if output_file is not None:
        with output_file.open("wb") as fp:
            pickle.dump(additional_features, fp)
        LOGGER.info(f"Precalculated features successfully stored at {output_file}")

    return additional_features


def calculate_dyn_climate_indices(precip: pd.Series,
                                  tmax: pd.Series,
                                  tmin: pd.Series,
                                  pet: pd.Series,
                                  window_length: int,
                                  raise_nan=False) -> pd.DataFrame:
    """Calculate dynamic climate indices.

    Compared to the long-term static climate indices included in the CAMELS dataset, this function computes the same
    climate indices by a moving window approach over the entire dataset. That is, for each time step, the climate
    indices are re-computed from the last `window_length` time steps.

    Parameters
    ----------
    precip : pd.Series
        Time-indexed series of precipitation.
    tmax : pd.Series
        Time-indexed series of maximum temperature.
    tmin : pd.Series
        Time-indexed series of minimum temperature.
    pet : pd.Series
        Time-indexed series of potential evapotranspiration.
    window_length : int
        Look-back period to use to compute the climate indices.
    raise_nan : bool, optional
        If True, will raise a ValueError if a climate index is NaN. Default: False.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame of climate indices. By definition, the climate indices for a given day in the
        DataFrame are computed from the `window_length` previous time steps (including the given day).

    Raises
    ------
    ValueError
        If `raise_nan` is True and a calculated climate index is NaN at any point in time.
    """
    x = np.array([precip.values, tmax.values, tmin.values, pet.values]).T

    new_features = _numba_climate_indexes(x, window_length=window_length)

    df = pd.DataFrame(
        {
            'p_mean_dyn': new_features[:, 0],
            'pet_mean_dyn': new_features[:, 1],
            'aridity_dyn': new_features[:, 2],
            't_mean_dyn': new_features[:, 3],
            'frac_snow_dyn': new_features[:, 4],
            'high_prec_freq_dyn': new_features[:, 5],
            'high_prec_dur_dyn': new_features[:, 6],
            'low_prec_freq_dyn': new_features[:, 7],
            'low_prec_dur_dyn': new_features[:, 8]
        },
        index=precip.iloc[min(window_length, len(precip)) - 1:].index)

    if raise_nan and np.any(df.isna()):
        raise ValueError(f"NaN in climate indices {[col for col in df.columns[df.isna().any()]]}")

    return df


@njit
def _numba_climate_indexes(features: np.ndarray, window_length: int) -> np.ndarray:
    # features shape is (#timesteps, 4), where 4 breaks down into: (prcp, tmax, tmin, pet)
    n_samples = features.shape[0]
    window_length = min(n_samples, window_length)
    new_features = np.zeros((n_samples - window_length + 1, 9))

    for i in range(new_features.shape[0]):
        x = features[i:i + window_length]

        p_mean = np.mean(x[:, 0])
        pet_mean = np.mean(x[:, -1])
        aridity = pet_mean / p_mean if p_mean > 0 else np.nan
        t_mean = (np.mean(x[:, 1]) + np.mean(x[:, 2])) / 2

        # fraction of precipitation falling as snow
        if np.sum(x[:, 0]) > 0:
            mean_temp = (x[:, 1] + x[:, 2]) / 2
            frac_snow = np.sum(x[mean_temp <= 0, 0]) / np.sum(x[:, 0])
        else:
            frac_snow = 0.0

        high_prec_freq = np.sum(x[:, 0] >= 5 * p_mean) / x.shape[0]
        low_prec_freq = np.sum(x[:, 0] < 1) / x.shape[0]

        idx = np.where(x[:, 0] < 1)[0]
        groups = _split_list(idx)
        low_prec_dur = np.mean(np.array([len(p) for p in groups]))

        idx = np.where(x[:, 0] >= 5 * p_mean)[0]
        groups = _split_list(idx)
        high_prec_dur = np.mean(np.array([len(p) for p in groups]))

        new_features[i, 0] = p_mean
        new_features[i, 1] = pet_mean
        new_features[i, 2] = aridity
        new_features[i, 3] = t_mean
        new_features[i, 4] = frac_snow
        new_features[i, 5] = high_prec_freq
        new_features[i, 6] = high_prec_dur
        new_features[i, 7] = low_prec_freq
        new_features[i, 8] = low_prec_dur

    return new_features


@njit
def _split_list(a_list: List) -> List:
    new_list = []
    start = 0
    for index, value in enumerate(a_list):
        if index < len(a_list) - 1:
            if a_list[index + 1] > value + 1:
                end = index + 1
                new_list.append(a_list[start:end])
                start = end
        else:
            new_list.append(a_list[start:len(a_list)])
    return new_list
