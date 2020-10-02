import logging
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
from numba import njit
from tqdm import tqdm

from .pet import get_priestley_taylor_pet
from .utils import load_camels_us_attributes, load_camels_us_forcings, load_basin_file

LOGGER = logging.getLogger(__name__)


def precalculate_dyn_climate_indices(data_dir: Path, basin_file: Path, window_length: int, forcings: str):
    basins = load_basin_file(basin_file=basin_file)
    camels_attributes = load_camels_us_attributes(data_dir=data_dir, basins=basins)
    additional_features = {}
    new_columns = [
        'p_mean_dyn', 'pet_mean_dyn', 'aridity_dyn', 't_mean_dyn', 'frac_snow_dyn', 'high_prec_freq_dyn',
        'high_prec_dur_dyn', 'low_prec_freq_dyn', 'low_prec_dur_dyn'
    ]
    for basin in tqdm(basins, file=sys.stdout):
        df, _ = load_camels_us_forcings(data_dir=data_dir, basin=basin, forcings=forcings)
        lat = camels_attributes.loc[camels_attributes.index == basin, 'gauge_lat'].values
        elev = camels_attributes.loc[camels_attributes.index == basin, 'elev_mean'].values
        df["PET(mm/d)"] = get_priestley_taylor_pet(t_min=df["tmin(C)"].values,
                                                   t_max=df["tmax(C)"].values,
                                                   s_rad=df["srad(W/m2)"].values,
                                                   lat=lat,
                                                   elev=elev,
                                                   doy=df.index.dayofyear.values)

        for col in new_columns:
            df[col] = np.nan

        x = np.array([
            df['prcp(mm/day)'].values, df['srad(W/m2)'].values, df['tmax(C)'].values, df['tmin(C)'].values,
            df['vp(Pa)'].values, df['PET(mm/d)'].values
        ]).T

        new_features = numba_climate_indexes(x, window_length=window_length)

        if np.sum(np.isnan(new_features)) > 0:
            raise ValueError(f"NaN in new features of basin {basin}")

        df.loc[df.index[window_length - 1]:, 'p_mean_dyn'] = new_features[:, 0]
        df.loc[df.index[window_length - 1]:, 'pet_mean_dyn'] = new_features[:, 1]
        df.loc[df.index[window_length - 1]:, 'aridity_dyn'] = new_features[:, 2]
        df.loc[df.index[window_length - 1]:, 't_mean_dyn'] = new_features[:, 3]
        df.loc[df.index[window_length - 1]:, 'frac_snow_dyn'] = new_features[:, 4]
        df.loc[df.index[window_length - 1]:, 'high_prec_freq_dyn'] = new_features[:, 5]
        df.loc[df.index[window_length - 1]:, 'high_prec_dur_dyn'] = new_features[:, 6]
        df.loc[df.index[window_length - 1]:, 'low_prec_freq_dyn'] = new_features[:, 7]
        df.loc[df.index[window_length - 1]:, 'low_prec_dur_dyn'] = new_features[:, 8]

        drop_cols = [c for c in df.columns if c not in new_columns]

        df = df.drop(drop_cols, axis=1)

        additional_features[basin] = df

    filename = f"dyn_climate_indices_{forcings}_{len(basins)}basins_{window_length}lookback.p"

    output_file = Path(__file__).parent.parent.parent / 'data' / filename

    with output_file.open("wb") as fp:
        pickle.dump(additional_features, fp)

    LOGGER.info(f"Precalculated features successfully stored at {output_file}")

    return additional_features


@njit
def numba_climate_indexes(features: np.ndarray, window_length: int) -> np.ndarray:
    n_samples = features.shape[0]
    new_features = np.zeros((n_samples - 365 + 1, 9))

    for i in range(new_features.shape[0]):
        x = features[i:i + window_length]

        p_mean = np.mean(x[:, 0])
        pet_mean = np.mean(x[:, -1])
        aridity = pet_mean / p_mean
        t_mean = (np.mean(x[:, 1]) + np.mean(x[:, 2])) / 2

        precip_days = x[x[:, 0] > 0]
        frac_snow = precip_days[precip_days[:, 2] < 0, :].shape[0] / precip_days.shape[0]

        high_prec_freq = precip_days[precip_days[:, 0] > 5 * p_mean].shape[0] / precip_days.shape[0]
        low_prec_freq = precip_days[precip_days[:, 0] < 1].shape[0] / precip_days.shape[0]

        idx = np.where(x[:, 0] < 1)[0]
        groups = split_list(idx)
        low_prec_dur = np.mean(np.array([len(p) for p in groups]))

        idx = np.where(x[:, 0] >= 5 * p_mean)[0]
        groups = split_list(idx)
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
def split_list(alist: List) -> List:
    newlist = []
    start = 0
    end = 0
    for index, value in enumerate(alist):
        if index < len(alist) - 1:
            if alist[index + 1] > value + 1:
                end = index + 1
                newlist.append(alist[start:end])
                start = end
        else:
            newlist.append(alist[start:len(alist)])
    return newlist
