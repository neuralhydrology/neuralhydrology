import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from neuralhydrology.datautils import utils

LOGGER = logging.getLogger(__name__)


def shift_discharge(data_dir: Path, basins: List[str], dataset: str, shift: int = 1,
                    output_file: Path = None) -> Dict[str, pd.DataFrame]:
    """Return shifted discharge data.
    
    This function returns the shifted discharge data for each basin in `basins`. Useful when training
    models with lagged discharge as input. Use the `output_file` argument to store the resulting dictionary as 
    pickle dump to disk. This pickle file can be used with the config argument `additional_feature_files` to make the
    data available as input feature.
    For CAMELS GB, the 'discharge_spec' column is used.
    
    Parameters
    ----------
    data_dir : Path
        Path to the dataset directory.
    basins : List[str]
        List of basin ids.
    dataset : {'camels_us', 'camels_gb', 'hourly_camels_us'} 
        Which data set to use.
    shift : int, optional
        Number of discharge lag in time steps, by default 1.
    output_file : Path, optional
        If specified, stores the resulting dictionary of DataFrames to this location as a pickle dump.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with one time-indexed DataFrame per basin. The lagged discharge column is named according to the 
        discharge column name, with '_t-SHIFT' as suffix, where 'SHIFT' corresponds to the argument `shift`.
    """
    data = {}
    for basin in tqdm(basins, file=sys.stdout):

        # load discharge data
        if dataset == "camels_us":
            df, area = utils.load_camels_us_forcings(data_dir=data_dir, basin=basin, forcings="daymet")
            df["QObs(mm/d)"] = utils.load_camels_us_discharge(data_dir=data_dir, basin=basin, area=area)
            discharge_col = "QObs(mm/d)"
        elif dataset == "camels_gb":
            df = utils.load_camels_gb_timeseries(data_dir=data_dir, basin=basin)
            discharge_col = "discharge_spec"
        elif dataset == "hourly_camels_gb":
            df = utils.load_hourly_us_discharge(data_dir=data_dir, basin=basin)
            discharge_col = "QObs(mm/h)"

        # shift discharge data by `shift` time steps
        df[f"{discharge_col}_t-{shift}"] = df[discharge_col].shift(shift)

        # remove all columns from data set except shifted discharge
        drop_columns = [col for col in df.columns if col != f"{discharge_col}_t-{shift}"]
        data[basin] = df.drop(labels=drop_columns, axis=1)

    if output_file is not None:
        # store pickle dump
        with output_file.open("wb") as fp:
            pickle.dump(data, fp)

        LOGGER.info(f"Data successfully stored at {output_file}")

    return data
