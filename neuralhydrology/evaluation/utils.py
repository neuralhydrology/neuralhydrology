import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import xarray
from ruamel.yaml import YAML


def load_scaler(run_dir: Path) -> Dict[str, Union[pd.Series, xarray.Dataset]]:
    """Load feature scaler from run directory.

    Checks run directory for scaler file in yaml format (new) or pickle format (old).

    Parameters
    ----------
    run_dir: Path
        Run directory. Has to contain a folder 'train_data' that contains the 'train_data_scaler' file.

    Returns
    -------
    Dictionary, containing the feature scaler for static and dynamic features.
    
    Raises
    ------
    FileNotFoundError
        If neither a 'train_data_scaler.yml' or 'train_data_scaler.p' file is found in the 'train_data' folder of the 
        run directory.
    """
    scaler_file = run_dir / "train_data" / "train_data_scaler.yml"

    if scaler_file.is_file():
        # read scaler from disk
        with scaler_file.open("r") as fp:
            yaml = YAML(typ="safe")
            scaler_dump = yaml.load(fp)

        # transform scaler into the format expected by NeuralHydrology
        scaler = {}
        for key, value in scaler_dump.items():
            if key in ["attribute_means", "attribute_stds", "camels_attr_means", "camels_attr_stds"]:
                scaler[key] = pd.Series(value)
            elif key in ["xarray_feature_scale", "xarray_feature_center"]:
                scaler[key] = xarray.Dataset.from_dict(value).astype(np.float32)

        return scaler

    else:
        scaler_file = run_dir / "train_data" / "train_data_scaler.p"

        if scaler_file.is_file():
            with scaler_file.open('rb') as fp:
                scaler = pickle.load(fp)
            return scaler
        else:
            raise FileNotFoundError(f"No scaler file found in {scaler_file.parent}. "
                                    "Looked for (new) yaml file or (old) pickle file")


def load_basin_id_encoding(run_dir: Path) -> Dict[str, int]:
    id_to_int_file = run_dir / "train_data" / "id_to_int.yml"
    if id_to_int_file.is_file():
        with id_to_int_file.open("r") as fp:
            yaml = YAML(typ="safe")
            id_to_int = yaml.load(fp)
        return id_to_int

    else:
        id_to_int_file = run_dir / "train_data" / "id_to_int.p"
        if id_to_int_file.is_file():
            with id_to_int_file.open("rb") as fp:
                id_to_int = pickle.load(fp)
            return id_to_int
        else:
            raise FileNotFoundError(f"No id-to-int file found in {id_to_int_file.parent}. "
                                    "Looked for (new) yaml file or (old) pickle file")
