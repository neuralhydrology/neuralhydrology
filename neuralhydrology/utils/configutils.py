"""Utility script to generate config files from a base config and a defined set of variations"""
import itertools
from pathlib import Path
from typing import Dict

from neuralhydrology.utils.config import Config


def create_config_files(base_config_path: Path, modify_dict: Dict[str, list], output_dir: Path):
    """Create configs, given a base config and a dictionary of parameters to modify.
    
    This function will create one config file for each combination of parameters defined in the modify_dict.
    
    Parameters
    ----------
    base_config_path : Path
        Path to a base config file (.yml)
    modify_dict : dict
        Dictionary, mapping from parameter names to lists of possible parameter values.
    output_dir : Path 
        Path to a folder where the generated configs will be stored
    """
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # load base config as dictionary
    base_config = Config(base_config_path)
    experiment_name = base_config.experiment_name
    option_names = list(modify_dict.keys())

    # iterate over each possible combination of hyper parameters
    for i, options in enumerate(itertools.product(*[val for val in modify_dict.values()])):

        base_config.update_config(dict(zip(option_names, options)))

        # create a unique run name
        name = experiment_name
        for key, val in zip(option_names, options):
            name += f"_{key}{val}"
        base_config.update_config({"experiment_name": name})

        base_config.dump_config(output_dir, f"config_{i+1}.yml")

    print(f"Finished. Configs are stored in {output_dir}")
