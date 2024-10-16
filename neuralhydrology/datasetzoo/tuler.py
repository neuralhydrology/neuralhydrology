from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config



class Tuler(BaseDataset):
    """Data set class for the USACE data set by [#]_ and [#]_.
    
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
        loaded from the dataset and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).
        
    References
    ----------
    .. [#] Zhang, L., Moges, E., Kirchner, J. W., Coda, E., Liu, T., Wymore, A. S., et al. (2021). 
    CHOSEN: A synthesis of hydrometeorological data from intensively monitored catchments and comparative analysis of hydrologic extremes. 
    Hydrological Processes, 35(11), e14429. https://doi.org/10.1002/hyp.14429
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(Tuler, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from csv files."""

        #select which columns you want for each basin by column name
        forcings_d = {  'Tuler': ['MF_TuleR_S10ET-POTENTIAL', 'MF_TuleR_S10FLOW',
                                    'MF_TuleR_S10PRECIP-INC', 'MF_TuleR_S10SATURATION FRACTION',
                                    'MF_TuleR_S10STORAGE-SOIL', 'MF_TuleR_S10SWE-OBSERVED',
                                    'MF_TuleR_S10TEMPERATURE-AIR', 'MF_TuleR_S20ET-POTENTIAL',
                                    'MF_TuleR_S20FLOW', 'MF_TuleR_S20PRECIP-INC',
                                    'MF_TuleR_S20SATURATION FRACTION', 'MF_TuleR_S20STORAGE-SOIL',
                                    'MF_TuleR_S20SWE-OBSERVED', 'MF_TuleR_S20TEMPERATURE-AIR',
                                    'NF_TuleR_S10ET-POTENTIAL', 'NF_TuleR_S10FLOW',
                                    'NF_TuleR_S10PRECIP-INC', 'NF_TuleR_S10SATURATION FRACTION',
                                    'NF_TuleR_S10STORAGE-SOIL', 'NF_TuleR_S10SWE-OBSERVED',
                                    'NF_TuleR_S10TEMPERATURE-AIR', 'ReservoirInflowFLOW',
                                    'SF_TuleR_S10ET-POTENTIAL',
                                    'SF_TuleR_S10FLOW', 'SF_TuleR_S10PRECIP-INC',
                                    'SF_TuleR_S10SATURATION FRACTION', 'SF_TuleR_S10STORAGE-SOIL',
                                    'SF_TuleR_S10SWE-OBSERVED', 'SF_TuleR_S10TEMPERATURE-AIR',
                                    'TuleR_S10ET-POTENTIAL', 'TuleR_S10FLOW', 'TuleR_S10PRECIP-INC',
                                    'TuleR_S10SATURATION FRACTION', 'TuleR_S10STORAGE-SOIL',
                                    'TuleR_S10SWE-OBSERVED', 'TuleR_S10TEMPERATURE-AIR',
                                    'TuleR_S20ET-POTENTIAL', 'TuleR_S20FLOW', 'TuleR_S20PRECIP-INC',
                                    'TuleR_S20SATURATION FRACTION', 'TuleR_S20STORAGE-SOIL',
                                    'TuleR_S20SWE-OBSERVED', 'TuleR_S20TEMPERATURE-AIR']
                        }

        discharge_d = { 'Tuler': ['ReservoirInflowFLOW-OBSERVED']
                        }
       
        #load data into df from csv
        df = pd.read_csv(str(self.cfg.data_dir) + f'/HMS_inflow_results_data.csv', index_col = 'Date', parse_dates=True)
    
        #only select the forcings, observed data you want by only selecting the columns that correspond to the above dictionary for that basin
        df = df.loc[:, forcings_d[basin]+discharge_d[basin]]

        #rename date index
        df = df.rename_axis('date')
        
        return df


    #def _load_attributes(self) -> pd.DataFrame:
     #   return load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)