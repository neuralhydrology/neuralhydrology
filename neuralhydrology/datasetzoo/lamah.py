from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datautils import utils
from neuralhydrology.utils.config import Config

_SUBDATASET_TO_DIRECTORY = {
    'lamah_a': 'A_basins_total_upstrm',
    'lamah_b': 'B_basins_intermediate_all',
    'lamah_c': 'C_basins_intermediate_lowimp'
}


class LamaH(BaseDataset):
    """Data set class for the LamaH-CE dataset by [#]_.
    
    The LamaH-CE dataset consists of three different catchment delineations, each with dedicated forcing time series and
    catchment attributes. These subdatasets are stored in the folder 'A_basins_total_upstrm', 
    'B_basins_intermediate_all', and 'C_basins_intermediate_lowimp'. The different datasets can be used by setting the
    config argument `dataset` to `lamah_a`, `lamah_b` or `lamah_c` for 'A_basins_total_upstrm', 'B_basins_intermediate_all',
    or 'C_basins_intermediate_lowimp', respectively. Furthermore, if you download the full dataset, each of these 
    subdatasets, as well as the streamflow data, comes at hourly and daily resolution. Based on the config argument 
    `use_frequencies` this dataset class will load daily data (for daily resolutions or lower), or hourly data (for all
    temporal resolutions higher than daily). If nothing is specified in `use_frequencies`, daily data is loaded by
    default. Also note: discharge data in the LamaH dataset is provided in m3s-1. This dataset class will transform
    discharge into mmd-1 (for daily data) or mmh-1 (for hourly data), using the 'area_gov' provided in the attributes
    file.

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
    .. [#]  Klingler, C., Schulz, K., and Herrnegger, M.: LamaH-CE: LArge-SaMple DAta for Hydrology and Environmental 
        Sciences for Central Europe, Earth Syst. Sci. Data, 13, 4529-4565, https://doi.org/10.5194/essd-13-4529-2021, 
        2021. 
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # Discharge is provided in m3/s in the data set as 'Qobs [m3/s]'. We allow to use 'Qobs [mm/h]' or 'Qobs [mm/d]'
        # in the config and in this case will normalize on the fly. For that, we need the basin area
        self._all_variables = self._get_list_of_all_variables(cfg)
        if any([f.startswith("qobs") for f in self._all_variables]):
            df = load_lamah_attributes(cfg.data_dir, sub_dataset=cfg.dataset)
            self._basin_area = df["area_gov"]

        # initialize parent class
        super(LamaH, self).__init__(cfg=cfg,
                                    is_train=is_train,
                                    period=period,
                                    basin=basin,
                                    additional_features=additional_features,
                                    id_to_int=id_to_int,
                                    scaler=scaler)

    @staticmethod
    def _get_list_of_all_variables(cfg: Config) -> List[str]:
        all_variables = []
        if isinstance(cfg.target_variables, dict):
            for val in cfg.target_variables.values():
                all_variables = all_variables + val
        else:
            all_variables = all_variables + cfg.target_variables
        if isinstance(cfg.dynamic_inputs, dict):
            for val in cfg.dynamic_inputs.values():
                all_variables = all_variables + val
        else:
            all_variables = all_variables + cfg.dynamic_inputs
        return all_variables

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""

        # Determine if hourly or daily data should be loaded, by default daily.
        temporal_resolution = '1D'
        if self.cfg.use_frequencies:
            if any([utils.compare_frequencies(freq, '1D') == 1 for freq in self.cfg.use_frequencies]):
                temporal_resolution = '1H'

        df = load_lamah_forcing(data_dir=self.cfg.data_dir,
                                basin=basin,
                                sub_dataset=self.cfg.dataset,
                                temporal_resolution=temporal_resolution)
        if any([f.startswith('qobs') for f in self._all_variables]):
            # We normalize discharge here to make use of the cached basin areas.
            discharge = load_lamah_discharge(data_dir=self.cfg.data_dir,
                                             basin=basin,
                                             temporal_resolution=temporal_resolution,
                                             normalize_discharge=False)
            df["qobs"] = _normalize_discharge(ser=discharge["qobs"],
                                              area=self._basin_area.loc[basin],
                                              temporal_resolution=temporal_resolution)

        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_lamah_attributes(self.cfg.data_dir, sub_dataset=self.cfg.dataset, basins=self.basins)


def load_lamah_forcing(data_dir: Path, basin: str, sub_dataset: str, temporal_resolution: str = '1D') -> pd.DataFrame:
    """Load forcing data of the LamaH data set.

    Parameters
    ----------
    data_dir : Path
        Path to the LamaH directory. 
    basin : str
        Basin identifier number as string.
    sub_dataset: str
        One of {'lamah_a', 'lamah_b', 'lamah_c'}, defining which of the three catchment delinations/sub-datasets 
        (A_basins_total_upstrm, B_basins_intermediate_all, or C_basins_intermediate_lowimp) will be loaded.
    temporal_resolution: str, optional
        Defines if either daily ('1D', default) or hourly ('1H') timeseries data will be loaded.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcings data.

    Raises
    ------
    ValueError
        If 'sub_dataset' is not one of {'lamah_a', 'lamah_b', 'lamah_c'}.
    ValueError
        If 'temporal_resolution' is not one of ['1H', '1D'].
    """
    if sub_dataset not in _SUBDATASET_TO_DIRECTORY:
        raise ValueError(
            f"{sub_dataset} is not a valid choice for 'sub_dataset'. Must be one of {_SUBDATASET_TO_DIRECTORY.keys()}.")

    if temporal_resolution not in ['1D', '1H']:
        raise ValueError(
            f"{temporal_resolution} is not a valid choice for 'temporal_resolution'. Must be one of '1H', '1D'.")

    temporal_resolution_directory = 'daily' if temporal_resolution == '1D' else 'hourly'

    # Load forcing data
    forcing_dir = data_dir / _SUBDATASET_TO_DIRECTORY[sub_dataset] / '2_timeseries' / temporal_resolution_directory
    return _load_lamah_timeseries_csv_file(forcing_dir / f"ID_{basin}.csv", temporal_resolution)


def load_lamah_discharge(data_dir: Path,
                         basin: str,
                         temporal_resolution: str = '1D',
                         normalize_discharge: bool = False) -> pd.DataFrame:
    """Load discharge data of the LamaH data set.

    Parameters
    ----------
    data_dir : Path
        Path to the LamaH directory. 
    basin : str
        Basin identifier number as string.
    temporal_resolution: str, optional
        Defines if either daily ('1D', default) or hourly ('1H') timeseries data will be loaded.
    normalize_discharge: bool, optional
        If true, normalizes discharge data by basin area, using the 'area_gov' attribute from attribute file.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the forcings data.

    Raises
    ------
    ValueError
        If 'temporal_resolution' is not one of ['1H', '1D'].
    """
    if temporal_resolution not in ['1D', '1H']:
        raise ValueError(
            f"{temporal_resolution} is not a valid choice for 'temporal_resolution'. Must be one of '1H', '1D'.")

    temporal_resolution_directory = 'daily' if temporal_resolution == '1D' else 'hourly'
    streamflow_dir = data_dir / 'D_gauges' / '2_timeseries' / temporal_resolution_directory
    df = _load_lamah_timeseries_csv_file(streamflow_dir / f"ID_{basin}.csv", temporal_resolution)

    # Replace missing discharge values (indicated by -999) to NaN
    df.loc[df["qobs"] < 0, "qobs"] = np.nan

    # If normalize_discharge is True, load attributes to extract upstream area.
    if normalize_discharge:
        attributes = load_lamah_attributes(data_dir, sub_dataset='lamah_a', basins=[basin])
        area = attributes.loc[basin, 'area_gov']
        df["qobs"] = _normalize_discharge(df["qobs"], area, temporal_resolution)

    return df


def load_lamah_attributes(data_dir: Path, sub_dataset: str, basins: List[str] = []) -> pd.DataFrame:
    """Load LamaH catchment attributes.

    Parameters
    ----------
    data_dir : Path
        Path to the LamaH-CE directory.
    sub_dataset: str
        One of {'lamah_a', 'lamah_b', 'lamah_c'}, defining which of the three catchment delinations/sub-datasets 
        (A_basins_total_upstrm, B_basins_intermediate_all, or C_basins_intermediate_lowimp) will be loaded.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes of the sub-dataset as well as the gauge attributes.

    Raises
    ------
    ValueError
        If any of the basin ids is not in the basin index.
    """
    # Load catchment attributes for sub_dataset.
    file_path = data_dir / _SUBDATASET_TO_DIRECTORY[sub_dataset] / "1_attributes" / "Catchment_attributes.csv"
    df_catchment = _load_lamah_attribute_csv_file(file_path)

    # Load gauge attributes
    file_path = data_dir / 'D_gauges' / "1_attributes" / "Gauge_attributes.csv"
    df_gauge = _load_lamah_attribute_csv_file(file_path)

    # Combine both DatFrames.
    df = pd.concat([df_catchment, df_gauge], axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df


def _load_lamah_timeseries_csv_file(filepath: Path, temporal_resolution: str) -> pd.DataFrame:
    """Helper function to load lamah data into time indexed dataframe."""
    df = pd.read_csv(filepath, sep=';', dtype={'YYYY': str, 'MM': str, 'DD': str})
    df["date"] = pd.to_datetime(df.YYYY.map(str) + "/" + df.MM.map(str) + "/" + df.DD.map(str), format="%Y/%m/%d")
    if temporal_resolution == "1D":
        df = df.drop(['YYYY', 'MM', 'DD'], axis=1)
    else:
        df["date"] = df["date"] + pd.to_timedelta(df['hh'], unit='h')
        df = df.drop(['YYYY', 'MM', 'DD', 'hh', 'mm'], axis=1)
    return df.set_index('date')


def _load_lamah_attribute_csv_file(file_path: Path) -> pd.DataFrame:
    """Helper function to load lamah attribute files into basin indexed dataframes."""
    df = pd.read_csv(file_path, sep=";", dtype={'ID': str})
    df = df.set_index("ID")
    df.index.name = "gauge_id"
    return df


def _normalize_discharge(ser: pd.Series, area: float, temporal_resolution: str) -> pd.Series:
    """Helper function to normalize discharge data by basin area"""
    if temporal_resolution == "1H":
        return ser / (area * 1e6) * 1000 * 3600
    else:
        return ser / (area * 1e6) * 1000 * 86400