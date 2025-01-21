from typing import Type

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.datasetzoo.camelsaus import CamelsAUS
from neuralhydrology.datasetzoo.camelsbr import CamelsBR
from neuralhydrology.datasetzoo.camelscl import CamelsCL
from neuralhydrology.datasetzoo.camelsgb import CamelsGB
from neuralhydrology.datasetzoo.camelsus import CamelsUS
from neuralhydrology.datasetzoo.caravan import Caravan
from neuralhydrology.datasetzoo.genericdataset import GenericDataset
from neuralhydrology.datasetzoo.hourlycamelsus import HourlyCamelsUS
from neuralhydrology.datasetzoo.lamah import LamaH
from neuralhydrology.utils.config import Config


def get_dataset(cfg: Config,
                is_train: bool,
                period: str,
                basin: str = None,
                additional_features: list = [],
                id_to_int: dict = {},
                scaler: dict = {}) -> BaseDataset:
    """Get data set instance, depending on the run configuration.

    Currently implemented datasets by default are 'caravan', 'camels_aus', 'camels_br', 'camels_cl', 'camels_gb', 'camels_us', and 
    'hourly_camels_us', as well as the 'generic' dataset class that can be used for any kind of dataset as long as it is
    in the correct format.

    New dataset can be added at the begining of runtime using the function register_dataset().

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
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) is(are) read from the appropriate
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

    Returns
    -------
    BaseDataset
        A new data set instance, depending on the run configuration.

    Raises
    ------
    NotImplementedError
        If no data set class is implemented for the 'dataset' argument in the config.
    """
    global _datasetZooRegistry

    return _datasetZooRegistry.instantiate_dataset(cfg, is_train, period, basin, additional_features, id_to_int, scaler)


def register_dataset(key: str, new_class: Type):
    """Register a datatet class that can be used by neuralhydrology. 

    This class must derive from BaseDataset. New dataset calss has to be added at the begining of runtime.

    Parameters
    ----------
    key : str 
        the key of the dataset that is set in the configuration file.

    new_class : Type 
        The new Dataset class to register 

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the new class is not derived from BaseDataset
    """
    global _datasetZooRegistry
    _datasetZooRegistry.register_dataset_class(key, new_class)


class DatasetZooRegistry:
    """Class that register dataset classes taht can be used with neuralhydrology.

       This class is an implementation detatil that must not be used outside this file.
    """

    def __init__(self):
        self.__dataset_class = {}

        self.__dataset_class["camels_us"] = CamelsUS
        self.__dataset_class["camels_gb"] = CamelsGB
        self.__dataset_class["camels_aus"] = CamelsAUS
        self.__dataset_class["camels_br"] = CamelsBR
        self.__dataset_class["hourly_camels_us"] = HourlyCamelsUS
        self.__dataset_class["camels_cl"] = CamelsCL
        self.__dataset_class["generic"] = GenericDataset
        self.__dataset_class["lamah_a"] = LamaH
        self.__dataset_class["lamah_b"] = LamaH
        self.__dataset_class["lamah_c"] = LamaH
        self.__dataset_class["caravan"] = Caravan

    def register_dataset_class(self, key: str, new_class: Type):
        if issubclass(new_class, BaseDataset):
            self.__dataset_class[key, new_class]
        else:
            raise TypeError(f"Class {type(new_class)} is not a subclass of BaseDataset.")

    def instantiate_dataset(self,
                            cfg: Config,
                            is_train: bool,
                            period: str,
                            basin: str = None,
                            additional_features: list = [],
                            id_to_int: dict = {},
                            scaler: dict = {}):
        dataset_key = cfg.dataset.lower()
        Dataset = self.__dataset_class.get(dataset_key, None)
        if Dataset is None:
            raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

        return Dataset(cfg=cfg,
                       is_train=is_train,
                       period=period,
                       basin=basin,
                       additional_features=additional_features,
                       id_to_int=id_to_int,
                       scaler=scaler)


_datasetZooRegistry = DatasetZooRegistry()
