from typing import Type

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class DatasetRegistry:
    """Class that registers dataset classes that can be used with neuralhydrology."""

    def __init__(self):
        self.__dataset_class = {}

    def register_dataset_class(self, key: str, new_class: Type):
        """Adds a new dataset class to the dataset registry.

        Parameters
        ----------
        key : str
            The unique identifier for the dataset class. This key will be used in configuration files
            to specify which dataset to use.
        new_class : Type
            The dataset class to register. Must be a subclass of BaseDataset.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the provided class is not a subclass of BaseDataset.

        Examples
        --------
        >>> registry = DatasetZooRegistry()
        >>> registry.register_dataset_class("my_dataset", MyCustomDataset)
        """
        if not issubclass(new_class, BaseDataset):
            raise TypeError(f"Class {type(new_class)} is not a subclass of BaseDataset.")
        self.__dataset_class[key] = new_class

    def instantiate_dataset(self,
                            cfg: Config,
                            is_train: bool,
                            period: str,
                            basin: str = None,
                            additional_features: list = [],
                            id_to_int: dict = {},
                            compute_scaler: bool = True) -> BaseDataset:
        """Creates and returns an instance of a dataset class based on the configuration.

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
        compute_scaler : bool
            Forces the dataset to calculate a new scaler instead of loading a precalculated scaler. Used during training, but
            not finetuning.

        Returns
        -------
        BaseDataset
            An instance of the appropriate dataset class based on the configuration.

        Raises
        ------
        NotImplementedError
            If no dataset class is implemented for the dataset specified in the configuration.
        """
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
                       compute_scaler=compute_scaler)
