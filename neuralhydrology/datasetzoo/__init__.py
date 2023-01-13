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

    Currently implemented datasets are 'caravan', 'camels_aus', 'camels_br', 'camels_cl', 'camels_gb', 'camels_us', and 
    'hourly_camels_us', as well as the 'generic' dataset class that can be used for any kind of dataset as long as it is
    in the correct format.

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
    if cfg.dataset.lower() == "camels_us":
        Dataset = CamelsUS
    elif cfg.dataset.lower() == "camels_gb":
        Dataset = CamelsGB
    elif cfg.dataset.lower() == "camels_aus":
        Dataset = CamelsAUS
    elif cfg.dataset.lower() == "camels_br":
        Dataset = CamelsBR
    elif cfg.dataset.lower() == "hourly_camels_us":
        Dataset = HourlyCamelsUS
    elif cfg.dataset.lower() == "camels_cl":
        Dataset = CamelsCL
    elif cfg.dataset.lower() == "generic":
        Dataset = GenericDataset
    elif cfg.dataset.lower() in ["lamah_a", "lamah_b", "lamah_c"]:
        Dataset = LamaH
    elif cfg.dataset.lower() == "caravan":
        Dataset = Caravan
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")

    ds = Dataset(cfg=cfg,
                 is_train=is_train,
                 period=period,
                 basin=basin,
                 additional_features=additional_features,
                 id_to_int=id_to_int,
                 scaler=scaler)
    return ds
