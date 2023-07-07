import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any

import pandas as pd
from ruamel.yaml import YAML


class Config(object):
    """Read run configuration from the specified path or dictionary and parse it into a configuration object.

    During parsing, config keys that contain 'dir', 'file', or 'path' will be converted to pathlib.Path instances.
    Configuration keys ending with '_date' will be parsed to pd.Timestamps. The expected format is DD/MM/YYYY.

    Parameters
    ----------
    yml_path_or_dict : Union[Path, dict]
        Either a path to the config file or a dictionary of configuration values.
    dev_mode : bool, optional
        If dev_mode is off, the config creation will fail if there are unrecognized keys in the passed config
        specification. dev_mode can be activated either through this parameter or by setting ``dev_mode: True``
        in `yml_path_or_dict`.

    Raises
    ------
    ValueError
        If the passed configuration specification is neither a Path nor a dict or if `dev_mode` is off (default) and
        the config file or dict contain unrecognized keys.
    """

    # Lists of deprecated config keys and purely informational metadata keys, needed when checking for unrecognized
    # config keys since these keys are not properties of the Config class.
    _deprecated_keys = [
        'static_inputs', 'camels_attributes', 'target_variable', 'embedding_hiddens', 'embedding_activation',
        'embedding_dropout'
    ]
    _metadata_keys = ['package_version', 'commit_hash']

    def __init__(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        if isinstance(yml_path_or_dict, Path):
            self._cfg = Config._read_and_parse_config(yml_path=yml_path_or_dict)
        elif isinstance(yml_path_or_dict, dict):
            self._cfg = Config._parse_config(yml_path_or_dict)
        else:
            raise ValueError(f'Cannot create a config from input of type {type(yml_path_or_dict)}.')

        if not (self._cfg.get('dev_mode', False) or dev_mode):
            Config._check_cfg_keys(self._cfg)

    def as_dict(self) -> dict:
        """Return run configuration as dictionary.
        
        Returns
        -------
        dict
            The run configuration, as defined in the .yml file.
        """
        return self._cfg

    def dump_config(self, folder: Path, filename: str = 'config.yml'):
        """Save the run configuration as a .yml file to disk.

        Parameters
        ----------
        folder : Path
            Folder in which the configuration will be stored.
        filename : str, optional
            Name of the file that will be stored. Default: 'config.yml'.

        Raises
        ------
        FileExistsError
            If the specified folder already contains a file named `filename`.
        """
        yml_path = folder / filename
        if not yml_path.exists():
            with yml_path.open('w') as fp:
                temp_cfg = {}
                for key, val in self._cfg.items():
                    if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(str(elem))
                            temp_cfg[key] = temp_list
                        else:
                            temp_cfg[key] = str(val)
                    elif key.endswith('_date'):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(elem.strftime(format="%d/%m/%Y"))
                            temp_cfg[key] = temp_list
                        else:
                            # Ignore None's due to e.g. using a per_basin_period_file
                            if isinstance(val, pd.Timestamp):
                                temp_cfg[key] = val.strftime(format="%d/%m/%Y")
                    else:
                        temp_cfg[key] = val

                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            raise FileExistsError(yml_path)

    def update_config(self, yml_path_or_dict: Union[Path, dict], dev_mode: bool = False):
        """Update config arguments.
        
        Useful e.g. in the context of fine-tuning or when continuing to train from a checkpoint to adapt for example the
        learning rate, train basin files or anything else.
        
        Parameters
        ----------
        yml_path_or_dict : Union[Path, dict]
            Either a path to the new config file or a dictionary of configuration values. Each argument specified in
            this file will overwrite the existing config argument.
        dev_mode : bool, optional
            If dev_mode is off, the config creation will fail if there are unrecognized keys in the passed config
            specification. dev_mode can be activated either through this parameter or by setting ``dev_mode: True``
            in `yml_path_or_dict`.

        Raises
        ------
        ValueError
            If the passed configuration specification is neither a Path nor a dict, or if `dev_mode` is off (default)
            and the config file or dict contain unrecognized keys.
        """
        new_config = Config(yml_path_or_dict, dev_mode=dev_mode)

        self._cfg.update(new_config.as_dict())

    def _get_value_verbose(self, key: str) -> Union[float, int, str, list, dict, Path, pd.Timestamp]:
        """Use this function internally to return attributes of the config that are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key]

    @staticmethod
    def _as_default_list(value: Any) -> list:
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    def _as_default_dict(value: Any) -> dict:
        if value is None:
            return {}
        elif isinstance(value, dict):
            return value
        else:
            raise RuntimeError(f"Incompatible type {type(value)}. Expected `dict` or `None`.")

    @staticmethod
    def _check_cfg_keys(cfg: dict):
        """Checks the config for unknown keys. """
        property_names = [p for p in dir(Config) if isinstance(getattr(Config, p), property)]

        unknown_keys = [
            k for k in cfg.keys()
            if k not in property_names and k not in Config._deprecated_keys and k not in Config._metadata_keys
        ]
        if unknown_keys:
            raise ValueError(f'{unknown_keys} are not recognized config keys.')

    @staticmethod
    def _parse_config(cfg: dict) -> dict:
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime indexs
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

            else:
                pass

        # Check forecast sequence length.
        if cfg.get('forecast_seq_length'):
            if cfg['forecast_seq_length'] >= cfg['seq_length']:
                raise ValueError('Forecast sequence length must be < sequence length.')
            if cfg.get('forecast_overlap'):
                if cfg['forecast_overlap'] > cfg['forecast_seq_length']:
                    raise ValueError('Forecast overlap must be <= forecast_seq_length.')

        # Check autoregressive inputs.
        if 'autoregressive_inputs' in cfg:
            if len(cfg['autoregressive_inputs']) > 1:
                raise ValueError('Currently only one autoregressive input is supported.')
            if cfg['autoregressive_inputs'] and len(cfg['target_variables']) > 1:
                raise ValueError('Autoregressive models currently only support a single target variable.')
            if not cfg['autoregressive_inputs'][0].startswith(cfg['target_variables'][0]):
                raise ValueError('Autoregressive input must be a lagged version of the target variable.')

        # Add more config parsing if necessary
        return cfg

    @staticmethod
    def _read_and_parse_config(yml_path: Path):
        if yml_path.exists():
            with yml_path.open('r') as fp:
                yaml = YAML(typ="safe")
                cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(yml_path)

        cfg = Config._parse_config(cfg)

        return cfg

    @property
    def additional_feature_files(self) -> List[Path]:
        return self._as_default_list(self._cfg.get("additional_feature_files", None))

    @property
    def allow_subsequent_nan_losses(self) -> int:
        return self._cfg.get("allow_subsequent_nan_losses", 0)

    @property
    def autoregressive_inputs(self) -> Union[List[str], Dict[str, List[str]]]:
        return self._as_default_list(self._cfg.get("autoregressive_inputs", []))

    @property
    def base_run_dir(self) -> Path:
        return self._get_value_verbose("base_run_dir")

    @base_run_dir.setter
    def base_run_dir(self, folder: Path):
        self._cfg["base_run_dir"] = folder

    @property
    def batch_size(self) -> int:
        return self._get_value_verbose("batch_size")

    @property
    def bidirectional_stacked_forecast_lstm(self) -> bool:
        return self._cfg.get("bidirectional_stacked_forecast_lstm", False)

    @property
    def cache_validation_data(self) -> bool:
        return self._cfg.get("cache_validation_data", True)

    @property
    def checkpoint_path(self) -> Path:
        return self._cfg.get("checkpoint_path", None)

    @property
    def clip_gradient_norm(self) -> float:
        return self._cfg.get("clip_gradient_norm", None)

    @property
    def clip_targets_to_zero(self) -> List[str]:
        return self._as_default_list(self._cfg.get("clip_targets_to_zero", []))

    @property
    def continue_from_epoch(self) -> int:
        return self._cfg.get("continue_from_epoch", None)

    @property
    def custom_normalization(self) -> dict:
        return self._as_default_dict(self._cfg.get("custom_normalization", {}))

    @property
    def data_dir(self) -> Path:
        return self._get_value_verbose("data_dir")

    @property
    def dataset(self) -> str:
        return self._get_value_verbose("dataset")

    @property
    def device(self) -> str:
        return self._cfg.get("device", None)

    @device.setter
    def device(self, device: str):
        if device == "cpu" or device.startswith("cuda:"):
            self._cfg["device"] = device
        else:
            raise ValueError("'device' must be either 'cpu' or a 'cuda:X', with 'X' being the GPU ID.")

    @property
    def duplicate_features(self) -> dict:
        duplicate_features = self._cfg.get("duplicate_features", {})
        if duplicate_features is None:
            return {}
        elif isinstance(duplicate_features, dict):
            return duplicate_features
        elif isinstance(duplicate_features, list):
            return {feature: 1 for feature in duplicate_features}
        elif isinstance(duplicate_features, str):
            return {duplicate_features: 1}
        else:
            raise RuntimeError(f"Unsupported type {type(duplicate_features)} for 'duplicate_features' argument.")

    @property
    def dynamic_inputs(self) -> Union[List[str], Dict[str, List[str]]]:
        return self._get_value_verbose("dynamic_inputs")

    @property
    def dynamics_embedding(self) -> dict:
        embedding_spec = self._cfg.get("dynamics_embedding", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def epochs(self) -> int:
        return self._get_value_verbose("epochs")

    @property
    def evolving_attributes(self) -> List[str]:
        if "evolving_attributes" in self._cfg.keys():
            return self._as_default_list(self._cfg["evolving_attributes"])
        elif "static_inputs" in self._cfg.keys():
            warnings.warn("'static_inputs' will be deprecated. Use 'evolving_attributes' in the future", FutureWarning)
            return self._as_default_list(self._cfg["static_inputs"])
        else:
            return []

    @property
    def experiment_name(self) -> str:
        if self._cfg.get("experiment_name", None) is None:
            return "run"
        else:
            return self._cfg["experiment_name"]

    @property
    def finetune_modules(self) -> Union[List[str], Dict[str, str]]:
        finetune_modules = self._cfg.get("finetune_modules", [])
        if finetune_modules is None:
            return []
        elif isinstance(finetune_modules, str):
            return [finetune_modules]
        elif isinstance(finetune_modules, dict) or isinstance(finetune_modules, list):
            return finetune_modules
        else:
            raise ValueError(f"Unknown data type {type(finetune_modules)} for 'finetune_modules' argument.")

    @property
    def forecast_network(self) -> dict:
        embedding_spec = self._cfg.get("forecast_network", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def forecast_hidden_size(self) -> int:
        return self._cfg.get("forecast_hidden_size", self.hidden_size)

    @property
    def forecast_inputs(self) -> List[str]:
        return self._cfg.get("forecast_inputs", [])

    @property
    def forecast_overlap(self) -> int:
        return self._cfg.get("forecast_overlap", None)

    @property
    def forecast_seq_length(self) -> int:
        return self._cfg.get("forecast_seq_length", None)

    @property
    def forcings(self) -> List[str]:
        return self._as_default_list(self._get_value_verbose("forcings"))

    @property
    def save_git_diff(self) -> bool:
        return self._cfg.get('save_git_diff', False)

    @property
    def state_handoff_network(self) -> dict:
        embedding_spec = self._cfg.get("state_handoff_network", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def head(self) -> str:
        if self.model == "mclstm":
            return ''
        else:
            return self._get_value_verbose("head")

    @property
    def hindcast_inputs(self) -> List[str]:
        return self._cfg.get("hindcast_inputs", [])

    @property
    def hidden_size(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("hidden_size")

    @property
    def hindcast_hidden_size(self) -> Union[int, Dict[str, int]]:
        return self._cfg.get("hindcast_hidden_size", self.hidden_size)

    @property
    def hydroatlas_attributes(self) -> List[str]:
        return self._as_default_list(self._cfg.get("hydroatlas_attributes", []))

    @property
    def img_log_dir(self) -> Path:
        return self._cfg.get("img_log_dir", None)

    @img_log_dir.setter
    def img_log_dir(self, folder: Path):
        self._cfg["img_log_dir"] = folder

    @property
    def initial_forget_bias(self) -> float:
        return self._cfg.get("initial_forget_bias", None)

    @property
    def is_continue_training(self) -> bool:
        return self._cfg.get("is_continue_training", False)

    @is_continue_training.setter
    def is_continue_training(self, flag: bool):
        self._cfg["is_continue_training"] = flag

    @property
    def is_finetuning(self) -> bool:
        return self._cfg.get("is_finetuning", False)

    @is_finetuning.setter
    def is_finetuning(self, flag: bool):
        self._cfg["is_finetuning"] = flag

    @property
    def lagged_features(self) -> dict:
        return self._as_default_dict(self._cfg.get("lagged_features", {}))

    @property
    def learning_rate(self) -> Dict[int, float]:
        if ("learning_rate" in self._cfg.keys()) and (self._cfg["learning_rate"] is not None):
            if isinstance(self._cfg["learning_rate"], float):
                return {0: self._cfg["learning_rate"]}
            elif isinstance(self._cfg["learning_rate"], dict):
                return self._cfg["learning_rate"]
            else:
                raise ValueError("Unsupported data type for learning rate. Use either dict (epoch to float) or float.")
        else:
            raise ValueError("No learning rate specified in the config (.yml).")

    @property
    def log_interval(self) -> int:
        return self._cfg.get("log_interval", 10)

    @property
    def log_n_figures(self) -> int:
        if (self._cfg.get("log_n_figures", None) is None) or (self._cfg["log_n_figures"] < 1):
            return 0
        else:
            return self._cfg["log_n_figures"]

    @property
    def log_tensorboard(self) -> bool:
        return self._cfg.get("log_tensorboard", True)

    @property
    def loss(self) -> str:
        return self._get_value_verbose("loss")

    @loss.setter
    def loss(self, loss: str):
        self._cfg["loss"] = loss

    @property
    def mass_inputs(self) -> List[str]:
        return self._as_default_list(self._cfg.get("mass_inputs", []))

    @property
    def mc_dropout(self) -> bool:
        return self._cfg.get("mc_dropout", False)

    @property
    def metrics(self) -> Union[List[str], Dict[str, List[str]]]:
        return self._cfg.get("metrics", [])

    @metrics.setter
    def metrics(self, metrics: Union[str, List[str], Dict[str, List[str]]]):
        self._cfg["metrics"] = metrics

    @property
    def model(self) -> str:
        return self._get_value_verbose("model")

    @property
    def n_distributions(self) -> int:
        return self._get_value_verbose("n_distributions")

    @property
    def n_samples(self) -> int:
        return self._get_value_verbose("n_samples")

    @property
    def n_taus(self) -> int:
        return self._get_value_verbose("n_taus")

    @property
    def negative_sample_handling(self) -> str:
        return self._cfg.get("negative_sample_handling", None)

    @property
    def negative_sample_max_retries(self) -> int:
        return self._get_value_verbose("negative_sample_max_retries")

    @property
    def no_loss_frequencies(self) -> list:
        return self._as_default_list(self._cfg.get("no_loss_frequencies", []))

    @property
    def num_workers(self) -> int:
        return self._cfg.get("num_workers", 0)

    @property
    def number_of_basins(self) -> int:
        return self._get_value_verbose("number_of_basins")

    @number_of_basins.setter
    def number_of_basins(self, num_basins: int):
        self._cfg["number_of_basins"] = num_basins

    @property
    def ode_method(self) -> str:
        return self._cfg.get("ode_method", "euler")

    @property
    def ode_num_unfolds(self) -> int:
        return self._cfg.get("ode_num_unfolds", 4)

    @property
    def ode_random_freq_lower_bound(self) -> str:
        return self._get_value_verbose("ode_random_freq_lower_bound")

    @property
    def optimizer(self) -> str:
        return self._get_value_verbose("optimizer")

    @property
    def output_activation(self) -> str:
        return self._cfg.get("output_activation", "linear")

    @property
    def output_dropout(self) -> float:
        return self._cfg.get("output_dropout", 0.0)

    @property
    def per_basin_test_periods_file(self) -> Path:
        return self._cfg.get("per_basin_test_periods_file", None)

    @property
    def per_basin_train_periods_file(self) -> Path:
        return self._cfg.get("per_basin_train_periods_file", None)

    @property
    def per_basin_validation_periods_file(self) -> Path:
        return self._cfg.get("per_basin_validation_periods_file", None)

    @property
    def predict_last_n(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("predict_last_n")

    @property
    def random_holdout_from_dynamic_features(self) -> Dict[str, float]:
        return self._as_default_dict(self._cfg.get("random_holdout_from_dynamic_features", {}))

    @property
    def rating_curve_file(self) -> Path:
        return self._get_value_verbose("rating_curve_file")

    @property
    def regularization(self) -> List[Union[str, Tuple[str, float]]]:
        return self._as_default_list(self._cfg.get("regularization", []))

    @property
    def run_dir(self) -> Path:
        return self._cfg.get("run_dir", None)

    @run_dir.setter
    def run_dir(self, folder: Path):
        self._cfg["run_dir"] = folder

    @property
    def save_train_data(self) -> bool:
        return self._cfg.get("save_train_data", False)

    @property
    def save_all_output(self) -> bool:
        return self._cfg.get('save_all_output', False)

    @property
    def save_validation_results(self) -> bool:
        return self._cfg.get("save_validation_results", False)

    @property
    def save_weights_every(self) -> int:
        return self._cfg.get("save_weights_every", 1)

    @property
    def seed(self) -> int:
        return self._cfg.get("seed", None)

    @property
    def transformer_nlayers(self) -> int:
        return self._get_value_verbose("transformer_nlayers")

    @property
    def transformer_positional_encoding_type(self) -> str:
        return self._get_value_verbose("transformer_positional_encoding_type")

    @property
    def transformer_dim_feedforward(self) -> int:
        return self._get_value_verbose("transformer_dim_feedforward")

    @property
    def transformer_positional_dropout(self) -> float:
        return self._get_value_verbose("transformer_positional_dropout")

    @property
    def transformer_dropout(self) -> float:
        return self._get_value_verbose("transformer_dropout")

    @property
    def transformer_nheads(self) -> int:
        return self._get_value_verbose("transformer_nheads")

    @seed.setter
    def seed(self, seed: int):
        if self._cfg.get("seed", None) is None:
            self._cfg["seed"] = seed
        else:
            raise RuntimeError("Seed was already specified and can't be replaced")

    @property
    def seq_length(self) -> Union[int, Dict[str, int]]:
        return self._get_value_verbose("seq_length")

    @property
    def shared_mtslstm(self) -> bool:
        return self._cfg.get("shared_mtslstm", False)

    @property
    def static_attributes(self) -> List[str]:
        if "static_attributes" in self._cfg.keys():
            return self._as_default_list(self._cfg["static_attributes"])
        elif "camels_attributes" in self._cfg.keys():
            warnings.warn("'camels_attributes' will be deprecated. Use 'static_attributes' in the future",
                          FutureWarning)
            return self._as_default_list(self._cfg["camels_attributes"])
        else:
            return []

    @property
    def statics_embedding(self) -> dict:
        embedding_spec = self._cfg.get("statics_embedding", None)

        if embedding_spec is None:
            return None
        return self._get_embedding_spec(embedding_spec)

    @property
    def target_loss_weights(self) -> List[float]:
        return self._cfg.get("target_loss_weights", None)

    @property
    def target_noise_std(self) -> float:
        if (self._cfg.get("target_noise_std", None) is None) or (self._cfg["target_noise_std"] == 0):
            return None
        else:
            return self._cfg["target_noise_std"]

    @property
    def target_variables(self) -> List[str]:
        if "target_variables" in self._cfg.keys():
            return self._cfg["target_variables"]
        elif "target_variable" in self._cfg.keys():
            warnings.warn("'target_variable' will be deprecated. Use 'target_variables' in the future", FutureWarning)
            return self._cfg["target_variable"]
        else:
            raise ValueError("No target variables ('target_variables') defined in the config.")

    @property
    def tau_down(self) -> float:
        return self._get_value_verbose("tau_down")

    @property
    def tau_up(self) -> float:
        return self._get_value_verbose("tau_up")

    @property
    def test_basin_file(self) -> Path:
        return self._get_value_verbose("test_basin_file")

    @property
    def test_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("test_end_date")

    @property
    def test_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("test_start_date")

    @property
    def timestep_counter(self) -> bool:
        return self._cfg.get("timestep_counter", False)
        
    @property
    def train_basin_file(self) -> Path:
        return self._get_value_verbose("train_basin_file")

    @property
    def train_data_file(self) -> Path:
        return self._cfg.get("train_data_file", None)

    @property
    def train_dir(self) -> Path:
        return self._cfg.get("train_dir", None)

    @train_dir.setter
    def train_dir(self, folder: Path):
        self._cfg["train_dir"] = folder

    @property
    def train_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("train_end_date")

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("train_start_date")

    @property
    def transfer_mtslstm_states(self) -> Dict[str, str]:
        return self._cfg.get("transfer_mtslstm_states", {'h': 'linear', 'c': 'linear'})

    @property
    def umal_extend_batch(self) -> bool:
        return self._cfg.get("umal_extend_batch", False)

    @property
    def use_basin_id_encoding(self) -> bool:
        return self._cfg.get("use_basin_id_encoding", False)

    @property
    def use_frequencies(self) -> List[str]:
        return self._as_default_list(self._cfg.get("use_frequencies", []))

    @property
    def validate_every(self) -> int:
        if (self._cfg.get("validate_every", None) is None) or (self._cfg["validate_every"] < 1):
            return None
        else:
            return self._cfg["validate_every"]

    @property
    def validate_n_random_basins(self) -> int:
        if (self._cfg.get("validate_n_random_basins", None) is None) or (self._cfg["validate_n_random_basins"] < 1):
            return 0
        else:
            return self._cfg["validate_n_random_basins"]

    @validate_n_random_basins.setter
    def validate_n_random_basins(self, n_basins: int):
        self._cfg["validate_n_random_basins"] = n_basins

    @property
    def validation_basin_file(self) -> Path:
        return self._get_value_verbose("validation_basin_file")

    @property
    def validation_end_date(self) -> pd.Timestamp:
        return self._get_value_verbose("validation_end_date")

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return self._get_value_verbose("validation_start_date")

    @property
    def verbose(self) -> int:
        """Defines level of verbosity.

        0: Only log info messages, don't show progress bars
        1: Log info messages and show progress bars

        Returns
        -------
        int
            Level of verbosity.
        """
        return self._cfg.get("verbose", 1)

    def _get_embedding_spec(self, embedding_spec: dict) -> dict:
        if isinstance(embedding_spec, bool) and embedding_spec:  #
            msg = [
                "The semantics of 'dynamics/statics_embedding' have changed, and the associated arguments "
                "'embedding_hiddens/activation/dropout' are deprecated. The old specifications may no longer work in "
                "the future. Specify embeddings as a dict in dynamics/statics_embedding instead."
            ]
            warnings.warn(" ".join(msg), FutureWarning)
            return {
                'type': 'fc',
                'hiddens': self._as_default_list(self._cfg.get("embedding_hiddens", [])),
                'activation': self._cfg.get("embedding_activation", "tanh"),
                'dropout': self._cfg.get("embedding_dropout", 0.0)
            }

        return {
            'type': embedding_spec.get('type', 'fc'),
            'hiddens': self._as_default_list(embedding_spec.get('hiddens', [])),
            'activation': embedding_spec.get('activation', 'tanh'),
            'dropout': embedding_spec.get('dropout', 0.0)
        }
