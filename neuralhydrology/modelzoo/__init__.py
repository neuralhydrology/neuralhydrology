import warnings

import torch.nn as nn

from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.gru import GRU
from neuralhydrology.modelzoo.mclstm import MCLSTM
from neuralhydrology.modelzoo.mtslstm import MTSLSTM
from neuralhydrology.modelzoo.odelstm import ODELSTM
from neuralhydrology.modelzoo.transformer import Transformer
from neuralhydrology.utils.config import Config

SINGLE_FREQ_MODELS = ["cudalstm", "ealstm", "customlstm", "embcudalstm", "gru", "transformer", "mclstm"]


def get_model(cfg: Config) -> nn.Module:
    """Get model object, depending on the run configuration.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    nn.Module
        A new model instance of the type specified in the config.
    """
    if cfg.model.lower() in SINGLE_FREQ_MODELS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"Model {cfg.model} does not support multiple frequencies.")

    if cfg.model.lower() != "mclstm" and cfg.mass_inputs:
        raise ValueError(f"The use of 'mass_inputs' with {cfg.model} is not supported.")

    if cfg.model.lower() == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "ealstm":
        model = EALSTM(cfg=cfg)
    elif cfg.model.lower() == "customlstm":
        model = CustomLSTM(cfg=cfg)
    elif cfg.model.lower() == "lstm":
        warnings.warn(
            "The `LSTM` class has been renamed to `CustomLSTM`. Support for `LSTM` will we dropped in the future.",
            FutureWarning)
        model = CustomLSTM(cfg=cfg)
    elif cfg.model.lower() == "gru":
        model = GRU(cfg=cfg)
    elif cfg.model.lower() == "embcudalstm":
        model = EmbCudaLSTM(cfg=cfg)
    elif cfg.model.lower() == "mtslstm":
        model = MTSLSTM(cfg=cfg)
    elif cfg.model.lower() == "odelstm":
        model = ODELSTM(cfg=cfg)
    elif cfg.model.lower() == "mclstm":
        model = MCLSTM(cfg=cfg)
    elif cfg.model.lower() == "transformer":
        model = Transformer(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
