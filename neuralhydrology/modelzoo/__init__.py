import warnings

import torch.nn as nn

from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.gru import GRU
from neuralhydrology.modelzoo.odelstm import ODELSTM
from neuralhydrology.modelzoo.mtslstm import MTSLSTM
from neuralhydrology.modelzoo.transformer import Transformer
from neuralhydrology.utils.config import Config

SINGLE_FREQ_MODELS = ["cudalstm", "ealstm", "customlstm", "embcudalstm", "gru"]


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
    if cfg.model in SINGLE_FREQ_MODELS and len(cfg.use_frequencies) > 1:
        raise ValueError(f"Model {cfg.model} does not support multiple frequencies.")

    if cfg.model == "cudalstm":
        model = CudaLSTM(cfg=cfg)
    elif cfg.model == "ealstm":
        model = EALSTM(cfg=cfg)
    elif cfg.model == "customlstm":
        model = CustomLSTM(cfg=cfg)
    elif cfg.model == "lstm":
        warnings.warn(
            "The `LSTM` class has been renamed to `CustomLSTM`. Support for `LSTM` will we dropped in the future.",
            FutureWarning)
        model = CustomLSTM(cfg=cfg)
    elif cfg.model == "gru":
        model = GRU(cfg=cfg)
    elif cfg.model == "embcudalstm":
        model = EmbCudaLSTM(cfg=cfg)
    elif cfg.model == "mtslstm":
        model = MTSLSTM(cfg=cfg)
    elif cfg.model == "odelstm":
        model = ODELSTM(cfg=cfg)
    elif cfg.model == "transformer":
        model = Transformer(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
