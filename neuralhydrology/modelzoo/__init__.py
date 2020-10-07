import torch.nn as nn

from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.modelzoo.embcudalstm import EmbCudaLSTM
from neuralhydrology.modelzoo.lstm import LSTM
from neuralhydrology.modelzoo.odelstm import ODELSTM
from neuralhydrology.modelzoo.mtslstm import MTSLSTM
from neuralhydrology.utils.config import Config

SINGLE_FREQ_MODELS = ["cudalstm", "ealstm", "lstm", "embcudalstm"]


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
    elif cfg.model == "lstm":
        model = LSTM(cfg=cfg)
    elif cfg.model == "embcudalstm":
        model = EmbCudaLSTM(cfg=cfg)
    elif cfg.model == "mtslstm":
        model = MTSLSTM(cfg=cfg)
    elif cfg.model == "odelstm":
        model = ODELSTM(cfg=cfg)
    else:
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_model()`")

    return model
