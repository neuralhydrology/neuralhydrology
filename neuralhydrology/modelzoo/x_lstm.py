from typing import Dict
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

# -------- Auto-setup CUDA env for xLSTM --------
# This block automatically sets CUDA_HOME and XLSTM_EXTRA_INCLUDE_PATHS so that xLSTM can find the correct CUDA headers.
# (Source: https://github.com/NX-AI/xlstm?tab=readme-ov-file#using-the-slstm-cuda-kernels) 
import os
import shutil
if "CUDA_HOME" not in os.environ:
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is not None:
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc_path))

if "XLSTM_EXTRA_INCLUDE_PATHS" not in os.environ:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is not None:
        include_path = os.path.join(cuda_home, "include")
        if os.path.exists(include_path):
            os.environ["XLSTM_EXTRA_INCLUDE_PATHS"] = include_path
# -----------------------------------------------

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
    XLSTM_AVAILABLE = True
except ModuleNotFoundError:
    XLSTM_AVAILABLE = False


class XLSTM(BaseModel):
    """Extended Long Short-Term Memory (xLSTM) backbone model class, which relies on https://github.com/NX-AI/xlstm
    
    This class implements the xLSTM backbone, consisting of stacked mLSTM and sLSTM blocks. 
    A transition layer (linear projection) ensures that the input feature dimension matches the hidden size required by the xLSTM stack. 

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ["embedding_net", "transition_layer", "xlstm", "dropout", "head"]

    def __init__(self, cfg: Config):
        super(XLSTM, self).__init__(cfg=cfg)

        if not XLSTM_AVAILABLE:
            raise ModuleNotFoundError("xlstm, and dependencies, required. Please install the xlstm package\
                                       from https://github.com/NX-AI/xLSTM (and ensure CUDA headers are available).")
            
        xlstm_cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=cfg.xlstm_kernel_size,
                    qkv_proj_blocksize=4,
                    num_heads=cfg.xlstm_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "vanilla",          
                    num_heads=cfg.xlstm_heads,
                    conv1d_kernel_size=cfg.xlstm_kernel_size,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=cfg.xlstm_proj_factor,
                    act_fn="gelu",
                ),
            ),
            context_length = cfg.seq_length, 
            num_blocks = cfg.xlstm_num_blocks,
            embedding_dim = cfg.hidden_size,      
            slstm_at = cfg.xlstm_slstm_at,
        )
        self.xlstm = xLSTMBlockStack(xlstm_cfg)
        
        self.embedding_net = InputLayer(cfg)

        # using a linear layer to move from the emdedded_layer dims to the specified hidden size
        self.transition_layer = nn.Linear(self.embedding_net.output_size, self.cfg.hidden_size)

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the xlstm model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            - 'y_hat': model predictions of shape [batch size, sequence length, number of target variables].
        """
        # possibly pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        # reshaping dimensions to what xlstm expects [batch size, sequence length, hidden size]:
        x_d = x_d.transpose(0, 1)
        x_d_transition = self.transition_layer(x_d)

        xlstm_output = self.xlstm(x_d_transition)

        pred = {'y_hat': xlstm_output}
        pred.update(self.head(self.dropout(xlstm_output)))
        return pred


    @staticmethod
    def _as_int(val):
        """NH seq_length could be dict like {'1h': 24}, here convert to int"""
        if isinstance(val, int):
            return val
        if isinstance(val, dict):
            if len(val) != 1:
                raise ValueError(f"Expected a single-valued dict (e.g. {{'1h': 24}}), got {val})")
            return next(iter(val.values()))
        raise TypeError(f"Expected int or single-valued dict, got {type(val)}")