from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.utils.config import Config


class BaseModel(nn.Module):

    def __init__(self, cfg: Config):
        super(BaseModel, self).__init__()
        self.cfg = cfg

        self.output_size = len(cfg.target_variables)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def sample(self, data: Dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
        predict_last_n = self.cfg.predict_last_n
        samples = torch.zeros(data['x_d'].shape[0], predict_last_n, n_samples)
        for i in range(n_samples):
            prediction = self.forward(data)
            samples[:, -predict_last_n:, i] = prediction['y_hat'][:, -predict_last_n:, 0]

        return samples