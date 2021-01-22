"""Test for checking that the outputs of the CustomLSTM match those of CudaLSTM and EmbCudaLSTM"""
from typing import Callable

import torch

from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.customlstm import CustomLSTM
from test import Fixture


def test_model_equality(get_config: Fixture[Callable[[str], dict]], custom_lstm_supported_models: Fixture[str]):
    config = get_config('daily_regression_with_embedding')

    # we only need to test for a single data set, input/output setting and model specifications
    config.update_config({
        'dataset': 'camels_us',
        'data_dir': config.data_dir / 'camels_us',
        'target_variables': 'QObs(mm/d)',
        'forcings': 'daymet',
        'dynamic_inputs': ['prcp(mm/day)', 'tmax(C)'],
        'model': custom_lstm_supported_models
    })

    # create random inputs
    data = {
        'x_d': torch.rand((config.batch_size, 50, len(config.dynamic_inputs))),
        'x_s': torch.rand((config.batch_size, len(config.static_attributes)))
    }

    # initialize two random models
    optimized_model = get_model(config)
    custom_lstm = CustomLSTM(config)

    # copy weights from optimized model into custom model implementation
    custom_lstm.copy_weights(optimized_model)

    # get model predictions
    optimized_model.eval()
    custom_lstm.eval()
    with torch.no_grad():
        pred_custom = custom_lstm(data)
        pred_optimized = optimized_model(data)

    # check for consistency in model outputs
    assert torch.allclose(pred_custom["y_hat"], pred_optimized["y_hat"], atol=1e-6)
