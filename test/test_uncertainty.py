"""Integration tests that perform full runs on the uncertainty estimation code. """
from typing import Callable

import pandas as pd
import numpy as np
import pytest

from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config

from test import Fixture

from test.test_config_runs import get_test_start_end_dates, get_basin_results


# Common to all uncertainty heads
common_uncertainty_config = {
    "n_samples": 10,
    "negative_sample_handling": "clip",
    "negative_sample_max_retries": 1,
    "mc_dropout": False
}

# Head-specific configs (only fields that differ)
head_configs = {
    "umal": {
        "head": "umal",
        "loss": "UMALLoss",
        "n_taus": 32,
        "umal_extend_batch": True,
        "tau_down": 0.1,
        "tau_up": 0.9,
    },
    "cmal": {
        "head": "cmal",
        "loss": "CMALLoss",
        "n_distributions": 3,
    },
    "gmm": {
        "head": "gmm",
        "loss": "GMMLoss",
        "n_distributions": 3,
    }
}

def build_full_config(head):
    """Builds a full config dictionary for the given head."""
    config = head_configs[head].copy()

    # Only add common uncertainty fields if the head supports them
    if head in ["umal", "cmal", "gmm"]:
        config.update(common_uncertainty_config)
    
    return config


@pytest.mark.parametrize("mc_dropout", [False, True])
@pytest.mark.parametrize("negative_sample_handling", ["none", "clip", "truncate"])
@pytest.mark.parametrize("head", ["umal", "cmal", "gmm"])
def test_daily_uncertainty(get_config: Fixture[Callable[[str], dict]],
                           daily_dataset: Fixture[str],
                           single_timescale_forcings: Fixture[str],
                           head: str,
                           negative_sample_handling: str,
                           mc_dropout: bool):
    """Test probabilistic output consistency across different heads, dropout settings, and negative sample handling modes.

    This test verifies that training and evaluation produce valid uncertainty outputs
    for UMAL, CMAL, and GMM heads under various negative sample handling strategies
    ('none', 'clip', 'truncate') and with or without Monte Carlo dropout.
    """
    
    config = get_config('daily_uncertainty')  # Load a generic daily config

    basin = '01022500'

    # Dynamically build the basic config
    update_dict = {
        'head': head,
        'dataset': daily_dataset['dataset'],
        'data_dir': config.data_dir / daily_dataset['dataset'],
        'negative_sample_handling': negative_sample_handling,
        'mc_dropout': mc_dropout,
        'target_variables': daily_dataset['target'],
        'forcings': single_timescale_forcings['forcings'],
        'dynamic_inputs': single_timescale_forcings['variables'],
    }

    config.update_config(update_dict)

    # Merge in the head-specific parameters dynamically
    head_specific_config = build_full_config(head)
    config.update_config(head_specific_config)

    # Start training and evaluation
    print(f"\n[TEST] head={head}, loss={config.head}, neg_sample_handling={negative_sample_handling}")
    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    # Sanity check of uncertainty outputs
    _check_uncertainty_output(config, basin, negative_sample_handling)


def _check_uncertainty_output(config: Config, basin: str, negative_sample_handling: str):
    """Perform sanity checks on uncertainty prediction outputs for a given basin.

    This function verifies that:
        - The results file contains the expected simulated target variable.
        - The simulated results have a 'samples' dimension with the correct number of samples.
        - The simulated results fully cover the configured test date range.
        - No NaN or infinite values are present in the simulated samples.
        - The 'samples' dimension exists and has no NaN entries.
        - If negative sample handling was set to 'clip', all negative values are within floating-point tolerance of zero.
        - negative sample handling = 'truncate' testing is not yet implemented

    Parameters
    ----------
    config : Config
        The configuration object used for model training and evaluation.
    basin : str
        The ID of the basin for which predictions are being checked.
    negative_sample_handling : str
        Strategy used to handle negative samples during training ("truncate" or "clip"),
        which determines whether non-negativity is strictly enforced in the output.
    """
    results = get_basin_results(config.run_dir, 1)[basin]['1D']['xr'].isel(time_step=-1)
    
    sample_key = f"{config.target_variables[0]}_sim"
    assert sample_key in results.data_vars, f"Expected {sample_key} in results, got {results.data_vars}"
    # The model evaluation should produce a 'samples' dimension (probabilistic output)
    assert "samples" in results[sample_key].dims, f'"samples" dimension not found in {sample_key}' 

    # Assert the number of samples in the output matches the config
    assert results[sample_key].shape[1] == config.n_samples, f'Expected {config.n_samples} samples, got {results[sample_key].shape[0]}'

    # Check that the results file has the correct date range
    test_start_date, test_end_date = get_test_start_end_dates(config)
    assert pd.to_datetime(results['date'].values[0]) == test_start_date.floor('D')
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.floor('D')

    # Check that no NaN values are present in the generated samples 
    test_dates = pd.date_range(test_start_date, test_end_date, freq='D')
    test_vals = results.sel(date=test_dates)
    # Assert all sample values are finite
    assert np.isfinite(test_vals[sample_key].values).all(), f'Found non-finite values in {sample_key}'

    negative_vals = test_vals[sample_key].values[test_vals[sample_key].values < 0]
    if negative_sample_handling == "clip":
        # For 'clip', we expect all non-negative values
        assert np.allclose(negative_vals, 0.0, atol=1e-6), (
            f"Found negative samples below tolerance. Smallest val: {np.min(negative_vals)}"
        )
    elif negative_sample_handling == "truncate":
        # TODO: Implement a more robust check for 'truncate' handling
        # where resampling is done to ensure non-negativity
        pass
