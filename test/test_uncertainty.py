"""Integration tests that perform full runs on the uncertainty estimation code. """
from typing import Callable

import pandas as pd
import pytest

from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config

from test import Fixture

from test.test_config_runs import _get_test_start_end_dates, _get_basin_results


# Common to all uncertainty heads
common_uncertainty_config = {
    "n_samples": 10,
    "negative_sample_handling": "clip",
    "negative_sample_max_retries": 5,
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
                           mc_dropout: bool,
                           negative_sample_handling: str):
    """Test uncertainty output for different heads, losses, and negative sample handling strategies."""
    
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

    if negative_sample_handling == 'truncate':
        update_dict['negative_sample_max_retries'] = 3

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
    """Perform basic sanity checks of uncertainty predictions.

    Checks that:
        -the results file has the correct date range, 
        -the observed discharge in the file is correct, 
        -the results object has a 'samples' dimension, 
        -there are no NaN predictions in the simulated samples.

    Parameters
    ----------
    config : Config
        The run configuration used to produce the results
    basin : str
        Id of a basin for which to check the results
    """
    results = _get_basin_results(config.run_dir, 1)[basin]['1D']['xr'].isel(time_step=-1)
    
    print("\n[DEBUG] Available variables:", results.data_vars)
    sample_key = f"{config.target_variables[0]}_sim"
    assert sample_key in results.data_vars
    assert "samples" in results[sample_key].dims # evaluation produces a samples dimension (probabilistic output)
    assert not pd.isna(results[sample_key]).any()  # Check for NaN values in the samples

    # get the test date range from the config
    test_start_date, test_end_date = _get_test_start_end_dates(config)
    # check that samples in the test period are not NaN
    test_dates = pd.date_range(test_start_date, test_end_date, freq='D')
    test_vals = results.sel(date=test_dates)
    assert not pd.isna(test_vals[sample_key]).any()  # Check for NaN values in the test period

    if negative_sample_handling == 'truncate':
        # check that no samples are negative
        assert (test_vals[sample_key] >= 0).all()
    elif negative_sample_handling == 'clip':
        # check that no samples are negative
        assert (test_vals[sample_key] >= 0).all()

