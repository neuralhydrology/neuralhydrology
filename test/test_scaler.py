"""Unit tests for Scaler."""
import os
from pathlib import Path
from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from neuralhydrology.datautils import utils
from neuralhydrology.datautils.scaler import Scaler, SCALER_FILE_NAME, _get_center, _get_scale

# Set a fixed seed for reproducible tests
np.random.seed(42)

# --- Fixtures for Test Data and Scaler Instances ---

@pytest.fixture
def sample_dataset_basic():
    """Provides a basic dataset for calculation and scaling."""
    data = {
        'temp': (('date', 'basin'), np.array([[10.0, 20.0], [15.0, 25.0], [12.0, 22.0]])),
        'pressure': (('date', 'basin'), np.array([[1000.0, 1010.0], [1005.0, 1015.0], [1002.0, 1012.0]])),
    }
    coords = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
        'basin': ['BASIN_1', 'BASIN_2'],
    }
    return xr.Dataset(data, coords=coords)

@pytest.fixture
def sample_dataset_with_constant_var():
    """Dataset with a constant variable for testing zero scale."""
    data = {
        'constant_val': (('x', 'y'), np.array([[5.0, 5.0], [5.0, 5.0]])),
        'varying_val': (('x', 'y'), np.array([[1.0, 2.0], [3.0, 4.0]])),
    }
    coords = {
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'basin': ['BASIN_1', 'BASIN_2'],
    }
    return xr.Dataset(data, coords=coords)

@pytest.fixture
def tmp_scaler_dir(tmp_path):
    """Provides a temporary directory for scaler file operations."""
    return tmp_path / "scaler_data"

@pytest.fixture
def scaler_instance_calculated(tmp_scaler_dir, sample_dataset_basic):
    """Provides a Scaler instance with calculated stats (no zero scales)."""
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=sample_dataset_basic)
    return scaler

# --- Test Helper Functions (_get_center, _get_scale) ---

def test_get_center_none():
    da = xr.DataArray(np.array([1, 2, 3]))
    assert _get_center(da, 'none') == 0.0

def test_get_center_mean():
    da = xr.DataArray(np.array([1, 2, 3]))
    np.testing.assert_allclose(_get_center(da, 'mean'), 2.0)

def test_get_center_median():
    da = xr.DataArray(np.array([1, 5, 2]))
    np.testing.assert_allclose(_get_center(da, 'median'), 2.0)

def test_get_center_min():
    da = xr.DataArray(np.array([1, 5, 2]))
    np.testing.assert_allclose(_get_center(da, 'min'), 1.0)

def test_get_center_unknown_type():
    da = xr.DataArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="Unknown centering method"):
        _get_center(da, 'invalid_type')

def test_get_scale_none():
    da = xr.DataArray(np.array([1, 2, 3]))
    assert _get_scale(da, 'none') == 1.0

def test_get_scale_std():
    da = xr.DataArray(np.array([1, 2, 3]))
    np.testing.assert_allclose(_get_scale(da, 'std'), np.std([1, 2, 3]))

def test_get_scale_minmax():
    da = xr.DataArray(np.array([1, 5, 2]))
    np.testing.assert_allclose(_get_scale(da, 'minmax'), 4.0)

def test_get_scale_unknown_type():
    da = xr.DataArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="Unknown scaling method"):
        _get_scale(da, 'invalid_type')

# --- Test Scaler.__init__ ---

def test_scaler_init_calculate_with_dataset(tmp_scaler_dir, sample_dataset_basic):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=sample_dataset_basic)
    assert scaler.scaler is not None
    assert isinstance(scaler.scaler, xr.Dataset)
    assert 'temp' in scaler.scaler.data_vars
    assert 'temp_obs' in scaler.scaler.data_vars
    assert 'temp_sim' in scaler.scaler.data_vars
    assert 'parameter' in scaler.scaler.coords
    assert set(scaler.scaler.coords['parameter'].values) == {'center', 'scale', 'mean', 'std'}

def test_scaler_init_calculate_no_dataset(tmp_scaler_dir):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    assert scaler.scaler is None

def test_scaler_init_load_with_dataset_raises_error(tmp_scaler_dir, sample_dataset_basic):
    with pytest.raises(ValueError, match="Do not pass a dataset if you are loading a pre-calculated scaler."):
        Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=False, dataset=sample_dataset_basic)

# --- Test Scaler.calculate ---

def test_scaler_calculate_parameters_correctness(tmp_scaler_dir, sample_dataset_basic):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    scaler.calculate(sample_dataset_basic)

    # Verify 'mean' and 'std' parameters
    expected_mean_temp = sample_dataset_basic['temp'].mean().item()
    expected_std_temp = sample_dataset_basic['temp'].std().item()
    expected_mean_pressure = sample_dataset_basic['pressure'].mean().item()
    expected_std_pressure = sample_dataset_basic['pressure'].std().item()

    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='mean').item(), expected_mean_temp, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='std').item(), expected_std_temp, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['pressure'].sel(parameter='mean').item(), expected_mean_pressure, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['pressure'].sel(parameter='std').item(), expected_std_pressure, rtol=1e-5)

    # Verify 'center' and 'scale' are also 'mean' and 'std' by default
    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='center').item(), expected_mean_temp, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='scale').item(), expected_std_temp, rtol=1e-5)

    # Verify _obs and _sim suffixes are present and refer to the same data
    assert 'temp_obs' in scaler.scaler.data_vars
    assert 'temp_sim' in scaler.scaler.data_vars
    np.testing.assert_allclose(scaler.scaler['temp_obs'].values, scaler.scaler['temp'].values)
    np.testing.assert_allclose(scaler.scaler['temp_sim'].values, scaler.scaler['temp'].values)

def test_scaler_calculate_custom_normalization(tmp_scaler_dir, sample_dataset_basic):
    custom_norm = {
        'temp': {'centering': 'min', 'scaling': 'minmax'},
        'pressure': {'centering': 'median', 'scaling': 'none'}
    }
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None, custom_normalization=custom_norm)
    scaler.calculate(sample_dataset_basic)

    # Verify custom 'center' and 'scale'
    expected_center_temp = sample_dataset_basic['temp'].min().item()
    expected_scale_temp = (sample_dataset_basic['temp'].max() - sample_dataset_basic['temp'].min()).item()
    expected_center_pressure = sample_dataset_basic['pressure'].median().item()
    expected_scale_pressure = 1.0 # 'none' scaling

    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='center').item(), expected_center_temp, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='scale').item(), expected_scale_temp, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['pressure'].sel(parameter='center').item(), expected_center_pressure, rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['pressure'].sel(parameter='scale').item(), expected_scale_pressure, rtol=1e-5)

    # Verify 'mean' and 'std' are always available
    np.testing.assert_allclose(scaler.scaler['temp'].sel(parameter='mean').item(), sample_dataset_basic['temp'].mean().item(), rtol=1e-5)
    np.testing.assert_allclose(scaler.scaler['pressure'].sel(parameter='std').item(), sample_dataset_basic['pressure'].std().item(), rtol=1e-5)

# --- MODIFIED TEST for `_check_zero_scale` being called by `calculate` ---
def test_scaler_calculate_raises_error_for_zero_scale(tmp_scaler_dir, sample_dataset_with_constant_var):
    # `constant_val` has a standard deviation of 0.0.
    # The calculate method should now raise a ValueError due to _check_zero_scale().
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    with pytest.raises(ValueError, match="Zero scale values found for features:"):
        scaler.calculate(sample_dataset_with_constant_var)

# --- NEW TEST: Direct test of _check_zero_scale ---
def test_check_zero_scale_method_raises_error(tmp_scaler_dir):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    # Manually set a scaler with a zero value for 'scale'
    scaler.scaler = xr.Dataset(
        {
            'feature_a': (('parameter',), np.array([0.0, 0.0, 1.0, 1.0])), # scale is 0
            'feature_b': (('parameter',), np.array([0.0, 5.0, 1.0, 1.0]))
        },
        coords={'parameter': ['center', 'scale', 'mean', 'std']}
    )
    with pytest.raises(ValueError, match="Zero scale values found for features:"):
        scaler._check_zero_scale()

def test_check_zero_scale_method_no_error_for_nonzero(tmp_scaler_dir):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    # Manually set a scaler with non-zero values for 'scale' and 'std'
    scaler.scaler = xr.Dataset(
        {
            'feature_a': (('parameter',), np.array([0.0, 1.0, 1.0, 1.0])),
            'feature_b': (('parameter',), np.array([0.0, 5.0, 1.0, 2.0]))
        },
        coords={'parameter': ['center', 'scale', 'mean', 'std']}
    )
    # Should not raise an error
    try:
        scaler._check_zero_scale()
    except ValueError:
        pytest.fail("`_check_zero_scale` raised ValueError unexpectedly for non-zero scale values.")

# --- Test Scaler.scale ---

def test_scaler_scale_basic_functionality(scaler_instance_calculated, sample_dataset_basic):
    scaled_ds = scaler_instance_calculated.scale(sample_dataset_basic)

    assert isinstance(scaled_ds, xr.Dataset)
    assert list(scaled_ds.dims.keys()) == list(sample_dataset_basic.dims.keys())
    assert set(scaled_ds.data_vars.keys()) == set(sample_dataset_basic.data_vars.keys())

    # Verify mean is approx 0 and std is approx 1 for scaled data
    # Note: This holds true when scaling the dataset used for calculation
    np.testing.assert_allclose(scaled_ds['temp'].mean().item(), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled_ds['temp'].std().item(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(scaled_ds['pressure'].mean().item(), 0.0, atol=1e-5)
    np.testing.assert_allclose(scaled_ds['pressure'].std().item(), 1.0, rtol=1e-5)


def test_scaler_scale_new_dataset(scaler_instance_calculated, sample_dataset_basic):
    new_ds = sample_dataset_basic * 2.0
    scaled_ds = scaler_instance_calculated.scale(new_ds)

    expected_scaled_temp = (new_ds['temp'] - scaler_instance_calculated.scaler['temp'].sel(parameter='center')) / scaler_instance_calculated.scaler['temp'].sel(parameter='scale')
    expected_scaled_pressure = (new_ds['pressure'] - scaler_instance_calculated.scaler['pressure'].sel(parameter='center')) / scaler_instance_calculated.scaler['pressure'].sel(parameter='scale')

    np.testing.assert_allclose(scaled_ds['temp'].values, expected_scaled_temp.values, rtol=1e-5)
    np.testing.assert_allclose(scaled_ds['pressure'].values, expected_scaled_pressure.values, rtol=1e-5)

def test_scaler_scale_raises_error_for_missing_features(scaler_instance_calculated):
    ds_with_extra = xr.Dataset({'extra_var': (('x',), [1, 2])})
    with pytest.raises(ValueError, match="Requesting to scale variables that are not part of the scaler:"):
        scaler_instance_calculated.scale(ds_with_extra)

# --- Test Scaler.unscale ---

def test_scaler_unscale_basic_functionality(scaler_instance_calculated, sample_dataset_basic):
    scaled_ds = scaler_instance_calculated.scale(sample_dataset_basic)
    unscaled_ds = scaler_instance_calculated.unscale(scaled_ds)

    assert isinstance(unscaled_ds, xr.Dataset)
    assert list(unscaled_ds.dims.keys()) == list(sample_dataset_basic.dims.keys())
    assert set(unscaled_ds.data_vars.keys()) == set(sample_dataset_basic.data_vars.keys())

    # Assert that unscaled data is very close to the original
    np.testing.assert_allclose(unscaled_ds['temp'].values, sample_dataset_basic['temp'].values, rtol=1e-5)
    np.testing.assert_allclose(unscaled_ds['pressure'].values, sample_dataset_basic['pressure'].values, rtol=1e-5)

def test_scaler_unscale_new_dataset(scaler_instance_calculated, sample_dataset_basic):
    new_ds = sample_dataset_basic * 2.0
    scaled_ds = scaler_instance_calculated.scale(new_ds)
    unscaled_ds = scaler_instance_calculated.unscale(scaled_ds)

    # Assert that unscaled data is very close to the input data for scaling
    np.testing.assert_allclose(unscaled_ds['temp'].values, new_ds['temp'].values, rtol=1e-5)
    np.testing.assert_allclose(unscaled_ds['pressure'].values, new_ds['pressure'].values, rtol=1e-5)

def test_scaler_unscale_raises_error_for_missing_features(scaler_instance_calculated):
    ds_with_extra = xr.Dataset({'extra_var': (('x',), [1, 2])})
    with pytest.raises(ValueError, match="Requesting to unscale variables that are not part of the scaler:"):
        scaler_instance_calculated.unscale(ds_with_extra)

# --- Test Scaler.save and Scaler.load ---

def test_scaler_save_and_load(tmp_scaler_dir, scaler_instance_calculated, sample_dataset_basic):
    scaler_instance_calculated.save()
    scaler_file_path = tmp_scaler_dir / SCALER_FILE_NAME
    assert scaler_file_path.exists()

    loaded_scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=False, dataset=None)

    assert loaded_scaler.scaler is not None
    np.testing.assert_allclose(loaded_scaler.scaler['temp'].values, scaler_instance_calculated.scaler['temp'].values)
    np.testing.assert_allclose(loaded_scaler.scaler['pressure'].values, scaler_instance_calculated.scaler['pressure'].values)

    scaled_via_loaded = loaded_scaler.scale(sample_dataset_basic)
    unscaled_via_loaded = loaded_scaler.unscale(scaled_via_loaded)
    np.testing.assert_allclose(unscaled_via_loaded['temp'].values, sample_dataset_basic['temp'].values, rtol=1e-5)

def test_scaler_save_raises_error_if_not_calculated(tmp_scaler_dir):
    scaler = Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=True, dataset=None)
    with pytest.raises(ValueError, match="You are trying to save a scaler that has not been computed."):
        scaler.save()

# --- NEW TESTS: `_check_zero_scale` interaction with `load` ---

def test_scaler_load_from_file_raises_error_for_zero_scale(tmp_scaler_dir):
    scaler_file_path = tmp_scaler_dir / SCALER_FILE_NAME
    # Create a dummy scaler file with zero scale value
    zero_scale_ds = xr.Dataset(
        {'test_var': (('parameter',), np.array([0.0, 0.0, 1.0, 1.0]))}, # scale is 0
        coords={'parameter': ['center', 'scale', 'mean', 'std']}
    )
    os.makedirs(tmp_scaler_dir, exist_ok=True)
    with open(scaler_file_path, 'wb') as f:
        zero_scale_ds.to_netcdf(f)

    # Now try to load this scaler and expect a ValueError
    with pytest.raises(ValueError, match="Zero scale values found for features:"):
        Scaler(scaler_dir=tmp_scaler_dir, calculate_scaler=False, dataset=None)
