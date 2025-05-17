import pytest
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict

from neuralhydrology.datautils.union_features import (
    _expand_lead_times,
    _union_features_with_same_dimensions,
    _union_lead_time_feature_with_non_lead_time_feature,
    _union_non_lead_time_feature_with_lead_time_feature,
    union_features
)


# --- Fixtures for common data structures ---

@pytest.fixture
def sample_dates():
    return pd.date_range("2000-01-01", "2000-01-05", freq="D")

@pytest.fixture
def sample_basins():
    return ['basin_A', 'basin_B']

@pytest.fixture
def sample_lead_times():
    return np.array([0, 1, 2])

@pytest.fixture
def base_dataset(sample_basins, sample_dates, sample_lead_times):
    """Provides a base dataset with various feature types."""
    ds = xr.Dataset(
        {
            "feature_scalar": (("basin",), np.array([10.0, 20.0])),
            "feature_2d_hindcast": (("basin", "date"), np.random.rand(2, 5) * 10),
            "feature_3d_forecast": (("basin", "date", "lead_time"), np.random.rand(2, 5, 3) * 100),
            "target": (("basin", "date"), np.random.rand(2, 5)),
            "mask_2d": (("basin", "date"), np.full((2, 5), np.nan)), # Will be used for masking
            "mask_3d": (("basin", "date", "lead_time"), np.full((2, 5, 3), np.nan)), # Will be used for masking
            "other_feature": (("basin", "date"), np.random.rand(2, 5)) # Unmasked feature
        },
        coords={
            "basin": sample_basins,
            "date": sample_dates,
            "lead_time": sample_lead_times
        }
    )
    # Introduce some NaNs for testing unioning
    ds['feature_2d_hindcast'].loc[{'basin': 'basin_A', 'date': '2000-01-02'}] = np.nan
    ds['feature_3d_forecast'].loc[{'basin': 'basin_B', 'date': '2000-01-03', 'lead_time': 1}] = np.nan
    return ds

def test_expand_lead_times_basic_expansion(sample_basins, sample_dates):
    """
    Tests basic functionality of _expand_lead_times to add a new 'lead_time' dimension.
    """
    original_da = xr.DataArray(
        [[10., 11., 12., 13., 14.], [20., 21., 22., 23., 24.]],
        coords={'basin': sample_basins, 'date': sample_dates},
        dims=['basin', 'date']
    )
    lead_time_max = 2 # Max lead time to expand to (0, 1, 2)

    expanded_da = _expand_lead_times(original_da, lead_time_max)

    # Assertions
    assert 'lead_time' in expanded_da.dims
    assert set(expanded_da.dims) == {'basin', 'date', 'lead_time'}
    assert np.array_equal(expanded_da['lead_time'].values, np.arange(lead_time_max + 1))
    
    # Check shape
    expected_shape = (len(sample_basins), len(sample_dates), lead_time_max + 1)
    assert sorted(expanded_da.shape) == sorted(expected_shape)

    # Verify values and NaN propagation due to shift
    # original_da.shift(date=-0) should be original_da itself
    xr.testing.assert_equal(expanded_da.sel(lead_time=0).drop('lead_time'), original_da)

    # original_da.shift(date=-1) should shift dates by 1, introducing NaN at the end
    expected_shifted_1 = original_da.shift(date=-1)
    xr.testing.assert_equal(expanded_da.sel(lead_time=1).drop('lead_time'), expected_shifted_1)
    assert np.isnan(expanded_da.isel(date=-1, lead_time=1)).all() # Last date should be NaN for lt=1

    # original_da.shift(date=-2) should shift dates by 2, introducing NaNs at the end
    expected_shifted_2 = original_da.shift(date=-2)
    xr.testing.assert_equal(expanded_da.sel(lead_time=2).drop('lead_time'), expected_shifted_2)
    assert np.isnan(expanded_da.isel(date=[-1, -2], lead_time=2)).all() # Last two dates should be NaN for lt=2


def test_expand_lead_times_raises_error_if_lead_time_exists(sample_basins, sample_dates, sample_lead_times):
    """
    Tests that _expand_lead_times raises a ValueError if 'lead_time' dim already exists.
    """
    # Create a DataArray that already has a 'lead_time' dimension
    da_with_lead_time = xr.DataArray(
        np.random.rand(len(sample_basins), len(sample_dates), len(sample_lead_times)),
        coords={'basin': sample_basins, 'date': sample_dates, 'lead_time': sample_lead_times},
        dims=['basin', 'date', 'lead_time']
    )
    
    with pytest.raises(ValueError, match='Trying to expand a dataarray that already has a lead time.'):
        _expand_lead_times(da_with_lead_time, 2)


def test_expand_lead_times_single_date(sample_basins):
    """Test expansion with a single date, checking NaN propagation."""
    single_date = pd.to_datetime(["2000-01-01"])
    original_da = xr.DataArray(
        [[10], [20]],
        coords={'basin': sample_basins, 'date': single_date},
        dims=['basin', 'date']
    )
    lead_time_max = 1 # Expand to lead_time 0, 1

    expanded_da = _expand_lead_times(original_da, lead_time_max)

    assert 'lead_time' in expanded_da.dims
    
    expected_shape = (len(sample_basins), 1, lead_time_max + 1)
    assert sorted(expanded_da.shape) == sorted(expected_shape)
    assert np.array_equal(expanded_da['lead_time'].values, np.arange(lead_time_max + 1))

    # Check values:
    # lead_time=0: original values
    xr.testing.assert_equal(expanded_da.sel(lead_time=0).drop('lead_time'), original_da)

    # lead_time=1: should be all NaN as there's no next date
    expected_shifted_1 = xr.DataArray(
        [[np.nan], [np.nan]],
        coords={'basin': sample_basins, 'date': single_date},
        dims=['basin', 'date']
    )
    xr.testing.assert_equal(expanded_da.sel(lead_time=1).drop('lead_time'), expected_shifted_1)
    assert np.isnan(expanded_da.sel(lead_time=1)).all()


 # --- Tests for helper functions ---

def test_union_features_with_same_dimensions():
    da1 = xr.DataArray(
        [[1, np.nan, 3], [4, 5, np.nan]],
        coords={'x': [0, 1], 'y': [0, 1, 2]},
        dims=['x', 'y']
    ).astype('float32')
    da2 = xr.DataArray(
        [[np.nan, 20, np.nan], [40, np.nan, 60]],
        coords={'x': [0, 1], 'y': [0, 1, 2]},
        dims=['x', 'y']
    ).astype('float32')
    expected = xr.DataArray(
        [[1, 20, 3], [4, 5, 60]],
        coords={'x': [0, 1], 'y': [0, 1, 2]},
        dims=['x', 'y']
    ).astype('float32')
    result = _union_features_with_same_dimensions(da1, da2)
    xr.testing.assert_equal(result, expected)

    # Test with one all-NaN array
    da_nan = xr.DataArray(
        [[np.nan, np.nan], [np.nan, np.nan]],
        coords={'x': [0, 1], 'y': [0, 1]},
        dims=['x', 'y']
    )
    da_values = xr.DataArray(
        [[1, 2], [3, 4]],
        coords={'x': [0, 1], 'y': [0, 1]},
        dims=['x', 'y']
    )
    xr.testing.assert_equal(_union_features_with_same_dimensions(da_nan, da_values), da_values)
    xr.testing.assert_equal(_union_features_with_same_dimensions(da_values, da_nan), da_values)

    
def test_union_lead_time_feature_with_non_lead_time_feature(sample_basins, sample_dates, sample_lead_times):
    # feature_da has lead_time, mask_feature_da does not
    feature_da = xr.DataArray(
        [[[10, np.nan, 30], [40, 50, 60]]],
        coords={
            'basin': sample_basins[:1],
            'date': sample_dates[:2],
            'lead_time': sample_lead_times
        },
        dims=['basin', 'date', 'lead_time']
    )
    mask_feature_da = xr.DataArray(
        [[1], [2]],
        coords={'date': sample_dates[:2], 'basin': sample_basins[:1]},
        dims=['date', 'basin']
    )
    # Expected: np.nan at lead_time=1 should be filled with the value from one date in the future.
    expected_data = [[[10, 2, 30], [40, 50, 60]]] # The mask_feature_da is broadcasted across lead_time
    expected = xr.DataArray(
        expected_data,
        coords={
            'basin': sample_basins[:1],
            'date': sample_dates[:2],
            'lead_time': sample_lead_times
        },
        dims=['basin', 'date', 'lead_time']
    )
    result = _union_lead_time_feature_with_non_lead_time_feature(feature_da, mask_feature_da)
    xr.testing.assert_equal(result, expected)

    # Expected: np.nan at lead_time=0 should be filled with the value from today.
    feature_da = xr.DataArray(
        [[[np.nan, 20, 30], [40, 50, 60]]],
        coords={
            'basin': sample_basins[:1],
            'date': sample_dates[:2],
            'lead_time': sample_lead_times
        },
        dims=['basin', 'date', 'lead_time']
    )
    expected_data = [[[1, 20, 30], [40, 50, 60]]] # The mask_feature_da is broadcasted across lead_time
    expected = xr.DataArray(
        expected_data,
        coords={
            'basin': sample_basins[:1],
            'date': sample_dates[:2],
            'lead_time': sample_lead_times
        },
        dims=['basin', 'date', 'lead_time']
    )
    result = _union_lead_time_feature_with_non_lead_time_feature(feature_da, mask_feature_da)
    xr.testing.assert_equal(result, expected)

    
def test_union_non_lead_time_feature_with_lead_time_feature(sample_basins, sample_dates, sample_lead_times):
    # feature_da does not have lead_time, mask_feature_da has lead_time
    feature_da = xr.DataArray(
        [[np.nan, 20]], # basin=A, date=2000-01-01, 2000-01-02
        coords={
            'basin': [sample_basins[0]],
            'date': sample_dates[:2]
        },
        dims=['basin', 'date']
    )
    # mask_feature_da has values at lead_time=0
    mask_feature_da = xr.DataArray(
        [[[100, 101, 102], [200, 201, 202]]], # basin=A, date=2000-01-01, 2000-01-02
        coords={
            'basin': [sample_basins[0]],
            'date': sample_dates[:2],
            'lead_time': sample_lead_times
        },
        dims=['basin', 'date', 'lead_time']
    )
    # Expected: np.nan at date=2000-01-01 should be filled with mask_feature_da.sel(lead_time=0) at 2000-01-01 (which is 100)
    expected_data = [[100, 20]]
    expected = xr.DataArray(
        expected_data,
        coords={
            'basin': [sample_basins[0]],
            'date': sample_dates[:2]
        },
        dims=['basin', 'date']
    )
    result = _union_non_lead_time_feature_with_lead_time_feature(feature_da, mask_feature_da)
    xr.testing.assert_equal(result, expected)

# --- Tests for union_features (main function) ---

def test_union_features_basic(base_dataset):
    # Test simple unioning of features with same dimensions
    union_mapping = {
        "feature_2d_hindcast": "mask_2d"
    }
    # Set a specific value in mask_2d to fill the NaN in feature_2d_hindcast
    base_dataset['mask_2d'].loc[{'basin': 'basin_A', 'date': '2000-01-02'}] = 99.0

    result_ds = union_features(base_dataset, union_mapping)

    # Expected feature_2d_hindcast: NaN at 2000-01-02 for basin_A should be 99.0
    expected_feature_2d_hindcast_data = base_dataset['feature_2d_hindcast'].values.copy()
    expected_feature_2d_hindcast_data[0, 1] = 99.0 # basin_A, 2000-01-02
    expected_feature_2d_hindcast = xr.DataArray(
        expected_feature_2d_hindcast_data,
        coords=base_dataset['feature_2d_hindcast'].coords,
        dims=base_dataset['feature_2d_hindcast'].dims
    )
    xr.testing.assert_equal(result_ds['feature_2d_hindcast'], expected_feature_2d_hindcast)
    assert 'other_feature' in result_ds # Ensure unmasked features are present

    
def test_union_features_mixed_dimensions(base_dataset):
    # Test a more complex scenario with mixed dimensions
    union_mapping = {
        "feature_3d_forecast": "mask_2d", # 3D feature masked by 2D feature
        "feature_2d_hindcast": "mask_3d"  # 2D feature masked by 3D feature
    }
    # Set specific values in masks to fill NaNs
    base_dataset['mask_2d'].loc[{'basin': 'basin_B', 'date': '2000-01-04'}] = 1000.0
    base_dataset['mask_3d'].loc[{'basin': 'basin_A', 'date': '2000-01-02', 'lead_time': 0}] = 2000.0

    result_ds = union_features(base_dataset, union_mapping)

    # Verify feature_3d_forecast (masked by mask_2d)
    # NaN at basin_B, 2000-01-03, lead_time=1 should be filled by mask_2d at basin_B, 2000-01-03 (1000.0)
    expected_3d_data = base_dataset['feature_3d_forecast'].values.copy()
    # The _union_lead_time_feature_with_non_lead_time_feature will broadcast the mask value across lead_time
    expected_3d_data[1, 2, 1] = 1000.0 # basin_B, 2000-01-03, all lead_times
    expected_feature_3d_forecast = xr.DataArray(
        expected_3d_data,
        coords=base_dataset['feature_3d_forecast'].coords,
        dims=base_dataset['feature_3d_forecast'].dims
    )
    xr.testing.assert_equal(result_ds['feature_3d_forecast'], expected_feature_3d_forecast)

    # Verify feature_2d_hindcast (masked by mask_3d)
    # NaN at basin_A, 2000-01-02 should be filled by mask_3d at basin_A, 2000-01-02, lead_time=0 (2000.0)
    expected_2d_data = base_dataset['feature_2d_hindcast'].values.copy()
    expected_2d_data[0, 1] = 2000.0 # basin_A, 2000-01-02
    expected_feature_2d_hindcast = xr.DataArray(
        expected_2d_data,
        coords=base_dataset['feature_2d_hindcast'].coords,
        dims=base_dataset['feature_2d_hindcast'].dims
    )
    xr.testing.assert_equal(result_ds['feature_2d_hindcast'], expected_feature_2d_hindcast)
    assert 'other_feature' in result_ds # Ensure unmasked features are present

    
def test_union_features_unmasked_features(base_dataset):
    # Test that features not in the mapping are still present
    union_mapping = {
        "feature_2d_hindcast": "mask_2d"
    }
    result_ds = union_features(base_dataset, union_mapping)
    assert "feature_scalar" in result_ds.data_vars
    assert "target" in result_ds.data_vars
    assert "other_feature" in result_ds.data_vars
    assert "mask_2d" in result_ds.data_vars # Masking features are also included
    assert "mask_3d" in result_ds.data_vars
    assert "feature_3d_forecast" in result_ds.data_vars # Not explicitly masked, so should be there

    
def test_union_features_self_union(base_dataset):
    # Test where a feature is mapped to itself (should result in no change for that feature)
    union_mapping = {
        "feature_2d_hindcast": "feature_2d_hindcast"
    }
    original_feature_2d = base_dataset['feature_2d_hindcast'].copy(deep=True)
    result_ds = union_features(base_dataset, union_mapping)
    xr.testing.assert_equal(result_ds['feature_2d_hindcast'], original_feature_2d)

    
def test_union_features_missing_feature_to_mask(base_dataset):
    """
    Test that it skips if the 'feature' to be masked is not in the dataset,
    and returns the original dataset unchanged (except for potential reordering).
    """
    union_mapping = {
        "non_existent_feature": "mask_2d"
    }
    result_ds = union_features(base_dataset, union_mapping)
    xr.testing.assert_equal(result_ds, base_dataset)


def test_union_features_missing_mask_feature_raises_error(base_dataset):
    # Test that it raises ValueError if the 'mask_feature' is not in the dataset
    union_mapping = {
        "feature_2d_hindcast": "non_existent_mask_feature"
    }
    with pytest.raises(ValueError, match='Masking feature non_existent_mask_feature not in dataset.'):
        union_features(base_dataset, union_mapping)

        
def test_union_features_empty_mapping(base_dataset):
    # Test with an empty union_feature_mapping
    union_mapping = {}
    result_ds = union_features(base_dataset, union_mapping)
    xr.testing.assert_equal(result_ds, base_dataset) # Should be identical as nothing was masked

    
def test_union_features_all_features_masked(base_dataset):
    # Test scenario where all features in the original dataset are part of the mapping
    # Create a mapping that includes all features from base_dataset
    all_features = list(base_dataset.data_vars.keys())
    union_mapping = {f: f for f in all_features} # Map each feature to itself for simplicity

    result_ds = union_features(base_dataset, union_mapping)
    xr.testing.assert_equal(result_ds, base_dataset) # Should be identical if mapped to self

    # Now, a more realistic scenario where some are actually masked
    base_dataset['mask_2d'].loc[{'basin': 'basin_A', 'date': '2000-01-02'}] = 99.0
    union_mapping_real = {
        "feature_2d_hindcast": "mask_2d",
        "feature_scalar": "feature_scalar" # Self-union
    }
    result_ds_real = union_features(base_dataset, union_mapping_real)
    
    # Check the masked feature
    expected_feature_2d_hindcast_data = base_dataset['feature_2d_hindcast'].values.copy()
    expected_feature_2d_hindcast_data[0, 1] = 99.0 # basin_A, 2000-01-02
    expected_feature_2d_hindcast = xr.DataArray(
        expected_feature_2d_hindcast_data,
        coords=base_dataset['feature_2d_hindcast'].coords,
        dims=base_dataset['feature_2d_hindcast'].dims
    )
    xr.testing.assert_equal(result_ds_real['feature_2d_hindcast'], expected_feature_2d_hindcast)
    
    # Check other features are still present and unchanged if not explicitly masked/changed
    assert 'feature_scalar' in result_ds_real
    xr.testing.assert_equal(result_ds_real['feature_scalar'], base_dataset['feature_scalar'])
    assert 'target' in result_ds_real
    xr.testing.assert_equal(result_ds_real['target'], base_dataset['target'])
    assert 'other_feature' in result_ds_real
    xr.testing.assert_equal(result_ds_real['other_feature'], base_dataset['other_feature'])
    assert 'feature_3d_forecast' in result_ds_real
    xr.testing.assert_equal(result_ds_real['feature_3d_forecast'], base_dataset['feature_3d_forecast'])

