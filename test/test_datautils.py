"""Unit tests for datautils functions. """
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuralhydrology.datasetzoo.camelsus import load_camels_us_attributes
from neuralhydrology.datautils.climateindices import calculate_camels_us_dyn_climate_indices
from neuralhydrology.datautils.pet import _get_clear_sky_rad, get_priestley_taylor_pet
from neuralhydrology.datautils.utils import (get_frequency_factor, infer_frequency, sort_frequencies, _ME_FREQ,
                                             _QE_FREQ, _YE_FREQ)


@pytest.mark.parametrize("elev, expected", [(0, np.array([7.5])), (824, np.array([7.6648000000000005])),
                                            (3000, np.array([8.100000000000001]))])
def test_clear_sky_radiation(elev, expected):
    """Test clear-sky radiation at selected elevations."""
    et_rad = np.array([10.0])

    np.testing.assert_array_equal(_get_clear_sky_rad(elev, et_rad), expected)


def test_priestley_taylor_pet():
    """Test Priestley-Taylor PET with the inputs of FAO-56 Example 18 (Uccle, 6 July)."""
    # The example's solar radiation is 22.07 MJm-2day-1; the function expects the corresponding 24-hour mean in Wm-2.
    pet = get_priestley_taylor_pet(t_min=np.array([12.3]),
                                   t_max=np.array([21.5]),
                                   s_rad=np.array([22.07 / 0.0864]),
                                   lat=50.8,
                                   elev=100.0,
                                   doy=np.array([187]))

    # alpha * delta / (delta + gamma) * Rn / lambda with the values of the example: delta = 0.122 kPa/degree C,
    # gamma = 0.0666 kPa/degree C, Rn = 13.28 MJm-2day-1, and 1 / lambda = 0.408. The example derives the actual vapor
    # pressure from humidity, the function from t_min, which makes the result about 0.3 % higher.
    expected = 1.26 * 0.122 / (0.122 + 0.0666) * 13.28 * 0.408
    np.testing.assert_allclose(pet, expected, rtol=0.01)


@pytest.mark.parametrize("forcings", ['daymet', 'maurer_extended', 'nldas'])
def test_camels_us_dyn_climate_indices_pet(forcings):
    """Test that the PET of the CAMELS US climate indices is close to the PET published with the CAMELS attributes.

    The CAMELS US forcings report solar radiation as the average over the daylight period, which has to be converted to
    a 24-hour average before PET is calculated.
    """
    data_dir = Path('./test/test_data/camels_us')
    basins = ['01022500', '01547700', '02064000', '03015500']
    climate_indices = calculate_camels_us_dyn_climate_indices(data_dir, basins, window_length=1096, forcings=forcings)
    published_pet = load_camels_us_attributes(data_dir, basins=basins)['pet_mean']

    # The published values were estimated with a Priestley-Taylor coefficient calibrated for each catchment, from
    # Daymet data of 1989-2009 (Addor et al., 2017), so they agree only approximately.
    for basin in basins:
        ratio = climate_indices[basin]['pet_mean_dyn'].iloc[-1] / published_pet[basin]
        assert 0.8 < ratio < 1.25, f"Basin {basin}: PET is {ratio:.2f} times the published value"


def test_sort_frequencies():
    """Test the sorting of frequencies. """
    assert sort_frequencies(['1D', '1h', '2D', '3h']) == ['2D', '1D', '3h', '1h']
    assert sort_frequencies(['1' + _ME_FREQ, '1' + _YE_FREQ]) == ['1' + _YE_FREQ, '1' + _ME_FREQ]
    assert sort_frequencies(['1D', '48h']) == ['48h', '1D']
    assert sort_frequencies(['1D']) == ['1D']
    assert sort_frequencies([]) == []

    pytest.raises(ValueError, sort_frequencies, ['1D', '1XYZ'])  # not a frequency


def test_infer_frequency():
    """Test the logic to infer frequencies. """
    assert infer_frequency(pd.date_range('2000-01-01', '2000-01-10', freq='D')) == '1D'
    assert infer_frequency(pd.date_range('2000-01-01', '2000-01-10', freq='48h')) == '2D'
    assert infer_frequency(pd.date_range('2000-01-01', '2000-03-01', freq='W-MON')) == '7D'

    # just a single date
    pytest.raises(ValueError, infer_frequency, [pd.to_datetime('2000-01-01')])
    # frequency of zero
    pytest.raises(ValueError, infer_frequency,
                  [pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-01')])
    # irregular dates
    pytest.raises(ValueError, infer_frequency,
                  [pd.to_datetime('2000-01-01'),
                   pd.to_datetime('2000-01-02'),
                   pd.to_datetime('2000-01-04')])


def test_get_frequency_factor():
    """Test the logic that calculates the ratio between two frequencies. """
    assert get_frequency_factor('1h', '1h') == 1
    assert get_frequency_factor('1A', '1' + _YE_FREQ) == 1
    assert get_frequency_factor('1' + _YE_FREQ, '4' + _QE_FREQ) == 1
    assert get_frequency_factor('1h', '1D') == 1 / 24
    assert get_frequency_factor('1D', '1h') == 24
    assert get_frequency_factor('2D', '12h') == 4
    assert get_frequency_factor('1W', '1D') == 7
    assert get_frequency_factor('1W-MON', '1D') == 7
    assert get_frequency_factor('1' + _YE_FREQ, '1' + _ME_FREQ) == 12
    assert get_frequency_factor('0D', '0h') == 1

    pytest.raises(ValueError, get_frequency_factor, '1YS', '1' + _ME_FREQ)  # year-start vs. month-end
    pytest.raises(ValueError, get_frequency_factor, '1' + _QE_FREQ, '1W')  # quarter vs. week
    pytest.raises(ValueError, get_frequency_factor, '1XYZ', '1D')  # not a frequency
    pytest.raises(ValueError, get_frequency_factor, '1' + _YE_FREQ,
                  '1D')  # disallowed because to_timedelta('1Y') is deprecated
    pytest.raises(ValueError, get_frequency_factor, '1' + _ME_FREQ,
                  '1D')  # disallowed because to_timedelta('1M') is deprecated
    pytest.raises(NotImplementedError, get_frequency_factor, '-1D', '1h')  # we should never need negative frequencies
