import numpy as np
import pandas as pd
import pytest
import xarray

from neuralhydrology.evaluation.metrics import (calculate_all_metrics, calculate_metrics, crps, get_available_metrics,
                                                mpiw, picp)


def test_ensemble_metrics():
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [1.0, 3.0]], dims=['datetime', 'samples'])

    assert crps(obs, sim) == pytest.approx(0.5)
    assert picp(obs, sim, alpha=0.1) == pytest.approx(1.0)
    assert mpiw(obs, sim, alpha=0.1) == pytest.approx(1.8)


def test_ensemble_metrics_ignore_invalid_samples():
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [np.nan, 3.0]], dims=['datetime', 'samples'])

    assert crps(obs, sim) == pytest.approx(0.5)
    assert picp(obs, sim) == pytest.approx(1.0)
    assert mpiw(obs, sim) == pytest.approx(1.8)


def test_calculate_metrics_mixes_deterministic_and_ensemble_metrics():
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [1.0, 3.0]], dims=['datetime', 'samples'])

    metrics = calculate_metrics(obs, sim, metrics=['MSE', 'CRPS', 'PICP', 'MPIW'])

    assert metrics['MSE'] == pytest.approx(0.0)
    assert metrics['CRPS'] == pytest.approx(0.5)
    assert metrics['PICP'] == pytest.approx(1.0)
    assert metrics['MPIW'] == pytest.approx(1.8)


def test_calculate_all_metrics_includes_ensemble_metrics_for_samples():
    dates = pd.date_range('2000-01-01', periods=4)
    obs = xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=['date'], coords={'date': dates})
    sim = xarray.DataArray([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]],
                           dims=['date', 'samples'],
                           coords={
                               'date': dates,
                               'samples': [0, 1]
                           })

    metrics = calculate_all_metrics(obs, sim, datetime_coord='date')

    assert metrics['MSE'] == pytest.approx(0.0)
    assert metrics['CRPS'] == pytest.approx(0.5)
    assert metrics['PICP'] == pytest.approx(1.0)
    assert metrics['MPIW'] == pytest.approx(1.8)


def test_calculate_all_metrics_excludes_ensemble_metrics_without_samples():
    dates = pd.date_range('2000-01-01', periods=4)
    obs = xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=['date'], coords={'date': dates})
    sim = xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=['date'], coords={'date': dates})

    metrics = calculate_all_metrics(obs, sim, datetime_coord='date')

    assert set(metrics) == set(get_available_metrics())


@pytest.mark.parametrize('metric', [crps, picp, mpiw])
@pytest.mark.parametrize('obs_values, sim_values', [
    ([np.nan, np.nan], [[1.0, 2.0], [2.0, 3.0]]),
    ([1.0, 2.0], [[np.nan, np.nan], [np.nan, np.nan]]),
])
def test_ensemble_metrics_return_nan_when_masking_removes_all_values(metric, obs_values, sim_values):
    obs = xarray.DataArray(obs_values, dims=['datetime'])
    sim = xarray.DataArray(sim_values, dims=['datetime', 'samples'])

    assert np.isnan(metric(obs, sim))


def test_available_metrics_only_include_probabilistic_metrics_on_request():
    assert 'CRPS' not in get_available_metrics()
    assert 'CRPS' in get_available_metrics(include_probabilistic=True)
