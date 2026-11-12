import numpy as np
import pandas as pd
import pytest
import xarray

from neuralhydrology.evaluation.metrics import (alpha_reliability, calculate_all_metrics, calculate_metrics, crps,
                                                get_available_metrics, mpiw, nse, picp, pit)


@pytest.mark.parametrize('metric, expected', [(crps, 0.5), (picp, 1.0), (mpiw, 1.8)])
def test_ensemble_metrics(metric, expected):
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [1.0, 3.0]], dims=['datetime', 'samples'])

    assert metric(obs, sim) == pytest.approx(expected)


@pytest.mark.parametrize('metric, expected', [(crps, 0.5), (picp, 1.0), (mpiw, 1.8)])
def test_ensemble_metrics_ignore_invalid_samples(metric, expected):
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [np.nan, 3.0]], dims=['datetime', 'samples'])

    assert metric(obs, sim) == pytest.approx(expected)


def test_calculate_metrics_mixes_deterministic_and_ensemble_metrics():
    obs = xarray.DataArray([1.0, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [1.0, 3.0]], dims=['datetime', 'samples'])

    metrics = calculate_metrics(obs, sim, metrics=['MSE', 'CRPS', 'PICP', 'MPIW', 'alpha-reliability'])

    assert metrics['MSE'] == pytest.approx(0.0)
    assert metrics['CRPS'] == pytest.approx(0.5)
    assert metrics['PICP'] == pytest.approx(1.0)
    assert metrics['MPIW'] == pytest.approx(1.8)
    assert metrics['Alpha-Reliability'] == pytest.approx(2 / 3)


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
    assert metrics['Alpha-Reliability'] == pytest.approx(0.6)


def test_calculate_all_metrics_excludes_ensemble_metrics_without_samples():
    dates = pd.date_range('2000-01-01', periods=4)
    obs = xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=['date'], coords={'date': dates})
    sim = xarray.DataArray([1.0, 2.0, 3.0, 4.0], dims=['date'], coords={'date': dates})

    metrics = calculate_all_metrics(obs, sim, datetime_coord='date')

    assert set(metrics) == set(get_available_metrics())


@pytest.mark.parametrize('metric', [crps, picp, mpiw, alpha_reliability])
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


@pytest.mark.parametrize('shape', [(5, 1), (1, 1)])
def test_metrics_reject_2d_inputs(shape):
    obs = xarray.DataArray(np.ones(shape), dims=['date', 'time_step'])

    with pytest.raises(RuntimeError, match='1d'):
        nse(obs, obs)


def test_pit_counts_ties_as_half():
    obs = xarray.DataArray([1.0, 0.0, 5.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 4.0]],
                           dims=['datetime', 'samples'])

    assert pit(obs, sim).values == pytest.approx([0.375, 0.375, 1.0])


def test_pit_ignores_invalid_values():
    obs = xarray.DataArray([1.0, np.nan, 2.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0], [0.0, 2.0], [np.nan, 3.0]], dims=['datetime', 'samples'])

    assert pit(obs, sim).values == pytest.approx([0.5])


def test_alpha_reliability_is_high_for_calibrated_and_low_for_overconfident_ensembles():
    rng = np.random.default_rng(42)
    n_steps, n_samples = 2000, 100
    obs = xarray.DataArray(rng.normal(size=n_steps), dims=['datetime'])
    calibrated = xarray.DataArray(rng.normal(size=(n_steps, n_samples)), dims=['datetime', 'samples'])
    overconfident = xarray.DataArray(0.3 * rng.normal(size=(n_steps, n_samples)), dims=['datetime', 'samples'])

    assert alpha_reliability(obs, calibrated) > 0.95
    assert alpha_reliability(obs, overconfident) < 0.7


def test_alpha_reliability_is_one_when_pit_values_match_uniform_plotting_positions():
    # PIT values 0.25, 0.5 and 0.75 equal the plotting positions t / (T + 1) for T = 3.
    obs = xarray.DataArray([1.0, 2.0, 3.0], dims=['datetime'])
    sim = xarray.DataArray([[0.0, 2.0, 3.0, 4.0], [0.0, 1.0, 3.0, 4.0], [0.0, 1.0, 2.0, 4.0]],
                           dims=['datetime', 'samples'])

    assert pit(obs, sim).values == pytest.approx([0.25, 0.5, 0.75])
    assert alpha_reliability(obs, sim) == pytest.approx(1.0)
