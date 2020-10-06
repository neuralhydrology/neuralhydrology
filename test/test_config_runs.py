"""Integration tests that perform full runs. """
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Callable

import pandas as pd
import pytest
from pytest import approx

from neuralhydrology.data.utils import load_camels_us_forcings, load_camels_us_discharge, load_hourly_us_netcdf
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config
from test import Fixture


def test_daily_regression(get_config: Fixture[Callable[[str], dict]], single_timescale_model: Fixture[str],
                          daily_dataset: Fixture[str], single_timescale_forcings: Fixture[str]):
    """Test regression training and evaluation for daily predictions.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]
        Method that returns a run configuration to test.
    single_timescale_model : Fixture[str]
        Model to test.
    daily_dataset : Fixture[str]
        Daily dataset to use.
    single_timescale_forcings : Fixture[str]
        Daily forcings set to use.
    """
    config = get_config('daily_regression')
    config.log_only('model', single_timescale_model)
    config.log_only('dataset', daily_dataset['dataset'])
    config.log_only('data_dir', config.data_dir / daily_dataset['dataset'])
    config.log_only('target_variables', daily_dataset['target'])
    config.log_only('forcings', single_timescale_forcings['forcings'])
    config.log_only('dynamic_inputs', single_timescale_forcings['variables'])

    basin = '01022500'
    test_start_date, test_end_date = _get_test_start_end_dates(config)

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    results = _get_basin_results(config.run_dir, 1)[basin]['1D']['xr'].isel(time_step=-1)

    assert pd.to_datetime(results['date'].values[0]) == test_start_date.date()
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.date()

    discharge = _get_discharge(config, basin)

    assert discharge.loc[test_start_date:test_end_date].values \
           == approx(results[f'{config.target_variables[0]}_obs'].values.reshape(-1), nan_ok=True)

    # CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(results[f'{config.target_variables[0]}_sim']).any()


def test_daily_regression_additional_features(get_config: Fixture[Callable[[str], dict]]):
    """Tests #38 (training and testing with additional_features).

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    """
    config = get_config('daily_regression_additional_features')

    basin = '01022500'
    test_start_date, test_end_date = _get_test_start_end_dates(config)

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    results = _get_basin_results(config.run_dir, 1)[basin]['1D']['xr'].isel(time_step=-1)

    assert pd.to_datetime(results['date'].values[0]) == test_start_date.date()
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.date()

    discharge = _get_discharge(config, basin)

    assert discharge.loc[test_start_date:test_end_date].values \
           == approx(results[f'{config.target_variables[0]}_obs'].values.reshape(-1), nan_ok=True)

    # CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(results[f'{config.target_variables[0]}_sim']).any()


def test_multi_timescale_regression(get_config: Fixture[Callable[[str], dict]], multi_timescale_model: Fixture[str]):
    """Test regression training and evaluation for multi-timescale predictions.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]
        Method that returns a run configuration to test.
    multi_timescale_model : Fixture[str]
        Model to test.
    """
    config = get_config('multi_timescale_regression')
    config.log_only('model', multi_timescale_model)

    basin = '01022500'
    test_start_date, test_end_date = _get_test_start_end_dates(config)

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    results = _get_basin_results(config.run_dir, 1)[basin]
    discharge = load_hourly_us_netcdf(config.data_dir, config.forcings[0]) \
        .sel(basin=basin, date=slice(test_start_date, test_end_date))['qobs_mm_per_hour']

    hourly_results = results['1H']['xr'].to_dataframe().reset_index()
    hourly_results.index = hourly_results['date'] + hourly_results['time_step']
    assert hourly_results.index[0] == test_start_date
    assert hourly_results.index[-1] == test_end_date.floor('H')

    daily_results = results['1D']['xr']
    assert pd.to_datetime(daily_results['date'].values[0]) == test_start_date
    assert pd.to_datetime(daily_results['date'].values[-1]) == test_end_date.date()
    assert len(daily_results['qobs_mm_per_hour_obs']) == len(discharge) // 24

    assert len(discharge) == len(hourly_results)
    assert discharge.values \
           == approx(hourly_results['qobs_mm_per_hour_obs'].values, nan_ok=True)

    # Hourly CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(hourly_results['qobs_mm_per_hour_sim']).any()
    assert not pd.isna(daily_results['qobs_mm_per_hour_sim'].values).any()


def _get_test_start_end_dates(config: Config) -> Tuple[datetime, datetime]:
    test_start_date = pd.to_datetime(config.test_start_date, format='%d/%m/%Y')
    test_end_date = pd.to_datetime(config.test_end_date, format='%d/%m/%Y') + pd.Timedelta(days=1, seconds=-1)

    return test_start_date, test_end_date


def _get_basin_results(run_dir: Path, epoch: int) -> Dict:
    results_file = list(run_dir.glob(f'test/model_epoch{str(epoch).zfill(3)}/test_results.p'))
    if len(results_file) != 1:
        pytest.fail(f'Results file not found.')

    return pickle.load(open(str(results_file[0]), 'rb'))


def _get_discharge(config: Config, basin: str) -> pd.Series:
    if config.dataset == 'camels_us':
        _, area = load_camels_us_forcings(config.data_dir, basin, 'daymet')
        return load_camels_us_discharge(config.data_dir, basin, area)
    else:
        raise NotImplementedError
