"""Integration tests that perform full runs. """
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Callable

import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from pytest import approx

from neuralhydrology.datasetzoo import camelsus, hourlycamelsus
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
    config.update_config({
        'model': single_timescale_model,
        'dataset': daily_dataset['dataset'],
        'data_dir': config.data_dir / daily_dataset['dataset'],
        'target_variables': daily_dataset['target'],
        'forcings': single_timescale_forcings['forcings'],
        'dynamic_inputs': single_timescale_forcings['variables']
    })

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    _check_results(config, '01022500')


def test_daily_regression_additional_features(get_config: Fixture[Callable[[str], dict]]):
    """Tests #38 (training and testing with additional_features).

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    """
    config = get_config('daily_regression_additional_features')

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    _check_results(config, '01022500')


def test_autoregression_daily_regression(get_config: Fixture[Callable[[str], dict]],
                                         daily_dataset: Fixture[str], single_timescale_forcings: Fixture[str]):

    """Tests training and testing with arlstm.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    daily_dataset : Fixture[str]
        Daily dataset to use.
    single_timescale_forcings : Fixture[str]
        Daily forcings set to use.
    """
    config = get_config('autoregression_daily_regression')
    config.update_config({
        'model': 'arlstm',
        'dataset': daily_dataset['dataset'],
        'data_dir': config.data_dir / daily_dataset['dataset'],
        'target_variables': daily_dataset['target'],
        'forcings': single_timescale_forcings['forcings'],
        'dynamic_inputs': single_timescale_forcings['variables']
    })

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    _check_results(config, '01022500')


def test_daily_regression_with_embedding(get_config: Fixture[Callable[[str], dict]],
                                         single_timescale_model: Fixture[str]):
    """Tests training and testing with static and dynamic embedding network.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    single_timescale_model : Fixture[str]
        Name of a single-timescale model
    """
    config = get_config('daily_regression_with_embedding')
    config.update_config({'model': single_timescale_model})

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    _check_results(config, '01022500')


def test_transformer_daily_regression(get_config: Fixture[Callable[[str], dict]]):
    """Tests training and testing with a transformer model.

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    """
    config = get_config('transformer_daily_regression')

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    _check_results(config, '01022500')


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
    config.update_config({'model': multi_timescale_model})

    basin = '01022500'
    test_start_date, test_end_date = _get_test_start_end_dates(config)

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')

    results = _get_basin_results(config.run_dir, 1)[basin]
    discharge = hourlycamelsus.load_hourly_us_netcdf(config.data_dir, config.forcings[0]) \
        .sel(basin=basin, date=slice(test_start_date, test_end_date))['qobs_mm_per_hour']

    hourly_results = results['1H']['xr'].to_dataframe().reset_index()
    hourly_results.index = hourly_results['date'] + hourly_results['time_step'] * to_offset('H')
    assert (results['1H']['xr']['time_step'].values == list(range(24))).all()
    assert hourly_results.index[0] == test_start_date
    assert hourly_results.index[-1] == test_end_date.floor('H')

    daily_results = results['1D']['xr']
    assert (results['1D']['xr']['time_step'].values == [0]).all()
    assert pd.to_datetime(daily_results['date'].values[0]) == test_start_date
    assert pd.to_datetime(daily_results['date'].values[-1]) == test_end_date.date()
    assert len(daily_results['qobs_mm_per_hour_obs']) == len(discharge) // 24

    assert len(discharge) == len(hourly_results)
    assert discharge.values == approx(hourly_results['qobs_mm_per_hour_obs'].values, nan_ok=True)

    # Hourly CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(hourly_results['qobs_mm_per_hour_sim']).any()
    assert not pd.isna(daily_results['qobs_mm_per_hour_sim'].values).any()


def test_daily_regression_nan_targets(get_config: Fixture[Callable[[str], dict]]):
    """Tests #112 (evaluation when target values are NaN).

    Parameters
    ----------
    get_config : Fixture[Callable[[str], dict]]
        Method that returns a run configuration
    """
    config = get_config('daily_regression_nan_targets')

    start_training(config)
    start_evaluation(cfg=config, run_dir=config.run_dir, epoch=1, period='test')


    # the fact that the targets are NaN should not lead the model to create NaN outputs.
    # however, we do need to pass discharge as an NaN series, because the camels discharge loader would return [],
    # as the test period is outside the part of the discharge time series that is stored on disk.
    discharge = pd.Series(float('nan'), index=pd.date_range(*_get_test_start_end_dates(config)))
    _check_results(config, '01022500', discharge=discharge)


def _check_results(config: Config, basin: str, discharge: pd.Series = None):
    """Perform basic sanity checks of model predictions.

    Checks that the results file has the correct date range, that the observed discharge in the file is correct, and
    that there are no NaN predictions.

    Parameters
    ----------
    config : Config
        The run configuration used to produce the results
    basin : str
        Id of a basin for which to check the results
    discharge : pd.Series, optional
        If provided, will check that the stored discharge obs match this series. Else, will compare to the discharge
        loaded from disk.
    """
    test_start_date, test_end_date = _get_test_start_end_dates(config)

    results = _get_basin_results(config.run_dir, 1)[basin]['1D']['xr'].isel(time_step=-1)

    assert pd.to_datetime(results['date'].values[0]) == test_start_date.date()
    assert pd.to_datetime(results['date'].values[-1]) == test_end_date.date()

    if discharge is None:
        discharge = _get_discharge(config, basin)

    assert discharge.loc[test_start_date:test_end_date].values \
           == approx(results[f'{config.target_variables[0]}_obs'].values.reshape(-1), nan_ok=True)

    # CAMELS forcings have no NaNs, so there should be no NaN predictions
    assert not pd.isna(results[f'{config.target_variables[0]}_sim']).any()


def _get_test_start_end_dates(config: Config) -> Tuple[datetime, datetime]:
    test_start_date = pd.to_datetime(config.test_start_date, format='%d/%m/%Y')
    test_end_date = pd.to_datetime(config.test_end_date, format='%d/%m/%Y') + pd.Timedelta(days=1, seconds=-1)

    return test_start_date, test_end_date


def _get_basin_results(run_dir: Path, epoch: int) -> Dict:
    results_file = list(run_dir.glob(f'test/model_epoch{str(epoch).zfill(3)}/test_results.p'))
    if len(results_file) != 1:
        pytest.fail('Results file not found.')

    return pickle.load(open(str(results_file[0]), 'rb'))


def _get_discharge(config: Config, basin: str) -> pd.Series:
    if config.dataset == 'camels_us':
        _, area = camelsus.load_camels_us_forcings(config.data_dir, basin, 'daymet')
        return camelsus.load_camels_us_discharge(config.data_dir, basin, area)
    else:
        raise NotImplementedError
