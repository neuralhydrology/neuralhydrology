"""Utility script to average the predictions of several runs. """
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from neuralhydrology.datautils.utils import get_frequency_factor, sort_frequencies
from neuralhydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from neuralhydrology.evaluation.utils import metrics_to_dataframe
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import AllNaNError


def create_results_ensemble(run_dirs: List[Path],
                            best_k: int = None,
                            metrics: List[str] = None,
                            period: str = 'test',
                            epoch: int = None) -> dict:
    """Average the predictions of several runs for the specified period and calculate new metrics.

    If `best_k` is provided, only the k runs with the best validation NSE will be used in the generated ensemble.
    
    Parameters
    ----------
    run_dirs : List[Path]
        List of directories of the runs to be merged
    best_k : int, optional
        If provided, will only merge the k best runs based on validation NSE.
    metrics : List[str], optional
        Use this parameter to override the metrics from the config files in the run directories.
    period : {'test', 'validation', 'train'}, optional
        One of train, val, test. If best_k is used, only 'test' is allowed. 
        The run_directories must contain results files for the specified period.
    epoch : int, optional
        If provided, will ensemble the model predictions of this epoch otherwise of the last epoch
    
    Returns
    -------
    dict
        Dictionary of ensemble predictions and metrics per basin and frequency.
    """
    if len(run_dirs) < 2:
        raise ValueError('Need to provide at least two run directories to be merged.')

    if period not in ['train', 'validation', 'test']:
        raise ValueError(f'Unknown period {period}.')
    if best_k is not None:
        if period != 'test':
            raise ValueError('If best_k is specified, the period must be test.')
        print('Searching for best validation runs.')
        best_val_runs = _get_best_validation_runs(run_dirs, best_k, epoch)
        best_runs = [_get_results_file(run_dir, period, epoch) for run_dir in best_val_runs]
    else:
        best_runs = [_get_results_file(run_dir, period, epoch) for run_dir in run_dirs]

    config = Config(run_dirs[0] / 'config.yml')
    if metrics is not None:
        # override metrics from config
        config.metrics = metrics

    # get frequencies from a results file.
    # (they might not be stored in the config if the native data frequency was used)
    run_results = pickle.load(open(best_runs[0], 'rb'))
    frequencies = list(run_results[list(run_results.keys())[0]].keys())

    return _create_ensemble(best_runs, frequencies, config)


def _create_ensemble(results_files: List[Path], frequencies: List[str], config: Config) -> dict:
    """Averages the predictions of the passed runs and re-calculates metrics. """
    lowest_freq = sort_frequencies(frequencies)[0]
    ensemble_sum = defaultdict(dict)
    target_vars = config.target_variables

    print('Loading results for each run.')
    for run in tqdm(results_files):
        run_results = pickle.load(open(run, 'rb'))
        for basin, basin_results in run_results.items():
            for freq in frequencies:
                freq_results = basin_results[freq]['xr']

                # sum up the predictions of all basins
                if freq not in ensemble_sum[basin]:
                    ensemble_sum[basin][freq] = freq_results
                else:
                    for target_var in target_vars:
                        ensemble_sum[basin][freq][f'{target_var}_sim'] += freq_results[f'{target_var}_sim']

    # divide the prediction sum by number of runs to get the mean prediction for each basin and frequency
    print('Combining results and calculating metrics.')
    ensemble = defaultdict(lambda: defaultdict(dict))
    for basin in tqdm(ensemble_sum.keys()):
        for freq in frequencies:
            ensemble_xr = ensemble_sum[basin][freq]

            # combine date and time to a single index to calculate metrics
            # create datetime range at the current frequency, removing time steps that are not being predicted
            frequency_factor = int(get_frequency_factor(lowest_freq, freq))
            # make sure the last day is fully contained in the range
            freq_date_range = pd.date_range(start=ensemble_xr.coords['date'].values[0],
                                            end=ensemble_xr.coords['date'].values[-1] \
                                                + pd.Timedelta(days=1, seconds=-1),
                                            freq=freq)
            mask = np.ones(frequency_factor).astype(bool)
            mask[:-len(ensemble_xr.coords['time_step'])] = False
            freq_date_range = freq_date_range[np.tile(mask, len(ensemble_xr.coords['date']))]

            ensemble_xr = ensemble_xr.isel(time_step=slice(-frequency_factor, None)) \
                .stack(datetime=['date', 'time_step']) \
                .drop_vars({'datetime', 'date', 'time_step'})
            ensemble_xr['datetime'] = freq_date_range
            for target_var in target_vars:
                # average predictions
                ensemble_xr[f'{target_var}_sim'] = ensemble_xr[f'{target_var}_sim'] / len(results_files)

                # clip predictions to zero
                sim = ensemble_xr[f'{target_var}_sim']
                if target_var in config.clip_targets_to_zero:
                    sim = xr.where(sim < 0, 0, sim)

                # calculate metrics
                metrics = config.metrics if isinstance(config.metrics, list) else config.metrics[target_var]
                if 'all' in metrics:
                    metrics = get_available_metrics()
                try:
                    ensemble_metrics = calculate_metrics(ensemble_xr[f'{target_var}_obs'],
                                                         sim,
                                                         metrics=metrics,
                                                         resolution=freq)
                except AllNaNError as err:
                    msg = f'Basin {basin} ' \
                        + (f'{target_var} ' if len(target_vars) > 1 else '') \
                        + (f'{freq} ' if len(frequencies) > 1 else '') \
                        + str(err)
                    print(msg)
                    ensemble_metrics = {metric: np.nan for metric in metrics}

                # add variable identifier to metrics if needed
                if len(target_vars) > 1:
                    ensemble_metrics = {f'{target_var}_{key}': val for key, val in ensemble_metrics.items()}
                # add frequency identifier to metrics if needed
                if len(frequencies) > 1:
                    ensemble_metrics = {f'{key}_{freq}': val for key, val in ensemble_metrics.items()}
                for metric, val in ensemble_metrics.items():
                    ensemble[basin][freq][metric] = val

            ensemble[basin][freq]['xr'] = ensemble_xr

    return dict(ensemble)


def _get_medians(results: dict, metric='NSE') -> dict:
    """Calculates median metric across all basins. """
    medians = {}
    key = metric
    frequencies = list(results[list(results.keys())[0]].keys())
    for freq in frequencies:
        if len(frequencies) > 1:
            key = f'{metric}_{freq}'
        metric_values = [v[freq][key] for v in results.values() if freq in v.keys() and key in v[freq].keys()]
        medians[freq] = np.nanmedian(metric_values)

    return medians


def _get_best_validation_runs(run_dirs: List[Path], k: int, epoch: int = None) -> List[Path]:
    """Returns the k run directories with the best median validation metrics. """
    val_files = list(zip(run_dirs, [_get_results_file(run_dir, 'validation', epoch) for run_dir in run_dirs]))

    # get validation medians
    median_sums = {}
    for run_dir, val_file in val_files:
        val_results = pickle.load(open(val_file, 'rb'))
        val_medians = _get_medians(val_results)
        print('validation', val_file, val_medians)
        median_sums[run_dir] = sum(val_medians.values())

    if k > len(run_dirs):
        raise ValueError(f'best_k k is larger than number of runs {len(val_files)}.')
    return sorted(median_sums, key=median_sums.get, reverse=True)[:k]


def _get_results_file(run_dir: Path, period: str = 'test', epoch: int = None) -> Path:
    """Returns the path of the results file in the given run directory. """
    if epoch is not None:
        dir_results_files = list(Path(run_dir).glob(f'{period}/model_epoch{str(epoch).zfill(3)}/{period}_results.p'))
    else:
        dir_results_files = list(Path(run_dir).glob(f'{period}/model_epoch*/{period}_results.p'))
    if len(dir_results_files) == 0:
        raise ValueError(f'{run_dir} is missing {period} results.')
    return sorted(dir_results_files)[-1]


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dirs', type=str, nargs='+', help='Directories of the runs to be averaged.')
    parser.add_argument('--period', type=str, choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--output-dir', type=str, help='Path to directory, where results are stored.')
    parser.add_argument('--metrics',
                        type=str,
                        nargs='+',
                        required=False,
                        help='Option to override the metrics from the config.')
    parser.add_argument('--best-k',
                        type=int,
                        required=False,
                        help='If provided, will only use the k results files with the best median validation NSEs.')
    parser.add_argument('--epoch',
                        type=int,
                        required=False,
                        help='If provided, will return results of this specific epoch otherwise of the last epoch')
    args = vars(parser.parse_args())

    run_dirs = [Path(f) for f in args['run_dirs']]
    ensemble_results = create_results_ensemble(run_dirs,
                                               args['best_k'],
                                               metrics=args['metrics'],
                                               period=args['period'],
                                               epoch=args['epoch'])
    output_dir = Path(args['output_dir']).absolute()

    metrics = args['metrics']
    if metrics is None:
        metrics = Config(run_dirs[0] / 'config.yml').metrics
    try:
        df = metrics_to_dataframe(ensemble_results, metrics)
        file_name = output_dir / f"{args['period']}_ensemble_metrics.csv"
        df.to_csv(file_name)
        print(f"Stored metrics of ensemble run to {file_name}")
    except RuntimeError as err:
        # in case no metrics were computed
        pass

    file_name = output_dir / f"{args['period']}_ensemble_results.p"
    pickle.dump(ensemble_results, open(file_name, 'wb'))
    print(f'Successfully written results to {file_name}')


if __name__ == '__main__':
    _main()
