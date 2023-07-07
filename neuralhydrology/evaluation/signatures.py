import warnings
from datetime import datetime
from typing import Dict, List, Tuple

from dateutil.relativedelta import relativedelta

import numpy as np
from numba import njit
from xarray.core.dataarray import DataArray

from neuralhydrology.datautils import utils


def get_available_signatures() -> List[str]:
    """Return a list of available signatures.

    Returns
    -------
    List[str]
        List of all available signatures.
    """
    signatures = [
        "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq", "q95", "q5", "q_mean", "hfd_mean",
        "baseflow_index", "slope_fdc", "stream_elas", "runoff_ratio"
    ]
    return signatures


def calculate_all_signatures(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> Dict[str, float]:
    """Calculate all signatures with default values.

    Parameters
    ----------
    da : DataArray
        Array of discharge values for which the signatures will be calculated.
    prcp : DataArray
        Array of precipitation values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.

    Returns
    -------
    Dict[str, float]
        Dictionary with signature names as keys and signature values as values.
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    results = {
        "high_q_freq": high_q_freq(da, datetime_coord=datetime_coord),
        "high_q_dur": high_q_dur(da),
        "low_q_freq": low_q_freq(da, datetime_coord=datetime_coord),
        "low_q_dur": low_q_dur(da),
        "zero_q_freq": zero_q_freq(da),
        "q95": q95(da),
        "q5": q5(da),
        "q_mean": q_mean(da),
        "hfd_mean": hfd_mean(da, datetime_coord=datetime_coord),
        "baseflow_index": baseflow_index(da, datetime_coord=datetime_coord)[0],
        "slope_fdc": slope_fdc(da),
        "stream_elas": stream_elas(da, prcp, datetime_coord=datetime_coord),
        "runoff_ratio": runoff_ratio(da, prcp, datetime_coord=datetime_coord)
    }
    return results


def calculate_signatures(da: DataArray,
                         signatures: List[str],
                         datetime_coord: str = None,
                         prcp: DataArray = None) -> Dict[str, float]:
    """Calculate the specified signatures with default values.

    Parameters
    ----------
    da : DataArray
        Array of discharge values for which the signatures will be calculated.
    signatures : List[str]
        List of names of the signatures to calculate.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.
    prcp : DataArray, optional
        Array of precipitation values. Required for signatures 'runoff_ratio' and 'streamflow_elas'.

    Returns
    -------
    Dict[str, float]
        Dictionary with signature names as keys and signature values as values.

    Raises
    ------
    ValueError
        If a passed signature name does not exist.
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    values = {}
    for signature in signatures:
        if signature == "high_q_freq":
            values["high_q_freq"] = high_q_freq(da, datetime_coord=datetime_coord)
        elif signature == "high_q_dur":
            values["high_q_dur"] = high_q_dur(da)
        elif signature == "low_q_freq":
            values["low_q_freq"] = low_q_freq(da, datetime_coord=datetime_coord)
        elif signature == "low_q_dur":
            values["low_q_dur"] = low_q_dur(da)
        elif signature == "zero_q_freq":
            values["zero_q_freq"] = zero_q_freq(da)
        elif signature == "q95":
            values["q95"] = q95(da)
        elif signature == "q5":
            values["q5"] = q5(da)
        elif signature == "q_mean":
            values["q_mean"] = q_mean(da)
        elif signature == "hfd_mean":
            values["hfd_mean"] = hfd_mean(da, datetime_coord=datetime_coord)
        elif signature == "baseflow_index":
            values["baseflow_index"] = baseflow_index(da, datetime_coord=datetime_coord)[0]
        elif signature == "slope_fdc":
            values["slope_fdc"] = slope_fdc(da)
        elif signature == "runoff_ratio":
            values["runoff_ratio"] = runoff_ratio(da, prcp, datetime_coord=datetime_coord)
        elif signature == "stream_elas":
            values["stream_elas"] = stream_elas(da, prcp, datetime_coord=datetime_coord)
        else:
            ValueError(f"Unknown signatures {signature}")
    return values


@njit
def _split_list(alist: list, min_length: int = 0) -> list:
    """Split a list of indices into lists of consecutive indices of at least length `min_length`. """
    newlist = []
    start = 0
    for index, value in enumerate(alist):
        if index < len(alist) - 1:
            if alist[index + 1] > value + 1:
                end = index + 1
                if end - start >= min_length:
                    newlist.append(alist[start:end])
                start = end
        else:
            if len(alist) - start >= min_length:
                newlist.append(alist[start:len(alist)])
    return newlist


def high_q_dur(da: DataArray, threshold: float = 9.) -> float:
    """Calculate high-flow duration.

    Average duration of high-flow events (number of consecutive steps >`threshold` times the median flow) [#]_,
    [#]_ (Table 2).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    threshold : float, optional
        High-flow threshold. Values larger than ``threshold * median`` are considered high flows.

    Returns
    -------
    float
        High-flow duration

    References
    ----------
    .. [#] Clausen, B. and Biggs, B. J. F.: Flow variables for ecological studies in temperate streams: groupings based
        on covariance. Journal of Hydrology, 2000, 237, 184--197, doi:10.1016/S0022-1694(00)00306-1
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    median_flow = float(da.median())
    idx = np.where(da.values > threshold * median_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        hqd = np.mean([len(p) for p in periods])
    else:
        hqd = np.nan
    return hqd


def low_q_dur(da: DataArray, threshold: float = 0.2) -> float:
    """Calculate low-flow duration.

    Average duration of low-flow events (number of consecutive steps <`threshold` times the median flow) [#]_,
    [#]_ (Table 2).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    threshold : float, optional
        Low-flow threshold. Values below ``threshold * median`` are considered low flows.

    Returns
    -------
    float
        Low-flow duration

    References
    ----------
    .. [#] Olden, J. D. and Poff, N. L.: Redundancy and the choice of hydrologic indices for characterizing streamflow
        regimes. River Research and Applications, 2003, 19, 101--121, doi:10.1002/rra.700
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    mean_flow = float(da.mean())
    idx = np.where(da.values < threshold * mean_flow)[0]
    if len(idx) > 0:
        periods = _split_list(idx)
        lqd = np.mean([len(p) for p in periods])
    else:
        lqd = np.nan
    return lqd


def zero_q_freq(da: DataArray) -> float:
    """Calculate zero-flow frequency.

    Frequency of steps with zero discharge.

    Parameters
    ----------
    da : DataArray
        Array of flow values.

    Returns
    -------
    float
        Zero-flow frequency.
    """
    # number of steps with zero flow
    n_steps = (da == 0).sum()

    return float(n_steps / len(da))


def high_q_freq(da: DataArray, datetime_coord: str = None, threshold: float = 9.) -> float:
    """Calculate high-flow frequency.

    Frequency of high-flow events (>`threshold` times the median flow) [#]_, [#]_ (Table 2).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.
    threshold : float, optional
        High-flow threshold. Values larger than ``threshold * median`` are considered high flows.

    Returns
    -------
    float
        High-flow frequency

    References
    ----------
    .. [#] Clausen, B. and Biggs, B. J. F.: Flow variables for ecological studies in temperate streams: groupings based
        on covariance. Journal of Hydrology, 2000, 237, 184--197, doi:10.1016/S0022-1694(00)00306-1
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # determine the date of the first January 1st in the data period
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # end date of the first full year period
    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # determine the median flow over the entire period
    median_flow = da.median(skipna=True)

    hqfs = []
    while end_date < last_date:

        data = da.sel({datetime_coord: slice(start_date, end_date)})

        # number of steps with discharge higher than threshold * median in a one year period
        n_steps = (data > (threshold * median_flow)).sum()

        hqfs.append(float(n_steps))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(hqfs)


def low_q_freq(da: DataArray, datetime_coord: str = None, threshold: float = 0.2) -> float:
    """Calculate Low-flow frequency.

    Frequency of low-flow events (<`threshold` times the median flow) [#]_, [#]_ (Table 2).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.
    threshold : float, optional
        Low-flow threshold. Values below ``threshold * median`` are considered low flows.

    Returns
    -------
    float
        Low-flow frequency

    References
    ----------
    .. [#] Olden, J. D. and Poff, N. L.: Redundancy and the choice of hydrologic indices for characterizing streamflow
        regimes. River Research and Applications, 2003, 19, 101--121, doi:10.1002/rra.700
    .. [#] Westerberg, I. K. and McMillan, H. K.: Uncertainty in hydrological signatures.
        Hydrology and Earth System Sciences, 2015, 19, 3951--3968, doi:10.5194/hess-19-3951-2015
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # determine the date of the first January 1st in the data period
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date == datetime.strptime(f'{first_date.year}-01-01', '%Y-%m-%d'):
        start_date = first_date
    else:
        start_date = datetime.strptime(f'{first_date.year + 1}-01-01', '%Y-%m-%d')

    # end date of the first full year period
    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # determine the mean flow over the entire period
    mean_flow = da.mean(skipna=True)

    lqfs = []
    while end_date < last_date:

        data = da.sel({datetime_coord: slice(start_date, end_date)})

        # number of steps with discharge lower than threshold * median in a one year period
        n_steps = (data < (threshold * mean_flow)).sum()

        lqfs.append(float(n_steps))

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(lqfs)


def hfd_mean(da: DataArray, datetime_coord: str = None) -> float:
    """Calculate mean half-flow duration.

    Mean half-flow date (step on which the cumulative discharge since October 1st
    reaches half of the annual discharge) [#]_.

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.

    Returns
    -------
    float
        Mean half-flow duration

    References
    ----------
    .. [#] Court, A.: Measures of streamflow timing. Journal of Geophysical Research (1896-1977), 1962, 67, 4335--4339,
        doi:10.1029/JZ067i011p04335
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # determine the date of the first October 1st in the data period
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    doys = []
    while end_date < last_date:

        # compute cumulative sum for the selected period
        data = da.sel({datetime_coord: slice(start_date, end_date)})
        cs = data.cumsum(skipna=True)

        # find steps with more cumulative discharge than the half annual sum
        hf_steps = np.where(~np.isnan(cs.where(cs > data.sum(skipna=True) / 2).values))[0]

        # ignore days without discharge
        if len(hf_steps) > 0:
            # store the first step in the result array
            doys.append(hf_steps[0])

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.mean(doys)


def q5(da: DataArray) -> float:
    """Calculate 5th flow quantile.

    Parameters
    ----------
    da : DataArray
        Array of flow values.

    Returns
    -------
    float
        5th flow quantile.
    """
    return float(da.quantile(0.05))


def q95(da: DataArray) -> float:
    """Calculate 95th flow quantile.

    Parameters
    ----------
    da : DataArray
        Array of flow values.

    Returns
    -------
    float
        95th flow quantile.
    """
    return float(da.quantile(0.95))


def q_mean(da: DataArray) -> float:
    """Calculate mean discharge.

    Parameters
    ----------
    da : DataArray
        Array of flow values.

    Returns
    -------
    float
        Mean discharge.
    """
    return float(da.mean())


@njit
def _baseflow_index_jit(streamflow: np.ndarray, alpha: float, warmup: int, n_passes: int) -> Tuple[float, np.ndarray]:
    non_nan_indices = np.where(~np.isnan(streamflow))[0]
    non_nan_streamflow_runs = _split_list(non_nan_indices, min_length=warmup + 1)
    if len(non_nan_streamflow_runs) == 0:
        # no consecutive run of non-NaN values is long enough to calculate baseflow.
        return np.nan, np.full_like(streamflow, np.nan)

    streamflow_sum = 0
    overall_baseflow = np.full_like(streamflow, np.nan)
    for non_nan_run in non_nan_streamflow_runs:
        streamflow_run = streamflow[non_nan_run]

        # mirror discharge of length 'window' at the start and end
        padded_streamflow = np.zeros((len(streamflow_run) + 2 * warmup))
        padded_streamflow[warmup:-warmup] = streamflow_run
        padded_streamflow[:warmup] = streamflow_run[1:warmup + 1][::-1]
        padded_streamflow[-warmup:] = streamflow_run[-warmup - 1:-1][::-1]

        baseflow = padded_streamflow
        for _ in range(n_passes):
            new_baseflow = np.zeros_like(padded_streamflow)
            quickflow = baseflow[0]
            for i in range(1, len(padded_streamflow)):
                quickflow = alpha * quickflow + (1 + alpha) * (baseflow[i] - baseflow[i - 1]) / 2
                if quickflow > 0:
                    new_baseflow[i] = baseflow[i] - quickflow
                else:
                    new_baseflow[i] = baseflow[i]

            # switch between forward and backward passes. Next iteration's input is the baseflow generated in this iteration
            baseflow = np.flip(new_baseflow)

        overall_baseflow[non_nan_run] = baseflow[warmup:-warmup]
        streamflow_sum += np.sum(streamflow_run)

    bf_index = np.nansum(overall_baseflow) / streamflow_sum

    return bf_index, overall_baseflow


def baseflow_index(da: DataArray,
                   alpha: float = 0.98,
                   warmup: int = 30,
                   n_passes: int = None,
                   datetime_coord: str = None) -> Tuple[float, DataArray]:
    """Calculate baseflow index.

    Ratio of mean baseflow to mean discharge [#]_. If `da` contains NaN values, the baseflow is calculated for each
    consecutive segment of more than `warmup` non-NaN values.

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    alpha : float, optional
        alpha filter parameter.
    warmup : int, optional
        Number of warmup steps.
    n_passes : int, optional
        Number of passes (alternating forward and backward) to perform. Should be an odd number. If None, will use
        3 for daily and 9 for hourly data and fail for all other input frequencies.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified. Used to infer the 
        frequency if `n_passes` is None.

    Returns
    -------
    Tuple[float, DataArray]
        Baseflow index and baseflow array. The baseflow array contains NaNs wherever no baseflow was
        calculated due to NaNs in `da`.

    Raises
    ------
    ValueError
        If `da` has a frequency other than daily or hourly and `n_passes` is None.

    References
    ----------
    .. [#] Ladson, T. R., Brown, R., Neal, B., and Nathan, R.: A Standard Approach to Baseflow Separation Using The
        Lyne and Hollick Filter. Australasian Journal of Water Resources, Taylor & Francis, 2013, 17, 25--34,
        doi:10.7158/13241583.2013.11465417
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    if n_passes is None:
        freq = utils.infer_frequency(da[datetime_coord].values)
        if freq == '1D':
            n_passes = 3
        elif freq == '1H':
            n_passes = 9
        else:
            raise ValueError(f'For frequencies other than daily or hourly, n_passes must be specified.')
    if n_passes % 2 != 1:
        warnings.warn('n_passes should be an even number. The returned baseflow will be reversed.')

    # call jit compiled function to calculate baseflow
    bf_index, baseflow = _baseflow_index_jit(da.values, alpha, warmup, n_passes)

    # parse baseflow as a DataArray using the coordinates of the streamflow array
    da_baseflow = da.copy()
    da_baseflow.data = baseflow

    return bf_index, da_baseflow


def slope_fdc(da: DataArray, lower_quantile: float = 0.33, upper_quantile: float = 0.66) -> float:
    """Calculates flow duration curve slope.

    Slope of the flow duration curve (between the log-transformed `lower_quantile` and `upper_quantile`) [#]_ (Eq. 3).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    lower_quantile : float, optional
        Lower quantile to use in slope calculation.
    upper_quantile : float, optional
        Upper quantile to use in slope calculation.

    Returns
    -------
    float
        Slope of the flow duration curve.

    References
    ----------
    .. [#] Sawicz, K., Wagener, T., Sivapalan, M., Troch, P. A., and Carrillo, G.: Catchment classification: empirical
        analysis of hydrologic similarity based on catchment function in the eastern USA.
        Hydrology and Earth System Sciences, 2011, 15, 2895--2911, doi:10.5194/hess-15-2895-2011
    """
    # sort discharge by descending order
    fdc = da.sortby(da, ascending=False)

    # get idx of lower and upper quantile
    idx_lower = np.round(lower_quantile * len(fdc)).astype(int)
    idx_upper = np.round(upper_quantile * len(fdc)).astype(int)

    value = (np.log(fdc[idx_lower].values + 1e-8) - np.log(fdc[idx_upper].values + 1e-8)) / (upper_quantile -
                                                                                             lower_quantile)

    return value


def runoff_ratio(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> float:
    """Calculate runoff ratio.

    Runoff ratio (ratio of mean discharge to mean precipitation) [#]_ (Eq. 2).

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    prcp : DataArray
        Array of precipitation values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.

    Returns
    -------
    float
        Runoff ratio.

    References
    ----------
    .. [#] Sawicz, K., Wagener, T., Sivapalan, M., Troch, P. A., and Carrillo, G.: Catchment classification: empirical
        analysis of hydrologic similarity based on catchment function in the eastern USA.
        Hydrology and Earth System Sciences, 2011, 15, 2895--2911, doi:10.5194/hess-15-2895-2011
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # rename precip coordinate name (to avoid problems with 'index' or 'date')
    prcp = prcp.rename({list(prcp.coords.keys())[0]: datetime_coord})

    # slice prcp to the same time window as the discharge
    prcp = prcp.sel({datetime_coord: slice(da.coords[datetime_coord][0], da.coords[datetime_coord][-1])})

    # calculate runoff ratio
    value = da.mean() / prcp.mean()

    return float(value)


def stream_elas(da: DataArray, prcp: DataArray, datetime_coord: str = None) -> float:
    """Calculate stream elasticity.

    Streamflow precipitation elasticity (sensitivity of streamflow to changes in precipitation at
    the annual time scale) [#]_.

    Parameters
    ----------
    da : DataArray
        Array of flow values.
    prcp : DataArray
        Array of precipitation values.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.

    Returns
    -------
    float
        Stream elasticity.

    References
    ----------
    .. [#] Sankarasubramanian, A., Vogel, R. M., and Limbrunner, J. F.: Climate elasticity of streamflow in the
        United States. Water Resources Research, 2001, 37, 1771--1781, doi:10.1029/2000WR900330
    """
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(da)

    # rename precip coordinate name (to avoid problems with 'index' or 'date')
    prcp = prcp.rename({list(prcp.coords.keys())[0]: datetime_coord})

    # slice prcp to the same time window as the discharge
    prcp = prcp.sel({datetime_coord: slice(da.coords[datetime_coord][0], da.coords[datetime_coord][-1])})

    # determine the date of the first October 1st in the data period
    first_date = da.coords[datetime_coord][0].values.astype('datetime64[s]').astype(datetime)
    last_date = da.coords[datetime_coord][-1].values.astype('datetime64[s]').astype(datetime)

    if first_date > datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d'):
        start_date = datetime.strptime(f'{first_date.year + 1}-10-01', '%Y-%m-%d')
    else:
        start_date = datetime.strptime(f'{first_date.year}-10-01', '%Y-%m-%d')

    end_date = start_date + relativedelta(years=1) - relativedelta(seconds=1)

    # mask only valid time steps (only discharge has missing values)
    idx = (da >= 0) & (~da.isnull())
    da = da[idx]
    prcp = prcp[idx]

    # calculate long-term means
    q_mean_total = da.mean()
    p_mean_total = prcp.mean()

    values = []
    while end_date < last_date:
        q = da.sel({datetime_coord: slice(start_date, end_date)})
        p = prcp.sel({datetime_coord: slice(start_date, end_date)})

        val = (q.mean() - q_mean_total) / (p.mean() - p_mean_total) * (p_mean_total / q_mean_total)
        values.append(val)

        start_date += relativedelta(years=1)
        end_date += relativedelta(years=1)

    return np.median([float(v) for v in values])
