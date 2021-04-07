import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats, signal
from xarray.core.dataarray import DataArray

from neuralhydrology.datautils import utils
from neuralhydrology.utils.errors import AllNaNError

LOGGER = logging.getLogger(__name__)


def get_available_metrics() -> List[str]:
    """Get list of available metrics.

    Returns
    -------
    List[str]
        List of implemented metric names.
    """
    metrics = ["NSE", "MSE", "RMSE", "KGE", "Alpha-NSE", "Pearson-r", "Beta-NSE", "FHV", "FMS", "FLV", "Peak-Timing"]
    return metrics


def _validate_inputs(obs: DataArray, sim: DataArray):
    if obs.shape != sim.shape:
        raise RuntimeError("Shapes of observations and simulations must match")

    if (len(obs.shape) > 1) and (obs.shape[1] > 1):
        raise RuntimeError("Metrics only defined for time series (1d or 2d with second dimension 1)")


def _mask_valid(obs: DataArray, sim: DataArray) -> Tuple[DataArray, DataArray]:
    # mask of invalid entries. NaNs in simulations can happen during validation/testing
    idx = (~sim.isnull()) & (~obs.isnull())

    obs = obs[idx]
    sim = sim[idx]

    return obs, sim


def _get_fdc(da: DataArray) -> np.ndarray:
    return da.sortby(da, ascending=False).values


def nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate Nash-Sutcliffe Efficiency [#]_
    
    Nash-Sutcliffe Efficiency is the R-square between observed and simulated discharge.
    
    .. math:: \text{NSE} = 1 - \frac{\sum_{t=1}^{T}(Q_m^t - Q_o^t)^2}{\sum_{t=1}^T(Q_o^t - \overline{Q}_o)^2},
    
    where :math:`Q_m` are the simulations (here, `sim`) and :math:`Q_o` are observations (here, `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Nash-Sutcliffe Efficiency 
        
    References
    ----------
    .. [#] Nash, J. E.; Sutcliffe, J. V. (1970). "River flow forecasting through conceptual models part I - A 
        discussion of principles". Journal of Hydrology. 10 (3): 282-290. doi:10.1016/0022-1694(70)90255-6.

    """

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    denominator = ((obs - obs.mean())**2).sum()
    numerator = ((sim - obs)**2).sum()

    value = 1 - numerator / denominator

    return float(value)


def mse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate mean squared error.
    
    .. math:: \text{MSE} = \frac{1}{T}\sum_{t=1}^T (\widehat{y}_t - y_t)^2,
    
    where :math:`\widehat{y}` are the simulations (here, `sim`) and :math:`y` are observations 
    (here, `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Mean squared error. 

    """

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(((sim - obs)**2).mean())


def rmse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate root mean squared error.
    
    .. math:: \text{RMSE} = \sqrt{\frac{1}{T}\sum_{t=1}^T (\widehat{y}_t - y_t)^2},
    
    where :math:`\widehat{y}` are the simulations (here, `sim`) and :math:`y` are observations 
    (here, `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Root mean sqaured error.

    """

    return np.sqrt(mse(obs, sim))


def alpha_nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the alpha NSE decomposition [#]_
    
    The alpha NSE decomposition is the fraction of the standard deviations of simulations and observations.
    
    .. math:: \alpha = \frac{\sigma_s}{\sigma_o},
    
    where :math:`\sigma_s` is the standard deviation of the simulations (here, `sim`) and :math:`\sigma_o` is the 
    standard deviation of the observations (here, `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Alpha NSE decomposition.
        
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.

    """

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(sim.std() / obs.std())


def beta_nse(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the beta NSE decomposition [#]_

    The beta NSE decomposition is the difference of the mean simulation and mean observation divided by the standard 
    deviation of the observations.

    .. math:: \beta = \frac{\mu_s - \mu_o}{\sigma_o},
    
    where :math:`\mu_s` is the mean of the simulations (here, `sim`), :math:`\mu_o` is the mean of the observations 
    (here, `obs`) and :math:`\sigma_o` the standard deviation of the observations.

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Beta NSE decomposition.

    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.

    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float((sim.mean() - obs.mean()) / obs.std())


def beta_kge(obs: DataArray, sim: DataArray) -> float:
    r"""Calculate the beta KGE term [#]_
    
    The beta term of the Kling-Gupta Efficiency is defined as the fraction of the means.
    
    .. math:: \beta_{\text{KGE}} = \frac{\mu_s}{\mu_o},
    
    where :math:`\mu_s` is the mean of the simulations (here, `sim`) and :math:`\mu_o` is the mean of the observations 
    (here, `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Beta NSE decomposition.

    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.

    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    return float(sim.mean() / obs.mean())


def kge(obs: DataArray, sim: DataArray, weights: List[float] = [1., 1., 1.]) -> float:
    r"""Calculate the Kling-Gupta Efficieny [#]_
    
    .. math:: 
        \text{KGE} = 1 - \sqrt{[ s_r (r - 1)]^2 + [s_\alpha ( \alpha - 1)]^2 + 
            [s_\beta(\beta_{\text{KGE}} - 1)]^2},
            
    where :math:`r` is the correlation coefficient, :math:`\alpha` the :math:`\alpha`-NSE decomposition, 
    :math:`\beta_{\text{KGE}}` the fraction of the means and :math:`s_r, s_\alpha, s_\beta` the corresponding weights
    (here the three float values in the `weights` parameter).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    weights : List[float]
        Weighting factors of the 3 KGE parts, by default each part has a weight of 1.

    Returns
    -------
    float
        Kling-Gupta Efficiency
    
    References
    ----------
    .. [#] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error 
        and NSE performance criteria: Implications for improving hydrological modelling. Journal of hydrology, 377(1-2),
        80-91.

    """
    if len(weights) != 3:
        raise ValueError("Weights of the KGE must be a list of three values")

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 2:
        return np.nan

    r, _ = stats.pearsonr(obs.values, sim.values)

    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()

    value = (weights[0] * (r - 1)**2 + weights[1] * (alpha - 1)**2 + weights[2] * (beta - 1)**2)

    return 1 - np.sqrt(float(value))


def pearsonr(obs: DataArray, sim: DataArray) -> float:
    """Calculate pearson correlation coefficient (using scipy.stats.pearsonr)

    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.

    Returns
    -------
    float
        Pearson correlation coefficient

    """

    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 2:
        return np.nan

    r, _ = stats.pearsonr(obs.values, sim.values)

    return float(r)


def fdc_fms(obs: DataArray, sim: DataArray, lower: float = 0.2, upper: float = 0.7) -> float:
    r"""Calculate the slope of the middle section of the flow duration curve [#]_
    
    .. math:: 
        \%\text{BiasFMS} = \frac{\left | \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right | - 
            \left | \log(Q_{o,\text{lower}}) - \log(Q_{o,\text{upper}}) \right |}{\left | 
            \log(Q_{s,\text{lower}}) - \log(Q_{s,\text{upper}}) \right |} \times 100,
            
    where :math:`Q_{s,\text{lower/upper}}` corresponds to the FDC of the simulations (here, `sim`) at the `lower` and
    `upper` bound of the middle section and :math:`Q_{o,\text{lower/upper}}` similarly for the observations (here, 
    `obs`).
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    lower : float, optional
        Lower bound of the middle section in range ]0,1[, by default 0.2
    upper : float, optional
        Upper bound of the middle section in range ]0,1[, by default 0.7
        
    Returns
    -------
    float
        Slope of the middle section of the flow duration curve.
    
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model 
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, 
        doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 1:
        return np.nan

    if any([(x <= 0) or (x >= 1) for x in [upper, lower]]):
        raise ValueError("upper and lower have to be in range ]0,1[")

    if lower >= upper:
        raise ValueError("The lower threshold has to be smaller than the upper.")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    # calculate fms part by part
    qsm_lower = np.log(sim[np.round(lower * len(sim)).astype(int)])
    qsm_upper = np.log(sim[np.round(upper * len(sim)).astype(int)])
    qom_lower = np.log(obs[np.round(lower * len(obs)).astype(int)])
    qom_upper = np.log(obs[np.round(upper * len(obs)).astype(int)])

    fms = ((qsm_lower - qsm_upper) - (qom_lower - qom_upper)) / (qom_lower - qom_upper + 1e-6)

    return fms * 100


def fdc_fhv(obs: DataArray, sim: DataArray, h: float = 0.02) -> float:
    r"""Calculate the peak flow bias of the flow duration curve [#]_
    
    .. math:: \%\text{BiasFHV} = \frac{\sum_{h=1}^{H}(Q_{s,h} - Q_{o,h})}{\sum_{h=1}^{H}Q_{o,h}} \times 100,
    
    where :math:`Q_s` are the simulations (here, `sim`), :math:`Q_o` the observations (here, `obs`) and `H` is the upper
    fraction of flows of the FDC (here, `h`). 
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    h : float, optional
        Fraction of upper flows to consider as peak flows of range ]0,1[, be default 0.02.
        
    Returns
    -------
    float
        Peak flow bias.
    
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model 
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, 
        doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 1:
        return np.nan

    if (h <= 0) or (h >= 1):
        raise ValueError("h has to be in range ]0,1[. Consider small values, e.g. 0.02 for 2% peak flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / np.sum(obs)

    return fhv * 100


def fdc_flv(obs: DataArray, sim: DataArray, l: float = 0.3) -> float:
    r"""Calculate the low flow bias of the flow duration curve [#]_
    
    .. math:: 
        \%\text{BiasFMS} = -1 \frac{\sum_{l=1}^{L}[\log(Q_{s,l}) - \log(Q_{s,L})] - \sum_{l=1}^{L}[\log(Q_{o,l})
            - \log(Q_{o,L})]}{\sum_{l=1}^{L}[\log(Q_{o,l}) - \log(Q_{o,L})]} \times 100,
    
    where :math:`Q_s` are the simulations (here, `sim`), :math:`Q_o` the observations (here, `obs`) and `L` is the lower
    fraction of flows of the FDC (here, `l`). 
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    l : float, optional
        Fraction of lower flows to consider as low flows of range ]0,1[, be default 0.3.
        
    Returns
    -------
    float
        Low flow bias.
    
    References
    ----------
    .. [#] Yilmaz, K. K., Gupta, H. V., and Wagener, T. ( 2008), A process-based diagnostic approach to model 
        evaluation: Application to the NWS distributed hydrologic model, Water Resour. Res., 44, W09417, 
        doi:10.1029/2007WR006716. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations
    obs, sim = _mask_valid(obs, sim)

    if len(obs) < 1:
        return np.nan

    if (l <= 0) or (l >= 1):
        raise ValueError("l has to be in range ]0,1[. Consider small values, e.g. 0.3 for 30% low flows")

    # get arrays of sorted (descending) discharges
    obs = _get_fdc(obs)
    sim = _get_fdc(sim)

    # for numerical reasons change 0s to 1e-6. Simulations can still contain negatives, so also reset those.
    sim[sim <= 0] = 1e-6
    obs[obs == 0] = 1e-6

    obs = obs[-np.round(l * len(obs)).astype(int):]
    sim = sim[-np.round(l * len(sim)).astype(int):]

    # transform values to log scale
    obs = np.log(obs)
    sim = np.log(sim)

    # calculate flv part by part
    qsl = np.sum(sim - sim.min())
    qol = np.sum(obs - obs.min())

    flv = -1 * (qsl - qol) / (qol + 1e-6)

    return flv * 100


def mean_peak_timing(obs: DataArray,
                     sim: DataArray,
                     window: int = None,
                     resolution: str = '1D',
                     datetime_coord: str = None) -> float:
    """Mean difference in peak flow timing.
    
    Uses scipy.find_peaks to find peaks in the observed time series. Starting with all observed peaks, those with a
    prominence of less than the standard deviation of the observed time series are discarded. Next, the lowest peaks
    are subsequently discarded until all remaining peaks have a distance of at least 100 steps. Finally, the
    corresponding peaks in the simulated time series are searched in a window of size `window` on either side of the
    observed peaks and the absolute time differences between observed and simulated peaks is calculated.
    The final metric is the mean absolute time difference across all peaks. For more details, see Appendix of [#]_
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    window : int, optional
        Size of window to consider on each side of the observed peak for finding the simulated peak. That is, the total
        window length to find the peak in the simulations is :math:`2 * \\text{window} + 1` centered at the observed
        peak. The default depends on the temporal resolution, e.g. for a resolution of '1D', a window of 3 is used and 
        for a resolution of '1H' the the window size is 12.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1H' for hourly.
    datetime_coord : str, optional
        Name of datetime coordinate. Tried to infer automatically if not specified.
        

    Returns
    -------
    float
        Mean peak time difference.

    References
    ----------
    .. [#] Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple 
        meteorological datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss., 
        https://doi.org/10.5194/hess-2020-221, in review, 2020. 
    """
    # verify inputs
    _validate_inputs(obs, sim)

    # get time series with only valid observations (scipy's find_peaks doesn't guarantee correctness with NaNs)
    obs, sim = _mask_valid(obs, sim)

    # heuristic to get indices of peaks and their corresponding height.
    peaks, _ = signal.find_peaks(obs.values, distance=100, prominence=np.std(obs.values))

    # infer name of datetime index
    if datetime_coord is None:
        datetime_coord = utils.infer_datetime_coord(obs)

    if window is None:
        # infer a reasonable window size
        window = max(int(utils.get_frequency_factor('12H', resolution)), 3)

    # evaluate timing
    timing_errors = []
    for idx in peaks:
        # skip peaks at the start and end of the sequence and peaks around missing observations
        # (NaNs that were removed in obs & sim would result in windows that span too much time).
        if (idx - window < 0) or (idx + window >= len(obs)) or (pd.date_range(obs[idx - window][datetime_coord].values,
                                                                              obs[idx + window][datetime_coord].values,
                                                                              freq=resolution).size != 2 * window + 1):
            continue

        # check if the value at idx is a peak (both neighbors must be smaller)
        if (sim[idx] > sim[idx - 1]) and (sim[idx] > sim[idx + 1]):
            peak_sim = sim[idx]
        else:
            # define peak around idx as the max value inside of the window
            values = sim[idx - window:idx + window + 1]
            peak_sim = values[values.argmax()]

        # get xarray object of qobs peak, for getting the date and calculating the datetime offset
        peak_obs = obs[idx]

        # calculate the time difference between the peaks
        delta = peak_obs.coords[datetime_coord] - peak_sim.coords[datetime_coord]

        timing_error = np.abs(delta.values / pd.to_timedelta(resolution))

        timing_errors.append(timing_error)

    return np.mean(timing_errors) if len(timing_errors) > 0 else np.nan


def calculate_all_metrics(obs: DataArray,
                          sim: DataArray,
                          resolution: str = "1D",
                          datetime_coord: str = None) -> Dict[str, float]:
    """Calculate all metrics with default values.
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1H' for hourly.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys corresponding to metric name and values corresponding to metric values.

    Raises
    ------
    AllNaNError
        If all observations or all simulations are NaN.
    """
    _check_all_nan(obs, sim)

    results = {
        "NSE": nse(obs, sim),
        "MSE": mse(obs, sim),
        "RMSE": rmse(obs, sim),
        "KGE": kge(obs, sim),
        "Alpha-NSE": alpha_nse(obs, sim),
        "Beta-NSE": beta_nse(obs, sim),
        "Pearson-r": pearsonr(obs, sim),
        "FHV": fdc_fhv(obs, sim),
        "FMS": fdc_fms(obs, sim),
        "FLV": fdc_flv(obs, sim),
        "Peak-Timing": mean_peak_timing(obs, sim, resolution=resolution, datetime_coord=datetime_coord)
    }

    return results


def calculate_metrics(obs: DataArray,
                      sim: DataArray,
                      metrics: List[str],
                      resolution: str = "1D",
                      datetime_coord: str = None) -> Dict[str, float]:
    """Calculate specific metrics with default values.
    
    Parameters
    ----------
    obs : DataArray
        Observed time series.
    sim : DataArray
        Simulated time series.
    metrics : List[str]
        List of metric names.
    resolution : str, optional
        Temporal resolution of the time series in pandas format, e.g. '1D' for daily and '1H' for hourly.
    datetime_coord : str, optional
        Datetime coordinate in the passed DataArray. Tried to infer automatically if not specified.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys corresponding to metric name and values corresponding to metric values.

    Raises
    ------
    AllNaNError
        If all observations or all simulations are NaN.
    """
    if 'all' in metrics:
        return calculate_all_metrics(obs, sim, resolution=resolution)

    _check_all_nan(obs, sim)

    values = {}
    for metric in metrics:
        if metric.lower() == "nse":
            values["NSE"] = nse(obs, sim)
        elif metric.lower() == "mse":
            values["MSE"] = mse(obs, sim)
        elif metric.lower() == "rmse":
            values["RMSE"] = rmse(obs, sim)
        elif metric.lower() == "kge":
            values["KGE"] = kge(obs, sim)
        elif metric.lower() == "alpha-nse":
            values["Alpha-NSE"] = alpha_nse(obs, sim)
        elif metric.lower() == "beta-nse":
            values["Beta-NSE"] = beta_nse(obs, sim)
        elif metric.lower() == "pearson-r":
            values["Pearson-r"] = pearsonr(obs, sim)
        elif metric.lower() == "fhv":
            values["FHV"] = fdc_fhv(obs, sim)
        elif metric.lower() == "fms":
            values["FMS"] = fdc_fms(obs, sim)
        elif metric.lower() == "flv":
            values["FLV"] = fdc_flv(obs, sim)
        elif metric.lower() == "peak-timing":
            values["Peak-Timing"] = mean_peak_timing(obs, sim, resolution=resolution, datetime_coord=datetime_coord)
        else:
            raise RuntimeError(f"Unknown metric {metric}")

    return values


def _check_all_nan(obs: DataArray, sim: DataArray):
    """Check if all observations or simulations are NaN and raise an exception if this is the case.

    Raises
    ------
    AllNaNError
        If all observations or all simulations are NaN.
    """
    if all(obs.isnull()):
        raise AllNaNError("All observed values are NaN, thus metrics will be NaN, too.")
    if all(sim.isnull()):
        raise AllNaNError("All simulated values are NaN, thus metrics will be NaN, too.")
