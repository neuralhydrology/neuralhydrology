from pathlib import Path
import xarray as xr

from neuralhydrology.scalerzoo.featurescaler import ALLOWED_TYPES, FeatureScaler


class MinMaxScaler(FeatureScaler):
    """Bounded scaling for a single feature.
    
    The default bounds are [0, 1], but this can be adjusted ith the 'min_bound' argument.
    The upper bound is always 1. 
    
    Parameters
    ----------
    feature : str
        Name of the feature that this scaler applies to.
    run_path : Path
        Path to the model run for saving and loading scaler.
    min_bound : float
        Lower bound for the scaled range.
    calculate : bool
        Force the scaler to (re)calculate parameters instead of loading a precalculated scaler.
    load : bool
        Force the scaler to load precalculated parameters and throw an error if none exist.
    da : xr.DataArray
        An optional xarray data array for calculating scaler parameters immediately. Alternatively, the calculate
        method can be applied after instantiation.
    """

    def __init__(
        self,
        feature: str,
        run_path: Path,
        min_bound: float = 0,
        calculate: bool = False,
        load: bool = False,
        da: xr.DataArray | None = None,
    ):
        super(MinMaxScaler, self).__init__(
            feature=feature,
            run_path=run_path,
            calculate=calculate,
            load=load,
            da=da,
        )
        self.min_bound = min_bound
        self.scale = 1 + min_bound
        
    def calculate(
        self,
        da: xr.DataArray,
    ):
        """Calculate scaling parameters."""
        self.parameters = {
            'min': da.min(skipna=True).values.item(),
            'max': da.max(skipna=True).values.item(),
        }
        self.parameters['range'] = self.parameters['max'] - self.parameters['min']
        self._calculate_mean_and_std(da)
        self.save()

    def scale(
        self,
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Scale the feature in a single xarray data array."""
        self._check_set()
        scaled_data = (data - self.parameters['min']) / self.parameters['range']
        return scaled_data * self.scale + self.min_bound
            
    def unscale(
        self,
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Unscale the feature in a single tensor."""
        self._check_set()
        shifted_data = (data - self.min_bound) / self.scale
        return shifted_data * self.parameters['range'] + self.parameters['min']
