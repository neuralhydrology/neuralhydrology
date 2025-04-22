from typing import Optional
from pathlib import Path
import xarray as xr

from neuralhydrology.scalerzoo.featurescaler import ALLOWED_TYPES, FeatureScaler


class NormalizationScaler(FeatureScaler):
    """Normalization-based scaling for a single feature.
    
    This scaler removes (subtracts) a center value and divides by a scale value.
    The default center is the data mean and the default scale is the standard deviaion.
    
    Parameters
    ----------
    feature : str
        Name of the feature that this scaler applies to.
    run_path : Path
        Path to the model run for saving and loading scaler.
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
        center: Optional[str] = 'mean',
        scale: Optional[str] = 'std',
        calculate: bool = False,
        load: bool = False,
        da: Optional[xr.DataArray] = None,
    ):
        self.centering = center
        self.scaling = scale
        super(NormalizationScaler, self).__init__(
            feature=feature,
            run_path=run_path,
            calculate=calculate,
            load=load,
            da=da,
        )
        
    def calculate(
        self,
        da: xr.DataArray,
    ):
        """Calculate scaling parameters."""
        self._calculate_mean_and_std(da)

        if self.centering == 'mean':
            center = self.mean
        elif self.centering == 'median':
            center = da.median(skipna=True).values.item()
        elif self.centering == 'min':
            center = da.min(skipna=True).values.item()
        elif self.centering.lower() == 'none':
            center = 0
        elif self.centering is None:
            center = 0
        else:
            raise ValueError(f'Normalization center type {self.center} not recognized.')

        if self.scaling == 'std':
            scale = self.std
        elif self.scaling == 'minmax':
            min_value = da.min(skipna=True).values.item()
            max_value = da.max(skipna=True).values.item()
            scale = max_value - min_value
        elif self.scaling.lower() == 'none':
            scale = 1.
        elif self.scaling is None:
            scale = 1.
        else:
            raise ValueError(f'Normalization center type {self.center} not recognized.')

        self.parameters = {
            'center': center,
            'scale': scale,
        }       
        self.save()

    def scale(
        self,
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Scale the feature in a single xarray data array."""
        self._check_set()
        return (data - self.parameters['center']) / self.parameters['scale']
            
    def unscale(
        self,
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Unscale the feature in a single tensor."""
        self._check_set()
        return data * self.parameters['scale'] + self.parameters['center']
