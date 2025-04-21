from pathlib import Path
import xarray as xr

from neuralhydrology.scalerzoo.featurescaler import ALLOWED_TYPES, FeatureScaler


class NormalizationScaler(FeatureScaler):
    """Normalization-based scaling for a single feature.
    
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
        center: str = 'mean',
        calculate: bool = False,
        load: bool = False,
        da: xr.DataArray | None = None,
    ):
        self.center = center
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

        if self.center == 'mean':
            center = self.mean
        elif self.center == 'median':
            center = da.median(skipna=True).values.item()
        elif self.center == 'min':
            center = da.min(skipna=True).values.item()
        else:
            raise ValueError(f'Normalization center type {self.center} not recognized.')

        self.parameters = {
            'center': center,
            'scale': self.std,
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
