from pathlib import Path
import torch
import xarray as xr

from neuralhydrology.scalerzoo.featurescaler import FeatureScaler


class IdentityScaler(FeatureScaler):
    """Identity scaling for a single feature.
    
    Parameters
    ----------
    feature : str
        Name of the feature that this scaler applies to.
    run_path : Path
        Path to the model run for saving and loading scaler.
    force_calculate : bool
        Force the scaler to recalculate parameters instead of loading a precalculated scaler, even if one exits.
    da : xr.DataArray
        An optional xarray data array for calculating scaler parameters immediately. Alternatively, the calculate
        method can be applied after instantiation.
    """

    def __init__(
        self,
        feature: str,
        run_path: Path,
        force_calculate: bool = False,
        da: xr.DataArray | None = None,
    ):
        super(IdentityScaler, self).__init__(
            feature=feature,
            run_path=run_path,
            force_calculate=force_calculate,
            da=da,
        )
        
    def calculate(
        self,
        da: xr.DataArray,
    ):
        """Calculate scaling parameters."""
        self.parameters = {'bias': 0, 'scale': 1}
        self._calculate_mean_and_std(da)
        self.save()

    def scale(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Scale the feature in a single tensor."""
        self._check_set()
        return data
            
    def unscale(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Unscale the feature in a single tensor."""
        self._check_set()
        return data
