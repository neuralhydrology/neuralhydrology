import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from neuralhydrology.datautils.utils import load_scaler as old_load_scaler

SCALER_FILE_NAME = 'scaler.nc'


def _get_center(feature_da: xr.DataArray, centering_type: str) -> float:
    """Canonical selector for currently-supported center types."""
    if (centering_type is None) or (centering_type.lower() == 'none'):
        return np.float32(0.0)
    elif centering_type.lower() == 'median':
        return feature_da.median(skipna=True)
    elif centering_type.lower() == 'min':
        return feature_da.min(skipna=True)
    elif centering_type.lower() == 'mean':
        return feature_da.mean(skipna=True)
    else:
        raise ValueError(f'Unknown centering method {centering_type}')

        
def _get_scale(feature_da: xr.DataArray, scaling_type: str) -> float:
    """Canonical selector for currently-supported scale types."""
    if (scaling_type is None) or (scaling_type.lower() == 'none'):
        return np.float32(1.0)
    elif scaling_type.lower() == 'minmax':
        return feature_da.max(skipna=True) - feature_da.min(skipna=True)
    elif scaling_type.lower() == 'std':
        return feature_da.std(skipna=True)
    else:
        raise ValueError(f'Unknown scaling method {scaling_type}')


class Scaler():
    """Scaler for a dataset that contains multiple features.
    
    Parameters
    ----------
    scaler_dir : pathlib.Path
        Directory for loading a pre-calculated scaler or saving this scaler if it is calculated.
    calculate_scaler : bool
        Flag to indicate if the scaler should be computed (the alternative is to load an existing scaler file).
    custom_normalization : Dict[str, Dict[str, float]]
        Feature-specific scaling instructions as a mapping from feature name to centering and/or scaling type.
        See docs for a list of accepted types and their meaning.
    dataset : Optional[xr.Dataset]
        Dataset to use for calculating a new scaler. Cannot be supplied if `calculate_scaler` is False.
        
    Raises
    -------
    ValueError for incompatible loading/calculating instructions.
    """
    
    def __init__(
        self,
        scaler_dir: Path,
        calculate_scaler,
        custom_normalization: Dict[str, Dict[str, float]] = {},
        dataset: Optional[xr.Dataset] = None,
    ):
        # Consistency check.
        if not calculate_scaler and dataset is not None:
            raise ValueError('Do not pass a dataset if you are loading a pre-calculated scaler.')

        # Load or calculate scaling parameters.
        self.scaler = None
        self.scaler_dir = scaler_dir
        if not calculate_scaler:
            self.load()
        else:
            self._custom_normalization = custom_normalization
            if dataset is not None:
                self.calculate(dataset)
   
    def load(self):
        scaler_file = self.scaler_dir / SCALER_FILE_NAME
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.scaler = xr.load_dataset(f)
            self._check_zero_scale()
            return
        else:
            scaler = old_load_scaler(self.scaler_dir)
            # Add target scalers for tester. Necessary for fine tuning.
            obs_scaler = scaler.copy().rename({feature: feature + '_obs' for feature in scaler.data_vars})
            sim_scaler = scaler.copy().rename({feature: feature + '_sim' for feature in scaler.data_vars})
            self.scaler = xr.merge([scaler, obs_scaler, sim_scaler])
            self._check_zero_scale()
            return

    def save(self):
        if self.scaler is None:
            raise ValueError('You are trying to save a scaler that has not been computed.')
        os.makedirs(self.scaler_dir, exist_ok=True)
        scaler_file = self.scaler_dir / SCALER_FILE_NAME
        with open(scaler_file, 'wb') as f:
            self.scaler.to_netcdf(f)
        
    def calculate(
        self,
        dataset: xr.Dataset,
    ):
        
        # Option for custom scaling for each feature.
        centering_types = {feature: 'mean' for feature in dataset.data_vars}
        scaling_types = {feature: 'std' for feature in dataset.data_vars}
        for feature in self._custom_normalization:
            if 'centering' in self._custom_normalization[feature]:
                centering_types[feature] = self._custom_normalization[feature]['centering']
            if 'scaling' in self._custom_normalization[feature]:
                scaling_types[feature] = self._custom_normalization[feature]['scaling']

        # Target noise distrbutions (in BaseTrainer) require means and standard deviations.
        # Force these parameters to be avaiable, even if the targets use a different center
        # and scale type. Could restrict these extra calculations to only when necessary 
        # (i.e., targets only, and only when using other scaling types), but this adds
        # complication and these extra mean and std calcs & storage are cheap.
        parameters = {
            feature: (('parameter',), [
                _get_center(dataset[feature], centering_types[feature]),
                _get_scale(dataset[feature], scaling_types[feature]),
                _get_center(dataset[feature], 'mean'),
                _get_scale(dataset[feature], 'std')
            ]) for feature in dataset.data_vars
        }
        
        # Expand the scaler dataset to include 'obs' and 'sim' versions of all variables.
        # As above, this is only necessary for target variables, but a check here adds
        # complexity for little benefit.
        parameters.update({f'{feature}_obs': parameters[feature] for feature in parameters})
        parameters.update({f'{feature}_sim': parameters[feature] for feature in parameters})      
        
        # Put the calculated parameters into an xarray dataset.
        coords = {'parameter': ['center', 'scale', 'mean', 'std']}
        scaler = xr.Dataset(parameters, coords=coords).astype('float32')

        # Handle cases where part of the scaler is already calculated. Simply add new features.
        if self.scaler is not None:
            self.scaler = xr.merge([self.scaler, scaler])
        else:
            self.scaler = scaler

        # Ensure that there are no zero-valued scale parameters, as this will cause NaN's.
        self._check_zero_scale()

    def _check_zero_scale(self):
        """Raises an error if the scale is zero for any feature."""
        zero_scale_features = [
            feature for feature, da in self.scaler.sel(parameter=['scale', 'std']).data_vars.items()
            if (da == 0).any()
        ]
        if any(zero_scale_features):
             raise ValueError(f'Zero scale values found for features: {zero_scale_features}.')
        
    def scale(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Scale a data set with a precaculated_scaler.
        
        $$ unscaled_dataset = (dataset - center) + scale $$

        Applies a linear transformation to the features (data_vars) in an xr.Dataset.
        This transformation is the inverse of the one applied by self.unscale().
        Agnostic to the dimensions and coordinates of the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be scaled.
        
        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are scaled.
        
        Raises
        ------
        ValueError if the dataset contains features that are not in the scaler parameters.
        """
        missing_features = [feature for feature in dataset if feature not in self.scaler.data_vars]
        if any(missing_features):
            raise ValueError(f'Requesting to scale variables that are not part of the scaler: {missing_features}')
        return (dataset - self.scaler.sel(parameter='center')) / self.scaler.sel(parameter='scale')

    def unscale(
        self,
        dataset: xr.Dataset
    ) -> xr.Dataset:
        """Un-scale a data set with a precalculated scaler.
        
        $$ scaled_dataset = dataset * scale + center $$
        
        Applies a linear transformation to the features (data_vars) in an xr.Dataset.
        This transformation is the inverse of the one applied by self.scale().
        Agnostic to the dimensions and coordinates of the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be un-scaled.
        
        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are un-scaled. 
        
        Raises
        ------
        ValueError if the dataset contains features that are not in the scaler parameters.
        """
        missing_features = [feature for feature in dataset if feature not in self.scaler.data_vars]
        if any(missing_features):
            raise ValueError(f'Requesting to unscale variables that are not part of the scaler: {missing_features}')
        return dataset * self.scaler.sel(parameter='scale') + self.scaler.sel(parameter='center')
