import os
import pandas as pd
from pathlib import Path
import torch
import xarray as xr

from neuralhydrology.utils.config import Config

BANNED_FILENAME_CHARACTERS = ['/', '(', ')']
REPLACEMENT_FILENAME_CHARACTER = '_'
SCALER_SUBPATH = 'scaler'
SCALER_INDEX_COLUMN_NAME = 'parameter'
SAMPLER_PREFIX = 'sampler'


class FeatureScaler():
    """Scaler for a single feature.
    
    Use subclasses of this class for specific scaling functions.

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
        self.feature = feature
        self._file_name(run_path=run_path, feature=feature)
        
        self.parameters = None
        self.mean = None
        self.std = None
        if not force_calculate:
            self.load()
        if self.parameters is None and da is not None:
            self.calculate(da)
        
    def _check_set(self):
        if self.parameters is None:
            raise AttributeError(f'Scaler parameters are not set for feature {self.feature}.')

    def _file_name(
        self,
        run_path: str,
        feature: str
    ) -> str:
        """Construct the file path and file name for saving parameters."""
        scaler_path = run_path / SCALER_SUBPATH
        os.makedirs(scaler_path, exist_ok=True)

        feature_str = feature
        for character in BANNED_FILENAME_CHARACTERS:
            feature_str = feature_str.replace(character, REPLACEMENT_FILENAME_CHARACTER)
        
        self.scaler_file = scaler_path / f'{self.__class__.__name__}_{feature_str}.csv'

    def load(self):
        """Loads precalculated scaling parameters."""
        if os.path.exists(self.scaler_file):
            with open(self.scaler_file, 'rt') as f:
                df = pd.read_csv(f, index_col=SCALER_INDEX_COLUMN_NAME)
            self.parameters = {
                name: row[self.feature]
                for name, row in df.iterrows()
                if not name.startswith(SAMPLER_PREFIX)
            }
            self.mean = df.loc[f'{SAMPLER_PREFIX}_mean']
            self.std = df.loc[f'{SAMPLER_PREFIX}_std']            

    def save(self):
        parameters = self.parameters.copy()
        parameters[f'{SAMPLER_PREFIX}_mean'] = self.mean
        parameters[f'{SAMPLER_PREFIX}_std'] = self.std
        scaler_df = pd.Series(
            data=parameters.values(),
            index=parameters.keys(),
            name=self.feature
        )
        scaler_df.index.name = 'parameter'
        with open(self.scaler_file, 'wt') as f:
            scaler_df.to_csv(f)
                
    def calculate(
        self,
        da: xr.DataArray,
    ):
        """Calculate scaling parameters."""
        raise NotImplementedError

    def _calculate_mean_and_std(
        self,
        da: xr.DataArray
    ):
        """Calculate mean and standard deviation.
        
        These statistics are needed for noise sampling, so they are always calculated.
        """
        self.mean = da.mean(skipna=True).values.item()
        self.std = da.std(skipna=True).values.item()
        
    def scale(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Scale the feature in a single xarray data array."""
        self._check_set()
        raise NotImplementedError
            
    def unscale(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Unscale the feature in a single tensor."""
        self._check_set()
        raise NotImplementedError
