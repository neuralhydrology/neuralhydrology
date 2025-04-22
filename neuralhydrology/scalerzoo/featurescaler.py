import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from typing import Union
import xarray as xr

from neuralhydrology.datautils.utils import load_scaler as old_load_scaler

BANNED_FILENAME_CHARACTERS = ['/', '(', ')']
REPLACEMENT_FILENAME_CHARACTER = '_'
SCALER_SUBPATH = 'scaler'
SCALER_INDEX_COLUMN_NAME = 'parameter'
SAMPLER_PREFIX = 'sampler'

ALLOWED_TYPES = Union[
    xr.DataArray,
    pd.Series,
    torch.Tensor,
    np.ndarray
]

OLD_SCALER_SUBPATH = 'train_data'
OLD_SCALER_COMPONENTS = {
    'center': [
        'attribute_means',
        'xarray_means',
        'xarray_feature_center',
    ],
    'scale': [
        'xarray_feature_scale',
        'attribute_stds',
        'xarray_stds',
    ]
} 


class FeatureScaler():
    """Scaler for a single feature.
    
    Use subclasses of this class for specific scaling functions.

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
        calculate: bool = False,
        load: bool = False,
        da: xr.DataArray | None = None,
    ):
        if load and calculate:
            raise ValueError('Cannot both load and calculate the scaler.')
        if not load and not calculate:
            raise ValueError('Must either load or calculate the scaler.')
        
        self.feature = feature
        self._file_name(run_path=run_path, feature=feature)
        # `scaler_path` is only used for backward compatibility with old scaler files.
        self._scaler_path = run_path

        self.parameters = None
        self.mean = None
        self.std = None
        if calculate and da is not None:
            self.calculate(da)
        elif load:
            self.load()
        
    def _check_set(self):
        if self.parameters is None:
            raise AttributeError(f'Scaler parameters are not set for feature {self.feature}.')

    def _file_name(
        self,
        run_path: Path,
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

        # Try loading the old type of scaler file.
        else:
            old_scaler = None
            try:
                old_scaler = old_load_scaler(self._scaler_path)
            except FileNotFoundError:
                pass
            if old_scaler is not None:
                self.parameters = {}
                for parameter in OLD_SCALER_COMPONENTS:
                    for data_key in OLD_SCALER_COMPONENTS[parameter]:
                        if data_key in old_scaler:
                            data = old_scaler[data_key]
                            if isinstance(data, pd.Series):
                                if self.feature in data.index:
                                    self.parameters[parameter] = data.loc[self.feature]
                            elif isinstance(data, xr.Dataset):
                                if self.feature in data.data_vars:
                                    self.parameters[parameter] = data[self.feature].values.item()
                # This part is not exactly backward compatable in cases where center and scale from
                # the old file were not mean and standard deviation.
                self.mean = self.parameters['center']
                self.std = self.parameters['scale']
        
        # If neither the new nor old scaler files exist, throw an error.
        if self.parameters is None:
            raise ValueError(f'Scaler file not found: {self.scaler_file}.')
            
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
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Scale the feature in a single xarray data array."""
        self._check_set()
        raise NotImplementedError
            
    def unscale(
        self,
        data: ALLOWED_TYPES,
    ) -> ALLOWED_TYPES:
        """Unscale the feature in a single tensor."""
        self._check_set()
        raise NotImplementedError
