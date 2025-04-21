import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Union
import xarray as xr

from neuralhydrology.scalerzoo.featurescaler import ALLOWED_TYPES as FEATURE_SCALER_ALLOWED_TYPES
from neuralhydrology.scalerzoo.featurescaler import FeatureScaler
from neuralhydrology.scalerzoo.normalization import NormalizationScaler
from neuralhydrology.scalerzoo.minimax import MinMaxScaler
from neuralhydrology.scalerzoo.identity import IdentityScaler
from neuralhydrology.utils.config import Config

DEFAULT_SCALER_TYPE = 'normalization'

TYPES_OF_FEATURES = [
    'target_variables',
    'hindcast_inputs',
    'forecast_inputs',
    'dynamic_inputs',
    'static_attributes',
    'evolving_attributes',
    'hydroatlas_attributes',
]

ALLOWED_TYPES_FOR_CALCULATING = Union[
    xr.Dataset,
    xr.DataArray,
    pd.DataFrame,
    pd.Series,
    List[
        Union[
            xr.Dataset,
            xr.DataArray,
            pd.DataFrame,
            pd.Series,
        ]
    ]
]

ALLOWED_TYPES_FOR_SCALING = Union[
    xr.DataArray,
    pd.Series,
    xr.Dataset,
    pd.DataFrame,
    Dict[str, Union[torch.Tensor, np.ndarray]],
]


def _get_feature_scaler(
    cfg: Config,
    scaler_type: str,
    feature: str,
    calculate: bool,
    load: bool,
    da: xr.DataArray | None = None,
) -> FeatureScaler:
    """Instantiates a FeatureScaler for a single feature."""
    
    feature_scaler_args = {
        'feature': feature,
        'run_path': cfg.train_dir,
        'calculate': calculate,
        'load': load,
        'da': da
    }

    if scaler_type.lower() == 'normalization':
        return NormalizationScaler(**feature_scaler_args)
    elif scaler_type.lower() == 'normalization_median':
        return NormalizationScaler(**feature_scaler_args.update({'center': 'median'}))
    elif scaler_type.lower() == 'normalization_min':
        return NormalizationScaler(**feature_scaler_args.update({'center': 'min'}))
    elif scaler_type.lower() == 'minmax':
        return MinMaxScaler(**feature_scaler_args)
    elif scaler_type.lower() == 'minmax_negative_one':
        return MinMaxScaler(**feature_scaler_args.update({'min_bound': -1}))
    elif scaler_type.lower() == 'identity':
        return IdentityScaler(**feature_scaler_args)
    else:
        raise NotImplementedError(f'Asking to calculate scaling parameters for a feature that is not in the initialized scaler: {feature}.')


class Scaler():
    """Scaler for a full dataset that contains multiple features.
    
    Each feature has its own scaling function.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.
    features : List[str]
        List of all features that should be in this scaler. If not provided, then the list is generated from the
        run config.
    force_calculate : bool
        Optionally force the scaler to not load a precalculated scaler on initialization, even if one exists.
    force_load : bool
        Optionally force the scaler to load and not calculate.
    """
    
    def __init__(
        self,
        cfg: Config,
        features: List[str] | None = None,
        calculate: bool = False,
        load: bool = False,
    ):
        if load and calculate:
            raise ValueError('Cannot both load and calculate the scaler.')
        if not load and not calculate:
            raise ValueError('Must either load or calculate the scaler.')
        self._load = load

        #TODO(gsnearing) :: Needs to be able to handle duplicate_features.
        self.features = []
        if features is None:
            for feature_type in TYPES_OF_FEATURES:
                if getattr(cfg, feature_type) is not None:
                    self.features += getattr(cfg, feature_type)
        else:
            self.features = features
        self.features = list(set(self.features))

        self.feature_scalers = {}
        for feature in self.features:
            if feature in cfg.custom_normalization:
                scaler_type = cfg.custom_normalization[feature]
            else:
                scaler_type = DEFAULT_SCALER_TYPE
            self.feature_scalers[feature] = _get_feature_scaler(
                cfg=cfg,
                scaler_type=scaler_type,
                feature=feature,
                calculate=calculate,
                load=load,
            )
        
        self.target_means = {} 
        self.target_stds = {}
        for feature, scaler in self.feature_scalers.items():
            if feature not in cfg.target_variables:
                continue
            self.target_means[feature] = scaler.mean
            self.target_stds[feature] = scaler.std
    
    def calculate(
        self,
        data: ALLOWED_TYPES_FOR_CALCULATING
    ):
        if self._load:
            raise ValueError('Cannot calculate parameters for a scaler that was loaded.')
            
        if not isinstance(data, list):
            data = [data]

        das = {}
        for data_object in data:
            if isinstance(data_object, xr.Dataset):
                for feature in data_object.data_vars:
                    das[feature] = data_object[feature]
            elif isinstance(data_object, pd.DataFrame):
                for feature in data_object:
                    das[feature] = data_object[feature].to_xarray()
            elif isinstance(data_object, xr.DataArray):
                feature = data_object.name
                das[feature] = data_object
            elif isinstance(data_object, pd.Series):
                feature = data_object.name
                das[feature] = data_object.to_xarray()

        for feature, da in das.items():
            if feature not in self.features:
                raise ValueError(f'Asking to calculate scaling parameters for a feature that is not in the initialized scaler: {feature}.')
            self.feature_scalers[feature].calculate(da)

    def _scale_or_unscale_feature(
        self,
        feature: str,
        data: FEATURE_SCALER_ALLOWED_TYPES,
        unscale: bool,
    ) -> FEATURE_SCALER_ALLOWED_TYPES:
        if feature in self.features:
            if not unscale:
                return self.feature_scalers[feature].scale(data)
            else:
                return self.feature_scalers[feature].unscale(data)
        else:
            return data
                
    def _scale_dataframe(
        self,
        data: pd.DataFrame,
        unscale: bool,
    ) -> pd.DataFrame:
        scaled_data = {}
        for feature in data:
            scaled_data[feature] = self._scale_or_unscale_feature(
                feature=feature,
                data=data[feature],
                unscale=unscale
            )
        return pd.concat(scaled_data, axis=1)

    def _scale_dataset(
        self,
        data: xr.Dataset,
        unscale: bool,
    ) -> xr.Dataset:
        scaled_data = {}
        for feature in data:
            scaled_data[feature] = self._scale_or_unscale_feature(
                feature=feature,
                data=data[feature],
                unscale=unscale
            )
        return xr.merge(scaled_data.values())

    def _scale_dataarray_or_series(
        self,
        data: pd.Series | xr.DataArray,
        unscale: bool,
    ) -> pd.Series | xr.DataArray:
        return self._scale_or_unscale_feature(
            feature=data.name,
            data=data,
            unscale=unscale
        )

    def _scale_array_dict(
        self,
        data: Dict[str, np.ndarray | torch.Tensor],
        unscale: bool,
    ) -> Dict[str, np.ndarray | torch.Tensor]:
        scaled_data = {}
        for feature, array in data.items():
            scaled_data[feature] = self._scale_or_unscale_feature(
                feature=feature,
                data=array,
                unscale=unscale
            )
        return scaled_data
 
                
    def scale(
        self,
        data: ALLOWED_TYPES_FOR_SCALING,
        unscale: bool = False
    ) -> ALLOWED_TYPES_FOR_SCALING:
        """Scale all features in a data set."""
        if isinstance(data, pd.DataFrame):
            return self._scale_dataframe(data, unscale=unscale)
        elif isinstance(data, xr.Dataset):
            return self._scale_dataset(data, unscale=unscale)
        elif isinstance(data, pd.Series) or isinstance(data, xr.DataArray):
            return self._scale_dataarray_or_series(data, unscale=unscale)
        elif isinstance(data, Dict):
            return self._scale_array_dict(data, unscale=unscale)
        else:
            raise ValueError(f'Unrecognized data type: {type(data)}.')
      

    def unscale(
        self,
        data: ALLOWED_TYPES_FOR_SCALING,
    ) -> ALLOWED_TYPES_FOR_SCALING:
        """Un-scale all features in a data set."""
        return self.scale(data, unscale=True)
        
            
       