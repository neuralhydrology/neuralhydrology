import torch
from typing import Dict, List, Union
import xarray as xr

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

ALLOWED_TYPES_FOR_CALCULATING = [
    xr.Dataset,
    xr.DataArray,
    pd.DataFrame,
    pd.Series,
]

def _get_feature_scaler(
    cfg: Config,
    scaler_type: str,
    feature: str,
    force_calculate: bool = False,
    da: xr.DataArray | None = None,
) -> FeatureScaler:
    """Instantiates a FeatureScaler for a single feature."""
    
    feature_scaler_args = {
        'feature': feature,
        'run_path': cfg.train_dir,
        'force_calculate': force_calculate,
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
        raise NotImplementedError(f"{cfg.model} not implemented or not linked in `get_scaler()`")


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
    """
    
    def __init__(
        self,
        cfg: Config,
        features: List[str] | None = None,
        force_calculate: bool = False,
    ):
        self.force_calculate = force_calculate
        
        #TODO(gsnearing) :: Needs to be able to handle duplicate_features.
        if features is None:
            for feature_type in TYPES_OF_FEATURES:
                if cfg.get(feature_types) is not None:
                    self.features += cfg.get(feature_types)
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
                force_calculate=force_calculate
            )
        
        self.target_means = {} 
        self.target_stds = {}
        if feature, scaler in self.feature_scalers.items():
            if feature not in cfg.target_variables:
                continue
            self.target_means[feature] = scaler[feature].mean
            self.target_stds[feature] = scaler[feature].std
    
    def calculate(
        self,
        dataset: Union[*ALLOWED_TYPES_FOR_CALCULATING, List[Union[*ALLOWED_TYPES_FOR_CALCULATING]]]
    ):
        if not isinstance(dataset, list):
            dataset = [dataset]

        def _calc(
            feature: str,
            scaler: FeatureScaler,
            da: xr.DataArray
        ):
            if feature not in self.features:
                raise ValueError(f'Asking to scale a feature that is not in the initialized scaler: {feature}.')
            if self.force_calculate or not self.feature_scalers[feature].parameters:
                self.feature_scalers[feature].calculate(da)
            
        for data_object in datasets:
            if isinstance(data_object, xr.Dataset):
                for feature in ds.data_vars:
                    _calc(
                        feature=feature,
                        scaler=self.feature_scalers[feature],
                        da=data_object[feature]
                    )
            elif isinstance(data_object, xr.DataArray):
                _calc(
                    feature=data_object.name,
                    scaler=self.feature_scalers[feature],
                    da=data_object
                )
            elif isinstance(data_object, pd.DataFrame):
                for feature in ds:
                    _calc(
                        feature=feature,
                        scaler=self.feature_scalers[feature],
                        da=data_object[feature].to_xarray()
                    )
            elif isinstance(data_object, pd.Series):
                _calc(
                    feature=data_object.name,
                    scaler=self.feature_scalers[feature],
                    da=data_object.to_xarray()
                )

    def scale(
        self,
        feature_arrays: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        scaled_feature_arrays = {}
        for feature in self.features:
            if feature not in feature_arrays:
                continue
            scaled_feature_arrays[feature] = self.feature_scalers[feature].scale(feature_arrays[feature])
        return scaled_feature_arrays
    
    def unscale(
        self,
        feature_arrays: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        scaled_feature_arrays = {}
        for feature in self.features:
            if feature not in feature_arrays:
                continue
            scaled_feature_arrays[feature] = self.feature_scalers[feature].unscale(feature_arrays[feature])
        return scaled_feature_arrays
        
            
       