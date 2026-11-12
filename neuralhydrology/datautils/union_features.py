from typing import Dict

import numpy as np
import xarray as xr
        

def _expand_lead_times(da: xr.DataArray, lead_time: int) -> xr.DataArray:
    if 'lead_time' in da.dims:
        raise ValueError('Trying to expand a dataarray that already has a lead time.')
    # TODO (future) :: This assumes daily data.
    lead_time_int = int(lead_time / np.timedelta64(1, 'D'))
    lt_das, lead_times = [], []
    # TODO (future) :: This assumes a minimum lead time of 1.
    for lt in range(1, lead_time_int+1):
        # TODO (future) :: This assumes daily data.
        lead_times.append(np.timedelta64(lt, 'D'))
        lt_das.append(da.shift(date=-lt))
    lt_da = xr.concat(lt_das, dim='lead_time')
    # TODO (future) :: Like many forecast models (unlike ForecastDataset), this assumes that
    # all lead times are present.
    return lt_da.assign_coords(lead_time=lead_times)

    
def _union_features_with_same_dimensions(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask one feature with another when both are the same size."""
    return feature_da.combine_first(mask_feature_da)


def _union_lead_time_feature_with_non_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask a lead-time feature with a non-lead-time feature."""
    target_dates = feature_da['date']
    all_needed_dates = np.unique(target_dates.values)
    lead_time = feature_da.lead_time.max().values
    lt_mask_da = _expand_lead_times(mask_feature_da, lead_time)
    reindexed_mask_feature = lt_mask_da.reindex(date=all_needed_dates, fill_value=np.nan)
    mask_values = reindexed_mask_feature.sel(date=target_dates, basin=feature_da['basin'])
    return _union_features_with_same_dimensions(feature_da, mask_values)


def _union_non_lead_time_feature_with_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask a non-lead-time feature with a lead-time feature."""
    min_lead_time = mask_feature_da['lead_time'].isel(lead_time=0).values
    target_issue_dates = [d - min_lead_time for d in feature_da['date'].values]
    all_needed_issue_dates = np.unique(target_issue_dates)
    reindexed_mask_feature = mask_feature_da.reindex(date=all_needed_issue_dates, fill_value=np.nan)
    min_lead_time_mask_feature = reindexed_mask_feature.sel(lead_time=min_lead_time)
    mask_values = min_lead_time_mask_feature.sel(
        date=target_issue_dates,
        basin=feature_da['basin'],
    )
    return _union_features_with_same_dimensions(feature_da, mask_values).drop('lead_time').sel(date=feature_da['date'])


def union_features(
    dataset: xr.Dataset,
    union_feature_mapping: Dict[str, str]
) -> xr.Dataset:
    """Replace missing values in various features with values from another feature.
    
    Works with features that do and do not have a `lead_time` dimension.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset of features to be masked *and* features used for masking.
    union_feature_mapping : Dict[str, str]
        A 1-to-1 mapping from a feature that will be masked by another feature. 
    
    Returns
    -------
    Dataset of all features from the original dataset where the features that are keys in `union_feature_mapping'
        are masked by the features in their respective dictionary values.
    
    Raises
    ------
    ValueError if a masking feature is not found.
    """
    masked_das = []

    # Keep track of features that have been explicitly processed (i.e., were the 'feature' key)
    processed_feature_names = set()

    for feature, mask_feature in union_feature_mapping.items():
        
        # Don't bother masking a feature with itself.
        if feature == mask_feature:
            masked_das.append(dataset[feature])
        # We don't care if a feature is not present.
        if feature not in dataset.data_vars:
            continue
        # We DO care if a masking feature is not present because the user might not notice
        # and believe it was masked.
        if mask_feature not in dataset.data_vars:
            raise ValueError(f'Masking feature {mask_feature} not in dataset.')
        
        # Mask the feature depending on their dimensions.
        feature_da = dataset[feature]
        mask_feature_da = dataset[mask_feature]
        
        if 'lead_time' in feature_da.dims and 'lead_time' not in mask_feature_da.dims:
            masked_das.append(
                _union_lead_time_feature_with_non_lead_time_feature(
                    feature_da=feature_da,
                    mask_feature_da=mask_feature_da
                )
            )
        
        elif 'lead_time' not in feature_da.dims and 'lead_time' in mask_feature_da.dims:
            masked_das.append(
                _union_non_lead_time_feature_with_lead_time_feature(
                    feature_da=feature_da,
                    mask_feature_da=mask_feature_da
                )
            )
    
        else:
            masked_das.append(
                _union_features_with_same_dimensions(
                    feature_da=feature_da,
                    mask_feature_da=mask_feature_da
                )
            )
    
    # Collect all features from the original dataset that were NOT targets of a union operation.
    # These should be included in the final output as they were.
    for original_feature_name in dataset.data_vars:
        if original_feature_name not in processed_feature_names:
            masked_das.append(dataset[original_feature_name])
    
    # Concatenate everything back into a single dataset.
    return xr.merge(masked_das)
