import re

import pandas as pd

from glyc_processing.misc import cast_to_float_nan_none
from glyc_processing import cf


##### Basic fields validation #####

def valid_site_composition(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['site_composition'], str) and re.match(cf.SITE_COMPOSITION_REGEX,
                                                                   row['site_composition']) is not None)
            or
            (na_allowed and pd.isna(row['site_composition']))
    )


def valid_source(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['source'], str) and re.match(cf.SOURCE_REGEX, row['source']) is not None)
            or
            (na_allowed and pd.isna(row['source']))
    )



def valid_quantification(row: pd.Series, na_allowed: bool = True) -> bool:
    float_quantification = cast_to_float_nan_none(row['quantification'], na_allowed)
    if float_quantification is None:
        return isinstance(row['quantification'], str) and re.match(cf.QUANTIFICATION_REGEX,
                                                                   row['quantification']) is not None

    return (
            (0 < float_quantification)
            or
            (na_allowed and pd.isna(row['quantification']))
    )


def valid_quantification_channels(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['quantification_channels'], str) and re.match(cf.QUANTIFICATION_CHANNELS_REGEX,
                                                                          row['quantification_channels']) is not None)
            or
            (na_allowed and pd.isna(row['quantification_channels']))
    )


def valid_site_ambiguity(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['site_ambiguity'], str) and re.match(cf.SITE_AMBIGUITY_REGEX,
                                                                 row['site_ambiguity']) is not None)
            or
            (na_allowed and pd.isna(row['site_ambiguity']))
    )


def valid_quantification_confidence(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['quantification_confidence'], str) and re.match(cf.QUANTIFICATION_CONFIDENCE_REGEX, row[
                'quantification_confidence']) is not None)
            or
            (na_allowed and pd.isna(row['quantification_confidence']))
    )


def valid_composition(row: pd.Series) -> bool:
    return isinstance(row['composition'], str) and re.match(cf.COMPOSITION_REGEX, row['composition']) is not None


def valid_activation(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['activation'], str) and re.match(cf.ACTIVATION_REGEX, row['activation']) is not None)
            or
            (na_allowed and pd.isna(row['activation']))
    )


def valid_dataset(row: pd.Series) -> bool:
    return isinstance(row['dataset'], str) and re.match(r"\S", row['dataset']) is not None


def valid_origin(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['origin'], str) and re.match(cf.ORIGIN_REGEX, row['origin']) is not None)
            or
            (na_allowed and pd.isna(row['origin']))
    )


def valid_lectin(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['lectin'], str) and re.match(cf.LECTIN_REGEX, row['lectin']) is not None)
            or
            (na_allowed and pd.isna(row['lectin']))
    )


def valid_proteases(row: pd.Series, na_allowed: bool = True) -> bool:
    return (
            (isinstance(row['proteases'], str) and re.match(cf.PROTEASES_REGEX, row['proteases']) is not None)
            or
            (na_allowed and pd.isna(row['proteases']))
    )


##### Consistent fields validation #####

def consistent_origin_lectin_proteases(row: pd.Series) -> bool:
    return (
            (not pd.isna(row['origin']) and not pd.isna(row['lectin']))
            or
            (not pd.isna(row['lectin']) and not pd.isna(row['proteases']))
            or
            (pd.isna(row['origin']) and pd.isna(row['lectin']) and pd.isna(row['proteases']))
    )
