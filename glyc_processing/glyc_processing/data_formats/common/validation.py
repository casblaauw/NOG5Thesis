import re

import pandas as pd

from glyc_processing.misc import cast_to_int_nan_none
from glyc_processing import cf


##### Basic fields validation #####

def valid_uniprot(row: pd.Series) -> bool:
    """Check that uniprot ID is valid"""
    return isinstance(row['uniprot'], str) and re.match(cf.UNIPROT_REGEX, row['uniprot']) is not None


def valid_peptide_id(row: pd.Series) -> bool:
    """Check that peptide ID is valid UUID"""
    return isinstance(row['peptide_id'], str) and re.match(cf.UUID_REGEX, row['peptide_id']) is not None


def valid_peptide(row: pd.Series, na_allowed: bool = True) -> bool:
    """Check that peptide sequence is valid"""
    return (
        (isinstance(row['peptide'], str) and re.match(cf.PEPTIDE_REGEX, row['peptide']) is not None)
        or
        (na_allowed and pd.isna(row['peptide']))
    )


def valid_peptide_range(row: pd.Series, na_allowed: bool = True) -> bool:
    """Check that peptide absolute start/end positions are valid integers and a proper range"""
    int_peptide_start = cast_to_int_nan_none(row['peptide_start'], na_allowed)
    if int_peptide_start is None:
        return False

    int_peptide_end = cast_to_int_nan_none(row['peptide_end'], na_allowed)
    if int_peptide_end is None:
        return False

    return (
            (0 < int_peptide_start < int_peptide_end)
            or
            (na_allowed and pd.isna(int_peptide_start) and pd.isna(int_peptide_end))
    )


def valid_single_site(row: pd.Series, na_allowed: bool = True) -> bool:
    """Check that absolute single site position is a valid integer"""
    int_site = cast_to_int_nan_none(row['single_site'], na_allowed)
    if int_site is None:
        return False

    return (
            (0 < int_site)
            or
            (na_allowed and pd.isna(int_site))
    )


def valid_unclear_site_range(row: pd.Series, na_allowed: bool = True) -> bool:
    """Check that absolute unclear site start/end positions are valid integers and a proper range"""
    int_unclear_site_start = cast_to_int_nan_none(row['unclear_site_start'], na_allowed)
    if int_unclear_site_start is None:
        return False

    int_unclear_site_end = cast_to_int_nan_none(row['unclear_site_end'], na_allowed)
    if int_unclear_site_end is None:
        return False

    return (
            (0 < int_unclear_site_start < int_unclear_site_end)
            or
            (na_allowed and pd.isna(int_unclear_site_start) and pd.isna(int_unclear_site_end))
    )


##### Consistent fields validation #####

def consistent_site_or_unclear_range(row: pd.Series) -> bool:
    """Check that row has either single site or unclear site but not both"""
    return (
            (pd.isna(row['single_site']) and not pd.isna(row['unclear_site_start']) and not pd.isna(row['unclear_site_end']))
            or
            (not pd.isna(row['single_site']) and pd.isna(row['unclear_site_start']) and pd.isna(row['unclear_site_end']))
    )


def consistent_negative_data_peptide_info(row: pd.Series) -> bool:
    """Check that rows without site info (negative data) has needed peptide range info"""
    return consistent_site_or_unclear_range(row) or valid_peptide_range(row, False)


def consistent_peptide_length(row: pd.Series, na_allowed: bool = True) -> bool:
    """Check that peptide length is consistent with peptide start/end positions"""
    if not valid_peptide(row, False) or not valid_peptide_range(row, False):
        return True

    int_peptide_start = cast_to_int_nan_none(row['peptide_start'], na_allowed)
    int_peptide_end = cast_to_int_nan_none(row['peptide_end'], na_allowed)

    return (
            (len(row['peptide']) == (int_peptide_end - int_peptide_start + 1))
            or
            (na_allowed and pd.isna(row['peptide']) and pd.isna(row['peptide_start']) and pd.isna(row['peptide_end']))
    )


def consistent_sites_position(row: pd.Series) -> bool:
    """Check that site positions are within peptide start/end range"""
    if not valid_peptide_range(row, False):
        return True

    int_peptide_start = cast_to_int_nan_none(row['peptide_start'])
    int_peptide_end = cast_to_int_nan_none(row['peptide_end'])

    if not valid_single_site(row, False):
        consistent_site = True
    else:
        int_site = cast_to_int_nan_none(row['single_site'])
        consistent_site = (int_peptide_start <= int_site <= int_peptide_end)

    if not valid_unclear_site_range(row, False):
        consistent_unclear_site_range = True
    else:
        int_unclear_site_start = cast_to_int_nan_none(row['unclear_site_start'])
        int_unclear_site_end = cast_to_int_nan_none(row['unclear_site_end'])
        consistent_unclear_site_range = (
                    int_peptide_start <= int_unclear_site_start < int_unclear_site_end <= int_peptide_end)

    return consistent_site and consistent_unclear_site_range


def consistent_sites_aa(row: pd.Series) -> bool:
    """Check that site positions are allowed amino acids in peptide"""
    if (not valid_peptide(row, False) or not valid_peptide_range(row, False) or
        not consistent_peptide_length(row, False) or not consistent_sites_position(row)):
        return True

    int_peptide_start = cast_to_int_nan_none(row['peptide_start'])

    if not valid_single_site(row, False):
        consistent_site = True
    else:
        int_site = cast_to_int_nan_none(row['single_site'])
        relative_site = int_site - int_peptide_start
        consistent_site = row['peptide'][relative_site] in cf.ALLOWED_AA

    if not valid_unclear_site_range(row, False):
        consistent_unclear_site_range = True
    else:
        int_unclear_site_start = cast_to_int_nan_none(row['unclear_site_start'])
        int_unclear_site_end = cast_to_int_nan_none(row['unclear_site_end'])
        relative_unclear_site_start = int_unclear_site_start - int_peptide_start
        relative_unclear_site_end = int_unclear_site_end - int_peptide_start
        consistent_unclear_site_range = (row['peptide'][relative_unclear_site_start] in cf.ALLOWED_AA) and (
                    row['peptide'][relative_unclear_site_end] in cf.ALLOWED_AA)

    return consistent_site and consistent_unclear_site_range


def consistent_id_common_info(id_column_name: str, consistent_columns_df: pd.DataFrame) -> pd.Series:
    """Check that data that should be consistent for a given peptide_id really is consistent"""
    value_combos_per_id = consistent_columns_df.drop_duplicates().groupby(id_column_name, as_index=False).size()
    nonconsistent_ids = set(value_combos_per_id.loc[value_combos_per_id['size'] > 1, id_column_name])
    valid_rows = ~(consistent_columns_df[id_column_name].isin(nonconsistent_ids))
    return valid_rows
