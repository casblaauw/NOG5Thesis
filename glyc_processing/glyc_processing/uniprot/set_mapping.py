from collections import Counter
from typing import Dict, List, Set

import pandas as pd

from glyc_processing.uniprot.find_mapping import find_new_uniprot_id, find_missing_peptide_sequence, \
    find_uniprot_isoforms_containing_peptide, find_correct_peptide_range_and_sites, find_correct_peptide_range


def set_new_uniprot_ids(df: pd.DataFrame, mappings_dict: Dict[str, List[str]]):
    """
    Sets new UniProt ID mappings for dataframe.
    :param df: dataframe (w. 'uniprot' column).
    :param mappings_dict: Uniprot ID mapping dict from get_uniprot_id_mappings().
    """
    df['uniprot'] = df.apply(find_new_uniprot_id, mappings_dict=mappings_dict, axis=1)


def set_missing_peptide_sequences(df: pd.DataFrame, isoform_seqs: Dict[str, str]):
    """
    Set recovered missing peptide sequences for dataframe.
    :param df: dataframe (w. 'uniprot', 'peptide_start', 'peptide_end' columns).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts())
    """
    df['peptide'] = df.apply(find_missing_peptide_sequence, isoform_seqs=isoform_seqs, axis=1)


def set_correct_peptide_ranges(df: pd.DataFrame, isoform_seqs: Dict[str, str]):
    df[['peptide_start', 'peptide_end']] = df \
        .apply(find_correct_peptide_range, isoform_seqs=isoform_seqs, axis=1, result_type='expand') \
        .rename(columns={0: 'peptide_start', 1: 'peptide_end'})


def set_correct_peptide_ranges_and_sites(df: pd.DataFrame, isoform_seqs: Dict[str, str]):
    df[['peptide_start', 'peptide_end', 'single_site', 'unclear_site_start', 'unclear_site_end']] = df \
        .apply(find_correct_peptide_range_and_sites, isoform_seqs=isoform_seqs, axis=1, result_type='expand') \
        .rename(columns={0: 'peptide_start', 1: 'peptide_end', 2: 'single_site', 3: 'unclear_site_start', 4: 'unclear_site_end'})


def set_uniprot_isoforms_containing_peptides(df, entry_isoforms: Dict[str, Set[str]], isoform_seqs: Dict[str, str],
                                             majority_resolution: bool = True):
    """
    Sets correct isoforms containing the peptides for dataframe.
    :param df: dataframe (w. 'uniprot', 'peptide_id', 'peptide', 'peptide_start', 'peptide_end' columns).
    :param entry_isoforms: dictionary of isoforms for each entry (see get_entry_isoforms_dicts()).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts()).
    :param majority_resolution: If true, picks the most commonly matched isoform in case there are multiple matches. If false, picks the lowest numbered isoform.
    """
    unique_peptide_rows = df[['uniprot', 'peptide_id', 'peptide', 'peptide_start', 'peptide_end']].drop_duplicates()
    if (unique_peptide_rows.value_counts(dropna=False) > 1).sum() > 0:
        raise ValueError(
            "Nonconsistent columns that should be consistent for the peptide id were found. Please fix/remove these.")

    new_uniprot_id_lists = unique_peptide_rows.apply(find_uniprot_isoforms_containing_peptide,
                                                     entry_isoforms=entry_isoforms,
                                                     isoform_seqs=isoform_seqs, axis=1)

    if majority_resolution:
        id_counter = Counter(uniprot_id for uniprot_id_list in new_uniprot_id_lists for uniprot_id in uniprot_id_list)
        new_uniprot_ids = new_uniprot_id_lists.apply(
            lambda uniprot_id_list: max(
                [(uniprot_id, id_counter[uniprot_id]) for uniprot_id in uniprot_id_list],
                key=lambda x: x[1]
            )[0]
        )
    else:
        new_uniprot_ids = new_uniprot_id_lists.apply(lambda uniprot_id_list: uniprot_id_list[0])

    new_uniprot_dict = dict(zip(unique_peptide_rows['peptide_id'], new_uniprot_ids))

    df['uniprot'] = df.apply(lambda row: new_uniprot_dict[row['peptide_id']], axis=1)

    # We have to update the peptide ranges & site positions as well if they are included in the dataframe
    if 'single_site' in df.columns and 'unclear_site_start' in df.columns and 'unclear_site_end' in df.columns:
        set_correct_peptide_ranges_and_sites(df, isoform_seqs)
    elif 'single_site' in df.columns or 'unclear_site_start' in df.columns or 'unclear_site_end' in df.columns:
        raise KeyError("Dataframe must have all or none of the following rows: 'single_site', 'unclear_site_start', 'unclear_site_end'")
    else:
        set_correct_peptide_ranges(df, isoform_seqs)
