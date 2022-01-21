from typing import Collection, Dict, List, Set

import pandas as pd

from glyc_processing.data_formats.common.validation import valid_uniprot, valid_peptide, valid_peptide_range, valid_single_site, \
    valid_unclear_site_range
from glyc_processing.misc import cast_to_int_nan_none, split_uniprot_id
from glyc_processing import cf


def check_uniprot_idmapping(uniprot_ids: Collection[str], mappings_dict: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    Checks if any uniprot IDs have been merged/demerged/deleted in mapping
    :param uniprot_ids: Collection of Uniprot IDs.
    :param mappings_dict: Uniprot ID mapping dict from get_uniprot_id_mappings().
    :return: Sets of new- and non-mappings
    """
    uniprot_protein_ids = {split_uniprot_id(uniprot_id)[0] for uniprot_id in uniprot_ids}
    new_mapping_ids = set()
    non_mapping_ids = set()
    for protein_id in uniprot_protein_ids:
        if protein_id in mappings_dict:
            if mappings_dict[protein_id] != [protein_id]:
                new_mapping_ids.add(protein_id)
        else:
            non_mapping_ids.add(protein_id)

    return {'new_mapping_ids': new_mapping_ids, 'non_mapping_ids': non_mapping_ids}


def consistent_entry_peptide(row: pd.Series, isoform_seqs) -> bool:
    """
    Checks that peptide sequence is found at right position in protein sequence for row
    :param row: row from DataFrame (w. 'uniprot', 'peptide', 'peptide_start', 'peptide_end' columns).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts())
    :return:
        True if:
            peptide sequence found at expected position in protein sequence
            or these are not valid: 'uniprot' or 'peptide' or 'peptide_start' or 'peptide_end'
        False if:
            none of the above
            or 'uniprot' is not in isoform_seqs
    """
    if not valid_uniprot(row) or not valid_peptide(row, False) or not valid_peptide_range(row, False):
        return True

    if row['uniprot'] not in isoform_seqs:
        return False

    seq = isoform_seqs[row['uniprot']]
    idx_peptide_start = cast_to_int_nan_none(row['peptide_start']) - 1
    idx_peptide_end = cast_to_int_nan_none(row['peptide_end']) - 1

    return seq[idx_peptide_start:idx_peptide_end + 1] == row['peptide']


def consistent_entry_sites(row: pd.Series, isoform_seqs) -> bool:
    """
    Checks that sites are the allowed amino acids in protein sequence for row
    :param row: row from DataFrame (w. 'uniprot', 'single_site', 'unclear_site_start', 'unclear_site_end' columns).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts())
    :return:
        True if:
            (
                right amino acids are found at single_site positions in protein sequence
                or these are not valid: 'single_site'
            ) and (
                right amino acids are found at unclear_site_start and 'unclear_site_end' positions in protein sequence
                or these are not valid: 'unclear_site_start' or 'unclear_site_end'
            ) or these are not valid: 'uniprot'
        False if:
            none of the above
            or 'uniprot' is not in isoform_seqs
    """
    if not valid_uniprot(row):
        return True

    if row['uniprot'] not in isoform_seqs:
        return False

    seq = isoform_seqs[row['uniprot']]
    seq_length = len(seq)

    if not valid_single_site(row, False):
        consistent_site = True
    else:
        idx_site = cast_to_int_nan_none(row['single_site']) - 1
        consistent_site = (idx_site < seq_length) and (seq[idx_site] in cf.ALLOWED_AA)

    if not valid_unclear_site_range(row, False):
        consistent_unclear_range_sites = True
    else:
        idx_unclear_site_start = cast_to_int_nan_none(row['unclear_site_start']) - 1
        idx_unclear_site_end = cast_to_int_nan_none(row['unclear_site_end']) - 1
        consistent_unclear_range_sites = (
                (idx_unclear_site_start < seq_length) and (seq[idx_unclear_site_start] in cf.ALLOWED_AA)
                and (idx_unclear_site_end < seq_length) and (seq[idx_unclear_site_end] in cf.ALLOWED_AA))

    return consistent_site and consistent_unclear_range_sites
