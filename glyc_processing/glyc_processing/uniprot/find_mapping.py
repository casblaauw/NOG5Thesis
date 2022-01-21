import re
import warnings
from typing import Dict, List, Union, Set, Tuple, Optional

import numpy as np
import pandas as pd

from glyc_processing.misc import split_uniprot_id, cast_to_int_nan_none
from glyc_processing import cf


def find_new_uniprot_id(row: pd.Series, mappings_dict: Dict[str, List[str]]) -> Union[str]:
    """
    Tries to map Uniprot ID to new ID. (Used by set_new_uniprot_ids())
    :param row: row from DataFrame (w. uniprot column).
    :param mappings_dict: Uniprot ID mapping dict from get_uniprot_id_mappings().
    :return: New ID if the accession changed (isoform ID removed) or else the old ID (with isoform ID kept).
    """
    uniprot_id, isoform_id = split_uniprot_id(row['uniprot'])
    if uniprot_id in mappings_dict:
        new_uniprot_ids = mappings_dict[uniprot_id]
        if len(new_uniprot_ids) == 1:
            new_uniprot_id = new_uniprot_ids[0]
            if new_uniprot_id == uniprot_id:
                return row['uniprot']
            else:
                return new_uniprot_id
        elif len(new_uniprot_ids) > 1:
            warnings.warn(
                f"UniProt ID {row['uniprot']} has several possible mappings due to entry demerging in UniProt, please resolve this manually")
            return row['uniprot']
    else:
        warnings.warn(
            f"UniProt ID {row['uniprot']} Seems to be incorrect or has been deleted in UniProt, please resolve this manually")
        return row['uniprot']


def find_correct_peptide_range(row: pd.Series, isoform_seqs: Dict[str, str]):
    if pd.isna(row['uniprot']):
        return row['peptide_start'], row['peptide_end']

    # First check if peptide range is already correct
    int_peptide_start = cast_to_int_nan_none(row['peptide_start'], False)
    int_peptide_end = cast_to_int_nan_none(row['peptide_end'], False)
    if int_peptide_start is not None and int_peptide_end is not None:
        idx_peptide_start = int_peptide_start - 1
        idx_peptide_end = int_peptide_end - 1
        if isoform_seqs[row['uniprot']][idx_peptide_start:idx_peptide_end + 1] == row['peptide']:
            return row['peptide_start'], row['peptide_end']

    # If current peptide range is incorrect, try to find the right range
    matches = [match.start() for match in re.finditer(fr"(?=({row['peptide']}))", isoform_seqs[row['uniprot']])]
    if len(matches) == 0:  # If there are no matches, return NA
        return np.nan, np.nan
    elif len(matches) == 1:  # If there is one match, return it
        return matches[0] + 1, matches[0] + len(row['peptide'])
    else: # If there are more than one match, we ignore the row
        return row['peptide_start'], row['peptide_end']


def find_correct_peptide_range_and_sites(row: pd.Series, isoform_seqs: Dict[str, str]):
    old_peptide_start, old_peptide_end = row['peptide_start'], row['peptide_end']
    new_peptide_start, new_peptide_end = find_correct_peptide_range(row, isoform_seqs=isoform_seqs)

    if old_peptide_start != new_peptide_start:
        int_old_peptide_start = cast_to_int_nan_none(old_peptide_start, False)
        int_new_peptide_start = cast_to_int_nan_none(new_peptide_start, False)
        if int_old_peptide_start is not None and int_new_peptide_start is not None:
            peptide_position_change = int_new_peptide_start - int_old_peptide_start

            int_old_single_site = cast_to_int_nan_none(row['single_site'], False)
            if int_old_single_site is not None:
                new_single_site = int_old_single_site + peptide_position_change
            else:
                new_single_site = row['single_site']

            int_old_unclear_site_start = cast_to_int_nan_none(row['unclear_site_start'], False)
            if int_old_unclear_site_start is not None:
                new_unclear_site_start = int_old_unclear_site_start + peptide_position_change
            else:
                new_unclear_site_start = row['unclear_site_start']

            int_old_unclear_site_end = cast_to_int_nan_none(row['unclear_site_end'], False)
            if int_old_unclear_site_end is not None:
                new_unclear_site_end = int_old_unclear_site_end + peptide_position_change
            else:
                new_unclear_site_end = row['unclear_site_end']

            return new_peptide_start, new_peptide_end, new_single_site, new_unclear_site_start, new_unclear_site_end

    return row['peptide_start'], row['peptide_end'], row['single_site'], row['unclear_site_start'], row['unclear_site_end']


def find_missing_peptide_sequence(row: pd.Series, isoform_seqs: Dict[str, str]):
    """
    Find missing peptide sequences where peptide_start/end are known for row. (Used by set_missing_peptide_sequences())
    :param row: row from DataFrame (w. 'uniprot', 'peptide_start', 'peptide_end' columns).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts())
    :return:
    """
    int_peptide_start = cast_to_int_nan_none(row['peptide_start'], False)
    int_peptide_end = cast_to_int_nan_none(row['peptide_end'], False)

    if int_peptide_start is None or int_peptide_end is None or pd.notna(row['peptide']):
        return row['peptide']

    idx_peptide_start = int_peptide_start - 1
    idx_peptide_end = int_peptide_end - 1

    return isoform_seqs[row['uniprot']][idx_peptide_start:idx_peptide_end + 1]


def _isoform_sorting_predicate(uniprot_id: str) -> Union[int, float]:
    protein_id, isoform_id = split_uniprot_id(uniprot_id)
    return int(isoform_id) if isoform_id is not None else float('-inf')


def find_uniprot_isoforms_containing_peptide(row: pd.Series, entry_isoforms: Dict[str, Set[str]],
                                            isoform_seqs: Dict[str, str]) -> List[str]:
    """
    Finds a list of matching isoforms containing the peptide for row. (Used by set_uniprot_isoforms_containing_peptides())
    :param row: row from DataFrame (w. 'uniprot', 'peptide_start', 'peptide_end' columns).
    :param entry_isoforms: dictionary of isoforms for each entry (see get_entry_isoforms_dicts()).
    :param isoform_seqs: dictionary of sequences for all entry isoforms (see get_entry_isoforms_dicts()).
    :return: A sorted list of matching isoforms.
    """
    protein_id, isoform_id = split_uniprot_id(row['uniprot'])
    if protein_id is None:
        warnings.warn(f"UniProt ID {row['uniprot']} Could not be parsed, skipping row")
        return [row['uniprot']]

    try:
        isoforms = entry_isoforms[protein_id]
    except KeyError:
        warnings.warn(f"UniProt ID {row['uniprot']} not found in entry_isoforms, skipping row")
        return [row['uniprot']]

    int_peptide_start = cast_to_int_nan_none(row['peptide_start'], False)
    int_peptide_end = cast_to_int_nan_none(row['peptide_end'], False)
    if int_peptide_start is None or int_peptide_end is None:
        return [row['uniprot']]
    idx_peptide_start = int_peptide_start - 1
    idx_peptide_end = int_peptide_end - 1

    seq_matching_entries = []
    seq_and_pos_matching_entries = []
    for entry in isoforms:
        if isoform_seqs[entry][idx_peptide_start:idx_peptide_end + 1] == row['peptide']:
            seq_and_pos_matching_entries.append(entry)
        else:
            matches = [match.start() for match in re.finditer(fr"(?=({row['peptide']}))", isoform_seqs[entry])]
            if len(matches) > 0:
                seq_matching_entries.append(entry)

    # Prioritize isoforms that already have the peptide in the right position
    if len(seq_and_pos_matching_entries) > 0:
        matching_entries = seq_and_pos_matching_entries
    # If there are none, then look at isoforms with the peptide in a different position
    elif len(seq_and_pos_matching_entries) > 0:
        matching_entries = seq_matching_entries
    else:  # If no matches, return the original ID
        return [row['uniprot']]

    if len(matching_entries) == 1:  # If there is only one match, return it
        return matching_entries
    elif row['uniprot'] in matching_entries:  # If there are multiple matches, return the original ID if possible
        return [row['uniprot']]
    else:  # If multiple isoforms match, just return all possible isoforms sorted
        return sorted(matching_entries, key=_isoform_sorting_predicate)


def extract_fasta_header_id(header: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract Uniprot ID (and isoform ID) from Uniprot fasta header
    :param header: Uniprot format fasta entry header
    :return: Uniprot ID (and potentially isoform ID)
    """
    split_header = header.split('|')
    if len(split_header) < 2 or re.match(cf.UNIPROT_REGEX, split_header[1]) is None:
        warnings.warn(f"Could not parse following entry header:\n{header}\nSkipping entry")
        return None, None
    protein_id, isoform_id = split_uniprot_id(split_header[1])
    if protein_id is None:
        warnings.warn(f"Could not parse following entry header:\n{header}\nSkipping entry")
        return None, None
    return protein_id, isoform_id
