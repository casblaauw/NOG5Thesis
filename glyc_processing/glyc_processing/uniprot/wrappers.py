import gzip
import pickle
from typing import Collection, Dict, List, Tuple, Set

from Bio import SeqIO
from Bio.bgzf import BgzfReader

from glyc_processing.uniprot.latest_release import fetch_latest_uniprot_idmappings, fetch_latest_uniprot_deleted_ids, \
    fetch_latest_uniprot_entries, fetch_latest_uniprot_isoforms
from glyc_processing.uniprot.specific_release import fetch_uniprot_release_sprot, fetch_uniprot_release_docs, extract_all_uniprot_entries, \
    extract_uniprot_id_mappings, extract_uniprot_deleted_ids, extract_uniprot_entries, extract_all_uniprot_isoforms, \
    extract_uniprot_isoforms
from glyc_processing.uniprot.find_mapping import extract_fasta_header_id
from glyc_processing.misc import split_uniprot_id
from glyc_processing import cf


##### Wrapper functions #####

def get_uniprot_id_mappings(uniprot_ids: Collection[str]) -> Dict[str, List[str]]:
    """
    Downloads/Extracts necessary info from uniprot to get new id mappings for uniprot IDs.
    :param uniprot_ids: Collection of Uniprot IDs.
    :return:
        Dictionary where current valid uniprot IDs (without isoform IDs) are keys, and lists of new IDs are values.
        If a list has multiple IDs, the ID was demerged.
        If a list has only np.nan, the ID was deleted.
    """
    if not cf.IGNORE_EXISTING_FILES and cf.UNIPROT_IDMAPPING_FILE.exists():
        with gzip.open(cf.UNIPROT_IDMAPPING_FILE, 'rb') as f:
            return pickle.load(f)

    uniprot_protein_ids = {split_uniprot_id(uniprot_id)[0] for uniprot_id in uniprot_ids}
    if cf.UNIPROT_RELEASE == 'latest':
        mappings_dict = fetch_latest_uniprot_idmappings(uniprot_protein_ids)
        deleted_mappings_dict = fetch_latest_uniprot_deleted_ids(uniprot_protein_ids)
    else:
        fetch_uniprot_release_sprot()
        fetch_uniprot_release_docs()
        extract_all_uniprot_entries()
        mappings_dict = extract_uniprot_id_mappings(uniprot_protein_ids)
        deleted_mappings_dict = extract_uniprot_deleted_ids(uniprot_protein_ids)

    mappings_dict.update(deleted_mappings_dict)

    with gzip.open(cf.UNIPROT_IDMAPPING_FILE, 'wb') as f:
        pickle.dump(mappings_dict, f, protocol=4)
    return mappings_dict


def get_uniprot_entries(uniprot_ids: Collection[str]):
    """
    Downloads/Extracts necessary files from uniprot to get entries for each uniprot ID
    :param uniprot_ids: Collection of Uniprot IDs.
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ENTRIES_FILE.exists():
        if cf.UNIPROT_RELEASE == 'latest':
            fetch_latest_uniprot_entries(uniprot_ids)
        else:
            uniprot_protein_ids = {split_uniprot_id(uniprot_id)[0] for uniprot_id in uniprot_ids}
            fetch_uniprot_release_sprot()
            extract_all_uniprot_entries()
            extract_uniprot_entries(uniprot_protein_ids)


def get_uniprot_isoforms(uniprot_ids: Collection[str]):
    """
    Downloads/Extracts necessary files from uniprot to get isoform sequences for each uniprot ID
    :param uniprot_ids: Collection of Uniprot IDs.
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ISOFORMS_FILE.exists():
        if cf.UNIPROT_RELEASE == 'latest':
            fetch_latest_uniprot_isoforms(uniprot_ids)
        else:
            uniprot_protein_ids = {split_uniprot_id(uniprot_id)[0] for uniprot_id in uniprot_ids}
            fetch_uniprot_release_sprot()
            extract_all_uniprot_isoforms()
            extract_uniprot_isoforms(uniprot_protein_ids)


##### Dicts for fast access to just entry & all isoform sequences #####

def get_entry_seqs_dict() -> Dict[str, str]:
    """
    Downloads/Extracts canonical sequences for each entry (first run get_uniprot_entries())
    :return: Dictionary where Uniprot ID (without isoform ID) is key, and canonical sequence is value
    """
    if not cf.IGNORE_EXISTING_FILES and cf.UNIPROT_ENTRY_SEQS_FILE.exists():
        with gzip.open(cf.UNIPROT_ENTRY_SEQS_FILE, 'rb') as f:
            return pickle.load(f)

    if not cf.UNIPROT_ENTRIES_FILE.exists():
        raise TypeError(
            f"get_entry_seqs_dict() requires the file '{cf.UNIPROT_ENTRIES_FILE}' to exist if the file '{cf.UNIPROT_ENTRY_SEQS_FILE}' does not exist"
        )

    entry_seqs = {}
    with BgzfReader(cf.UNIPROT_ENTRIES_FILE, 'rt') as f:
        for entry in SeqIO.parse(f, 'swiss'):
            entry_seqs[entry.id] = str(entry.seq).upper()

    with gzip.open(cf.UNIPROT_ENTRY_SEQS_FILE, 'wb') as f:
        pickle.dump(entry_seqs, f, protocol=4)

    return entry_seqs


def get_entry_isoforms_dicts() -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """
    Downloads/Extracts both canonical and isoform sequences for each entry (first run get_uniprot_isoforms())
    :return: A tuple of two dictionaries:
        entry_isoforms: Dictionary where Uniprot ID (without isoform ID) is key, and a list of both canonical and isoform IDs is value
        isoform_seqs: Dictionary where Uniprot ID (with/without isoform ID) is key, and sequence is value
    """
    if not cf.IGNORE_EXISTING_FILES and cf.UNIPROT_ISOFORM_SEQS_FILE.exists():
        with gzip.open(cf.UNIPROT_ISOFORM_SEQS_FILE, 'rb') as f:
            return pickle.load(f)

    if not cf.UNIPROT_ISOFORMS_FILE.exists():
        raise TypeError(
            f"get_entry_isoforms_dicts() requires the file '{cf.UNIPROT_ISOFORMS_FILE}' to exist if the file '{cf.UNIPROT_ISOFORM_SEQS_FILE}' does not exist"
        )

    entry_isoforms = {}
    isoform_seqs = {}
    with BgzfReader(cf.UNIPROT_ISOFORMS_FILE, 'rt') as f:
        for entry in SeqIO.parse(f, 'fasta'):
            protein_id, isoform_id = extract_fasta_header_id(entry.id)
            if protein_id is None:
                continue
            full_id = protein_id + (f"-{isoform_id}" if isoform_id is not None else '')
            if protein_id not in entry_isoforms:
                entry_isoforms[protein_id] = set()
            entry_isoforms[protein_id].add(full_id)
            isoform_seqs[full_id] = str(entry.seq).upper()

    if cf.UNIPROT_RELEASE != 'latest':  # Add missing canonical sequences if pulling from varsplic instead of API
        entry_seqs = get_entry_seqs_dict()
        for protein_id in entry_seqs:
            if protein_id not in entry_isoforms:
                entry_isoforms[protein_id] = set()
            entry_isoforms[protein_id].add(protein_id)
            isoform_seqs[protein_id] = entry_seqs[protein_id]

    with gzip.open(cf.UNIPROT_ISOFORM_SEQS_FILE, 'wb') as f:
        pickle.dump((entry_isoforms, isoform_seqs), f, protocol=4)

    return entry_isoforms, isoform_seqs
