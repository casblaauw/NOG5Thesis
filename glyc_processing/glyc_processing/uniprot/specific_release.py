import re
import gzip
import math
from functools import partial
import tarfile
from typing import Collection, List, Dict

import numpy as np
from Bio import SeqIO
from Bio.bgzf import BgzfReader, BgzfWriter
from tqdm.auto import tqdm

from glyc_processing.uniprot.find_mapping import extract_fasta_header_id
from glyc_processing.misc import download, get_uncompressed_size
from glyc_processing import cf


##### Common fetching functions #####

def fetch_uniprot_release_sprot():
    """
    Fetch tar.gz file with all entries for specific Uniprot release.
    """
    if not cf.UNIPROT_RELEASE_SPROT_TAR_FILE.exists():
        download(url=cf.UNIPROT_RELEASE_SPROT_URL, stream_to_file=cf.UNIPROT_RELEASE_SPROT_TAR_FILE)


def extract_all_uniprot_entries():
    """
    Extract all Uniprot entries from tar.gz file.
    """
    chunk_size = 65536
    if not cf.UNIPROT_RELEASE_SPROT_BGZ_FILE.exists():
        with tarfile.open(cf.UNIPROT_RELEASE_SPROT_TAR_FILE, 'r') as tar_f:
            with tar_f.extractfile(cf.UNIPROT_RELEASE_SPROT_TAR_ENTRIES_LOCATION) as zip_f:
                with gzip.open(zip_f, 'rb') as in_f, BgzfWriter(cf.UNIPROT_RELEASE_SPROT_BGZ_FILE, 'wb') as out_f:
                    for chunk in tqdm(iterable=iter(partial(in_f.read, chunk_size), b''),
                                      desc=f"BGZ encoding Uniprot entries from {cf.UNIPROT_RELEASE_SPROT_TAR_FILE}",
                                      total=math.ceil(get_uncompressed_size(zip_f) / chunk_size)):
                        out_f.write(chunk)


##### ID Map fetching functions #####

def fetch_uniprot_release_docs():
    """
    Fetch tar.gz file with misc. info for specific Uniprot release (including deleted Uniprot IDs).
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_RELEASE_DOCS_TAR_FILE.exists():
        download(url=cf.UNIPROT_RELEASE_DOCS_URL, stream_to_file=cf.UNIPROT_RELEASE_DOCS_TAR_FILE)


def extract_uniprot_id_mappings(uniprot_protein_ids: Collection[str]) -> Dict[str, List[str]]:
    """
    Extract Uniprot ID mappings for specific Uniprot release.
    :param uniprot_protein_ids: Collection of Uniprot IDs (without isoform ID).
    :return: Dictionary of ID mappings.
    """
    mappings_dict = {}
    with BgzfReader(cf.UNIPROT_RELEASE_SPROT_BGZ_FILE, 'rt') as f, tqdm(total=len(uniprot_protein_ids),
                                                                        desc=f"Extracting ID mappings from {cf.UNIPROT_RELEASE_SPROT_BGZ_FILE}") as t:
        for entry in SeqIO.parse(f, 'swiss'):
            dataset_accessions = (uniprot_id for uniprot_id in entry.annotations["accessions"] if
                                  uniprot_id in uniprot_protein_ids)
            for dataset_ac in dataset_accessions:
                if dataset_ac not in mappings_dict:
                    mappings_dict[dataset_ac] = []
                    t.update()
                mappings_dict[dataset_ac].append(entry.id)
    return mappings_dict


def extract_uniprot_deleted_ids(uniprot_protein_ids: Collection[str]) -> Dict[str, List]:
    """
    Extract deleted Uniprot IDs from specific Uniprot release .tar.gz file.
    :param uniprot_protein_ids: Collection of Uniprot IDs (without isoform ID).
    :return: Dictionary where keys are deleted Uniprot IDs and values are np.nan.
    """
    valid_uniprot_id_regex = re.compile(cf.UNIPROT_REGEX, re.MULTILINE)
    with tarfile.open(cf.UNIPROT_RELEASE_DOCS_TAR_FILE, 'r') as tar_f:
        with tar_f.extractfile(cf.UNIPROT_RELEASE_DOCS_TAR_DELETEDIDS_LOCATION) as f:
            all_deleted_ids = valid_uniprot_id_regex.findall(f.read().decode('utf-8'))
            deleted_mappings_dict = {uniprot_id: [np.nan] for uniprot_id in all_deleted_ids if uniprot_id in uniprot_protein_ids}
    return deleted_mappings_dict


##### Entry sequence fetching functions #####

def extract_uniprot_entries(uniprot_protein_ids: Collection[str]):
    """
    Extract Uniprot entries .dat.bgz file from specific Uniprot release .dat.bgz file.
    :param uniprot_protein_ids: Collection of Uniprot IDs (without isoform ID).
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ENTRIES_FILE.exists():
        all_entries = SeqIO.index(str(cf.UNIPROT_RELEASE_SPROT_BGZ_FILE), 'swiss')
        with BgzfWriter(cf.UNIPROT_ENTRIES_FILE, 'wb') as out_f:
            for entry in tqdm(iterable=uniprot_protein_ids,
                              desc=f"Extracting data-specific Uniprot entries from {cf.UNIPROT_RELEASE_SPROT_BGZ_FILE}"):
                out_f.write(all_entries.get_raw(entry))
        all_entries.close()


##### Isoforms fetching functions #####

def extract_all_uniprot_isoforms(chunk_size: int = 65536):
    """
    Extract all Uniprot isoform sequences .fasta.bgz file from specific Uniprot release.
    :param chunk_size: Bytes that will be downloaded and written to the file at a time (65536 is BGZF max block size)
    """
    if not cf.UNIPROT_RELEASE_ISOFORMS_BGZ_FILE.exists():
        with tarfile.open(cf.UNIPROT_RELEASE_SPROT_TAR_FILE, 'r') as tar_f:
            with tar_f.extractfile(cf.UNIPROT_RELEASE_SPROT_TAR_ISOFORMS_LOCATION) as zip_f:
                with gzip.open(zip_f, 'rb') as in_f, BgzfWriter(cf.UNIPROT_RELEASE_ISOFORMS_BGZ_FILE, 'wb') as out_f:
                    for chunk in tqdm(iterable=iter(partial(in_f.read, chunk_size), b''),
                                      desc=f"BGZ encoding Uniprot entry isoforms from {cf.UNIPROT_RELEASE_SPROT_TAR_FILE}",
                                      total=math.ceil(get_uncompressed_size(zip_f) / chunk_size)):
                        out_f.write(chunk)


def extract_uniprot_isoforms(uniprot_protein_ids: Collection[str]):
    """
    Extract Uniprot isoform sequences .fasta.bgz file from specific Uniprot release .fasta.bgz file.
    :param uniprot_protein_ids: Collection of Uniprot IDs (without isoform ID).
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ISOFORMS_FILE.exists():
        all_entries = SeqIO.index(str(cf.UNIPROT_RELEASE_ISOFORMS_BGZ_FILE), 'fasta')
        with BgzfWriter(cf.UNIPROT_ISOFORMS_FILE, 'wt') as out_f, tqdm(total=len(uniprot_protein_ids),
                                                                       desc=f"Extracting data-specific Uniprot entry isoforms from {cf.UNIPROT_RELEASE_ISOFORMS_BGZ_FILE}") as t:
            for entry in all_entries:
                protein_id, isoform_id = extract_fasta_header_id(entry)
                if protein_id in uniprot_protein_ids:
                    out_f.write(all_entries.get_raw(entry))
                    t.update()
        all_entries.close()
