import re
from io import StringIO
from typing import Collection, List, Dict

import numpy as np
import pandas as pd
from Bio.bgzf import BgzfWriter

from glyc_processing.misc import split_uniprot_id, download
from glyc_processing import cf


##### ID Map fetching functions #####

def fetch_latest_uniprot_idmappings(uniprot_ids: Collection[str]) -> Dict[str, List[str]]:
    """
    Fetch Uniprot ID mappings to latest Uniprot release.
    :param uniprot_ids: Collection of Uniprot IDs.
    :return: Dictionary of ID mappings. Isoforms are removed, and there can be multiple or no mappings for some IDs.
    """
    data = {
        'from': 'ACC+ID',
        'to': 'ACC',
        'format': 'tab',
        'query': ' '.join(uniprot_ids),
    }
    response = download(url='https://www.uniprot.org/uploadlists/', post_data=data)
    mappings_df = pd.read_table(StringIO(response))
    mappings_df['From'] = mappings_df['From'].apply(lambda From: split_uniprot_id(From)[0])
    mappings_dict = mappings_df.groupby('From').agg(lambda df: df.unique().tolist()).to_dict()['To']
    return mappings_dict


def fetch_latest_uniprot_deleted_ids(uniprot_protein_ids: Collection[str]) -> Dict[str, List]:
    """
    Fetch deleted Uniprot IDs from latest Uniprot release.
    :param uniprot_protein_ids: Collection of Uniprot IDs (without isoform ID).
    :return: Dictionary where keys are deleted Uniprot IDs and values are np.nan.
    """
    valid_uniprot_id_regex = re.compile(cf.UNIPROT_REGEX, re.MULTILINE)
    response = download(url=cf.UNIPROT_LATEST_DELETEDIDS_URL)
    deleted_mappings_list = valid_uniprot_id_regex.findall(response)
    deleted_mappings_dict = {uniprot_id: [np.nan] for uniprot_id in deleted_mappings_list
                             if uniprot_id in uniprot_protein_ids}
    return deleted_mappings_dict


##### Entry sequence fetching functions #####

def fetch_latest_uniprot_entries(uniprot_ids: Collection[str]):
    """
    Fetch Uniprot entries .dat.bgz file from latest Uniprot release.
    :param uniprot_ids: Collection of Uniprot IDs.
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ENTRIES_FILE.exists():
        data = {
            'from': 'ACC+ID',
            'to': 'ACC',
            'format': 'txt',
            'query': ' '.join(uniprot_ids),
        }
        download(url='https://www.uniprot.org/uploadlists/', post_data=data,
                 stream_to_file=cf.UNIPROT_ENTRIES_FILE, chunk_size=65536, open_func=BgzfWriter)


##### Isoforms fetching functions #####

def fetch_latest_uniprot_isoforms(uniprot_ids: Collection[str]):
    """
    Fetch Uniprot isoform sequences .fasta.bgz file from latest Uniprot release.
    :param uniprot_ids: Collection of Uniprot IDs.
    """
    if cf.IGNORE_EXISTING_FILES or not cf.UNIPROT_ISOFORMS_FILE.exists():
        data = {
            'from': 'ACC+ID',
            'to': 'ACC',
            'format': 'fasta',
            'include': 'yes',
            'query': ' '.join(uniprot_ids),
        }
        download(url='https://www.uniprot.org/uploadlists/', post_data=data,
                 stream_to_file=cf.UNIPROT_ISOFORMS_FILE, chunk_size=65536, open_func=BgzfWriter)
