import subprocess
from os import PathLike
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from glyc_processing import cf
from glyc_processing.data_formats.common.validation import valid_peptide


def cluster_peptides(df: pd.DataFrame, cdhit_args: List[Union[str, PathLike[str]]] = None,
                     cdhit_path: PathLike[str] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Cluster peptide sequences using CD-HIT
    :param df: The dataframe (w. 'peptide' column)
    :param cdhit_args: Extra arguments for cd-hit in list form (default: ['-M', '4096', '-n', '5'])
    :param cdhit_path: If 'cd-hit' is not on the system/env path, it's location can be specified here.
    :return:
        A tuple of two Boolean Series - representative rows and redundant rows.
        Be aware that invalid 'peptide' rows are not included in either.
    """
    valid_rows = df.apply(valid_peptide, axis=1, na_allowed=False)

    with open(cf.CDHIT_PEPTIDES_INPUT_FILE, "w") as f:
        for i, row in df[valid_rows].iterrows():
            SeqIO.write(SeqRecord(Seq(row['peptide']), str(i), '', ''), f, "fasta")

    if cdhit_path is not None:
        run_args = [cdhit_path]
    else:
        run_args = ['cd-hit']
    run_args.extend(['-i', cf.CDHIT_PEPTIDES_INPUT_FILE, '-o', cf.CDHIT_PEPTIDES_OUTPUT_FILE, '-d', '0'])
    if cdhit_args is not None and len(cdhit_args) > 0:
        run_args.extend(cdhit_args)
    else:
        run_args.extend(['-M', '4096', '-n', '5'])

    with open(cf.CDHIT_PEPTIDES_LOG_FILE, 'w') as f:
        output = subprocess.run(run_args, stdout=f, stderr=subprocess.STDOUT)
    if output.returncode > 0:
        raise Exception(f"An error was encountered while trying to run CD-HIT, please check the log at {cf.CDHIT_PEPTIDES_LOG_FILE}")

    clustered_seq_index = SeqIO.index(str(cf.CDHIT_PEPTIDES_OUTPUT_FILE), 'fasta')
    representative_rows = np.full(df.shape[0], True)
    for i in clustered_seq_index:
        representative_rows[int(i)] = False
    representative_rows = pd.Series(representative_rows)
    redundant_rows = valid_rows & ~representative_rows

    return representative_rows, redundant_rows
