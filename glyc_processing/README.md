# glyc_processing
Library for standardization, validation, Uniprot ID mapping, clustering & consolidation of peptide glycosylation data

## Purpose:
To be a reusable library for processing different MS glycosylation peptide data formats for machine learning.

## Features:
* *Standardization*: Bring different data formats to a similar format.
* *Uniprot mapping*: Mapping of data to Uniprot, with support for choosing specific uniprot releases for reproducability, fixing merges/de-merges/deletions of IDs & mapping to right isoforms if canonical sequences has changed.
* *Validation*: Validate and visualize erroneous data.
* *Clustering*: Cluster peptides using CD-HIT
* *Annotation*: Consolidation of peptide data into a format from which machine learning data can easily be exported.

## Data formats:
Different data formats can have different configurations, needs for standardization and validation. Therefore, each data format has a separate folder in *glyc_processing.data_formats* along with common functions in *glyc_processing.data_formats.common*

**Essential fields for data annotation:**

Some minimum of fields are needed to extract glycosylation data. If the data is in a different format, it should be converted in its specific standardization pipeline.

* **uniprot**: Valid UniProt accession string
* **peptide_id**: Unique UUID for Peptide (or Site if no peptide info) - must be a valid UUID string
* **peptide**: Peptide sequence - must be a valid AA sequence string w. only 20 basic AA and X (or NA if no peptide data is available)
* **peptide_start**: Absolute start position of peptide in protein sequence (or NA if no peptide data is available, should be 1-indexed)
* **peptide_end**: Absolute end position of peptide in protein sequence (or NA if no peptide data is available, should be 1-indexed and end-inclusive)
* **single_site**: Absolute position of single-site glycosite in protein sequence (or NA if site is ambiguous or negative data, should be 1-indexed)
* **unclear_site_start**: Absolute start position of ambiguous glycosite in protein AA sequence (or NA if site is single-site, should be 1-indexed)
* **unclear_site_end**: Absolute end position of ambiguous glycosite in protein AA sequence (or NA if site is single-site, should be 1-indexed and end-inclusive)

## Installation:
**If using venv, first create a new environment**

You need python >= 3.6 and pip as a minimum. Biopython may also need gcc.

Run the following in library root directory:

```python3
python3 -m venv .glyc_processing_env
source .glyc_processing_env/bin/activate
```

**If using Conda, first create a new environment**

You need conda or miniconda installed, then run:
```bash
conda create -n glyc_processing_env -c conda-forge 'python>=3.6' 'pip' 'gcc' # gcc may be needed by biopython
conda activate glyc_processing_env
```

**Install library:**

Go to the library root directory and run:
```python3
pip install -e .
```

to install glyc_processing and it's dependencies. The `-e` option specifies the installation as editable, so you can change the library code.

## Requirements:
* python>=3.6
* numpy
* pandas
* requests
* biopython
* tqdm
* ipython
* CD-HIT
