import builtins
import gzip
import math
from os import PathLike
from typing import Callable, Union, Optional, TypeVar, IO, Tuple

import requests
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def download(url: str, query_params: dict = None, post_data: dict = None, stream_to_file: str = None,
             chunk_size: int = 8192, open_func: Callable = builtins.open, method: str = None) -> Optional[str]:
    """
    Wrapper to allow for easy downloading of text or files.
    :param url: URL to request from
    :param query_params: GET/POST parameters to put in URL, e.g. {'foo': 'bar'} becomes google.com/?foo=bar
    :param post_data: POST parameters to send as JSON
    :param stream_to_file: If not None, is the file path that the download will be streamed to. Use for large downloads that may not fit in RAM.
    :param chunk_size: If stream_to_file is not None, is bytes that will be downloaded and written to the file at a time
    :param open_func: If stream_to_file is not None, is the function that creates a open file handle and allows to write to it
    :param method: Is used to override default HTTP method, which would otherwise be 'GET' or 'POST' if post_data is not None
    :return: The downloaded text string, or None if stream_to_file is not None
    """
    if method is not None:
        request_method = method
    else:
        request_method = 'GET' if post_data is None else 'POST'
    with requests.request(request_method, url=url, params=query_params, data=post_data,
                          stream=stream_to_file is not None) as r:
        r.raise_for_status()
        if stream_to_file is not None:
            with open_func(stream_to_file, 'wb') as f:
                for chunk in tqdm(iterable=r.iter_content(chunk_size=chunk_size),
                                  desc=f"Downloading to {stream_to_file}",
                                  total=math.ceil(int(r.headers[
                                                          'Content-Length'].strip()) / chunk_size) if 'Content-Length' in r.headers else None):
                    f.write(chunk)
        else:
            return r.text


def get_uncompressed_size(file: Union[PathLike[str], IO[bytes]]):
    """
    Method to estimate uncompressed size of gzipped files.
    Warning: size is mod 2^32 so doesn't work for files >4GB
    :param file: file path or handle for gzip file
    :return: uncompressed size in bytes
    """
    with gzip.open(file, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(0, 0)
    return size


def get_latest_uniprot_release() -> str:
    """
    Get release version of current Uniprot website
    :return: the release in (YYYY_MM) format
    """
    with requests.head('https://www.uniprot.org/') as r:
        r.raise_for_status()
        return r.headers['X-UniProt-Release']


def display_invalid_rows(df: pd.DataFrame, description: str, row_apply_func: Callable = None, df_func: Callable = None,
                         **kwargs) -> list:
    """
    Runs a validation function on a dataframe, and returns a list of True/false for each row while showing rows that did not pass.
    :param df: The dataframe
    :param description: Description of why invalid rows did not pass
    :param row_apply_func: The validation function, which should take a dataframe row as it's first param and return True/False
    :param df_func: The validation function, if it takes the full dataframe as first param and returns a list of Tru/False for each row
    :param kwargs: Additional arguments for the validation function
    :return: pd.Series of True/False for each row in df
    """
    if row_apply_func is not None and df_func is None:
        valid_rows = df.apply(row_apply_func, axis=1, **kwargs)
    elif row_apply_func is None and df_func is not None:
        valid_rows = df_func(**kwargs)
    else:
        raise TypeError(
            "display_invalid_rows() requires one of the following arguments, but not both: 'row_apply_func', 'df_func'")
    total_rows = len(df)
    n_invalid_rows = sum(~valid_rows)
    percent_invalid_rows = n_invalid_rows / total_rows * 100
    if n_invalid_rows > 0:
        display(Markdown(f"#### {description}: {n_invalid_rows} ({percent_invalid_rows:4.2f}%):"))
        display(df[~valid_rows])
    return valid_rows


def cast_to_float_nan_none(value, na_allowed: bool = False) -> Union[float, None]:
    """
    Cast a value to a float if possible or None otherwise, while optionally passing through NA
    :param value: Any value that should be tried to convert to a float
    :param na_allowed: If True, passes through NA values
    :return: a float if cast succeeded, else None or NA if allowed
    """
    if na_allowed and pd.isna(value):
        return np.nan
    elif isinstance(value, (int, str)):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def cast_to_int_nan_none(value, na_allowed: bool = False) -> Union[int, None]:
    """
    Cast a value to an int if possible or None otherwise, while optionally passing through NA
    :param value: Any value that should be tried to convert to an int
    :param na_allowed: If True, passes through NA values
    :return: an int if cast succeeded, else None or NA if allowed
    """
    if na_allowed and pd.isna(value):
        return np.nan
    elif isinstance(value, (int, float, str)):
        try:
            int_value = int(value)
            if isinstance(value, float) and not value == int_value:
                return None
            return int_value
        except ValueError:
            return None
    return None


T = TypeVar("T")


def na_to_none(value: T) -> Optional[T]:
    """
    Should be used on a singular value to convert a potential NA into None
    :param value: Any value that may be NA
    :return: None if value is NA, else None (or throws ValueError if value is not a singular value)
    """
    return value if pd.notna(value) is True else None


def split_uniprot_id(uniprot_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Split Uniprot accession ID into protein ID and isoform ID
    :param uniprot_id: A Uniprot accession ID
    :return: A tuple of protein ID and isoform ID, where the isoform ID may be None if input was the canonical isoform
    """
    protein_id_isoform_split = uniprot_id.split('-')
    if len(protein_id_isoform_split) == 1:
        return protein_id_isoform_split[0], None
    elif len(protein_id_isoform_split) == 2:
        return protein_id_isoform_split[0], protein_id_isoform_split[1]
    else:
        return None, None
