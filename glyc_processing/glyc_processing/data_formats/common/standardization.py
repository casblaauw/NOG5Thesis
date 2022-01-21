import numpy as np
import pandas as pd


def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strips leading and trailing whitespace from all string elements in DataFrame
    :param df: Dataframe
    :return: whitespace-stripped DataFrame
    """
    # Strip leading/trailing whitespace from string elements
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Convert whitespace-only string elements to NaN
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x == '' else x)
    return df


def standardize_peptide(df: pd.DataFrame):
    """
    Standardizes peptide sequences to Uppercase with no whitespace
    :param df: Dataframe
    """
    # All peptide sequences should be uppercase with no whitespace
    df['peptide'] = df['peptide'].str.upper()
    df['peptide'] = df['peptide'].replace(r"\s+", "", regex=True)
