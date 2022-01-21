import pandas as pd

from glyc_processing.data_formats.common.standardization import strip_string_whitespace, standardize_peptide
from glyc_processing.misc import cast_to_int_nan_none
from glyc_processing import cf


def rename_columns(df):
    df.rename(columns={
        'uniprot': 'uniprot',
        'peptide.id': 'peptide_id',
        'peptide': 'peptide',
        'peptide.start': 'peptide_start',
        'peptide.end': 'peptide_end',
        'site': 'single_site',
        'site.ambiguous.start': 'unclear_site_start',
        'site.ambiguous.end': 'unclear_site_end',
        'site.composition': 'site_composition',
        'source': 'source',
        'quantification': 'quantification',
        'quantification.channels': 'quantification_channels',
        'site.ambiguity': 'site_ambiguity',
        'quantification.confidence': 'quantification_confidence',
        'composition': 'composition',
        'activation': 'activation',
        'dataset': 'dataset',
    }, inplace=True)


def standardize_site_composition(df: pd.DataFrame):
    # Add 1x to singlet site compositions (eg. HexNac->1xHexNac)
    singlet_site_composition_rows = df['site_composition'].str.match(r"^\d+x") == False
    df.loc[singlet_site_composition_rows, 'site_composition'] = '1x' + df.loc[singlet_site_composition_rows, 'site_composition']


def standardize_source(df: pd.DataFrame):
    # Add _ to source when either origin or proteases is unknown (eg. 'sec_vva'->'sec_vva_'; 'pna_try,chy'->'_pna_try,chy')
    origin_lectin_source_rows = df['source'].str.match(cf.ORIGIN_LECTIN_REGEX) == True
    lectin_protease_source_rows = df['source'].str.match(cf.LECTIN_PROTEASES_REGEX) == True
    df.loc[origin_lectin_source_rows, 'source'] = df.loc[origin_lectin_source_rows, 'source'] + '_'
    df.loc[lectin_protease_source_rows, 'source'] = '_' + df.loc[lectin_protease_source_rows, 'source']

    # or

    # Split source into three separate columns
    # new_columns = df['source'].str.extractall(cf.SOURCE_REGEX).droplevel('match')
    # df = pd.concat([df, new_columns], axis=1)
    # return df


def recover_site_info(df: pd.DataFrame):
    # Recover 'single_site' or 'unclear_site' position information from the peptide amino acids and absolute position if possible
    def recover_site_columns(row: pd.Series):
        peptide_start = cast_to_int_nan_none(row['peptide_start'])
        if isinstance(peptide_start, int) and isinstance(row['peptide'], str) and not pd.isna(row['peptide']):
            relative_aa_positions = [i for i, aa in enumerate(row['peptide']) if aa in cf.ALLOWED_AA]
            if len(relative_aa_positions) == 1:
                return peptide_start + relative_aa_positions[0], row['unclear_site_start'], row['unclear_site_end']
            elif len(relative_aa_positions) > 1:
                return row['single_site'], peptide_start + relative_aa_positions[0], peptide_start + relative_aa_positions[-1]
        return row['single_site'], row['unclear_site_start'], row['unclear_site_end']

    recoverable_site_rows = (
        df['single_site'].isna() & df['unclear_site_start'].isna() & df['unclear_site_end'].isna() &
        df['peptide'].notna() & df['peptide_start'].notna() & df['peptide_end'].notna()
    )
    df.loc[recoverable_site_rows, ['single_site', 'unclear_site_start', 'unclear_site_end']] = df[
        recoverable_site_rows].apply(recover_site_columns, axis=1, result_type='expand').rename(
        columns={0: 'single_site', 1: 'unclear_site_start', 2: 'unclear_site_end'})


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Apply standardization functions to dataframe
    rename_columns(df)
    df = strip_string_whitespace(df)
    standardize_peptide(df)
    standardize_site_composition(df)
    standardize_source(df)
    recover_site_info(df)
    return df
