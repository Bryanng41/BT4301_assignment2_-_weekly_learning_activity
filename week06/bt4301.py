import pandas as pd



def set_default_pandas_options(max_columns=10, max_rows=2000, width=1000, max_colwidth=50):
    
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.width', width)
    pd.set_option('max_colwidth', max_colwidth)



def data_quality_report(df):
    
    if isinstance(df, pd.core.frame.DataFrame):
        
        descriptive_statistics = df.describe(include = 'all')
        data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).transpose()
        missing_value_counts = pd.DataFrame(df.isnull().sum(), columns=['Missing Values']).transpose()
        present_value_counts = pd.DataFrame(df.count(), columns=['Present Values']).transpose()
        data_report = pd.concat([descriptive_statistics, data_types, missing_value_counts, present_value_counts], axis=0)
        
        return data_report
    
    else:
    
        return None



# Data watermarking

import pandas as pd
import hashlib
import json
from decimal import Decimal



FINGERPRINT_COL = "_fp"



def normalize_value(v):
    """
    Normalize values for stable hashing.
    """
    if pd.isna(v):
        return None
    if isinstance(v, float):
        return format(v, ".15g")   # deterministic float rendering
    if isinstance(v, Decimal):
        return str(v)
    return v



def row_fingerprint(row: pd.Series, exclude_cols=None) -> str:
    """
    Generate SHA-256 fingerprint for a single row.
    """
    if exclude_cols is None:
        
        exclude_cols = set()

    # Build canonical dict with sorted keys
    record = {
        col: normalize_value(row[col])
        for col in sorted(row.index)
        if col not in exclude_cols
    }

    # Deterministic JSON encoding
    canonical = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def add_fingerprint_column(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Adds a fingerprint column to the dataframe.
    """
    df = df.copy()
    df[FINGERPRINT_COL] = df.apply(
        lambda row: row_fingerprint(row, exclude_cols=exclude_cols),
        axis=1
    )

    return df
