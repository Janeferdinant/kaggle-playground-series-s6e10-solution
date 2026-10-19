import pandas as pd
from typing import Dict, Any, List

class DataProfiler:
    """Profiles tabular datasets for dimensions, nulls, types, and duplicates."""

    def profile_dataframe(self, df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
        shape = df.shape
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        null_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
        null_pct = {col: round(float(cnt / max(1, shape[0]) * 100), 2) for col, cnt in null_counts.items()}
        
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category", "string"]).columns)
        duplicate_rows = int(df.duplicated().sum())

        return {
            "name": name,
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "null_counts": null_counts,
            "null_percentage": null_pct,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "duplicate_rows": duplicate_rows
        }
