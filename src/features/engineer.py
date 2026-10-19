import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class FeatureEngineer:
    """Preprocesses tabular airline passenger dataset and extracts domain features."""

    RATINGS_COLS = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service",
        "Baggage handling", "Checkin service", "Inflight service", "Cleanliness"
    ]

    CAT_MAPPINGS = {
        "Gender": {"Male": 1, "Female": 0},
        "Customer Type": {"Loyal Customer": 1, "disloyal Customer": 0},
        "Type of Travel": {"Business travel": 1, "Personal Travel": 0},
        "Class": {"Business": 2, "Eco Plus": 1, "Eco": 0}
    }

    def __init__(self):
        self.medians_: dict = {}
        self.feature_columns_: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        calc_df = df.drop(columns=[target_col], errors="ignore").drop(columns=["id"], errors="ignore")
        for col in calc_df.select_dtypes(include=["number"]).columns:
            self.medians_[col] = float(calc_df[col].median(skipna=True))
        return self

    def _transform_core(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "id" in out.columns:
            out = out.drop(columns=["id"])

        # Map categoricals
        for col, mapping in self.CAT_MAPPINGS.items():
            if col in out.columns:
                out[col] = out[col].map(mapping).fillna(0).astype(int)

        # Impute numeric missing values using training medians
        for col, med in self.medians_.items():
            if col in out.columns:
                out[col] = out[col].fillna(med)

        # Domain Feature Engineering
        dep_delay = out["Departure Delay in Minutes"] if "Departure Delay in Minutes" in out.columns else 0
        arr_delay = out["Arrival Delay in Minutes"] if "Arrival Delay in Minutes" in out.columns else 0
        out["total_delay"] = dep_delay + arr_delay
        
        dist = out["Flight Distance"] if "Flight Distance" in out.columns else 1000.0
        out["delay_ratio"] = out["total_delay"] / (dist + 1.0)

        # Service rating aggregations
        avail_ratings = [c for c in self.RATINGS_COLS if c in out.columns]
        if avail_ratings:
            out["mean_service_rating"] = out[avail_ratings].mean(axis=1)
            out["service_variance"] = out[avail_ratings].var(axis=1).fillna(0)
            out["dissatisfied_count"] = (out[avail_ratings] <= 2).sum(axis=1)
            out["delighted_count"] = (out[avail_ratings] >= 4).sum(axis=1)

        # Fill any remaining NaNs
        out = out.fillna(0)
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: str = "satisfaction") -> Tuple[pd.DataFrame, np.ndarray]:
        self.fit(df, target_col=target_col)
        y = None
        if target_col in df.columns:
            raw_y = df[target_col]
            if raw_y.dtype == object or isinstance(raw_y.iloc[0], str):
                y = (raw_y == "satisfied").astype(int).values
            else:
                y = raw_y.astype(int).values

        features_df = df.drop(columns=[target_col], errors="ignore")
        X = self._transform_core(features_df)
        self.feature_columns_ = list(X.columns)
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._transform_core(df)
        # Ensure column order matches training
        for col in self.feature_columns_:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_columns_]
