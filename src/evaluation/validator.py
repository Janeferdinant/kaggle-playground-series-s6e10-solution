import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

class SubmissionValidator:
    """Rigorous submission file validator against Kaggle competition requirements."""

    def __init__(self, sample_submission_path: Path | str):
        self.ref_path = Path(sample_submission_path)
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Reference sample submission file not found: {self.ref_path}")
        self.ref_df = pd.read_csv(self.ref_path)

    def validate(
        self,
        candidate_path: Path | str,
        allow_probabilities: bool = False
    ) -> Dict[str, Any]:
        p = Path(candidate_path)
        errors: List[str] = []
        checks: List[Dict[str, Any]] = []

        # Check 1: Existence and format
        exists = p.exists()
        is_csv = p.suffix.lower() == ".csv"
        checks.append({"check": "file_exists_and_csv", "passed": exists and is_csv})
        if not exists:
            return {"is_valid": False, "checks": checks, "errors": [f"File {p} does not exist."]}
        if not is_csv:
            errors.append(f"File must be a .csv format, but got {p.suffix}")

        try:
            cand_df = pd.read_csv(p)
        except Exception as e:
            return {"is_valid": False, "checks": checks, "errors": [f"Failed to read CSV: {e}"]}

        # Check 2: Row count
        row_match = len(cand_df) == len(self.ref_df)
        checks.append({
            "check": "row_count_match",
            "passed": row_match,
            "expected": len(self.ref_df),
            "actual": len(cand_df)
        })
        if not row_match:
            errors.append(f"Row count mismatch! Expected {len(self.ref_df)} rows, but got {len(cand_df)} rows.")

        # Check 3: Column names and order
        cols_match = list(cand_df.columns) == list(self.ref_df.columns)
        checks.append({
            "check": "column_names_and_order",
            "passed": cols_match,
            "expected": list(self.ref_df.columns),
            "actual": list(cand_df.columns)
        })
        if not cols_match:
            errors.append(f"Column headers or order mismatch! Expected {list(self.ref_df.columns)}, got {list(cand_df.columns)}")

        # Check 4: ID column alignment
        if "id" in self.ref_df.columns and "id" in cand_df.columns:
            id_match = cand_df["id"].equals(self.ref_df["id"])
            checks.append({"check": "id_alignment", "passed": id_match})
            if not id_match:
                errors.append("IDs do not match the required sample submission IDs sequence!")

        # Check 5: No missing values / NaNs
        null_count = int(cand_df.isnull().sum().sum())
        no_nans = null_count == 0
        checks.append({"check": "no_missing_values", "passed": no_nans, "null_count": null_count})
        if not no_nans:
            errors.append(f"Submission contains {null_count} missing (NaN/null) values!")

        # Check 6: Value validity
        if cols_match and len(cand_df) > 0:
            target_col = self.ref_df.columns[-1]
            cand_vals = cand_df[target_col]
            
            if pd.api.types.is_numeric_dtype(cand_vals):
                # Check numeric probability range
                in_range = bool(cand_vals.min() >= -1e-6 and cand_vals.max() <= 1.0 + 1e-6)
                checks.append({"check": "numeric_probability_range_0_1", "passed": in_range})
                if not in_range:
                    errors.append(f"Numeric predictions out of range [0, 1]! Min: {cand_vals.min()}, Max: {cand_vals.max()}")
            else:
                # String labels: check against reference categories
                ref_unique = set(self.ref_df[target_col].dropna().unique())
                cand_unique = set(cand_vals.dropna().unique())
                # If reference is dummy, check typical satisfaction classes
                allowed_labels = ref_unique.union({"satisfied", "neutral or dissatisfied"})
                valid_labels = cand_unique.issubset(allowed_labels)
                checks.append({"check": "categorical_labels_valid", "passed": valid_labels})
                if not valid_labels:
                    errors.append(f"Invalid categorical labels: {cand_unique - allowed_labels}")

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "row_count": len(cand_df),
            "columns": list(cand_df.columns),
            "checks": checks,
            "errors": errors
        }
