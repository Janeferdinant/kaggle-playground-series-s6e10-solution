import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.evaluation.validator import SubmissionValidator

@pytest.fixture
def sample_submission_ref(tmp_path):
    sub = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "satisfaction": ["neutral or dissatisfied"] * 5
    })
    path = tmp_path / "sample_submission.csv"
    sub.to_csv(path, index=False)
    return path

def test_submission_validator_valid_string_labels(sample_submission_ref, tmp_path):
    candidate = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "satisfaction": ["satisfied", "neutral or dissatisfied", "satisfied", "neutral or dissatisfied", "satisfied"]
    })
    cand_path = tmp_path / "submission.csv"
    candidate.to_csv(cand_path, index=False)

    validator = SubmissionValidator(sample_submission_path=sample_submission_ref)
    report = validator.validate(cand_path)

    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert report["row_count"] == 5

def test_submission_validator_valid_probabilities(sample_submission_ref, tmp_path):
    candidate = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "satisfaction": [0.85, 0.12, 0.94, 0.05, 0.77]
    })
    cand_path = tmp_path / "submission_prob.csv"
    candidate.to_csv(cand_path, index=False)

    validator = SubmissionValidator(sample_submission_path=sample_submission_ref)
    report = validator.validate(cand_path, allow_probabilities=True)

    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

def test_submission_validator_detects_row_mismatch(sample_submission_ref, tmp_path):
    candidate = pd.DataFrame({
        "id": [101, 102, 103], # only 3 rows instead of 5
        "satisfaction": ["satisfied"] * 3
    })
    cand_path = tmp_path / "sub_short.csv"
    candidate.to_csv(cand_path, index=False)

    validator = SubmissionValidator(sample_submission_path=sample_submission_ref)
    report = validator.validate(cand_path)

    assert report["is_valid"] is False
    assert any("row count" in err.lower() for err in report["errors"])

def test_submission_validator_detects_missing_column(sample_submission_ref, tmp_path):
    candidate = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "wrong_col": ["satisfied"] * 5
    })
    cand_path = tmp_path / "sub_bad_cols.csv"
    candidate.to_csv(cand_path, index=False)

    validator = SubmissionValidator(sample_submission_path=sample_submission_ref)
    report = validator.validate(cand_path)

    assert report["is_valid"] is False
    assert any("column" in err.lower() for err in report["errors"])

def test_submission_validator_detects_nan(sample_submission_ref, tmp_path):
    candidate = pd.DataFrame({
        "id": [101, 102, 103, 104, 105],
        "satisfaction": ["satisfied", None, "neutral or dissatisfied", "satisfied", "neutral or dissatisfied"]
    })
    cand_path = tmp_path / "sub_nan.csv"
    candidate.to_csv(cand_path, index=False)

    validator = SubmissionValidator(sample_submission_path=sample_submission_ref)
    report = validator.validate(cand_path)

    assert report["is_valid"] is False
    assert any("missing" in err.lower() or "nan" in err.lower() for err in report["errors"])
