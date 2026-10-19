import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.data.loader import DataLoader
from src.data.profiler import DataProfiler

@pytest.fixture
def sample_data_dir(tmp_path):
    train_df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "feature_a": [10.5, 20.0, None, 40.2, 50.1],
        "category_b": ["A", "B", "A", "B", "C"],
        "satisfaction": [1, 0, 1, 0, 1]
    })
    test_df = pd.DataFrame({
        "id": [6, 7, 8],
        "feature_a": [60.0, 70.5, 80.1],
        "category_b": ["A", "B", "A"]
    })
    sub_df = pd.DataFrame({
        "id": [6, 7, 8],
        "satisfaction": [0.5, 0.5, 0.5]
    })
    
    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)
    sub_df.to_csv(tmp_path / "sample_submission.csv", index=False)
    return tmp_path

def test_data_loader_success(sample_data_dir):
    loader = DataLoader(sample_data_dir)
    train, test, sub = loader.load_all()
    assert len(train) == 5
    assert len(test) == 3
    assert len(sub) == 3
    assert list(test["id"]) == list(sub["id"])

def test_data_loader_missing_file(tmp_path):
    loader = DataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_all()

def test_data_profiler(sample_data_dir):
    train_df = pd.read_csv(sample_data_dir / "train.csv")
    profiler = DataProfiler()
    profile = profiler.profile_dataframe(train_df, "train")
    
    assert profile["shape"] == (5, 4)
    assert "feature_a" in profile["null_counts"]
    assert profile["null_counts"]["feature_a"] == 1
    assert "feature_a" in profile["numeric_columns"]
    assert "category_b" in profile["categorical_columns"]
    assert profile["duplicate_rows"] == 0
