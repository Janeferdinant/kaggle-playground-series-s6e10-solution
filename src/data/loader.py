import pandas as pd
from pathlib import Path
from typing import Tuple

class DataLoader:
    """Loads and verifies competition datasets."""
    
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.sample_sub_path = self.data_dir / "sample_submission.csv"

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, test, and sample submission dataframes with integrity checks."""
        for p in [self.train_path, self.test_path, self.sample_sub_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required file {p.name} not found in {self.data_dir}")
                
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)
        sub_df = pd.read_csv(self.sample_sub_path)
        
        # Verify ID alignment between test and sample submission
        if "id" in test_df.columns and "id" in sub_df.columns:
            if not test_df["id"].equals(sub_df["id"]):
                raise ValueError("Mismatch between test IDs and sample submission IDs!")
                
        return train_df, test_df, sub_df
