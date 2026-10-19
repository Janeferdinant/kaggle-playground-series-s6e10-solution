import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.data.loader import DataLoader
from src.evaluation.validator import SubmissionValidator

def generate_and_validate_submission():
    raw_dir = repo_root / "data" / "raw"
    models_dir = repo_root / "models"
    subs_dir = repo_root / "submissions"
    reports_dir = repo_root / "reports"
    
    subs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SUBMISSION ARTIFACT GENERATOR & VALIDATOR")
    print("=" * 60)

    loader = DataLoader(raw_dir)
    _, test_df, sample_sub = loader.load_all()

    prob_path = models_dir / "test_probabilities.npy"
    if not prob_path.exists():
        raise FileNotFoundError(f"Model predictions not found at {prob_path}. Please run scripts/train.py first.")

    probs = np.load(prob_path)
    print(f"Loaded {len(probs)} test probabilities. Mean prob: {np.mean(probs):.4f}")

    # Inspect sample submission target column format
    target_col = sample_sub.columns[-1]
    ref_first_val = sample_sub[target_col].iloc[0]
    
    if isinstance(ref_first_val, str):
        # String categorical mapping
        preds = np.where(probs >= 0.5, "satisfied", "neutral or dissatisfied")
    else:
        # Numeric probabilities
        preds = np.round(probs, 5)

    sub_df = pd.DataFrame({
        "id": test_df["id"],
        target_col: preds
    })

    out_csv = subs_dir / "submission_verified.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"Generated candidate submission file at: {out_csv}")
    print(f"Shape: {sub_df.shape}")

    # Validate candidate submission
    print("\nRunning SubmissionValidator against sample_submission.csv...")
    validator = SubmissionValidator(raw_dir / "sample_submission.csv")
    report = validator.validate(out_csv)

    audit_out = reports_dir / "submission_audit.json"
    with open(audit_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Validation Status: {'PASSED [READY FOR SUBMISSION]' if report['is_valid'] else 'FAILED'}")
    for chk in report["checks"]:
        print(f"  [{'x' if chk['passed'] else ' '}] {chk['check']}")

    if not report["is_valid"]:
        print("ERRORS DETECTED:")
        for err in report["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SUBMISSION POLICY CHECK:")
    print("In compliance with Master Autonomous Agent Rule #10:")
    print("Artifact is verified locally and ready for submission.")
    print("Kaggle submission CLI command (requires explicit user confirmation):")
    print(f"  kaggle competitions submit -c playground-series-s6e10 -f {out_csv} -m \"ACO Ensemble Solution CV-AUC 0.86153\"")
    print("=" * 60)

if __name__ == "__main__":
    generate_and_validate_submission()
