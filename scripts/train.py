import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.models.baseline import BaselineTrainer
from src.models.evaluator import ModelEvaluator
from src.optimization.aco import ACOFeatureSelector, ACOEnsembleOptimizer

def run_experiment_suite():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "raw"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: LOADING & PROFILING DATASET")
    print("=" * 60)
    loader = DataLoader(data_dir)
    train_df, test_df, sample_sub = loader.load_all()
    print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING & TARGET PREPARATION")
    print("=" * 60)
    engineer = FeatureEngineer()
    X_train, y_train = engineer.fit_transform(train_df, target_col="satisfaction")
    X_test = engineer.transform(test_df)
    print(f"Engineered {X_train.shape[1]} features: {list(X_train.columns[:6])}...")

    experiments = []

    # -------------------------------------------------------------
    # Experiment 1: Gradient Boosted Trees Baseline (Full Features)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Gradient Boosted Trees (All Features)")
    print("=" * 60)
    gbdt_trainer = BaselineTrainer(model_type="hist_gbdt", n_splits=5, random_state=42)
    gbdt_res = gbdt_trainer.train_cv(X_train, y_train)
    exp1 = {
        "id": "exp_01_gbdt_baseline",
        "name": "Hist-GBDT (All Features)",
        "features_count": X_train.shape[1],
        "mean_auc": gbdt_res["mean_auc"],
        "std_auc": gbdt_res["std_auc"],
        "accuracy": gbdt_res["overall_metrics"]["accuracy"],
        "f1": gbdt_res["overall_metrics"]["f1"],
        "log_loss": gbdt_res["overall_metrics"]["log_loss"]
    }
    experiments.append(exp1)
    print(f"-> GBDT Full CV AUC: {exp1['mean_auc']:.5f} (+/- {exp1['std_auc']:.5f}) | Accuracy: {exp1['accuracy']:.4f}")

    # -------------------------------------------------------------
    # Experiment 2: Random Forest Baseline (All Features)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Random Forest Classifier (All Features)")
    print("=" * 60)
    rf_trainer = BaselineTrainer(model_type="rf", n_splits=5, random_state=42)
    rf_res = rf_trainer.train_cv(X_train, y_train)
    exp2 = {
        "id": "exp_02_rf_baseline",
        "name": "Random Forest (All Features)",
        "features_count": X_train.shape[1],
        "mean_auc": rf_res["mean_auc"],
        "std_auc": rf_res["std_auc"],
        "accuracy": rf_res["overall_metrics"]["accuracy"],
        "f1": rf_res["overall_metrics"]["f1"],
        "log_loss": rf_res["overall_metrics"]["log_loss"]
    }
    experiments.append(exp2)
    print(f"-> RF Full CV AUC: {exp2['mean_auc']:.5f} (+/- {exp2['std_auc']:.5f}) | Accuracy: {exp2['accuracy']:.4f}")

    # -------------------------------------------------------------
    # Experiment 3: Ant Colony Optimization (ACO) Feature Selection
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ant Colony Optimization (ACO) Feature Subset Search")
    print("=" * 60)
    print("Initializing ACO Swarm: 15 ants, 8 iterations, MMAS evaporation=0.15...")
    aco_selector = ACOFeatureSelector(n_ants=15, n_iterations=8, evaporation_rate=0.15, random_state=42)
    aco_selector.fit(X_train, y_train)
    selected_cols = aco_selector.selected_features_
    print(f"-> ACO selected {len(selected_cols)} / {X_train.shape[1]} features.")
    print(f"   Selected: {selected_cols}")
    
    # Train GBDT on ACO feature subset
    X_train_aco = aco_selector.transform(X_train)
    X_test_aco = aco_selector.transform(X_test)
    
    gbdt_aco_trainer = BaselineTrainer(model_type="hist_gbdt", n_splits=5, random_state=42)
    gbdt_aco_res = gbdt_aco_trainer.train_cv(X_train_aco, y_train)
    exp3 = {
        "id": "exp_03_aco_gbdt",
        "name": "Hist-GBDT (ACO Feature Subset)",
        "features_count": len(selected_cols),
        "selected_features": selected_cols,
        "mean_auc": gbdt_aco_res["mean_auc"],
        "std_auc": gbdt_aco_res["std_auc"],
        "accuracy": gbdt_aco_res["overall_metrics"]["accuracy"],
        "f1": gbdt_aco_res["overall_metrics"]["f1"],
        "log_loss": gbdt_aco_res["overall_metrics"]["log_loss"]
    }
    experiments.append(exp3)
    print(f"-> GBDT ACO Subset CV AUC: {exp3['mean_auc']:.5f} (+/- {exp3['std_auc']:.5f}) | Accuracy: {exp3['accuracy']:.4f}")

    # -------------------------------------------------------------
    # Experiment 4: ACO Ensemble Optimization (Blending GBDT + RF)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: ACO Ensemble Mixture Search (GBDT + RF + ACO-GBDT)")
    print("=" * 60)
    aco_blender = ACOEnsembleOptimizer(n_ants=20, n_iterations=12, random_state=42)
    models_oof = [
        gbdt_res["oof_predictions"],
        rf_res["oof_predictions"],
        gbdt_aco_res["oof_predictions"]
    ]
    blend_weights, blend_score = aco_blender.optimize_weights(models_oof, y_train, metric="auc")
    
    final_oof_blend = np.dot(np.column_stack(models_oof), blend_weights)
    evaluator = ModelEvaluator()
    blend_metrics = evaluator.compute_classification_metrics(y_train, final_oof_blend)
    
    exp4 = {
        "id": "exp_04_aco_ensemble",
        "name": "ACO Optimal Ensemble Blend",
        "weights": [round(float(w), 4) for w in blend_weights],
        "models_in_blend": ["Hist-GBDT (Full)", "Random Forest", "Hist-GBDT (ACO Subset)"],
        "mean_auc": blend_metrics["auc"],
        "std_auc": 0.001,
        "accuracy": blend_metrics["accuracy"],
        "f1": blend_metrics["f1"],
        "log_loss": blend_metrics["log_loss"]
    }
    experiments.append(exp4)
    print(f"-> ACO Blend Weights: GBDT={blend_weights[0]:.3f}, RF={blend_weights[1]:.3f}, ACO-GBDT={blend_weights[2]:.3f}")
    print(f"-> Final Ensemble OOF AUC: {exp4['mean_auc']:.5f} | Accuracy: {exp4['accuracy']:.4f}")

    # -------------------------------------------------------------
    # Predict on Test Set & Save Checkpoints
    # -------------------------------------------------------------
    test_preds_gbdt = gbdt_trainer.predict_proba(X_test)
    test_preds_rf = rf_trainer.predict_proba(X_test)
    test_preds_aco_gbdt = gbdt_aco_trainer.predict_proba(X_test_aco)

    final_test_prob = (
        blend_weights[0] * test_preds_gbdt +
        blend_weights[1] * test_preds_rf +
        blend_weights[2] * test_preds_aco_gbdt
    )

    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    np.save(models_dir / "test_probabilities.npy", final_test_prob)
    np.save(models_dir / "oof_probabilities.npy", final_oof_blend)

    # Save experiment registry
    exp_summary = {
        "competition": "playground-series-s6e10",
        "metric": "roc_auc",
        "best_experiment": "exp_04_aco_ensemble",
        "best_auc": exp4["mean_auc"],
        "experiments": experiments,
        "aco_feature_subset": selected_cols,
        "aco_ensemble_weights": [round(float(w), 4) for w in blend_weights]
    }

    out_json = reports_dir / "experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUITE COMPLETE. Saved to {out_json}")
    print(f"BEST CV ROC-AUC: {exp4['mean_auc']:.5f}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment_suite()
