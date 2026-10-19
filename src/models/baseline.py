import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from src.models.evaluator import ModelEvaluator

class BaselineTrainer:
    """Trains Gradient Boosted Trees and tree ensembles with Stratified K-Fold CV."""

    def __init__(
        self,
        model_type: str = "hist_gbdt",
        n_splits: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        self.model_type = model_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.params = kwargs
        self.models: List[Any] = []
        self.evaluator = ModelEvaluator()

    def _create_model(self, seed: int):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 150),
                max_depth=self.params.get("max_depth", 10),
                random_state=seed,
                n_jobs=-1
            )
        else: # hist_gbdt (High-performance gradient boosting)
            return HistGradientBoostingClassifier(
                max_iter=self.params.get("max_iter", 200),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_leaf_nodes=self.params.get("max_leaf_nodes", 31),
                random_state=seed
            )

    def train_cv(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros(len(y), dtype=float)
        fold_scores = []
        self.models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model = self._create_model(seed=self.random_state + fold)
            model.fit(X_tr, y_tr)
            
            # Predict probability of class 1
            val_probs = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_probs

            fold_auc = self.evaluator.compute_auc(y_val, val_probs)
            fold_scores.append(fold_auc)
            self.models.append(model)

        mean_auc = float(np.mean(fold_scores))
        std_auc = float(np.std(fold_scores))
        overall_metrics = self.evaluator.compute_classification_metrics(y, oof_preds)

        return {
            "oof_predictions": oof_preds,
            "fold_scores": fold_scores,
            "mean_auc": round(mean_auc, 5),
            "std_auc": round(std_auc, 5),
            "overall_metrics": overall_metrics,
            "models": self.models
        }

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Averages predicted probabilities across all CV fold models."""
        if not self.models:
            raise RuntimeError("Models have not been trained yet. Call train_cv first.")
        fold_preds = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(fold_preds, axis=0)
