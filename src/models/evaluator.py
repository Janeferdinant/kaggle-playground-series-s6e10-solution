import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from typing import Dict, Any

class ModelEvaluator:
    """Computes competition evaluation metrics and diagnostic classification stats."""

    @staticmethod
    def compute_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        try:
            return float(roc_auc_score(y_true, y_pred_prob))
        except Exception:
            return 0.5

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        auc = ModelEvaluator.compute_auc(y_true, y_pred_prob)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        loss = float(log_loss(y_true, np.clip(y_pred_prob, 1e-6, 1.0 - 1e-6)))

        return {
            "auc": round(auc, 5),
            "accuracy": round(acc, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "f1": round(f1, 5),
            "log_loss": round(loss, 5)
        }
