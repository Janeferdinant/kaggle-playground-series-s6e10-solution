import pytest
import numpy as np
import pandas as pd
from src.optimization.aco import AntColonyOptimizer, ACOFeatureSelector, ACOEnsembleOptimizer

def test_aco_base_optimizer():
    def fitness_fn(binary_vector):
        return float(np.sum(binary_vector))

    aco = AntColonyOptimizer(
        n_ants=10,
        n_iterations=8,
        evaporation_rate=0.15,
        alpha=1.0,
        beta=1.0,
        random_state=42
    )
    best_solution, best_fitness, history = aco.optimize(fitness_fn, n_dimensions=10)
    
    assert len(best_solution) == 10
    assert best_fitness >= 7.0
    assert len(history) == 8
    assert history[-1] >= history[0]

def test_aco_feature_selector_synthetic():
    np.random.seed(42)
    n_samples = 200
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples, 5)
    
    y_prob = 1 / (1 + np.exp(-(2.5 * x1 + 2.0 * x2 - 1.8 * x3)))
    y = (np.random.rand(n_samples) < y_prob).astype(int)
    
    X = np.column_stack([x1, x2, x3, noise])
    feature_names = [f"f_{i}" for i in range(8)]
    df_X = pd.DataFrame(X, columns=feature_names)
    
    selector = ACOFeatureSelector(
        n_ants=10,
        n_iterations=6,
        evaporation_rate=0.2,
        random_state=42
    )
    selector.fit(df_X, y)
    
    assert selector.selected_features_ is not None
    assert len(selector.selected_features_) > 0
    assert "f_0" in selector.selected_features_ or "f_1" in selector.selected_features_
    
    X_trans = selector.transform(df_X)
    assert X_trans.shape[1] == len(selector.selected_features_)

def test_aco_ensemble_optimizer():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    
    pred1 = np.clip(y_true * 0.8 + 0.1 + np.random.normal(0, 0.15, size=n), 0.01, 0.99)
    pred2 = np.clip(y_true * 0.3 + 0.35 + np.random.normal(0, 0.3, size=n), 0.01, 0.99)
    pred3 = np.random.uniform(0.1, 0.9, size=n)
    
    preds = [pred1, pred2, pred3]
    
    opt = ACOEnsembleOptimizer(n_ants=20, n_iterations=15, random_state=42)
    weights, best_score = opt.optimize_weights(preds, y_true, metric="auc")
    
    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0, atol=1e-3)
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]

def test_aco_vs_random_ablation():
    # Ablation study: ACO feature selection vs 20 random subset trials
    np.random.seed(123)
    n = 250
    # 4 strong features, 8 pure noise
    informative = np.random.randn(n, 4)
    noise = np.random.randn(n, 8)
    X = np.hstack([informative, noise])
    y_prob = 1 / (1 + np.exp(-(informative[:, 0] * 2.0 + informative[:, 1] * 1.5)))
    y = (np.random.rand(n) < y_prob).astype(int)
    
    df_X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(12)])
    
    selector = ACOFeatureSelector(n_ants=15, n_iterations=8, random_state=123)
    selector.fit(df_X, y)
    aco_score = selector.best_score_
    
    # 10 random subsets
    random_scores = []
    for _ in range(10):
        mask = np.random.choice([0, 1], size=12)
        if np.sum(mask) == 0:
            mask[0] = 1
        # Fast score proxy
        sub = X[:, mask == 1]
        diff = np.mean(sub[y == 1], axis=0) - np.mean(sub[y == 0], axis=0)
        proj = np.dot(sub, diff)
        from sklearn.metrics import roc_auc_score
        random_scores.append(roc_auc_score(y, proj))
        
    mean_random_score = np.mean(random_scores)
    # ACO should outperform average random subset
    assert aco_score >= mean_random_score
