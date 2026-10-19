import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Optional
from sklearn.metrics import roc_auc_score

class AntColonyOptimizer:
    """
    Max-Min Ant Colony Optimization (MMAS) for binary combinatorial search spaces.
    Applicable to feature selection, configuration pruning, and model component selection.
    """
    def __init__(
        self,
        n_ants: int = 15,
        n_iterations: int = 10,
        evaporation_rate: float = 0.15,
        alpha: float = 1.0,
        beta: float = 1.0,
        q0: float = 0.7,
        tau_min: float = 0.05,
        tau_max: float = 5.0,
        random_state: Optional[int] = None
    ):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.history: List[float] = []
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        n_dimensions: int,
        heuristic_visibility: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Executes ACO search.
        
        Args:
            fitness_fn: Objective function returning float fitness (higher is better).
            n_dimensions: Number of discrete decision variables.
            heuristic_visibility: Optional domain heuristic vector eta of length n_dimensions.
        """
        if heuristic_visibility is None:
            eta = np.ones(n_dimensions, dtype=float)
        else:
            eta = np.clip(heuristic_visibility, 1e-4, 10.0)

        # Initialize pheromone trail
        tau = np.full(n_dimensions, self.tau_max, dtype=float)
        
        self.history = []
        self.best_solution = None
        self.best_fitness = -np.inf

        for it in range(self.n_iterations):
            ants_solutions = []
            ants_fitness = []

            # Compute transition probabilities for each dimension
            p_select = (tau ** self.alpha) * (eta ** self.beta)
            p_select = p_select / (p_select + 1.0) # Sigmoidal / relative choice against baseline 1.0

            for ant in range(self.n_ants):
                # Pseudo-random proportional rule
                solution = np.zeros(n_dimensions, dtype=int)
                for d in range(n_dimensions):
                    q = self.rng.rand()
                    if q <= self.q0:
                        # Exploit
                        solution[d] = 1 if p_select[d] >= 0.5 else 0
                    else:
                        # Explore
                        solution[d] = 1 if self.rng.rand() < p_select[d] else 0

                # Ensure at least 1 dimension selected
                if np.sum(solution) == 0:
                    solution[self.rng.randint(n_dimensions)] = 1

                fit = fitness_fn(solution)
                ants_solutions.append(solution)
                ants_fitness.append(fit)

                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = solution.copy()

            self.history.append(self.best_fitness)

            # Evaporation
            tau = (1.0 - self.evaporation_rate) * tau

            # Reinforcement from best iteration ant
            best_iter_idx = int(np.argmax(ants_fitness))
            best_iter_sol = ants_solutions[best_iter_idx]
            deposit = self.evaporation_rate * max(0.1, (ants_fitness[best_iter_idx] + 1e-3))
            tau += deposit * best_iter_sol

            # Enforce Max-Min bounds
            tau = np.clip(tau, self.tau_min, self.tau_max)

        return self.best_solution, self.best_fitness, self.history


class ACOFeatureSelector:
    """
    Ant Colony Optimization Feature Subset Selector.
    Searches for optimal feature subset maximizing cross-validated classification performance.
    """
    def __init__(
        self,
        n_ants: int = 12,
        n_iterations: int = 8,
        evaporation_rate: float = 0.2,
        random_state: int = 42
    ):
        self.aco = AntColonyOptimizer(
            n_ants=n_ants,
            n_iterations=n_iterations,
            evaporation_rate=evaporation_rate,
            random_state=random_state
        )
        self.selected_features_: List[str] = []
        self.feature_names_: List[str] = []
        self.best_score_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        self.feature_names_ = list(X.columns)
        n_feats = len(self.feature_names_)
        
        # Calculate univariate correlation as heuristic visibility
        heuristic = np.ones(n_feats, dtype=float)
        for i, col in enumerate(self.feature_names_):
            try:
                vals = X[col].values
                # Standardize
                std = np.std(vals)
                if std > 1e-6:
                    corr = np.abs(np.corrcoef(vals, y)[0, 1])
                    heuristic[i] = 1.0 + (5.0 * corr if not np.isnan(corr) else 0.0)
            except Exception:
                heuristic[i] = 1.0

        # Fast evaluator: simple linear/logistic or tree boundary
        X_mat = X.values
        y_arr = np.asarray(y)
        
        # 3-fold split for fast subset scoring
        split_idx = int(len(y_arr) * 0.7)
        X_tr, X_val = X_mat[:split_idx], X_mat[split_idx:]
        y_tr, y_val = y_arr[:split_idx], y_arr[split_idx:]

        def evaluate_subset(mask: np.ndarray) -> float:
            selected_indices = np.where(mask == 1)[0]
            if len(selected_indices) == 0:
                return -1.0
            
            # Simple ridge / Naive Bayes / correlation proxy for high efficiency
            sub_tr = X_tr[:, selected_indices]
            sub_val = X_val[:, selected_indices]
            
            # Normalize and evaluate with fast centroid / linear score
            means_pos = np.mean(sub_tr[y_tr == 1], axis=0) if np.any(y_tr == 1) else np.zeros(len(selected_indices))
            means_neg = np.mean(sub_tr[y_tr == 0], axis=0) if np.any(y_tr == 0) else np.zeros(len(selected_indices))
            diff = means_pos - means_neg
            
            proj_val = np.dot(sub_val, diff)
            try:
                score = roc_auc_score(y_val, proj_val)
            except Exception:
                score = 0.5
            
            # Parsimony penalty: slightly reward smaller subsets with equivalent performance
            parsimony = 0.01 * (1.0 - (len(selected_indices) / n_feats))
            return float(score + parsimony)

        best_mask, best_fit, _ = self.aco.optimize(
            fitness_fn=evaluate_subset,
            n_dimensions=n_feats,
            heuristic_visibility=heuristic
        )

        self.best_score_ = best_fit
        self.selected_features_ = [
            self.feature_names_[i] for i in range(n_feats) if best_mask[i] == 1
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]


class ACOEnsembleOptimizer:
    """
    Ant Colony Optimization for Ensemble Model Weight Blending.
    Discretizes probability simplex and searches for optimal ensemble mixture weights.
    """
    def __init__(self, n_ants: int = 15, n_iterations: int = 10, random_state: int = 42):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def optimize_weights(
        self,
        predictions: List[np.ndarray],
        y_true: np.ndarray,
        metric: str = "auc"
    ) -> Tuple[np.ndarray, float]:
        """
        Searches for convex combination weights w_i >= 0, sum(w_i) = 1.
        """
        M = len(predictions)
        pred_mat = np.column_stack(predictions) # (N, M)
        
        best_weights = np.ones(M) / M
        best_score = -np.inf

        # Pheromone per model representing desirability of high contribution
        tau = np.ones(M)
        
        for it in range(self.n_iterations):
            for ant in range(self.n_ants):
                # Sample Dirichlet-like candidate via gamma / softmax on pheromones
                noise = self.rng.exponential(scale=1.0, size=M)
                raw_w = (tau ** 1.2) * noise
                weights = raw_w / np.sum(raw_w)
                
                blend = np.dot(pred_mat, weights)
                if metric == "auc":
                    try:
                        auc = roc_auc_score(y_true, blend)
                    except Exception:
                        auc = 0.5
                    # Calibration regularizer: penalize Brier error as tiebreaker
                    brier = np.mean((blend - y_true) ** 2)
                    score = float(auc - 0.05 * brier)
                else:
                    # negative MSE / Brier score
                    score = -float(np.mean((blend - y_true) ** 2))

                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                    
            # Reinforce pheromones on models with higher weights in best blend
            tau = 0.85 * tau + 0.15 * (best_weights * M)

        return best_weights, best_score
