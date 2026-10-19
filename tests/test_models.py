import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.engineer import FeatureEngineer
from src.models.evaluator import ModelEvaluator
from src.models.baseline import BaselineTrainer

@pytest.fixture
def mock_airline_data():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "id": range(n),
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "Customer Type": np.random.choice(["Loyal Customer", "disloyal Customer"], size=n),
        "Age": np.random.randint(18, 70, size=n),
        "Type of Travel": np.random.choice(["Business travel", "Personal Travel"], size=n),
        "Class": np.random.choice(["Business", "Eco", "Eco Plus"], size=n),
        "Flight Distance": np.random.randint(100, 3000, size=n),
        "Inflight wifi service": np.random.randint(0, 6, size=n),
        "Departure/Arrival time convenient": np.random.randint(0, 6, size=n),
        "Ease of Online booking": np.random.randint(0, 6, size=n),
        "Gate location": np.random.randint(0, 6, size=n),
        "Food and drink": np.random.randint(0, 6, size=n),
        "Online boarding": np.random.randint(0, 6, size=n),
        "Seat comfort": np.random.randint(0, 6, size=n),
        "Inflight entertainment": np.random.randint(0, 6, size=n),
        "On-board service": np.random.randint(0, 6, size=n),
        "Leg room service": np.random.randint(0, 6, size=n),
        "Baggage handling": np.random.randint(0, 6, size=n),
        "Checkin service": np.random.randint(0, 6, size=n),
        "Inflight service": np.random.randint(0, 6, size=n),
        "Cleanliness": np.random.randint(0, 6, size=n),
        "Departure Delay in Minutes": np.random.choice([0, 10, 30], size=n),
        "Arrival Delay in Minutes": [np.nan if i % 15 == 0 else float(i % 20) for i in range(n)],
        "satisfaction": np.random.choice(["satisfied", "neutral or dissatisfied"], size=n)
    })
    return df

def test_feature_engineering_pipeline(mock_airline_data):
    engineer = FeatureEngineer()
    train_df = mock_airline_data.iloc[:100]
    test_df = mock_airline_data.iloc[100:].drop(columns=["satisfaction"])
    
    X_train, y_train = engineer.fit_transform(train_df, target_col="satisfaction")
    X_test = engineer.transform(test_df)
    
    assert X_train.isnull().sum().sum() == 0, "X_train contains unhandled NaNs"
    assert X_test.isnull().sum().sum() == 0, "X_test contains unhandled NaNs"
    assert "total_delay" in X_train.columns
    assert "mean_service_rating" in X_train.columns
    assert len(y_train) == 100
    assert set(np.unique(y_train)).issubset({0, 1})

def test_model_evaluator():
    evaluator = ModelEvaluator()
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7])
    
    auc = evaluator.compute_auc(y_true, y_pred_prob)
    assert np.isclose(auc, 1.0)
    
    metrics = evaluator.compute_classification_metrics(y_true, y_pred_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

def test_baseline_trainer_cv(mock_airline_data):
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(mock_airline_data, target_col="satisfaction")
    
    trainer = BaselineTrainer(n_splits=3, random_state=42)
    results = trainer.train_cv(X, y)
    
    assert "oof_predictions" in results
    assert "mean_auc" in results
    assert len(results["oof_predictions"]) == len(mock_airline_data)
    assert 0.4 <= results["mean_auc"] <= 1.0
    assert len(results["models"]) == 3
