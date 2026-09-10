"""Training script for Iris RandomForest model."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import mlflow

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_model(base_dir: str | Path = ".", random_state: int = 42) -> dict[str, Any]:
    base = Path(base_dir)
    proc_dir = base / "data" / "processed"
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = proc_dir / "train.csv"
    test_path = proc_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed train/test CSVs not found. Run the data pipeline first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train["species"]
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test["species"]

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)

    model_path = models_dir / "iris_model.pkl"
    joblib.dump(clf, model_path)
    metrics = {"accuracy": acc, "classification_report": report}
    metrics_path = base / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    logger.info("Saved model to %s and metrics to %s", model_path, metrics_path)
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "accuracy": acc}


if __name__ == "__main__":
    train_model()
