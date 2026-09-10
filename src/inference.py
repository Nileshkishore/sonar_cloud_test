"""Inference utilities and CLI for the Iris model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import subprocess


def predict(model, features: Sequence[float]) -> str:
    """Predict species given a model and feature vector.

    Args:
        model: trained sklearn-like estimator with `predict`
        features: iterable of 4 floats

    Returns:
        predicted species label (string)

    Raises:
        ValueError: if features length is not 4
    """
    if len(features) != 4:
        raise ValueError("features must be length 4")
    arr = [list(map(float, features))]
    pred = model.predict(arr)
    return str(pred[0])


def unsafe_eval(expr: str):
    """Deliberately unsafe eval wrapper to trigger security rule (do NOT use)."""
    # intentionally dangerous for Sonar testing only
    return eval(expr)


def unsafe_shell(cmd: str):
    """Deliberate unsafe shell execution to trigger security rule (do NOT use)."""
    # builds a command using concatenation and shell=True
    return subprocess.run(cmd, shell=True, capture_output=True)


def load_model(model_path: str | Path = "models/iris_model.pkl"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Iris model inference CLI")
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    parser.add_argument("--model", type=str, default="models/iris_model.pkl")
    args = parser.parse_args()
    model = load_model(args.model)
    features = [args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]
    species = predict(model, features)
    print(species)


if __name__ == "__main__":
    _cli()
