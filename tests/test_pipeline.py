"""Pytest suite for the iris-mlops pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import pandas as pd

from src import data_processing, train, inference


@pytest.fixture(scope="module")
def iris_df() -> pd.DataFrame:
    return data_processing.load_data()


@pytest.fixture
def tmp_base(tmp_path) -> Path:
    return tmp_path


def test_load_data_shape(iris_df: pd.DataFrame):
    assert iris_df.shape == (150, 5)


def test_validate_data_pass(iris_df: pd.DataFrame):
    # should not raise
    data_processing.validate_data(iris_df)


def test_validate_data_fail_nulls():
    df = pd.DataFrame({"a": [1, None]})
    with pytest.raises(ValueError):
        data_processing.validate_data(df)


def test_split_ratio(iris_df: pd.DataFrame):
    train_df, test_df = data_processing.split_data(iris_df, test_size=0.2, random_state=0)
    assert len(train_df) + len(test_df) == 150
    assert abs(len(test_df) - 30) <= 1
    # ensure stratification roughly preserved
    species = iris_df["species"].value_counts(normalize=True)
    for s in species.index:
        prop_train = (train_df["species"] == s).mean()
        assert pytest.approx(species[s], rel=1e-1) == prop_train


def test_run_pipeline_writes_files(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    assert (tmp_base / "data" / "raw" / "raw.csv").exists()
    assert (tmp_base / "data" / "processed" / "train.csv").exists()
    assert (tmp_base / "data" / "processed" / "test.csv").exists()


def test_train_model_creates_artifacts(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    assert Path(res["model_path"]).exists()
    assert Path(res["metrics_path"]).exists()


def test_evaluation_accuracy_threshold(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    with open(res["metrics_path"]) as fh:
        metrics = json.load(fh)
    assert metrics["accuracy"] >= 0.9


def test_predict_valid(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    from joblib import load

    model = load(res["model_path"])  # type: ignore
    # pick a sample from iris data
    df = data_processing.load_data()
    sample = df.iloc[0][["sepal_length", "sepal_width", "petal_length", "petal_width"]].tolist()
    pred = inference.predict(model, sample)
    assert isinstance(pred, str)


def test_predict_invalid_length(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    from joblib import load

    model = load(res["model_path"])  # type: ignore
    with pytest.raises(ValueError):
        inference.predict(model, [1.0, 2.0, 3.0])


def test_pipeline_integration(tmp_base: Path):
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    assert float(res["accuracy"]) > 0.0


def test_load_model_api_and_missing(tmp_base: Path):
    """Verify `load_model` can load the trained model and errors on missing file."""
    data_processing.run_pipeline(tmp_base)
    res = train.train_model(tmp_base)
    # load via inference.load_model
    mdl = inference.load_model(res["model_path"])  # type: ignore
    assert hasattr(mdl, "predict")

    # missing file raises
    missing = tmp_base / "models" / "no_such_model.pkl"
    with pytest.raises(FileNotFoundError):
        inference.load_model(missing)
