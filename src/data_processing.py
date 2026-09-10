"""Data processing pipeline for the Iris dataset.

Loads the dataset into a pandas DataFrame, validates it, splits into
train/test, and saves CSVs to disk.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


REQUIRED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]


def load_data() -> pd.DataFrame:
    """Load the Iris dataset into a pandas DataFrame with named columns and string labels.

    Returns:
        pd.DataFrame: DataFrame with columns sepal_length, sepal_width, petal_length, petal_width, species
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    # map target numbers to names
    df["species"] = [iris.target_names[t] for t in iris.target]
    logger.info("Loaded Iris dataset with shape %s", df.shape)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate the DataFrame: not empty, required columns present, no nulls.

    Raises:
        ValueError: if validation fails
    """
    if df.empty:
        logger.error("DataFrame is empty")
        raise ValueError("DataFrame is empty")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        logger.error("Null values found in data")
        raise ValueError("Null values found in data")
    logger.info("Data validation passed")


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split.

    Returns:
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["species"],
    )
    logger.info("Split data into train=%s test=%s", train_df.shape, test_df.shape)
    return train_df, test_df


def save_csvs(base_dir: str | Path, raw_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    base = Path(base_dir)
    raw_dir = base / "data" / "raw"
    proc_dir = base / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "raw.csv"
    train_path = proc_dir / "train.csv"
    test_path = proc_dir / "test.csv"
    raw_df.to_csv(raw_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved raw to %s and processed to %s", raw_path, proc_dir)


def run_pipeline(base_dir: str | Path = ".") -> None:
    """End-to-end data pipeline entry point.

    Args:
        base_dir: base directory where `data/` will be written
    """
    logger.info("Running data pipeline with base_dir=%s", base_dir)
    df = load_data()
    validate_data(df)
    train_df, test_df = split_data(df)
    save_csvs(base_dir, df, train_df, test_df)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    run_pipeline()
