"""Data loading, leakage control, feature engineering, and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temperature gap [K]",
    "Mechanical power [W]",
    "Strain proxy [Nm min]",
]
CATEGORICAL_FEATURES = ["Type"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the UCI AI4I CSV and validate the expected schema."""
    frame = pd.read_csv(path)
    required = set(RAW_FEATURES + [TARGET] + ID_COLUMNS + LEAKAGE_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create physically meaningful, label-independent interaction features."""
    features = frame[RAW_FEATURES].copy()
    features["Temperature gap [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    features["Mechanical power [W]"] = (
        features["Torque [Nm]"]
        * features["Rotational speed [rpm]"]
        * 2.0
        * np.pi
        / 60.0
    )
    features["Strain proxy [Nm min]"] = (
        features["Torque [Nm]"] * features["Tool wear [min]"]
    )
    return features


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and target while explicitly excluding leakage columns."""
    return engineer_features(frame), frame[TARGET].astype("int64")


def build_preprocessor() -> ColumnTransformer:
    """Standardize numeric values and one-hot encode product quality type."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

