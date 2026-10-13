from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import (
    LEAKAGE_COLUMNS,
    TARGET,
    build_preprocessor,
    engineer_features,
    load_dataset,
    split_features_target,
)
from src.model import FailureMLP


ROOT = Path(__file__).resolve().parents[1]


def test_feature_engineering_blocks_leakage() -> None:
    frame = load_dataset(ROOT / "data" / "ai4i2020.csv")
    features, target = split_features_target(frame.head(100))
    assert TARGET not in features.columns
    assert not set(LEAKAGE_COLUMNS).intersection(features.columns)
    assert len(features) == len(target) == 100


def test_engineered_features_have_expected_values() -> None:
    raw = pd.DataFrame(
        [
            {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
            }
        ]
    )
    features = engineer_features(raw)
    assert features.loc[0, "Temperature gap [K]"] == 10.0
    assert np.isclose(features.loc[0, "Mechanical power [W]"], 6283.185307)
    assert features.loc[0, "Strain proxy [Nm min]"] == 4000.0


def test_preprocessor_and_network_shapes() -> None:
    frame = load_dataset(ROOT / "data" / "ai4i2020.csv").head(250)
    features, _ = split_features_target(frame)
    values = build_preprocessor().fit_transform(features).astype("float32")
    model = FailureMLP(values.shape[1])
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(values[:8]))
    assert logits.shape == (8,)

