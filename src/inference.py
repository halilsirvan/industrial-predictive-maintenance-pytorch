"""Reusable local inference API for the trained model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import engineer_features
from src.model import FailureMLP


class FailurePredictor:
    """Load preprocessing, PyTorch, calibration, and threshold artifacts once."""

    def __init__(self, artifact_dir: str | Path) -> None:
        root = Path(artifact_dir)
        checkpoint = torch.load(root / "failure_mlp.pt", map_location="cpu", weights_only=True)
        self.model = FailureMLP(checkpoint["input_dim"], checkpoint["dropout"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.preprocessor = joblib.load(root / "preprocessor.joblib")
        self.calibrator = joblib.load(root / "calibrator.joblib")
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    def predict(self, raw_inputs: pd.DataFrame) -> pd.DataFrame:
        engineered = engineer_features(raw_inputs)
        values = self.preprocessor.transform(engineered).astype("float32")
        with torch.no_grad():
            logits = self.model(torch.as_tensor(values)).numpy().reshape(-1, 1)
        probabilities = self.calibrator.predict_proba(logits)[:, 1]
        threshold = float(self.metadata["threshold"])
        return pd.DataFrame(
            {
                "failure_probability": probabilities,
                "maintenance_alert": probabilities >= threshold,
                "decision_threshold": threshold,
            }
        )

