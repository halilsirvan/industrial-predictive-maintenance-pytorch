"""Train, calibrate, evaluate, and persist the predictive-maintenance model."""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import build_preprocessor, load_dataset, split_features_target
from src.model import FailureMLP


SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_tensor(values: np.ndarray | pd.DataFrame | pd.Series) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    return model(to_tensor(values)).cpu().numpy()


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold that maximizes F1 for the rare class."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def metric_summary(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def plot_history(history: list[dict[str, float]], destination: Path) -> None:
    frame = pd.DataFrame(history)
    fig, left = plt.subplots(figsize=(8, 4.5))
    left.plot(frame["epoch"], frame["train_loss"], label="Training loss", color="#2563eb")
    left.set_xlabel("Epoch")
    left.set_ylabel("Weighted BCE loss")
    right = left.twinx()
    right.plot(frame["epoch"], frame["val_pr_auc"], label="Validation PR-AUC", color="#ef4444")
    right.set_ylabel("Validation PR-AUC")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="center right")
    left.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(y_true: np.ndarray, probabilities: np.ndarray, destination: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(recall, precision, color="#7c3aed", linewidth=2)
    axes[0].axhline(y_true.mean(), linestyle="--", color="#64748b", label="Prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve")
    axes[0].legend()
    axes[1].plot(fpr, tpr, color="#0891b2", linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#64748b")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, probabilities: np.ndarray, destination: Path) -> None:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", color="#059669", linewidth=2, label="PyTorch + Platt")
    axis.set(xlabel="Predicted probability", ylabel="Observed failure rate", title="Calibration curve")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gradient_importance(
    model: FailureMLP,
    x_test: np.ndarray,
    feature_names: list[str],
    destination: Path,
) -> list[dict[str, float | str]]:
    model.eval()
    inputs = to_tensor(x_test).requires_grad_(True)
    logits = model(inputs)
    logits.mean().backward()
    scores = np.mean(np.abs(inputs.grad.detach().cpu().numpy()), axis=0)
    order = np.argsort(scores)[::-1]
    records = [
        {"feature": feature_names[index], "mean_abs_gradient": float(scores[index])}
        for index in order
    ]
    top = records[:10][::-1]
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.barh(
        [item["feature"] for item in top],
        [item["mean_abs_gradient"] for item in top],
        color="#0f766e",
    )
    axis.set(xlabel="Mean absolute input gradient", title="Global model sensitivity (test set)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return records


def train(args: argparse.Namespace) -> None:
    set_seed()
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / args.data
    artifact_dir = project_root / args.artifacts
    asset_dir = project_root / args.assets
    artifact_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(data_path)
    features, target = split_features_target(frame)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=0.15,
        stratify=target,
        random_state=SEED,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.1764705882,
        stratify=y_train_full,
        random_state=SEED,
    )

    preprocessor = build_preprocessor()
    train_values = preprocessor.fit_transform(x_train).astype("float32")
    val_values = preprocessor.transform(x_val).astype("float32")
    test_values = preprocessor.transform(x_test).astype("float32")
    feature_names = preprocessor.get_feature_names_out().tolist()

    train_dataset = TensorDataset(to_tensor(train_values), to_tensor(y_train.values))
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    model = FailureMLP(input_dim=train_values.shape[1], dropout=args.dropout)
    negative, positive = np.bincount(y_train.values)
    positive_weight = torch.tensor([negative / positive], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_state = deepcopy(model.state_dict())
    best_score = -np.inf
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

        val_logits = predict_logits(model, val_values)
        val_probabilities = 1.0 / (1.0 + np.exp(-val_logits))
        val_pr_auc = average_precision_score(y_val, val_probabilities)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / len(train_dataset),
                "val_pr_auc": float(val_pr_auc),
            }
        )
        if val_pr_auc > best_score + 1e-4:
            best_score = val_pr_auc
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    model.load_state_dict(best_state)
    val_logits = predict_logits(model, val_values).reshape(-1, 1)
    test_logits = predict_logits(model, test_values).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, random_state=SEED)
    calibrator.fit(val_logits, y_val.values)
    val_probabilities = calibrator.predict_proba(val_logits)[:, 1]
    test_probabilities = calibrator.predict_proba(test_logits)[:, 1]
    threshold = choose_threshold(y_val.values, val_probabilities)

    metrics = metric_summary(y_test.values, test_probabilities, threshold)
    metrics.update(
        {
            "dataset_rows": int(len(frame)),
            "test_rows": int(len(y_test)),
            "test_failures": int(y_test.sum()),
            "failure_prevalence": float(target.mean()),
            "epochs_trained": int(len(history)),
            "validation_pr_auc_best": float(best_score),
            "decision_rule": "Threshold selected on validation data to maximize failure-class F1.",
        }
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(train_values.shape[1]),
            "dropout": float(args.dropout),
        },
        artifact_dir / "failure_mlp.pt",
    )
    joblib.dump(preprocessor, artifact_dir / "preprocessor.joblib")
    joblib.dump(calibrator, artifact_dir / "calibrator.joblib")
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {"threshold": threshold, "feature_names": feature_names, "metrics": metrics},
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(history).to_csv(asset_dir / "training_history.csv", index=False)
    plot_history(history, asset_dir / "training_history.png")
    plot_curves(y_test.values, test_probabilities, asset_dir / "evaluation_curves.png")
    plot_calibration(y_test.values, test_probabilities, asset_dir / "calibration_curve.png")
    importance = plot_gradient_importance(
        model, test_values, feature_names, asset_dir / "feature_importance.png"
    )
    pd.DataFrame(importance).to_csv(asset_dir / "feature_importance.csv", index=False)
    (asset_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(
        y_test.values,
        (test_probabilities >= threshold).astype(int),
        target_names=["normal", "failure"],
        digits=4,
    )
    (asset_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/ai4i2020.csv")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

