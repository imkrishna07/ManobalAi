"""Train a backend-contract-compatible model on reproducible synthetic data.

This script is for hackathon/demo use only. It generates synthetic labels from
an explicit simulation rule; its metrics are not evidence of real-world welfare
prediction performance.

Run inside the backend image (see README):
  python train_synthetic_model.py --output-dir /app/artifacts \
    --mirror-dir /app/model-artifacts-mirror
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

from lib.feature_engineering import (
    RAW_NUMERIC_FEATURES,
    STATIC_FEATURES,
    engineer_features,
    get_feature_columns,
)
from lib.preprocessing import preprocess_for_inference


SEED = 42
MODEL_VERSION = "0.3.0-synthetic-backend-compatible"
FEATURE_VERSION = "1.1.0"
BAND_ORDER = ["Low", "Moderate", "High"]
BAND_TO_INDEX = {band: index for index, band in enumerate(BAND_ORDER)}


def generate_synthetic_records(
    people_count: int = 1500,
    weeks: int = 26,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate simulated weekly raw records and intentionally synthetic labels."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    latent_scores: list[float] = []

    for person_index in range(people_count):
        personnel_id = f"synthetic-{person_index:05d}"
        service = float(rng.uniform(1, 30))
        hardship = int(rng.random() < 0.28)
        transfers = int(min(rng.poisson(1.3), 8))
        posting_years = float(rng.uniform(0.2, min(8.0, service)))
        vulnerability = float(rng.normal(0, 0.55))
        strain = float(rng.normal(0, 0.35) + 0.25 * hardship + 0.08 * transfers)

        for week in range(1, weeks + 1):
            strain = float(
                np.clip(
                    0.78 * strain + 0.18 * vulnerability + 0.10 * hardship + rng.normal(0, 0.42),
                    -2.3,
                    2.8,
                )
            )
            duty = float(np.clip(46 + 8.5 * strain + 1.5 * hardship + rng.normal(0, 7), 24, 96))
            overtime = float(np.clip(duty - 46 + rng.normal(0, 3.5), 0, 38))
            night_ratio = float(np.clip(0.18 + 0.11 * strain + rng.normal(0, 0.12), 0, 0.9))
            rest_gap = float(np.clip(5 + 2.8 * strain + 2.0 * hardship + rng.normal(0, 2), 0, 45))
            leave_gap = float(np.clip(54 + 13 * strain + 18 * hardship + rng.normal(0, 22), 0, 360))
            leave_balance = float(np.clip(30 - 2.8 * strain - 4 * hardship + rng.normal(0, 8), 0, 60))
            wellness = float(np.clip(7.4 - 1.15 * strain - 0.35 * hardship + rng.normal(0, 1.1), 0, 10))
            sleep_quality = float(np.clip(7.3 - 0.95 * strain + rng.normal(0, 1.15), 0, 10))
            resting_hr_trend = float(np.clip(0.55 * strain + 0.24 * hardship + rng.normal(0, 0.85), -3, 6))
            sleep_hours = float(np.clip(7.5 - 0.48 * strain + rng.normal(0, 0.75), 3, 10))

            # The synthetic target is defined by this simulation only. It is
            # deliberately not represented as a clinical or operational label.
            latent = (
                0.42 * strain
                + 0.20 * hardship
                + 0.055 * transfers
                + 0.12 * max(0.0, 6.5 - wellness)
                + 0.08 * max(0.0, 6.0 - sleep_quality)
                + rng.normal(0, 0.36)
            )
            latent_scores.append(float(latent))

            row: dict[str, Any] = {
                "personnel_id": personnel_id,
                "week": week,
                "years_of_service": service,
                "hardship_posting_flag": hardship,
                "transfer_count_24mo": transfers,
                "years_in_current_posting": posting_years,
                "weekly_duty_hours": duty,
                "night_shift_ratio": night_ratio,
                "overtime_hours": overtime,
                "days_since_last_rest": rest_gap,
                "days_since_last_leave": leave_gap,
                "leave_balance": leave_balance,
                "wellness_score_self_report": wellness,
                "sleep_quality_score_self_report": sleep_quality,
                "resting_hr_trend_biometric": resting_hr_trend,
                "sleep_hours_biometric": sleep_hours,
                "_latent_score": latent,
            }

            # Simulated non-response is slightly more common at higher strain.
            missing_probability = min(0.10, 0.025 + 0.012 * max(strain, 0))
            for feature in RAW_NUMERIC_FEATURES:
                if rng.random() < missing_probability:
                    row[feature] = np.nan
            rows.append(row)

    frame = pd.DataFrame(rows)
    scores = np.asarray(latent_scores)
    low_cut, high_cut = np.quantile(scores, [0.55, 0.83])
    frame["risk_label"] = np.select(
        [frame["_latent_score"] < low_cut, frame["_latent_score"] < high_cut],
        ["Low", "Moderate"],
        default="High",
    )
    return frame.drop(columns="_latent_score")


def split_personnel(frame: pd.DataFrame, seed: int = SEED) -> dict[str, set[str]]:
    """Create person-disjoint 70/15/15 train, validation, and test groups."""
    people = np.array(sorted(frame["personnel_id"].unique()))
    dummy = np.zeros((len(people), 1))
    first = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val_idx, test_idx = next(first.split(dummy, groups=people))
    train_val_people = people[train_val_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.1875, random_state=seed + 1)
    train_idx, val_idx = next(
        second.split(np.zeros((len(train_val_people), 1)), groups=train_val_people)
    )
    return {
        "train": set(train_val_people[train_idx]),
        "validation": set(train_val_people[val_idx]),
        "test": set(people[test_idx]),
    }


def engineer_split(frame: pd.DataFrame, people: set[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Engineer one person at a time to match the API's per-person inference path."""
    subset = frame[frame["personnel_id"].isin(people)].copy()
    parts: list[pd.DataFrame] = []
    for _, person_rows in subset.groupby("personnel_id", sort=False):
        parts.append(engineer_features(person_rows.drop(columns="risk_label")))
    features = pd.concat(parts, ignore_index=True)
    labels = subset[["personnel_id", "week", "risk_label"]]
    joined = features.merge(labels, on=["personnel_id", "week"], validate="one_to_one")
    x = joined[get_feature_columns()].astype(float)
    y = joined["risk_label"].map(BAND_TO_INDEX).to_numpy(dtype=int)
    return x, y


def metric_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1, 2],
        target_names=BAND_ORDER,
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = recall_score(
        y_true, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "recall_by_band": {band: float(per_class_recall[i]) for i, band in enumerate(BAND_ORDER)},
        "classification_report": report,
    }


def save_artifacts(
    output_dir: Path,
    risk_model: Any,
    baseline_model: Any,
    metrics: dict[str, Any],
    class_rates: dict[str, float],
    people_by_split: dict[str, int],
    rows_by_split: dict[str, int],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(risk_model, output_dir / "risk_model.pkl", compress=3)
    joblib.dump(baseline_model, output_dir / "baseline_model.pkl", compress=3)
    pipeline = FunctionTransformer(preprocess_for_inference, validate=False)
    joblib.dump(pipeline, output_dir / "preprocessing_pipeline.pkl", compress=3)
    metadata = {
        "product_name": "Manobal-AI",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_list_in_order": get_feature_columns(),
        "feature_count": len(get_feature_columns()),
        "band_order": BAND_ORDER,
        "decision_policy": "calibrated_probability_argmax (matches backend inference)",
        "training_data": {
            "source": "reproducibly generated synthetic records",
            "seed": seed,
            "people_by_split": people_by_split,
            "rows_by_split": rows_by_split,
            "class_rates": class_rates,
            "label_definition": "Simulated latent strain rule; not a clinical or real-personnel outcome.",
        },
        "evaluation": {
            "test_metrics": metrics,
            "limitation": "Metrics measure recovery of the synthetic simulation labels only.",
        },
        "environment": {"scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "notes": "Synthetic hackathon prototype. Not validated for real personnel or operational decisions.",
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument("--people", type=int, default=1500)
    parser.add_argument("--weeks", type=int, default=26)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    records = generate_synthetic_records(args.people, args.weeks, args.seed)
    splits = split_personnel(records, args.seed)
    split_names = list(splits)
    if any(
        splits[left] & splits[right]
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise RuntimeError("Personnel overlap detected between splits")

    engineered: dict[str, tuple[pd.DataFrame, np.ndarray]] = {
        name: engineer_split(records, people) for name, people in splits.items()
    }
    expected_order = get_feature_columns()
    if len(expected_order) != 44 or any(list(x.columns) != expected_order for x, _ in engineered.values()):
        raise RuntimeError("Generated feature order does not match the backend's 44-feature contract")

    x_train, y_train = engineered["train"]
    x_val, y_val = engineered["validation"]
    x_test, y_test = engineered["test"]

    base = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=20,
        min_child_samples=45,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=4,
        verbosity=-1,
    )
    base.fit(x_train, y_train)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(x_val, y_val)

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    baseline.fit(x_train, y_train)

    probabilities = calibrated.predict_proba(x_test)
    if probabilities.shape[1] != 3 or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise RuntimeError("Probability output failed the three-class normalization check")
    metrics = metric_summary(y_test, probabilities)
    class_rates = records[records["personnel_id"].isin(splits["train"])]["risk_label"].value_counts(normalize=True).to_dict()
    people_by_split = {name: len(people) for name, people in splits.items()}
    rows_by_split = {name: len(engineered[name][0]) for name in splits}

    save_artifacts(
        args.output_dir,
        calibrated,
        baseline,
        metrics,
        {str(k): float(v) for k, v in class_rates.items()},
        people_by_split,
        rows_by_split,
        args.seed,
    )
    if args.mirror_dir:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in ("risk_model.pkl", "baseline_model.pkl", "preprocessing_pipeline.pkl", "model_metadata.json"):
            shutil.copy2(args.output_dir / name, args.mirror_dir / name)

    print(f"Model: {MODEL_VERSION}; feature contract: {FEATURE_VERSION} ({len(expected_order)} features)")
    print(f"People by split: {people_by_split}; rows by split: {rows_by_split}")
    print("Test metrics on synthetic labels:")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
