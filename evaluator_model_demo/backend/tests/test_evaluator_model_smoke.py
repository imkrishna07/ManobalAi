"""Evaluator smoke test for the bundled synthetic model and backend contract."""

import math
from lib.inference import InferenceEngine
from lib.feature_engineering import get_feature_columns
from train_synthetic_model import generate_synthetic_records, MODEL_VERSION


def test_bundled_model_loads_and_predicts_from_raw_weekly_records():
    # Uses MODEL_ARTIFACT_DIR exactly as the running backend does in Compose.
    engine = InferenceEngine()
    engine.load()

    records = generate_synthetic_records(people_count=1, weeks=4, seed=42)
    records = records.drop(columns="risk_label").to_dict(orient="records")
    result, features = engine.predict_from_raw_records(records)

    assert engine.metadata["model_version"] == MODEL_VERSION
    assert engine.metadata["feature_version"] == "1.1.0"
    assert len(get_feature_columns()) == len(features) == 44
    assert list(features) == get_feature_columns()
    assert result["predicted_band"] in {"Low", "Moderate", "High"}
    assert result["model_version"] == MODEL_VERSION
    probabilities = result["class_probabilities"]
    assert len(probabilities) == 3
    assert math.isclose(sum(item["probability"] for item in probabilities), 1.0, abs_tol=1e-6)
    assert all(math.isfinite(item["probability"]) for item in probabilities)
