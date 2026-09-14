"""Flask analytics API for the final KSI fatal-vs-non-fatal SVM model."""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from schema import ALL_FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "metadata.json"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Deserialization: load the serialized fitted pipeline once at server startup.
try:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

try:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
except FileNotFoundError:
    metadata = {"category_options": {}, "numeric_ranges": {}}


def _predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if model is None:
        raise RuntimeError("Model not loaded. Run train_model.py first.")

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[ALL_FEATURES].copy()
    X = X.replace({None: np.nan})
    predictions = model.predict(X)

    # Final SVM uses probability=False, so this is a relative decision score.
    risk_score = model.decision_function(X)

    out = pd.DataFrame(index=df.index)
    out["predicted_label"] = np.where(predictions == 1, "Fatal", "Non-Fatal")
    out["predicted_fatal"] = predictions.astype(int)
    out["risk_score"] = risk_score.astype(float)
    return out


@app.get("/")
def index():
    return render_template(
        "index.html",
        numeric_features=metadata.get("numeric_ranges", {}),
        category_options=metadata.get("category_options", {}),
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(ALL_FEATURES),
    })


@app.get("/schema")
def schema():
    return jsonify({
        "features": ALL_FEATURES,
        "category_options": metadata.get("category_options", {}),
        "numeric_ranges": metadata.get("numeric_ranges", {}),
        "risk_score_note": "SVM decision_function score; not a calibrated probability.",
    })


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict):
        return jsonify({"error": "Body must be {'features': {...}}"}), 400

    try:
        result = _predict_dataframe(pd.DataFrame([features])).iloc[0]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({
        "predicted_label": result["predicted_label"],
        "predicted_fatal": int(result["predicted_fatal"]),
        "risk_score": float(result["risk_score"]),
        "risk_score_note": "Relative SVM decision score; not a probability.",
    })


@app.post("/predict_batch")
def predict_batch():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    payload = request.get_json(silent=True) or {}
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "Body must be {'records': [{...}, ...]}"}), 400

    try:
        results = _predict_dataframe(pd.DataFrame(records))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({"predictions": results.to_dict(orient="records")})


if __name__ == "__main__":
    # Assignment requirement: local-host deployment.
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
