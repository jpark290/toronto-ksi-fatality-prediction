"""Client that validates the API with the untouched held-out test set."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from schema import ALL_FEATURES, TARGET_COL

BASE_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:5000"
TEST_PATH = BASE_DIR / "test_data.csv"


def json_safe(record: dict) -> dict:
    return {
        k: (None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v))
        for k, v in record.items()
    }


def main() -> None:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    print("Health:", health.json())
    if not health.json().get("model_loaded"):
        sys.exit("Model is not loaded. Run train_model.py, then app.py.")

    test_df = pd.read_csv(TEST_PATH)
    print(f"Held-out test rows: {len(test_df)}")

    # Single-record API demonstration.
    one = json_safe(test_df.iloc[0][ALL_FEATURES].to_dict())
    resp = requests.post(f"{API_URL}/predict", json={"features": one}, timeout=10)
    resp.raise_for_status()
    print("Single prediction:", resp.json())

    # Full held-out evaluation through the actual HTTP API.
    records = [json_safe(r) for r in test_df[ALL_FEATURES].to_dict(orient="records")]
    resp = requests.post(f"{API_URL}/predict_batch", json={"records": records}, timeout=120)
    resp.raise_for_status()
    predictions = resp.json()["predictions"]

    y_true = test_df[TARGET_COL].astype(int).to_numpy()
    y_pred = np.array([p["predicted_fatal"] for p in predictions], dtype=int)
    scores = np.array([p["risk_score"] for p in predictions], dtype=float)

    print("\nClassification report from API responses:")
    print(classification_report(y_true, y_pred, target_names=["Non-Fatal", "Fatal"], zero_division=0))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_true, y_pred))
    print(f"ROC-AUC from API risk scores: {roc_auc_score(y_true, scores):.4f}")


if __name__ == "__main__":
    main()
