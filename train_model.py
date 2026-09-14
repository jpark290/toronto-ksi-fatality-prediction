"""Train and serialize the final KSI model selected in the notebook.

Run this OFFLINE with the same KSI.csv used by the notebook. It recreates the
notebook's 27-feature final model recipe and writes:
  model.pkl      fitted preprocessing -> SMOTE -> linear SVM pipeline
  metadata.json  UI/API schema metadata learned from X_train only
  test_data.csv  untouched 20% held-out split for API testing
"""

from pathlib import Path
import json
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from schema import (
    ALL_FEATURES, BEST_SVM_PARAMS, BINARY_FEATURES, CATEGORICAL_FEATURES,
    MODE_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET_COL, UNKNOWN_FEATURES,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "KSI.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "metadata.json"
TEST_PATH = BASE_DIR / "test_data.csv"


def load_and_prepare_collision_df(csv_path: Path) -> pd.DataFrame:
    """Mirror the notebook's cleaning, collision aggregation and time features."""
    df = pd.read_csv(csv_path)

    df["ACCLASS_CLEAN"] = df["ACCLASS"].astype(str).str.strip()
    df_clean = df[df["ACCLASS_CLEAN"].isin(["Fatal", "Non-Fatal Injury"])].copy()
    df_clean[TARGET_COL] = (df_clean["ACCLASS_CLEAN"] == "Fatal").astype(int)

    # Recover missing collision identifiers the same way as the notebook workflow.
    missing_mask = df_clean["ACCNUM"].isna()
    key_cols = ["DATE", "TIME", "STREET1", "STREET2"]
    recovery_key = df_clean.loc[missing_mask, key_cols].fillna("UNKNOWN")
    recovered_ids = recovery_key.groupby(key_cols).ngroup().astype(str).radd("recovered_")
    df_clean["ACCNUM"] = df_clean["ACCNUM"].astype("object")
    df_clean.loc[missing_mask, "ACCNUM"] = recovered_ids.values

    collision_features = [
        "DATE", "TIME", "ROAD_CLASS", "DISTRICT", "LATITUDE", "LONGITUDE",
        "ACCLOC", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "INITDIR",
        "DIVISION", "HOOD_158", "NEIGHBOURHOOD_158", "SPEEDING", "AG_DRIV",
        "REDLIGHT", "ALCOHOL", "DISABILITY",
    ]
    party_features = [
        "PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE", "TRUCK",
        "TRSN_CITY_VEH", "PASSENGER",
    ]

    collision_target = df_clean.groupby("ACCNUM")[TARGET_COL].max()
    collision_df = df_clean.groupby("ACCNUM")[collision_features].first()
    party_df = df_clean.groupby("ACCNUM")[party_features].max()
    collision_df = pd.concat([collision_df, party_df, collision_target], axis=1)

    # Notebook keeps this as a string during EDA, then drops it before modelling.
    collision_df["HOOD_158"] = collision_df["HOOD_158"].astype("string")

    collision_df["DATE"] = pd.to_datetime(collision_df["DATE"])
    collision_df["YEAR"] = collision_df["DATE"].dt.year
    collision_df["MONTH"] = collision_df["DATE"].dt.month
    collision_df["DAY_OF_WEEK"] = collision_df["DATE"].dt.dayofweek
    collision_df["HOUR"] = collision_df["TIME"] // 100
    collision_df.drop(columns=["DATE", "TIME"], inplace=True)

    for col in collision_df.select_dtypes(include="object").columns:
        collision_df[col] = collision_df[col].str.strip()

    return collision_df


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="No")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    mode_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    unknown_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("binary", binary_transformer, BINARY_FEATURES),
        ("mode", mode_transformer, MODE_FEATURES),
        ("unknown", unknown_transformer, UNKNOWN_FEATURES),
    ])


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CSV_PATH.name}. Put the same raw KSI.csv used by the notebook in {BASE_DIR}."
        )

    collision_df = load_and_prepare_collision_df(CSV_PATH)

    # Exact notebook feature-selection decision: drop both neighbourhood fields.
    model_df = collision_df.drop(columns=["HOOD_158", "NEIGHBOURHOOD_158"]).copy()
    X = model_df[ALL_FEATURES]
    y = model_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    final_pipeline = ImbPipeline([
        ("preprocessing", build_preprocessor()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", SVC(random_state=RANDOM_STATE, **BEST_SVM_PARAMS)),
    ])
    final_pipeline.fit(X_train, y_train)

    preds = final_pipeline.predict(X_test)
    scores = final_pipeline.decision_function(X_test)
    print(f"Held-out Fatal F1:     {f1_score(y_test, preds):.4f}")
    print(f"Held-out Fatal Recall: {recall_score(y_test, preds):.4f}")
    print(f"Held-out ROC-AUC:      {roc_auc_score(y_test, scores):.4f}")

    # Serialization: save the entire fitted end-to-end pipeline.
    with MODEL_PATH.open("wb") as f:
        pickle.dump(final_pipeline, f)

    def uniques(col: str):
        return sorted(str(x) for x in X_train[col].dropna().unique().tolist())

    metadata = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        # Explicit No/Yes gives the UI both valid states even when source absence is NaN.
        "category_options": {
            **{c: ["No", "Yes"] for c in BINARY_FEATURES},
            **{c: uniques(c) for c in MODE_FEATURES + UNKNOWN_FEATURES},
        },
        "numeric_ranges": {
            c: [float(X_train[c].min(skipna=True)), float(X_train[c].max(skipna=True))]
            for c in NUMERIC_FEATURES
        },
        "model": {"type": "SVC", "kernel": "linear", "C": 10.0},
        "feature_count": len(ALL_FEATURES),
        "risk_score_note": "SVM decision_function score; not a calibrated probability.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    test_out = X_test.copy()
    test_out[TARGET_COL] = y_test.values
    test_out.to_csv(TEST_PATH, index=False)

    print(f"Saved: {MODEL_PATH.name}, {METADATA_PATH.name}, {TEST_PATH.name}")
    print(f"Model input feature count: {len(ALL_FEATURES)}")


if __name__ == "__main__":
    main()
