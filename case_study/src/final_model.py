import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from geopy.distance import geodesic
from pydantic import BaseModel, Field, validator


MODEL_PATH = Path(__file__).resolve().parents[1] / "notebook" / "fraud_detection_model.pkl"

app = FastAPI(title="Fraud Detection API", version="1.0.0")


class PredictionRequest(BaseModel):
    transactions: List[Dict[str, Any]] = Field(..., min_items=1)

    @validator("transactions")
    def _timestamp_present(cls, value: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx, item in enumerate(value):
            if "timestamp" not in item:
                raise ValueError(f"Missing 'timestamp' key for transaction index {idx}")
        return value


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]


def featurize_input(input_data: Any) -> pd.DataFrame:
    """
    Expect a DataFrame-like object with columns such as:
    ['txn_id','timestamp','payer_upi','payee_upi','payer_device_id','payee_device_id',
     'payer_city','payee_city','payer_lat','payer_lon','payee_lat','payee_lon','amount',
     'txn_type','channel','status', ...]
    Returns a DataFrame of features ready for model.predict.
    """
    df = pd.DataFrame(input_data).copy()

    # Basic sanity checks
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Amount transform (handle missing)
    if "amount" in df.columns:
        df["log_amount"] = np.log1p(df["amount"].fillna(0))
    else:
        df["log_amount"] = 0.0

    # Distance feature (handle missing lat/lon)
    def calculate_distance(row):
        try:
            payer_coords = (float(row["payer_lat"]), float(row["payer_lon"]))
            payee_coords = (float(row["payee_lat"]), float(row["payee_lon"]))
            return geodesic(payer_coords, payee_coords).km
        except Exception:
            return np.nan

    if {"payer_lat", "payer_lon", "payee_lat", "payee_lon"}.issubset(df.columns):
        df["distance"] = df.apply(calculate_distance, axis=1)
    else:
        df["distance"] = np.nan

    # Device counts per upi (nunique)
    if "payer_upi" in df.columns and "payer_device_id" in df.columns:
        df["payer_device_count"] = (
            df.groupby("payer_upi")["payer_device_id"].transform("nunique").fillna(0)
        )
    else:
        df["payer_device_count"] = 0

    if "payee_upi" in df.columns and "payee_device_id" in df.columns:
        df["payee_device_count"] = (
            df.groupby("payee_upi")["payee_device_id"].transform("nunique").fillna(0)
        )
    else:
        df["payee_device_count"] = 0

    # Sort by timestamp for temporal features
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Transactions in "last hour" heuristic per payer/payee
    def last_hour_counts(ts_series: pd.Series) -> pd.Series:
        # Conservative heuristic: counts cumulative occurrences of diffs < 1 hour
        diffs = ts_series.diff().dt.total_seconds()
        recent_flags = diffs.lt(3600).fillna(False)
        return recent_flags.cumsum().fillna(0).astype(int)

    if "payer_upi" in df.columns and "timestamp" in df.columns:
        df["payer_txn_last_hour"] = df.groupby("payer_upi")["timestamp"].transform(
            last_hour_counts
        )
    else:
        df["payer_txn_last_hour"] = 0

    if "payee_upi" in df.columns and "timestamp" in df.columns:
        df["payee_txn_last_hour"] = df.groupby("payee_upi")["timestamp"].transform(
            last_hour_counts
        )
    else:
        df["payee_txn_last_hour"] = 0

    # Drop columns not used by model (only drop if they exist)
    drop_cols = [
        "txn_id",
        "timestamp",
        "payer_upi",
        "payee_upi",
        "payer_device_id",
        "payee_device_id",
        "payer_city",
        "payee_city",
        "payer_lat",
        "payer_lon",
        "payee_lat",
        "payee_lon",
        "amount",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # One-hot encode categorical features if present
    categorical_features = ["txn_type", "channel", "status", "hour", "day_of_week", "month"]
    categorical_present = [c for c in categorical_features if c in df.columns]
    if categorical_present:
        df = pd.get_dummies(df, columns=categorical_present, drop_first=True)

    # Fill remaining NaNs with 0 (models generally expect numeric input)
    df = df.fillna(0)

    return df


@lru_cache()
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    return model


@app.on_event("startup")
def _load_on_startup():
    # Ensure the model loads when the service starts so cold requests are faster.
    try:
        load_model()
    except Exception as exc:
        raise RuntimeError(f"Failed to load model during startup: {exc}") from exc


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = featurize_input(request.transactions)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = load_model()
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        features = features.reindex(columns=feature_names, fill_value=0)

    if len(features) == 0:
        raise HTTPException(status_code=400, detail="No usable transactions provided")

    predictions = model.predict(features).astype(int).tolist()
    probabilities: List[float]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1].astype(float).tolist()
    else:
        probabilities = [float(p) for p in predictions]

    return PredictionResponse(predictions=predictions, probabilities=probabilities)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("case_study.src.final_model:app", host="0.0.0.0", port=8000, reload=True)
