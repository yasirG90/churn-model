from __future__ import annotations

"""
FastAPI inference endpoint for the trained churn model.

Design goals:
- Keep API intentionally simple for interview demos.
- Validate inputs clearly and return explainable outputs.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException


app = FastAPI(title="Churn Model API", version="1.0.0")

# Global state loaded once at startup.
MODEL = None
METADATA: dict[str, Any] | None = None



def get_project_root() -> Path:
    """Resolve project root from this file's location."""
    return Path(__file__).resolve().parents[1]



def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load model pipeline and metadata from the models directory."""
    project_root = get_project_root()
    model_path = project_root / "models" / "churn_model.joblib"
    metadata_path = project_root / "models" / "model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run src/train_churn_model.py first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata



def align_features(
    records: list[dict[str, Any]],
    feature_columns: list[str],
    numeric_features: list[str],
) -> pd.DataFrame:
    """
    Convert incoming records to DataFrame and align to training feature schema.

    Notes:
    - Missing columns are added as None (handled by imputers in the pipeline).
    - Extra columns are ignored.
    """
    if not records:
        raise ValueError("Request must include at least one record.")

    df = pd.DataFrame(records)

    # Add missing required columns.
    for column in feature_columns:
        if column not in df.columns:
            df[column] = None

    # Keep only expected feature order.
    df = df[feature_columns]

    # Normalize blank strings and coerce numeric fields.
    df = df.replace(" ", None)
    for column in numeric_features:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@app.on_event("startup")
def startup_event() -> None:
    """Load model artifacts when API starts."""
    global MODEL, METADATA
    MODEL, METADATA = load_artifacts()


@app.get("/health")
def health() -> dict[str, Any]:
    """Simple health endpoint for deployment checks."""
    if MODEL is None or METADATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    return {
        "status": "ok",
        "model": METADATA.get("selected_model"),
        "threshold": METADATA.get("threshold"),
    }


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Predict churn for one or more records.

    Expected payload:
    {
      "records": [ { ...feature values... }, { ... } ],
      "return_probabilities": true
    }
    """
    if MODEL is None or METADATA is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    records = payload.get("records", [])
    return_probabilities = bool(payload.get("return_probabilities", True))

    try:
        features = align_features(
            records,
            METADATA["feature_columns"],
            METADATA.get("numeric_features", []),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probabilities = MODEL.predict_proba(features)[:, 1]
    threshold = float(METADATA.get("threshold", 0.5))
    predictions = (probabilities >= threshold).astype(int)

    response_items: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        item: dict[str, Any] = {
            "record_index": idx,
            "predicted_churn": bool(pred),
        }
        if return_probabilities:
            item["churn_probability"] = float(probabilities[idx])
        response_items.append(item)

    return {
        "model": METADATA.get("selected_model"),
        "threshold": threshold,
        "predictions": response_items,
    }
