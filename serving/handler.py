"""
Model inference Lambda — Person 2: ML Engineer (Model Deployment).

Exposes a `/predict` endpoint (API Gateway HTTP API → Lambda proxy integration)
that returns used-car price predictions from the best trained model.

Design / best practices:
- The model + manifest are downloaded from S3 once per container and cached
  across warm invocations (no per-request download).
- No credentials in code: the Lambda execution role grants
  ``s3:GetObject`` on ``artifacts/models/latest/*`` only.
- Reuses the exact training-time feature pipeline (``modeling.features.prepare_xy``)
  so train/serve preprocessing can never drift.
- Reads ``use_log_target`` and ``native_categorical_model`` from the manifest,
  so inference adapts automatically to whichever model won.

Environment variables:
  USED_CARS_BUCKET   S3 bucket (default: mlds423-used-cars-project)
  AWS_REGION         provided by the Lambda runtime
  MODEL_KEY          override best-model key (default: latest/best_model.pkl)
  MANIFEST_KEY       override manifest key (default: latest/model_manifest.json)
  CONFIG_PATH        path to config.yaml baked into the image (default: config.yaml)

Request body (API Gateway) — a single object or a batch:
  {"manufacturer": "toyota", "year": 2018, "odometer": 45000, ...}
  {"instances": [ {...}, {...} ]}

Response:
  {"predictions": [12345.67, ...], "model": "XGBoost", "run_id": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any

import boto3
import numpy as np
import pandas as pd
import yaml

from modeling.features import prepare_xy
from modeling.train import prepare_xgb_native_frame

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = os.environ.get("USED_CARS_BUCKET", "mlds423-used-cars-project")
MODEL_KEY = os.environ.get("MODEL_KEY", "artifacts/models/latest/best_model.pkl")
MANIFEST_KEY = os.environ.get(
    "MANIFEST_KEY", "artifacts/models/latest/model_manifest.json"
)
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

_s3 = boto3.client("s3")

# Cached across warm invocations.
_MODEL: Any | None = None
_MANIFEST: dict[str, Any] | None = None
_CONFIG: dict[str, Any] | None = None


def _load_artifacts() -> None:
    """Download + cache model, manifest, and config on the first (cold) call."""
    global _MODEL, _MANIFEST, _CONFIG
    if _MODEL is not None:
        return

    with open(CONFIG_PATH, encoding="utf-8") as fh:
        _CONFIG = yaml.safe_load(fh)

    model_obj = _s3.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    _MODEL = pickle.loads(model_obj["Body"].read())

    manifest_obj = _s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)
    _MANIFEST = json.loads(manifest_obj["Body"].read())

    logger.info(
        "Loaded model s3://%s/%s (best_model=%s, run_id=%s, native=%s, log_target=%s)",
        BUCKET,
        MODEL_KEY,
        _MANIFEST.get("best_model"),
        _MANIFEST.get("run_id"),
        _MANIFEST.get("native_categorical_model"),
        _MANIFEST.get("use_log_target"),
    )


def _records_from_event(event: dict | list) -> list[dict]:
    """Accept API Gateway proxy events, direct invokes, single rows, or batches."""
    body: Any = event
    if isinstance(event, dict) and "body" in event and event["body"] is not None:
        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)

    if isinstance(body, dict) and "instances" in body:
        records = body["instances"]
    elif isinstance(body, list):
        records = body
    else:
        records = [body]

    if not records:
        raise ValueError("No input records provided.")
    return records


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run training-time feature engineering. Adds a placeholder target since
    inference rows have no price (prepare_xy expects the target column)."""
    assert _CONFIG is not None
    target = _CONFIG["features"]["target_column"]
    if target not in df.columns:
        df = df.copy()
        df[target] = 0.0  # placeholder; dropped by prepare_xy
    X, _ = prepare_xy(df, _CONFIG)
    return X


def _predict(records: list[dict]) -> list[float]:
    assert _MODEL is not None and _MANIFEST is not None and _CONFIG is not None
    X = _build_features(pd.DataFrame(records))

    if _MANIFEST.get("native_categorical_model"):
        X_native, _, _ = prepare_xgb_native_frame(X, _CONFIG)
        raw = _MODEL.predict(X_native)
    else:
        raw = _MODEL.predict(X)

    if _MANIFEST.get("use_log_target"):
        raw = np.expm1(raw)
    return [round(float(v), 2) for v in np.asarray(raw).ravel()]


def _response(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def handler(event: dict, context: object) -> dict:
    try:
        _load_artifacts()
        records = _records_from_event(event)
        predictions = _predict(records)
        assert _MANIFEST is not None
        return _response(
            200,
            {
                "predictions": predictions,
                "model": _MANIFEST.get("best_model"),
                "run_id": _MANIFEST.get("run_id"),
            },
        )
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Bad request: %s", exc)
        return _response(400, {"error": f"Invalid input: {exc}"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return _response(500, {"error": str(exc)})
