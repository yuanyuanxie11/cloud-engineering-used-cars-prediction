"""
Train baseline (Ridge), tree baseline (Random Forest), and challenger (XGBoost)
models for used-car price prediction.

Usage (from the project root):

  python -m modeling.train --config config.yaml
  python -m modeling.train --config config.yaml --local-input ./out/vehicles_clean.parquet
  python -m modeling.train --config config.yaml --local-output ./out/models --run-id 20260528T120000Z

Environment variables (optional):
  USED_CARS_BUCKET   Override S3 bucket name
  AWS_REGION         Override AWS region
  TRAINING_RUN_ID    Override --run-id for versioned artifact paths

Credentials come from the EC2 instance IAM role (never from config).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from modeling.features import build_preprocessor, get_feature_names, prepare_xy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    """Load YAML config; env vars can override aws.bucket and run_id."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    bucket_override = os.environ.get("USED_CARS_BUCKET")
    if bucket_override:
        cfg["aws"]["bucket"] = bucket_override
        logger.info("Bucket overridden via env: %s", bucket_override)

    region_override = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region_override:
        cfg["aws"]["region"] = region_override

    return cfg


def resolve_run_id(cli_run_id: str | None) -> str:
    """UTC run id for versioned artifacts."""
    if cli_run_id:
        return cli_run_id
    env_id = os.environ.get("TRAINING_RUN_ID")
    if env_id:
        return env_id
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def config_hash(cfg: dict[str, Any]) -> str:
    """Short hash of training + feature config for manifest."""
    payload = {
        "features": cfg.get("features"),
        "training": cfg.get("training"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return digest[:12]


def artifact_keys(cfg: dict[str, Any], run_id: str) -> dict[str, str]:
    """Resolve S3/local keys for this run (versioned + latest + legacy)."""
    art = cfg["artifacts"]
    run_prefix = f"{art['runs_prefix'].rstrip('/')}/{run_id}/"
    latest = art["latest_prefix"].rstrip("/") + "/"
    return {
        "run_prefix": run_prefix,
        "ridge": run_prefix + "ridge.pkl",
        "random_forest": run_prefix + "random_forest.pkl",
        "xgboost": run_prefix + "xgboost.pkl",
        "xgboost_native": run_prefix + "xgboost_native.pkl",
        "best_model": run_prefix + "best_model.pkl",
        "metrics": run_prefix + "metrics.json",
        "manifest": run_prefix + "model_manifest.json",
        "feature_importance": run_prefix + "feature_importance.csv",
        "latest_ridge": latest + "ridge.pkl",
        "latest_random_forest": latest + "random_forest.pkl",
        "latest_xgboost": latest + "xgboost.pkl",
        "latest_xgboost_native": latest + "xgboost_native.pkl",
        "latest_best_model": latest + "best_model.pkl",
        "latest_metrics": latest + "metrics.json",
        "latest_manifest": latest + "model_manifest.json",
        "latest_feature_importance": latest + "feature_importance.csv",
        "legacy_ridge": art["legacy_ridge_key"],
        "legacy_random_forest": art["legacy_baseline_key"],
        "legacy_xgboost": art["legacy_challenger_key"],
        "legacy_xgboost_native": art.get(
            "legacy_native_challenger_key", "artifacts/models/xgboost_native.pkl"
        ),
        "legacy_best_model": art.get(
            "legacy_best_model_key", "artifacts/models/best_model.pkl"
        ),
        "legacy_metrics": art["legacy_metrics_key"],
        "legacy_manifest": art["legacy_manifest_key"],
        "legacy_feature_importance": art["legacy_importance_key"],
    }


def log_structured(event: str, payload: dict[str, Any]) -> None:
    """Single-line JSON log for CloudWatch / log aggregation."""
    record = {"event": event, **payload}
    logger.info("STRUCTURED %s", json.dumps(record, default=str))


def make_cv(cfg: dict[str, Any]) -> KFold:
    """Shuffled K-fold splitter so CV does not depend on row ordering."""
    train_cfg = cfg["training"]
    return KFold(
        n_splits=train_cfg.get("cv_folds", 5),
        shuffle=True,
        random_state=train_cfg.get("random_seed", 42),
    )


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _s3_client(cfg: dict[str, Any]):
    import boto3
    return boto3.client("s3", region_name=cfg["aws"]["region"])


def push_cloudwatch_metrics(
    cfg: dict[str, Any], run_id: str, best_name: str, best_metrics: dict[str, Any]
) -> None:
    """
    Publish the chosen model's test metrics to CloudWatch as custom metrics so a
    dashboard / alarm can track model quality over runs. Best-effort: any failure
    (no AWS creds, no permission, offline) is logged and swallowed so training
    output is never lost.
    """
    mon_cfg = cfg.get("monitoring", {})
    if not mon_cfg.get("push_metrics", False):
        return
    namespace = mon_cfg.get("cloudwatch_namespace", "UsedCarsML")
    try:
        import boto3

        cw = boto3.client("cloudwatch", region_name=cfg["aws"]["region"])
        dims = [{"Name": "BestModel", "Value": best_name}]
        test = best_metrics["test"]
        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {"MetricName": "TestR2", "Value": test["r2"], "Unit": "None", "Dimensions": dims},
                {"MetricName": "TestRMSE", "Value": test["rmse"], "Unit": "None", "Dimensions": dims},
                {"MetricName": "TestMAE", "Value": test["mae"], "Unit": "None", "Dimensions": dims},
                {"MetricName": "CVR2Mean", "Value": best_metrics["cv_r2_mean"], "Unit": "None", "Dimensions": dims},
            ],
        )
        logger.info("Published CloudWatch metrics to namespace %s (run_id=%s)", namespace, run_id)
    except Exception as exc:  # noqa: BLE001 — monitoring must never break training
        logger.warning("Skipping CloudWatch metric push: %s", exc)


def read_parquet_from_s3(cfg: dict[str, Any]) -> pd.DataFrame:
    bucket = cfg["aws"]["bucket"]
    key = cfg["aws"]["processed_key"]
    logger.info("Reading s3://%s/%s", bucket, key)
    s3 = _s3_client(cfg)
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    if df.empty:
        raise ValueError(f"Processed dataset is empty: s3://{bucket}/{key}")
    logger.info("Loaded %d rows, %d columns from S3", *df.shape)
    return df


def upload_bytes(cfg: dict[str, Any], key: str, data: bytes, content_type: str) -> None:
    bucket = cfg["aws"]["bucket"]
    _s3_client(cfg).put_object(
        Bucket=bucket, Key=key, Body=data, ContentType=content_type
    )
    logger.info("Uploaded s3://%s/%s (%d bytes)", bucket, key, len(data))


def save_model_to_s3(cfg: dict[str, Any], model: Any, key: str) -> None:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    upload_bytes(cfg, key, buf.getvalue(), "application/octet-stream")


def save_json_to_s3(cfg: dict[str, Any], payload: dict, key: str) -> None:
    upload_bytes(
        cfg,
        key,
        json.dumps(payload, indent=2).encode("utf-8"),
        "application/json",
    )


def save_csv_to_s3(cfg: dict[str, Any], df: pd.DataFrame, key: str) -> None:
    upload_bytes(cfg, key, df.to_csv(index=False).encode("utf-8"), "text/csv")


# ---------------------------------------------------------------------------
# Metrics & targets
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    )
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape}


def transform_target(y: pd.Series, use_log: bool) -> pd.Series:
    if use_log:
        return np.log1p(y.astype(float))
    return y.astype(float)


def inverse_transform_predictions(y_pred: np.ndarray, use_log: bool) -> np.ndarray:
    if use_log:
        return np.expm1(y_pred)
    return y_pred


# ---------------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------------

def build_ridge_pipeline(preprocessor: Any, cfg: dict[str, Any]) -> Pipeline:
    ridge_params = cfg["training"]["ridge"]
    model = Ridge(**ridge_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_rf_pipeline(preprocessor: Any, cfg: dict[str, Any]) -> Pipeline:
    rf_params = cfg["training"]["random_forest"]
    model = RandomForestRegressor(**rf_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_xgb_pipeline(preprocessor: Any, cfg: dict[str, Any]) -> Pipeline:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    xgb_cfg = cfg["training"]["xgboost"]
    xgb_params = {
        k: v
        for k, v in xgb_cfg.items()
        if k not in ("early_stopping_rounds", "validation_size")
    }
    model = XGBRegressor(**xgb_params)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
    *,
    use_log_target: bool,
    fit_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Pipeline]:
    """
    Fit pipeline with optional fit_kwargs (XGBoost early stopping), run CV,
    evaluate on test set in original price scale.
    """
    train_cfg = cfg["training"]
    cv_n_jobs = train_cfg.get("cv_n_jobs", 1)
    logger.info("=== Training %s ===", name)
    t0 = time.time()

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=make_cv(cfg),
        scoring="r2",
        n_jobs=cv_n_jobs,
    )
    logger.info(
        "%s CV R² — mean: %.4f  std: %.4f", name, cv_scores.mean(), cv_scores.std()
    )

    pipeline.fit(X_train, y_train, **(fit_kwargs or {}))
    elapsed = time.time() - t0

    y_pred_train = inverse_transform_predictions(
        pipeline.predict(X_train), use_log_target
    )
    y_pred_test = inverse_transform_predictions(
        pipeline.predict(X_test), use_log_target
    )
    y_train_orig = y_train.values if not use_log_target else np.expm1(y_train.values)
    y_test_orig = y_test.values if not use_log_target else np.expm1(y_test.values)

    metrics = {
        "model": name,
        "train": compute_metrics(y_train_orig, y_pred_train),
        "test": compute_metrics(y_test_orig, y_pred_test),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "training_time_seconds": round(elapsed, 2),
        "use_log_target": use_log_target,
    }

    logger.info(
        "%s test — RMSE: %.2f  MAE: %.2f  R²: %.4f",
        name,
        metrics["test"]["rmse"],
        metrics["test"]["mae"],
        metrics["test"]["r2"],
    )
    return metrics, pipeline


def fit_xgboost_with_early_stopping(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> None:
    """
    Fit XGBoost with early stopping.

    xgboost>=2.0 requires ``early_stopping_rounds`` to be a *constructor*
    argument (passing only ``eval_set`` to ``fit`` does nothing). We split a
    validation slice out of the training set first, fit the preprocessor on the
    training slice only (so validation stats never leak), then early-stop on the
    held-out slice.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    xgb_cfg = cfg["training"]["xgboost"]
    val_size = xgb_cfg.get("validation_size", 0.1)
    random_seed = cfg["training"]["random_seed"]

    preprocessor = pipeline.named_steps["preprocessor"]

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_seed,
    )
    X_tr = preprocessor.fit_transform(X_tr_raw, y_tr)
    X_val = preprocessor.transform(X_val_raw)

    # early_stopping_rounds must live on the constructor for xgboost>=2.0.
    model_params = {k: v for k, v in xgb_cfg.items() if k != "validation_size"}
    model = XGBRegressor(**model_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(
        "%s stopped at %s trees (best_iteration=%s of %s configured)",
        "XGBoost",
        getattr(model, "best_iteration", None),
        getattr(model, "best_iteration", None),
        xgb_cfg.get("n_estimators"),
    )
    pipeline.steps = [("preprocessor", preprocessor), ("model", model)]


def train_xgboost(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
    use_log_target: bool,
) -> tuple[dict[str, Any], Pipeline]:
    """XGBoost with early stopping when configured; otherwise standard path."""
    xgb_cfg = cfg["training"]["xgboost"]
    if xgb_cfg.get("early_stopping_rounds"):
        logger.info("=== Training %s (with early stopping) ===", name)
        t0 = time.time()

        # CV uses the fixed-n_estimators pipeline (no eval_set inside folds);
        # early stopping is applied only to the final refit below.
        cv_n_jobs = cfg["training"].get("cv_n_jobs", 1)
        cv_scores = cross_val_score(
            build_xgb_pipeline(
                build_preprocessor(X_train, cfg)[0], cfg
            ),
            X_train,
            y_train,
            cv=make_cv(cfg),
            scoring="r2",
            n_jobs=cv_n_jobs,
        )
        logger.info(
            "%s CV R² — mean: %.4f  std: %.4f",
            name,
            cv_scores.mean(),
            cv_scores.std(),
        )

        fit_xgboost_with_early_stopping(pipeline, X_train, y_train, cfg)
        elapsed = time.time() - t0

        y_pred_train = inverse_transform_predictions(
            pipeline.predict(X_train), use_log_target
        )
        y_pred_test = inverse_transform_predictions(
            pipeline.predict(X_test), use_log_target
        )
        y_train_orig = (
            y_train.values if not use_log_target else np.expm1(y_train.values)
        )
        y_test_orig = y_test.values if not use_log_target else np.expm1(y_test.values)

        metrics = {
            "model": name,
            "train": compute_metrics(y_train_orig, y_pred_train),
            "test": compute_metrics(y_test_orig, y_pred_test),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "training_time_seconds": round(elapsed, 2),
            "use_log_target": use_log_target,
            "early_stopping_rounds": xgb_cfg.get("early_stopping_rounds"),
        }
        logger.info(
            "%s test — RMSE: %.2f  MAE: %.2f  R²: %.4f",
            name,
            metrics["test"]["rmse"],
            metrics["test"]["mae"],
            metrics["test"]["r2"],
        )
        return metrics, pipeline

    return train_and_evaluate(
        name, pipeline, X_train, y_train, X_test, y_test, cfg,
        use_log_target=use_log_target,
    )


def _native_xgb_columns(X: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) columns for native-categorical XGBoost."""
    feat_cfg = cfg["features"]
    present = set(X.columns)
    numeric = [c for c in feat_cfg.get("numeric_features", []) if c in present]
    categorical = [
        c for c in feat_cfg.get("native_xgboost_categorical_features", [])
        if c in present and c not in numeric
    ]
    return numeric, categorical


def prepare_xgb_native_frame(
    X: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Prepare X for XGBoost native categorical mode.

    Unlike the shared sklearn preprocessor, this keeps categoricals as pandas
    category dtype and lets XGBoost split categories internally.
    """
    numeric_cols, categorical_cols = _native_xgb_columns(X, cfg)
    selected = numeric_cols + categorical_cols
    if not selected:
        raise ValueError("No columns selected for native XGBoost.")

    out = X[selected].copy()
    for col in numeric_cols:
        out[col] = out[col].astype(float)
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    for col in categorical_cols:
        out[col] = out[col].astype("string").fillna("missing").astype("category")

    return out, numeric_cols, categorical_cols


def build_xgb_native_model(cfg: dict[str, Any]):
    """Build XGBoost model with native categorical support."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Run: pip install xgboost") from exc

    native_cfg = cfg["training"]["xgboost_native"]
    params = {k: v for k, v in native_cfg.items() if k != "enabled"}
    params.setdefault("enable_categorical", True)
    params.setdefault("tree_method", "hist")
    return XGBRegressor(**params)


def train_xgboost_native(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: dict[str, Any],
    use_log_target: bool,
) -> tuple[dict[str, Any], Any, list[str]]:
    """Train and evaluate native-categorical XGBoost."""
    name = "XGBoostNativeCategorical"
    train_cfg = cfg["training"]
    cv_n_jobs = train_cfg.get("cv_n_jobs", 1)

    X_train_native, numeric_cols, categorical_cols = prepare_xgb_native_frame(X_train, cfg)
    X_test_native, _, _ = prepare_xgb_native_frame(X_test, cfg)
    feature_names = list(X_train_native.columns)

    logger.info(
        "=== Training %s (native categorical) === numeric=%s categorical=%s",
        name,
        numeric_cols,
        categorical_cols,
    )
    t0 = time.time()

    cv_model = build_xgb_native_model(cfg)
    cv_scores = cross_val_score(
        cv_model,
        X_train_native,
        y_train,
        cv=make_cv(cfg),
        scoring="r2",
        n_jobs=cv_n_jobs,
    )
    logger.info(
        "%s CV R² — mean: %.4f  std: %.4f", name, cv_scores.mean(), cv_scores.std()
    )

    model = build_xgb_native_model(cfg)
    model.fit(X_train_native, y_train)
    elapsed = time.time() - t0

    y_pred_train = inverse_transform_predictions(
        model.predict(X_train_native), use_log_target
    )
    y_pred_test = inverse_transform_predictions(
        model.predict(X_test_native), use_log_target
    )
    y_train_orig = y_train.values if not use_log_target else np.expm1(y_train.values)
    y_test_orig = y_test.values if not use_log_target else np.expm1(y_test.values)

    metrics = {
        "model": name,
        "train": compute_metrics(y_train_orig, y_pred_train),
        "test": compute_metrics(y_test_orig, y_pred_test),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "training_time_seconds": round(elapsed, 2),
        "use_log_target": use_log_target,
        "native_categorical": True,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    }
    logger.info(
        "%s test — RMSE: %.2f  MAE: %.2f  R²: %.4f",
        name,
        metrics["test"]["rmse"],
        metrics["test"]["mae"],
        metrics["test"]["r2"],
    )
    return metrics, model, feature_names


def feature_importance_df(model_or_pipeline: Any, feature_names: list[str]) -> pd.DataFrame:
    model = (
        model_or_pipeline.named_steps["model"]
        if hasattr(model_or_pipeline, "named_steps")
        else model_or_pipeline
    )
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model has no importances; skipping export")
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
    )


def build_manifest(
    cfg: dict[str, Any],
    run_id: str,
    best_name: str,
    all_metrics: dict[str, Any],
    raw_feature_columns: list[str],
    transformed_feature_names: list[str],
) -> dict[str, Any]:
    feat_cfg = cfg["features"]
    return {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "target_column": feat_cfg["target_column"],
        "use_log_target": cfg["training"].get("use_log_target", False),
        "config_hash": config_hash(cfg),
        "s3_bucket": cfg["aws"]["bucket"],
        "processed_key": cfg["aws"]["processed_key"],
        "raw_feature_columns": raw_feature_columns,
        "transformed_feature_names_sample": transformed_feature_names[:50],
        "transformed_feature_count": len(transformed_feature_names),
        "drop_columns": feat_cfg.get("drop_columns", []),
        "native_categorical_model": best_name == "XGBoostNativeCategorical",
        "native_xgboost_categorical_features": feat_cfg.get(
            "native_xgboost_categorical_features", []
        ),
        "metrics_summary": {
            m["model"]: m["test"] for m in all_metrics["models"]
        },
        "inference_note": (
            "If best_model is Ridge/RandomForest/XGBoost, load best_model.pkl as a "
            "sklearn Pipeline and pass prepared X from prepare_xy. If best_model is "
            "XGBoostNativeCategorical, convert the manifest categorical columns to "
            "pandas category dtype before prediction."
        ),
    }


def save_artifacts_local(
    out_dir: Path,
    artifacts: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, obj in artifacts.items():
        path = out_dir / filename
        if filename.endswith(".pkl"):
            with open(path, "wb") as fh:
                pickle.dump(obj, fh)
        elif filename.endswith(".json"):
            path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        elif filename.endswith(".csv") and isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)


def persist_artifacts(
    cfg: dict[str, Any],
    keys: dict[str, str],
    run_id: str,
    *,
    local_output: Path | None,
    ridge_pipeline: Pipeline,
    rf_pipeline: Pipeline,
    xgb_pipeline: Pipeline,
    xgb_native_model: Any | None,
    best_pipeline: Any,
    all_metrics: dict[str, Any],
    manifest: dict[str, Any],
    imp_df: pd.DataFrame,
) -> None:
    """Write to versioned run dir, latest/, legacy paths, and optional local dir."""
    bundle = {
        "ridge.pkl": ridge_pipeline,
        "random_forest.pkl": rf_pipeline,
        "xgboost.pkl": xgb_pipeline,
        **({"xgboost_native.pkl": xgb_native_model} if xgb_native_model else {}),
        "best_model.pkl": best_pipeline,
        "metrics.json": all_metrics,
        "model_manifest.json": manifest,
        **({"feature_importance.csv": imp_df} if not imp_df.empty else {}),
    }

    if local_output:
        for directory in (
            local_output / "runs" / run_id,
            local_output / "latest",
            local_output,
        ):
            save_artifacts_local(directory, bundle)
        logger.info("Artifacts written to %s (runs/%s, latest, root)", local_output, run_id)
        return

    def _upload_all(target_keys: dict[str, str]) -> None:
        save_model_to_s3(cfg, ridge_pipeline, target_keys["ridge"])
        save_model_to_s3(cfg, rf_pipeline, target_keys["random_forest"])
        save_model_to_s3(cfg, xgb_pipeline, target_keys["xgboost"])
        if xgb_native_model and "xgboost_native" in target_keys:
            save_model_to_s3(cfg, xgb_native_model, target_keys["xgboost_native"])
        save_model_to_s3(cfg, best_pipeline, target_keys["best_model"])
        save_json_to_s3(cfg, all_metrics, target_keys["metrics"])
        save_json_to_s3(cfg, manifest, target_keys["manifest"])
        if not imp_df.empty:
            save_csv_to_s3(cfg, imp_df, target_keys["feature_importance"])

    _upload_all(keys)
    _upload_all(
        {
            "ridge": keys["latest_ridge"],
            "random_forest": keys["latest_random_forest"],
            "xgboost": keys["latest_xgboost"],
            "xgboost_native": keys["latest_xgboost_native"],
            "best_model": keys["latest_best_model"],
            "metrics": keys["latest_metrics"],
            "manifest": keys["latest_manifest"],
            "feature_importance": keys["latest_feature_importance"],
        }
    )
    _upload_all(
        {
            "ridge": keys["legacy_ridge"],
            "random_forest": keys["legacy_random_forest"],
            "xgboost": keys["legacy_xgboost"],
            "xgboost_native": keys["legacy_xgboost_native"],
            "best_model": keys["legacy_best_model"],
            "metrics": keys["legacy_metrics"],
            "manifest": keys["legacy_manifest"],
            "feature_importance": keys["legacy_feature_importance"],
        }
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train used-cars price models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--local-input", help="Local Parquet (skips S3 download)")
    parser.add_argument(
        "--local-output",
        help="Directory for artifacts (mirrors S3 layout under this path)",
    )
    parser.add_argument(
        "--run-id",
        help="Version id for artifacts/models/runs/<run_id>/ (default: UTC timestamp)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_id = resolve_run_id(args.run_id)
    keys = artifact_keys(cfg, run_id)
    train_cfg = cfg["training"]
    use_log_target = bool(train_cfg.get("use_log_target", False))

    if args.local_input:
        logger.info("Loading local file: %s", args.local_input)
        df = pd.read_parquet(args.local_input)
        if df.empty:
            raise ValueError(f"Local input is empty: {args.local_input}")
        logger.info("Loaded %d rows, %d columns", *df.shape)
    else:
        df = read_parquet_from_s3(cfg)

    X, y_raw = prepare_xy(df, cfg)
    y = transform_target(y_raw, use_log_target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_seed"],
    )
    logger.info("Train/test split: %d train / %d test rows", len(X_train), len(X_test))

    log_structured(
        "training_start",
        {
            "run_id": run_id,
            "rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_columns": list(X.columns),
            "use_log_target": use_log_target,
            "bucket": cfg["aws"]["bucket"],
        },
    )

    preprocessor_ridge, _ = build_preprocessor(X_train, cfg)
    ridge_pipeline = build_ridge_pipeline(preprocessor_ridge, cfg)
    ridge_metrics, ridge_pipeline = train_and_evaluate(
        "Ridge",
        ridge_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        use_log_target=use_log_target,
    )

    preprocessor_rf, _ = build_preprocessor(X_train, cfg)
    rf_pipeline = build_rf_pipeline(preprocessor_rf, cfg)
    rf_metrics, rf_pipeline = train_and_evaluate(
        "RandomForest",
        rf_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        use_log_target=use_log_target,
    )

    preprocessor_xgb, _ = build_preprocessor(X_train, cfg)
    xgb_pipeline = build_xgb_pipeline(preprocessor_xgb, cfg)
    xgb_metrics, xgb_pipeline = train_xgboost(
        "XGBoost",
        xgb_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        cfg,
        use_log_target,
    )

    xgb_native_metrics: dict[str, Any] | None = None
    xgb_native_model: Any | None = None
    xgb_native_feature_names: list[str] = []
    if train_cfg.get("xgboost_native", {}).get("enabled", False):
        xgb_native_metrics, xgb_native_model, xgb_native_feature_names = (
            train_xgboost_native(
                X_train,
                y_train,
                X_test,
                y_test,
                cfg,
                use_log_target,
            )
        )

    candidates = {
        "Ridge": (ridge_metrics, ridge_pipeline),
        "RandomForest": (rf_metrics, rf_pipeline),
        "XGBoost": (xgb_metrics, xgb_pipeline),
    }
    if xgb_native_metrics and xgb_native_model:
        candidates["XGBoostNativeCategorical"] = (
            xgb_native_metrics,
            xgb_native_model,
        )
    # Select by cross-validated R² (computed on the training folds only) so the
    # held-out test set is never used for model selection — it is reported once
    # for the chosen model.
    best_name = max(candidates, key=lambda n: candidates[n][0]["cv_r2_mean"])
    best_metrics, best_pipeline = candidates[best_name]
    logger.info(
        "Best model by CV R²: %s (cv R²=%.4f, test R²=%.4f)",
        best_name,
        best_metrics["cv_r2_mean"],
        best_metrics["test"]["r2"],
    )

    if hasattr(best_pipeline, "named_steps"):
        feat_names = get_feature_names(best_pipeline.named_steps["preprocessor"])
    else:
        feat_names = xgb_native_feature_names
    imp_df = feature_importance_df(best_pipeline, feat_names)

    model_metrics = [ridge_metrics, rf_metrics, xgb_metrics]
    if xgb_native_metrics:
        model_metrics.append(xgb_native_metrics)

    all_metrics = {
        "run_id": run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": model_metrics,
        "best_model": best_name,
        "selection_metric": "cv_r2_mean",
        "config_snapshot": {
            "test_size": train_cfg["test_size"],
            "random_seed": train_cfg["random_seed"],
            "cv_folds": train_cfg["cv_folds"],
            "use_log_target": use_log_target,
            "config_hash": config_hash(cfg),
        },
    }
    manifest = build_manifest(
        cfg, run_id, best_name, all_metrics, list(X.columns), feat_names
    )

    local_out = Path(args.local_output) if args.local_output else None
    persist_artifacts(
        cfg,
        keys,
        run_id,
        local_output=local_out,
        ridge_pipeline=ridge_pipeline,
        rf_pipeline=rf_pipeline,
        xgb_pipeline=xgb_pipeline,
        xgb_native_model=xgb_native_model,
        best_pipeline=best_pipeline,
        all_metrics=all_metrics,
        manifest=manifest,
        imp_df=imp_df,
    )

    if local_out is None:
        push_cloudwatch_metrics(cfg, run_id, best_name, best_metrics)

    log_structured(
        "training_complete",
        {
            "run_id": run_id,
            "best_model": best_name,
            "test_r2": best_metrics["test"]["r2"],
            "test_rmse": best_metrics["test"]["rmse"],
            "models": {m["model"]: m["test"]["r2"] for m in all_metrics["models"]},
        },
    )
    logger.info("Training complete. run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
