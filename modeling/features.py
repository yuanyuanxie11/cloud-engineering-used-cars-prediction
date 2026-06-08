"""
Feature engineering pipeline for used-cars price prediction.

All parameters are read from config.yaml (or a dict passed at runtime) so
nothing is hard-coded.  The public API is:

    preprocessor = build_preprocessor(cfg)   # returns sklearn Pipeline
    X, y = prepare_xy(df, cfg)               # feature matrix + target
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw feature construction
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Apply feature engineering steps that create new columns or transform
    existing ones.  Returns a copy; does not modify the input.
    """
    df = df.copy()
    feat_cfg = cfg["features"]

    # -- vehicle_age: more interpretable than raw year ----------------------
    if "year" in df.columns:
        current_year = datetime.now(timezone.utc).year
        df["vehicle_age"] = current_year - df["year"].astype(float)
        logger.debug("Engineered vehicle_age (current_year=%d)", current_year)

    # -- normalise text categoricals (already done by pipeline.clean, but
    #    guard against re-use on unprocessed frames). Convert pandas nullable
    #    NA values to np.nan so sklearn's SimpleImputer can handle them.
    categorical_cols = feat_cfg.get("nominal_features", []) + [
        c for c in feat_cfg.get("ordinal_features", {}).keys()
    ]
    for col in categorical_cols:
        if col in df.columns:
            normalized = df[col].astype("string").str.strip().str.lower()
            df[col] = pd.Series(
                normalized.to_numpy(dtype=object, na_value=np.nan),
                index=df.index,
            )

    # -- drop high-cardinality / identifier columns -------------------------
    drop_cols = [c for c in feat_cfg.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.debug("Dropped columns: %s", drop_cols)

    return df


def prepare_xy(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and split into X (features) and y (target).
    Rows where the target is null are dropped with a warning.
    """
    target = cfg["features"]["target_column"]
    df = engineer_features(df, cfg)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found after engineering.")

    null_target = df[target].isna().sum()
    if null_target:
        logger.warning("Dropping %d rows with null target '%s'", null_target, target)
        df = df.dropna(subset=[target])

    y = df[target].astype(float)
    X = df.drop(columns=[target])
    logger.info("prepare_xy → X shape %s, y shape %s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# sklearn preprocessor
# ---------------------------------------------------------------------------


def _resolve_columns(X: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """
    Intersect configured feature lists with columns actually present in X.
    Returns a dict with keys: numeric, ordinal, nominal.
    """
    feat_cfg = cfg["features"]
    present = set(X.columns)

    numeric = [c for c in feat_cfg.get("numeric_features", []) if c in present]
    ordinal = [c for c in feat_cfg.get("ordinal_features", {}).keys() if c in present]
    nominal = [c for c in feat_cfg.get("nominal_features", []) if c in present]

    logger.debug(
        "Resolved features — numeric: %d, ordinal: %d, nominal: %d",
        len(numeric),
        len(ordinal),
        len(nominal),
    )
    return {"numeric": numeric, "ordinal": ordinal, "nominal": nominal}


def build_preprocessor(
    X: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[ColumnTransformer, dict[str, list[str]]]:
    """
    Build a ColumnTransformer that handles numeric, ordinal, and nominal
    features as configured in config.yaml.

    Returns (preprocessor, resolved_columns) so callers can inspect which
    columns ended up in each transformer.
    """
    feat_cfg = cfg["features"]
    cols = _resolve_columns(X, cfg)

    # -- Numeric pipeline ---------------------------------------------------
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=feat_cfg.get("numeric_impute_strategy", "median")),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    # -- Ordinal pipeline ---------------------------------------------------
    ordinal_categories = []
    for col in cols["ordinal"]:
        cats = feat_cfg["ordinal_features"][col].get("categories", ["unknown"])
        ordinal_categories.append([str(c) for c in cats])

    ordinal_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=feat_cfg.get("categorical_impute_strategy", "most_frequent")
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories if ordinal_categories else "auto",
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    # -- Nominal pipeline ---------------------------------------------------
    max_cats = feat_cfg.get("ohe_max_categories", 20)
    nominal_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy=feat_cfg.get("categorical_impute_strategy", "most_frequent")
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    max_categories=max_cats,
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers: list[tuple] = []
    if cols["numeric"]:
        transformers.append(("numeric", numeric_pipe, cols["numeric"]))
    if cols["ordinal"]:
        transformers.append(("ordinal", ordinal_pipe, cols["ordinal"]))
    if cols["nominal"]:
        transformers.append(("nominal", nominal_pipe, cols["nominal"]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Built preprocessor with %d transformer(s) covering %d columns",
        len(transformers),
        len(cols["numeric"]) + len(cols["ordinal"]) + len(cols["nominal"]),
    )
    return preprocessor, cols


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return a flat list of output feature names after fit."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
        names: list[str] = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "get_feature_names_out"):
                names.extend(trans.get_feature_names_out(cols))
            else:
                names.extend(cols if isinstance(cols, list) else list(cols))
        return names
