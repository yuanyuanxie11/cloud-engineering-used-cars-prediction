"""
Exploratory Data Analysis for the used-cars dataset.

Reads the cleaned Parquet from S3 (or a local file), produces a set of
diagnostic plots and summary statistics, then saves them to
s3://<bucket>/artifacts/eda/ (or a local directory).

Usage:
  # From S3
  python -m modeling.eda --config config.yaml

  # Local dry-run
  python -m modeling.eda --config config.yaml \
      --local-input /tmp/vehicles_clean.parquet \
      --local-output ./eda_output
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402 — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config with optional env-var overrides."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    bucket_override = os.environ.get("USED_CARS_BUCKET")
    if bucket_override:
        cfg["aws"]["bucket"] = bucket_override
    return cfg


def read_data(cfg: dict[str, Any], local_input: str | None) -> pd.DataFrame:
    if local_input:
        logger.info("Reading local file: %s", local_input)
        return pd.read_parquet(local_input)
    import boto3

    bucket = cfg["aws"]["bucket"]
    key = cfg["aws"]["processed_key"]
    logger.info("Reading s3://%s/%s", bucket, key)
    obj = boto3.client("s3", region_name=cfg["aws"]["region"]).get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _save_fig(
    fig: plt.Figure,
    filename: str,
    local_output: str | None,
    cfg: dict[str, Any],
) -> None:
    """Save figure locally or upload to S3."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    plt.close(fig)

    if local_output:
        out_path = Path(local_output) / filename
        out_path.write_bytes(buf.getvalue())
        logger.info("Saved %s", out_path)
    else:
        import boto3

        bucket = cfg["aws"]["bucket"]
        key = f"artifacts/eda/{filename}"
        boto3.client("s3", region_name=cfg["aws"]["region"]).put_object(
            Bucket=bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="image/png",
        )
        logger.info("Uploaded s3://%s/%s", bucket, key)


def _save_json(
    payload: dict,
    filename: str,
    local_output: str | None,
    cfg: dict[str, Any],
) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    if local_output:
        out_path = Path(local_output) / filename
        out_path.write_bytes(data)
        logger.info("Saved %s", out_path)
    else:
        import boto3

        bucket = cfg["aws"]["bucket"]
        key = f"artifacts/eda/{filename}"
        boto3.client("s3", region_name=cfg["aws"]["region"]).put_object(
            Bucket=bucket, Key=key, Body=data, ContentType="application/json"
        )
        logger.info("Uploaded s3://%s/%s", bucket, key)


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------


def plot_price_distribution(
    df: pd.DataFrame, local_output: str | None, cfg: dict[str, Any]
) -> None:
    """Price histogram + log-scale histogram side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price Distribution", fontsize=14)

    # Raw
    df["price"].clip(upper=100_000).plot.hist(bins=80, ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_title("Price (clipped at $100k)")

    # Log scale
    np.log1p(df["price"]).plot.hist(bins=80, ax=axes[1], color="darkorange")
    axes[1].set_xlabel("log(1 + Price)")
    axes[1].set_title("Log-transformed Price")

    _save_fig(fig, "price_distribution.png", local_output, cfg)


def plot_price_by_category(
    df: pd.DataFrame,
    col: str,
    local_output: str | None,
    cfg: dict[str, Any],
    top_n: int = 15,
) -> None:
    """Median price by category (bar chart)."""
    if col not in df.columns:
        logger.warning("Column '%s' not found; skipping plot", col)
        return

    order = (
        df.groupby(col)["price"].median().sort_values(ascending=False).head(top_n).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sub = df[df[col].isin(order)]
    sns.barplot(data=sub, x=col, y="price", order=order, estimator=np.median, ax=ax)
    ax.set_title(f"Median Price by {col.replace('_', ' ').title()} (top {top_n})")
    ax.set_xlabel(col)
    ax.set_ylabel("Median Price ($)")
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, f"price_by_{col}.png", local_output, cfg)


def plot_odometer_vs_price(df: pd.DataFrame, local_output: str | None, cfg: dict[str, Any]) -> None:
    """Scatter: odometer vs price (sampled for speed)."""
    if "odometer" not in df.columns:
        return

    sample = df[["odometer", "price"]].dropna().sample(min(10_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample["odometer"], sample["price"], alpha=0.15, s=5, color="teal")
    ax.set_xlabel("Odometer (miles)")
    ax.set_ylabel("Price ($)")
    ax.set_ylim(0, 100_000)
    ax.set_title("Odometer vs Price (sampled)")
    _save_fig(fig, "odometer_vs_price.png", local_output, cfg)


def plot_vehicle_age_vs_price(
    df: pd.DataFrame, local_output: str | None, cfg: dict[str, Any]
) -> None:
    """Box plot: vehicle age buckets vs price."""
    if "year" not in df.columns:
        return

    from datetime import datetime, timezone

    current_year = datetime.now(timezone.utc).year
    tmp = df.copy()
    tmp["vehicle_age"] = current_year - tmp["year"]
    tmp = tmp[(tmp["vehicle_age"] >= 0) & (tmp["vehicle_age"] <= 40)]
    age_labels = ["0–3", "4–7", "8–12", "13–20", "21+"]
    tmp["age_bucket"] = pd.cut(tmp["vehicle_age"], bins=[0, 3, 7, 12, 20, 40], labels=age_labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=tmp[tmp["price"] <= 80_000],
        x="age_bucket",
        y="price",
        ax=ax,
        color="mediumpurple",
    )
    ax.set_title("Price by Vehicle Age Group")
    ax.set_xlabel("Vehicle Age (years)")
    ax.set_ylabel("Price ($)")
    _save_fig(fig, "vehicle_age_vs_price.png", local_output, cfg)


def plot_missing_values(df: pd.DataFrame, local_output: str | None, cfg: dict[str, Any]) -> None:
    """Horizontal bar chart of missing-value percentages."""
    pct_missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    pct_missing = pct_missing[pct_missing > 0]
    if pct_missing.empty:
        logger.info("No missing values found; skipping missing-value plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(pct_missing) * 0.35)))
    pct_missing.plot.barh(ax=ax, color="salmon")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Value Rates by Column")
    _save_fig(fig, "missing_values.png", local_output, cfg)


def plot_correlation_heatmap(
    df: pd.DataFrame, local_output: str | None, cfg: dict[str, Any]
) -> None:
    """Correlation heatmap of numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Numeric Feature Correlation Matrix")
    _save_fig(fig, "correlation_heatmap.png", local_output, cfg)


def compute_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serialisable summary of the dataset."""
    price = df["price"].dropna()
    stats: dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "price": {
            "mean": round(float(price.mean()), 2),
            "median": round(float(price.median()), 2),
            "std": round(float(price.std()), 2),
            "min": round(float(price.min()), 2),
            "max": round(float(price.max()), 2),
            "p25": round(float(price.quantile(0.25)), 2),
            "p75": round(float(price.quantile(0.75)), 2),
        },
        "missing_pct": {
            col: round(float(pct), 4) for col, pct in (df.isnull().mean() * 100).items() if pct > 0
        },
        "top_manufacturers": (
            df["manufacturer"].value_counts().head(10).to_dict()
            if "manufacturer" in df.columns
            else {}
        ),
    }
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDA for used-cars dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--local-input", help="Path to local Parquet (skips S3 download)")
    parser.add_argument("--local-output", help="Directory to write plots locally")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.local_output:
        Path(args.local_output).mkdir(parents=True, exist_ok=True)

    df = read_data(cfg, args.local_input)
    logger.info("Dataset: %d rows × %d columns", *df.shape)

    # Summary statistics
    stats = compute_summary_stats(df)
    _save_json(stats, "summary_stats.json", args.local_output, cfg)
    logger.info("Price stats: median=$%(median).0f  mean=$%(mean).0f", stats["price"])

    # Plots
    plot_price_distribution(df, args.local_output, cfg)
    plot_missing_values(df, args.local_output, cfg)
    plot_correlation_heatmap(df, args.local_output, cfg)
    plot_odometer_vs_price(df, args.local_output, cfg)
    plot_vehicle_age_vs_price(df, args.local_output, cfg)

    for cat_col in ("manufacturer", "condition", "fuel", "type", "transmission"):
        plot_price_by_category(df, cat_col, args.local_output, cfg)

    logger.info("EDA complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
