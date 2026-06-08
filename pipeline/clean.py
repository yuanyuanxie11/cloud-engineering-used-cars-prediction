"""
Clean and preprocess the Craigslist vehicles dataset.

Run locally:
  python -m pipeline.clean --input /path/to/vehicles.csv --output ./out/vehicles_clean.parquet

Run from S3 (after upload):
  python -m pipeline.clean --s3-input s3://used-cars-project/raw/vehicles.csv \\
      --s3-output s3://used-cars-project/processed/vehicles_clean.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import (
    DROP_COLUMNS,
    MAX_ODOMETER,
    MAX_PRICE,
    MIN_PRICE,
    MIN_YEAR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    path = uri[5:]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return bucket, key


def read_input(path: str | None, s3_uri: str | None) -> pd.DataFrame:
    if s3_uri:
        import boto3

        bucket, key = _parse_s3_uri(s3_uri)
        logger.info("Reading s3://%s/%s", bucket, key)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return pd.read_csv(obj["Body"], low_memory=False)
    if path:
        logger.info("Reading %s", path)
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Provide --input or --s3-input")


def write_output(df: pd.DataFrame, path: str | None, s3_uri: str | None) -> None:
    if s3_uri:
        from io import BytesIO

        import boto3

        bucket, key = _parse_s3_uri(s3_uri)
        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info("Wrote s3://%s/%s (%d rows)", bucket, key, len(df))
        return
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, engine="pyarrow")
        logger.info("Wrote %s (%d rows)", out, len(df))
        return
    raise ValueError("Provide --output or --s3-output")


def clean_vehicles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply team-agreed cleaning rules; return cleaned frame and stats."""
    stats: dict[str, Any] = {"input_rows": len(df)}

    # Drop unused / all-null columns
    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        stats["dropped_columns"] = drop_cols

    # Deduplicate on listing id
    before = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    stats["duplicates_removed"] = before - len(df)

    # Price outliers ($0, $1B+, and unrealistic used-car range)
    price = df["price"]
    price_mask = (
        price.notna()
        & (price >= MIN_PRICE)
        & (price <= MAX_PRICE)
        & (price < 1_000_000_000)  # explicit $1B guard from project spec
    )
    stats["price_outliers_removed"] = int((~price_mask).sum())
    df = df.loc[price_mask]

    # Year: require valid model year for price prediction
    if "year" in df.columns:
        current_year = datetime.now(timezone.utc).year + 1
        year_mask = df["year"].notna() & (df["year"] >= MIN_YEAR) & (df["year"] <= current_year)
        stats["invalid_year_removed"] = int((~year_mask).sum())
        df = df.loc[year_mask]

    # Odometer: drop negative or absurd readings when present
    if "odometer" in df.columns:
        odo = df["odometer"]
        odo_mask = odo.isna() | ((odo >= 0) & (odo <= MAX_ODOMETER))
        stats["odometer_outliers_removed"] = int((~odo_mask).sum())
        df = df.loc[odo_mask]

    # Normalize text categoricals (optional; helps downstream encoding)
    for col in ("manufacturer", "model", "state", "condition", "fuel", "transmission"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()

    stats["output_rows"] = len(df)
    stats["columns"] = list(df.columns)
    return df.reset_index(drop=True), stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean vehicles listing data")
    parser.add_argument("--input", help="Local path to raw vehicles.csv")
    parser.add_argument("--output", help="Local path for cleaned Parquet")
    parser.add_argument("--s3-input", help="S3 URI for raw CSV")
    parser.add_argument("--s3-output", help="S3 URI for processed Parquet")
    parser.add_argument(
        "--stats-output",
        help="Write JSON cleaning report (local path or s3:// URI)",
    )
    args = parser.parse_args(argv)

    df = read_input(args.input, args.s3_input)
    cleaned, stats = clean_vehicles(df)
    write_output(cleaned, args.output, args.s3_output)

    if args.stats_output:
        report = json.dumps(stats, indent=2)
        if args.stats_output.startswith("s3://"):
            import boto3

            bucket, key = _parse_s3_uri(args.stats_output)
            boto3.client("s3").put_object(
                Bucket=bucket,
                Key=key,
                Body=report.encode("utf-8"),
                ContentType="application/json",
            )
        else:
            Path(args.stats_output).write_text(report, encoding="utf-8")
        logger.info("Stats: %s", stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
