"""
Lambda entry point: S3 ObjectCreated on raw/vehicles.csv triggers cleaning.

Deploy with a Lambda layer or package containing pandas, pyarrow, and boto3.
Set env vars: USED_CARS_BUCKET, RAW_PREFIX, PROCESSED_PREFIX, ARTIFACTS_PREFIX
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from urllib.parse import unquote_plus

import boto3
import pandas as pd
from pipeline.clean import clean_vehicles
from pipeline.config import ARTIFACTS_PREFIX, PROCESSED_PREFIX

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")


def _should_process(bucket: str, key: str, record: dict) -> bool:
    raw_prefix = os.environ.get("RAW_PREFIX", "raw/")
    if not key.startswith(raw_prefix):
        return False
    if not key.endswith(".csv"):
        return False
    # Ignore folder placeholders
    if key.endswith("/"):
        return False
    return True


def handler(event: dict, context: object) -> dict:
    """Process S3 upload events."""
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])

        if not _should_process(bucket, key, record):
            logger.info("Skipping s3://%s/%s", bucket, key)
            continue

        logger.info("Processing s3://%s/%s", bucket, key)

        with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
            s3.download_file(bucket, key, tmp.name)
            df = pd.read_csv(tmp.name, low_memory=False)

        cleaned, stats = clean_vehicles(df)

        base_name = key.rsplit("/", 1)[-1].replace(".csv", "")
        processed_key = f"{PROCESSED_PREFIX}{base_name}_clean.parquet"
        artifacts_prefix = os.environ.get("ARTIFACTS_PREFIX", ARTIFACTS_PREFIX)
        stats_key = (
            f"{artifacts_prefix}cleaning_reports/"
            f"{base_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        )

        parquet_path = "/tmp/cleaned.parquet"
        cleaned.to_parquet(parquet_path, index=False, engine="pyarrow")
        s3.upload_file(parquet_path, bucket, processed_key)

        s3.put_object(
            Bucket=bucket,
            Key=stats_key,
            Body=json.dumps(stats, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        results.append(
            {
                "source": f"s3://{bucket}/{key}",
                "processed": f"s3://{bucket}/{processed_key}",
                "stats": f"s3://{bucket}/{stats_key}",
                "output_rows": stats["output_rows"],
            }
        )

    return {"statusCode": 200, "body": json.dumps(results)}
