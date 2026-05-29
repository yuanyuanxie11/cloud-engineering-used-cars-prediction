"""Configuration for the used-cars data pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

# S3 layout (override via env for local testing)
DEFAULT_BUCKET = os.environ.get("USED_CARS_BUCKET", "used-cars-project")
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
ARTIFACTS_PREFIX = os.environ.get("ARTIFACTS_PREFIX", "artifacts/")

RAW_OBJECT_KEY = f"{RAW_PREFIX}vehicles.csv"
PROCESSED_OBJECT_KEY = f"{PROCESSED_PREFIX}vehicles_clean.parquet"

# Cleaning thresholds (document these in the team README for ML teammates)
MIN_PRICE = int(os.environ.get("MIN_PRICE", "100"))
MAX_PRICE = int(os.environ.get("MAX_PRICE", "500_000").replace("_", ""))
MIN_YEAR = int(os.environ.get("MIN_YEAR", "1980"))
MAX_ODOMETER = int(os.environ.get("MAX_ODOMETER", "1_000_000").replace("_", ""))

# Columns dropped before modeling (100% null in source data)
DROP_COLUMNS = ("county",)
