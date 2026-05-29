"""Upload raw vehicles.csv to S3 raw/ prefix."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import boto3

from pipeline.config import DEFAULT_BUCKET, RAW_OBJECT_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def upload_file(local_path: Path, bucket: str, key: str) -> None:
    client = boto3.client("s3")
    logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    client.upload_file(str(local_path), bucket, key)
    logger.info("Upload complete")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/Users/ericwuu/Downloads/vehicles.csv",
        help="Path to raw vehicles.csv",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--key", default=RAW_OBJECT_KEY)
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.is_file():
        logger.error("File not found: %s", path)
        return 1

    upload_file(path, args.bucket, args.key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
