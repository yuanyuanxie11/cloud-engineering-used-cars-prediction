#!/usr/bin/env bash
# =============================================================================
# EC2 Bootstrap Script — Person 2: ML Engineer
#
# Launch an EC2 instance with this script as user-data (or run manually after
# SSH-ing in).  The instance's IAM role must have:
#   - s3:GetObject on s3://mlds423-used-cars-project/processed/*
#   - s3:PutObject on s3://mlds423-used-cars-project/artifacts/*
#
# Recommended instance type: t3.xlarge or c5.2xlarge (4–8 vCPU, 16 GB RAM)
# AMI: Amazon Linux 2023 or Ubuntu 22.04 LTS
#
# Environment variables (never put secrets / access keys here):
#   USED_CARS_BUCKET  — S3 bucket (default: mlds423-used-cars-project)
#   AWS_REGION        — AWS region (default: us-east-1)
#   GIT_REPO_URL      — HTTPS clone URL of the project repo
#   TRAINING_RUN_ID   — optional version id for artifacts/models/runs/<id>/
# =============================================================================

set -euo pipefail

BUCKET="${USED_CARS_BUCKET:-mlds423-used-cars-project}"
REGION="${AWS_REGION:-us-east-1}"
REPO="${GIT_REPO_URL:-}"
RUN_ID="${TRAINING_RUN_ID:-}"
PROJECT_DIR="${PROJECT_DIR:-/home/ec2-user/used-cars-project}"
LOG_FILE="/var/log/train.log"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== EC2 training bootstrap started at $(date -u) ==="
echo "Bucket: $BUCKET  Region: $REGION  RunId: ${RUN_ID:-auto}"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "--- Installing system packages ---"
if command -v dnf &>/dev/null; then
    dnf update -y
    dnf install -y python3.11 python3.11-pip git
    PYTHON=python3.11
else
    apt-get update -y
    apt-get install -y python3 python3-pip python3-venv git
    PYTHON=python3
fi

# ---------------------------------------------------------------------------
# 2. Pull project code
# ---------------------------------------------------------------------------
if [[ -n "$REPO" ]]; then
    echo "--- Cloning repo: $REPO ---"
    git clone "$REPO" "$PROJECT_DIR"
else
    echo "--- GIT_REPO_URL not set; expecting code at $PROJECT_DIR ---"
fi

cd "$PROJECT_DIR"

# ---------------------------------------------------------------------------
# 3. Python virtual environment + dependencies
# ---------------------------------------------------------------------------
echo "--- Creating virtualenv ---"
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "--- Installing Python dependencies ---"
pip install --upgrade pip
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 4. Environment (credentials from instance profile only)
# ---------------------------------------------------------------------------
export USED_CARS_BUCKET="$BUCKET"
export AWS_DEFAULT_REGION="$REGION"
[[ -n "$RUN_ID" ]] && export TRAINING_RUN_ID="$RUN_ID"

# ---------------------------------------------------------------------------
# 5. Verify processed data exists before training
# ---------------------------------------------------------------------------
echo "--- Checking processed Parquet on S3 ---"
python - <<'PY'
import os, sys
import boto3

bucket = os.environ["USED_CARS_BUCKET"]
key = "processed/vehicles_clean.parquet"
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
s3 = boto3.client("s3", region_name=region)
try:
    head = s3.head_object(Bucket=bucket, Key=key)
    print(f"OK s3://{bucket}/{key} size={head['ContentLength']} bytes")
except Exception as exc:
    print(f"ERROR: missing processed data s3://{bucket}/{key}: {exc}", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
# 6. EDA → S3 artifacts/eda/
# ---------------------------------------------------------------------------
echo "--- Running EDA ---"
python -m modeling.eda --config config.yaml

# ---------------------------------------------------------------------------
# 7. Train → S3 artifacts/models/ (runs/, latest/, legacy flat keys)
# ---------------------------------------------------------------------------
echo "--- Running model training ---"
TRAIN_ARGS=(--config config.yaml)
[[ -n "$RUN_ID" ]] && TRAIN_ARGS+=(--run-id "$RUN_ID")
python -m modeling.train "${TRAIN_ARGS[@]}"

echo "=== Training complete at $(date -u) ==="
echo "Artifacts: s3://$BUCKET/artifacts/models/latest/"
echo "Run log: $LOG_FILE"
