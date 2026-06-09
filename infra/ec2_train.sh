#!/usr/bin/env bash
# =============================================================================
# EC2 Bootstrap Script — Person 2: ML Engineer
#
# Launch an EC2 instance with this script as user-data (or run manually after
# SSH-ing in).  The instance's IAM role must have (see iam_train_policy.json):
#   - s3:GetObject on s3://mlds423-used-cars-project/processed/*
#   - s3:PutObject on s3://mlds423-used-cars-project/artifacts/*
#   - cloudwatch:PutMetricData (training metrics namespace UsedCarsML)
#   - logs:PutLogEvents on log group /used-cars/ec2-train (CloudWatch Agent)
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
CW_LOG_GROUP="${CW_LOG_GROUP:-/used-cars/ec2-train}"
CW_AGENT_CONFIG="/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "--- Installing system packages ---"
if command -v dnf &>/dev/null; then
    dnf update -y
    dnf install -y python3.11 python3.11-pip git amazon-cloudwatch-agent curl
    PYTHON=python3.11
else
    apt-get update -y
    apt-get install -y python3 python3-pip python3-venv git curl
    PYTHON=python3
fi

# ---------------------------------------------------------------------------
# 1b. CloudWatch Agent — tail /var/log/train.log to CloudWatch Logs
#     Requires UsedCarsMLTrainRole logs:PutLogEvents (iam_train_policy.json).
# ---------------------------------------------------------------------------
setup_cloudwatch_agent() {
    if ! command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
        echo "WARN: amazon-cloudwatch-agent not installed; logs stay on disk only."
        return 0
    fi

    local instance_id stream_name
    instance_id="unknown"
    if command -v curl &>/dev/null; then
        local token
        token=$(curl -sSf -X PUT "http://169.254.169.254/latest/api/token" \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null) || token=""
        if [[ -n "$token" ]]; then
            instance_id=$(curl -sSf -H "X-aws-ec2-metadata-token: $token" \
                "http://169.254.169.254/latest/meta-data/instance-id" 2>/dev/null) || instance_id="unknown"
        fi
    fi
    stream_name="${instance_id}"
    [[ -n "$RUN_ID" ]] && stream_name="${instance_id}-${RUN_ID}"

    sudo touch "$LOG_FILE"
    sudo chmod 644 "$LOG_FILE"

    echo "--- Configuring CloudWatch Agent (log group: $CW_LOG_GROUP) ---"
    sudo tee "$CW_AGENT_CONFIG" >/dev/null <<EOF
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "$LOG_FILE",
            "log_group_name": "$CW_LOG_GROUP",
            "log_stream_name": "$stream_name",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
EOF

    sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
        -a fetch-config -m ec2 -c "file:$CW_AGENT_CONFIG" -s
    echo "CloudWatch Agent started; stream=$stream_name"
}

setup_cloudwatch_agent

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== EC2 training bootstrap started at $(date -u) ==="
echo "Bucket: $BUCKET  Region: $REGION  RunId: ${RUN_ID:-auto}"
echo "CloudWatch Logs: $CW_LOG_GROUP (file: $LOG_FILE)"

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
echo "Run log (local): $LOG_FILE"
echo "CloudWatch Logs: $CW_LOG_GROUP"
