#!/usr/bin/env bash
# One-time S3 bucket layout for the used-cars team project.
# Usage: ./infra/setup_bucket.sh used-cars-project us-east-1

set -euo pipefail

BUCKET="${1:-used-cars-project}"
REGION="${2:-us-east-1}"

echo "Creating bucket s3://${BUCKET} in ${REGION} (skip if exists)..."
aws s3api create-bucket \
  --bucket "${BUCKET}" \
  --region "${REGION}" \
  $( [[ "${REGION}" != "us-east-1" ]] && echo "--create-bucket-configuration LocationConstraint=${REGION}" ) \
  2>/dev/null || true

for prefix in raw processed artifacts; do
  echo "Ensuring prefix ${prefix}/"
  aws s3api put_object --bucket "${BUCKET}" --key "${prefix}/" --body /dev/null 2>/dev/null || \
    printf '' | aws s3 cp - "s3://${BUCKET}/${prefix}/" --region "${REGION}"
done

echo "Bucket ready:"
echo "  s3://${BUCKET}/raw/"
echo "  s3://${BUCKET}/processed/"
echo "  s3://${BUCKET}/artifacts/"
