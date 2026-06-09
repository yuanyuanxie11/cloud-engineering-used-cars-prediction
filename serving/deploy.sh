#!/usr/bin/env bash
# Build + push the inference container to ECR for the /predict Lambda.
# Usage: ./serving/deploy.sh us-east-1
#
# After pushing, create/update a container-image Lambda from this URI with:
#   - Execution role granting s3:GetObject on
#       arn:aws:s3:::mlds423-used-cars-project/artifacts/models/latest/*
#   - Env var USED_CARS_BUCKET=mlds423-used-cars-project
#   - Memory >= 1024 MB, timeout >= 30s (model load on cold start)
#   - Front it with an API Gateway HTTP API route: POST /predict

set -euo pipefail

REGION="${1:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_NAME="mlds423-used-cars-serving"
IMAGE_TAG="latest"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"

aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

# Use buildx with --provenance=false so the pushed image stays in the
# Docker V2 Schema 2 manifest format. Modern Docker (24+) defaults to
# OCI manifests with a provenance attestation, which AWS Lambda's
# container support currently rejects with:
#   "The image manifest, config or layer media type ... is not supported."
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --load \
  -f serving/Dockerfile \
  -t "${REPO_NAME}:${IMAGE_TAG}" .
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Image pushed: ${IMAGE_URI}"
echo "Create/update the inference Lambda from this URI (see header for IAM + API Gateway notes)."
