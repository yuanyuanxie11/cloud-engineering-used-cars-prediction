#!/usr/bin/env bash
# Build and push Lambda container image to ECR.
# Usage: ./lambda/deploy.sh us-east-1

set -euo pipefail

REGION="${1:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_NAME="mlds423-used-cars-clean"
IMAGE_TAG="latest"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"

aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${REGION}" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "${REPO_NAME}" --region "${REGION}"

aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}"

docker build --platform linux/amd64 -f lambda/Dockerfile -t "${REPO_NAME}:${IMAGE_TAG}" .
docker tag "${REPO_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Image pushed: ${IMAGE_URI}"
echo "Use this URI when creating/updating the Lambda function."
