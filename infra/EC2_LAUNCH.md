# EC2 launch checklist (Person 2)

Use this when running [`ec2_train.sh`](ec2_train.sh) for the Week 9 demo.

## Prerequisites

- [ ] S3 bucket `mlds423-used-cars-project` (or your team name) exists
- [ ] `s3://<bucket>/processed/vehicles_clean.parquet` uploaded by Person 1
- [ ] IAM role attached to EC2 with `GetObject` on `processed/*` and `PutObject` on `artifacts/*`

## Launch

1. AMI: Amazon Linux 2023
2. Instance: `t3.xlarge` (minimum)
3. IAM instance profile: role above
4. User data (example):

```bash
#!/bin/bash
export USED_CARS_BUCKET=mlds423-used-cars-project
export AWS_REGION=us-east-1
export GIT_REPO_URL=https://github.com/YOUR_ORG/mlds423-used-cars-project.git
export TRAINING_RUN_ID=demo-$(date -u +%Y%m%dT%H%M%SZ)
curl -fsSL https://raw.githubusercontent.com/YOUR_ORG/mlds423-used-cars-project/main/infra/ec2_train.sh | bash
```

Or clone the repo and run `infra/ec2_train.sh` after SSH.

## Verify

```bash
aws s3 ls s3://mlds423-used-cars-project/artifacts/models/latest/
aws s3 cp s3://mlds423-used-cars-project/artifacts/models/latest/metrics.json -
tail -50 /var/log/train.log
```

Expected keys: `best_model.pkl`, `model_manifest.json`, `metrics.json`, `ridge.pkl`, `random_forest.pkl`, `xgboost.pkl`, `xgboost_native.pkl`
