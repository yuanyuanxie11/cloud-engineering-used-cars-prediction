# Person 2: ML Engineer — Modeling, Training & Deployment

MSiA 423 used-cars team project. This deliverable covers **Week 6–8** (EDA, baselines, challenger), **Week 9** cloud training on **EC2 / ECS → S3**, and a **`/predict` inference service** (API Gateway + Lambda).

## What runs where

| Concern | Path | Code |
|---------|------|------|
| Train + store artifacts (EC2) | EC2 user-data → S3 | [`infra/ec2_train.sh`](infra/ec2_train.sh) |
| Train + store artifacts (ECS) | Fargate task → S3 | [`infra/Dockerfile.train`](infra/Dockerfile.train), [`infra/ECS_TRAIN.md`](infra/ECS_TRAIN.md) |
| Online inference | API Gateway → Lambda → S3 model | [`serving/`](serving/) |
| Monitoring | CloudWatch metrics + logs | `monitoring:` in `config.yaml` |

> **Source of truth:** the only official metrics + artifacts come from
> `python -m modeling.train` run on the **real** `processed/vehicles_clean.parquet`.
> The notebook and any local `out/` files (synthetic, git-ignored) are exploratory
> scratch — do **not** quote their numbers.

## Runbook — produce the official results (real data on AWS)

Follow this in order on demo day. Pick **one** of EC2 (simplest) or ECS Fargate.

### Step 0 — Prerequisites (verify before launching)

```bash
export BUCKET=mlds423-used-cars-project
export AWS_REGION=us-east-1

# (a) Real processed data exists in S3 (Person 1)
aws s3 ls s3://$BUCKET/processed/vehicles_clean.parquet

# (b) The training IAM role exists ...
aws iam get-role --role-name UsedCarsMLTrainRole \
  --query 'Role.[RoleName,Arn]' --output table

# ... with the right permissions attached (inline and/or managed policies)
aws iam list-role-policies          --role-name UsedCarsMLTrainRole   # inline
aws iam list-attached-role-policies --role-name UsedCarsMLTrainRole   # managed
# Inspect an inline policy's JSON (expect s3 Get on processed/*,
# s3 Put/List on artifacts/*, and cloudwatch:PutMetricData):
aws iam get-role-policy --role-name UsedCarsMLTrainRole --policy-name <policy-name>

# (c) For EC2, the role must be exposed via an instance profile:
aws iam list-instance-profiles-for-role --role-name UsedCarsMLTrainRole
```

If `get-role` returns `NoSuchEntity`, the role hasn't been created yet — see
[`infra/EC2_LAUNCH.md`](infra/EC2_LAUNCH.md) (EC2) or
[`infra/ECS_TRAIN.md`](infra/ECS_TRAIN.md) (ECS) for the policy JSON to attach.

### Step 1 — Get the code onto the instance (no GitHub needed)

`ec2_train.sh` normally `git clone`s the repo. Without a Git remote, ship the
code via S3 instead (the EC2 role can already read `artifacts/*`).

```bash
# On your laptop — package + upload
cd /path/to/parent-of-project
zip -r used-cars-project.zip used-cars-project \
  -x '*/.venv/*' '*/__pycache__/*' '*/out/*' '*/.git/*' '*.pyc' '*/.DS_Store'
aws s3 cp used-cars-project.zip \
  s3://mlds423-used-cars-project/artifacts/code/used-cars-project.zip
```

### Step 2 — Launch EC2 (Console)

EC2 → Launch instance, in **us-east-1**:

| Setting | Value |
|---------|-------|
| AMI | Amazon Linux 2023 |
| Instance type | t3.xlarge (4 vCPU / 16 GB) |
| Key pair | "Proceed without a key pair" (use EC2 Instance Connect) |
| Security group | Allow SSH (22) from *My IP* |
| **Advanced → IAM instance profile** | **`UsedCarsMLTrainProfile`** |

Launch → wait for *Running* → **Connect → EC2 Instance Connect**.

### Step 3 — Confirm the role works, then pull code + train (on the instance)

```bash
aws sts get-caller-identity                                  # expect .../UsedCarsMLTrainRole/...
aws s3 ls s3://mlds423-used-cars-project/processed/          # GetObject/List works

sudo dnf install -y unzip
aws s3 cp s3://mlds423-used-cars-project/artifacts/code/used-cars-project.zip /tmp/code.zip
unzip -o /tmp/code.zip -d /home/ec2-user/                    # → /home/ec2-user/used-cars-project
cd /home/ec2-user/used-cars-project

export USED_CARS_BUCKET=mlds423-used-cars-project
export AWS_REGION=us-east-1
export TRAINING_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
sudo -E bash infra/ec2_train.sh    # -E preserves env; runs EDA + train, reads S3, writes S3
```

**ECS path (alternative):** follow [`infra/ECS_TRAIN.md`](infra/ECS_TRAIN.md) (build image → register task → `run-task`); no zip needed since the code is baked into the image.

### Step 4 — Verify the official numbers + artifacts (back on your laptop)

```bash
aws s3 cp s3://mlds423-used-cars-project/artifacts/models/latest/metrics.json -
aws s3 ls s3://mlds423-used-cars-project/artifacts/models/latest/
# Expect: best_model.pkl, model_manifest.json, metrics.json, ridge/random_forest/xgboost(.native).pkl
```

`metrics.json` → the numbers for your deck (read `best_model` + its `test` block).
`best_model.pkl` → exactly what the `/predict` Lambda loads.

### Step 5 — (optional) Check monitoring landed

```bash
aws cloudwatch list-metrics --namespace UsedCarsML --region $AWS_REGION
tail -50 /var/log/train.log            # EC2; or CloudWatch Logs for ECS/Lambda
```

### Step 6 — Stop the instance (avoid idle charges)

Training is a one-off job, so **terminate** (or stop) the EC2 instance once
artifacts are in S3:

```bash
aws ec2 terminate-instances --instance-ids <instance-id> --region us-east-1
```

### Local pre-flight (optional, free)

Identical code + seed ⇒ identical metrics, so you can sanity-check on the **real**
parquet locally first (download it once), but this earns no cloud credit:

```bash
aws s3 cp s3://mlds423-used-cars-project/processed/vehicles_clean.parquet /tmp/real.parquet
python -m modeling.train --config config.yaml --local-input /tmp/real.parquet --local-output ./out/models
```

## S3 layout (shared bucket with Person 1)

Default bucket: **`mlds423-used-cars-project`** (override with `USED_CARS_BUCKET`).

| Prefix | Owner | Contents |
|--------|-------|----------|
| `processed/` | Person 1 | `vehicles_clean.parquet` |
| `artifacts/eda/` | Person 2 | PNG plots, `summary_stats.json` |
| `artifacts/models/runs/<run_id>/` | Person 2 | Versioned training run |
| `artifacts/models/latest/` | Person 2 | **Person 3 handoff** (always updated) |
| `artifacts/models/*.pkl` | Person 2 | Legacy flat paths (backward compatible) |

## Configuration

All hyperparameters and feature lists live in [`config.yaml`](config.yaml). Nothing is hardcoded in Python.

| Override | Purpose |
|----------|---------|
| `USED_CARS_BUCKET` | S3 bucket name |
| `AWS_REGION` | AWS region |
| `TRAINING_RUN_ID` | Version folder under `artifacts/models/runs/` |

## Local development (no AWS)

```bash
cd /path/to/mlds423-used-cars-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If Person 1 parquet is unavailable, generate a small sample:
python scripts/make_sample_parquet.py --output ./out/vehicles_clean.parquet --rows 8000

# Or use real cleaned data from Person 1:
# python -m pipeline.clean --input /path/to/vehicles.csv --output ./out/vehicles_clean.parquet

python -m modeling.eda --config config.yaml \
  --local-input ./out/vehicles_clean.parquet \
  --local-output ./out/eda

python -m modeling.train --config config.yaml \
  --local-input ./out/vehicles_clean.parquet \
  --local-output ./out/models
```

Outputs under `./out/models/latest/`: `best_model.pkl`, `model_manifest.json`, `metrics.json`, etc.

## EC2 training (production path)

1. Create IAM role **`UsedCarsMLTrainRole`** with:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::mlds423-used-cars-project/processed/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::mlds423-used-cars-project/artifacts/*",
        "arn:aws:s3:::mlds423-used-cars-project"
      ]
    }
  ]
}
```

2. Launch **t3.xlarge** (or c5.2xlarge) with the role attached.
3. User-data or SSH:

```bash
export USED_CARS_BUCKET=mlds423-used-cars-project
export AWS_REGION=us-east-1
export GIT_REPO_URL=https://github.com/your-team/mlds423-used-cars-project.git
export TRAINING_RUN_ID=20260528-demo
chmod +x infra/ec2_train.sh
sudo infra/ec2_train.sh
```

4. Verify in S3 console:

- `s3://mlds423-used-cars-project/artifacts/eda/`
- `s3://mlds423-used-cars-project/artifacts/models/latest/best_model.pkl`
- `s3://mlds423-used-cars-project/artifacts/models/latest/metrics.json`

Logs: `/var/log/train.log` on the instance (stdout includes `STRUCTURED {...}` JSON lines for CloudWatch).

## Models trained

| Model | Role | Config section |
|-------|------|----------------|
| Ridge | Week 7 linear baseline | `training.ridge` |
| Random Forest | Tree baseline using sklearn | `training.random_forest` |
| XGBoost (one-hot) | Week 8 challenger using shared sklearn preprocessing | `training.xgboost` (+ early stopping) |
| XGBoost (native categorical) | Extra challenger using pandas `category` dtype | `training.xgboost_native` |

The model with the highest **cross-validated** R² (training folds only) is saved as **`best_model.pkl`**; the held-out test set is reported once for the winner and never used for selection.

**XGBoost early stopping** is set via the constructor (`early_stopping_rounds`) as required by xgboost ≥ 2.0; the preprocessor is fit on the training slice only (the early-stopping validation slice is held out first) so validation statistics never leak. The chosen tree count is logged (`best_iteration`).

**Log target:** `training.use_log_target` is `true` — models fit on `log1p(price)` and predictions are inverted with `expm1`, which suits the right-skewed price distribution. The manifest records this so inference applies the inverse automatically.

## Feature and encoding strategy

All models predict the same target:

- `y`: cleaned `price`
- `X`: vehicle features after `vehicle_age = current_year - year` and configured column drops

Shared sklearn preprocessing for Ridge, Random Forest, and one-hot XGBoost:

| Feature group | Columns | Transformation |
|---------------|---------|----------------|
| Numeric | `odometer`, `vehicle_age` | Median imputation + `StandardScaler` |
| Ordinal categorical | `condition`, `cylinders`, `title_status` | Most-frequent imputation + `OrdinalEncoder` |
| Nominal categorical | `manufacturer`, `fuel`, `transmission`, `drive`, `type`, `paint_color`, `state` | Most-frequent imputation + `OneHotEncoder(max_categories=20)` |

Why this mix:

- Ridge requires numeric encoded inputs and benefits from scaling.
- sklearn `RandomForestRegressor` does **not** support raw string categoricals, so it still needs numeric encoding.
- One-hot XGBoost uses the same transformed matrix as Ridge/RF for a fair comparison.
- Native-categorical XGBoost is trained separately with pandas `category` columns and `enable_categorical=True`; it can also test `region` without one-hot expansion.

No embedding encoding is used. Embeddings would require a neural network and add unnecessary complexity for this cloud-engineering-focused project.

## Hyperparameter strategy

The defaults in `config.yaml` are conservative starting points, not claims of being globally optimal.

- Random Forest: tune `n_estimators`, `max_depth`, `min_samples_leaf`, and optionally `max_features`.
- XGBoost: tune `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
- Use cross-validation on the training set only; keep the held-out test set for final reporting.
- The notebook includes an optional `RandomizedSearchCV` section (`RUN_TUNING = True`) for experimenting before copying better values back into `config.yaml`.

## Handoff to Person 3 (inference)

- Load `artifacts/models/latest/best_model.pkl`
- Read `artifacts/models/latest/model_manifest.json` for `config_hash`, `use_log_target`, `raw_feature_columns`, and `native_categorical_model`
- Input: one row as a DataFrame with the same columns as **processed** parquet (before `prepare_xy` drops)
- Example:

```python
import pickle
import pandas as pd
from modeling.features import prepare_xy
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
with open("best_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

row = pd.read_parquet("vehicles_clean.parquet").iloc[[0]]
X, _ = prepare_xy(row, cfg)
print(pipeline.predict(X))
```

If `native_categorical_model` is true in the manifest, the best model is not a sklearn preprocessing pipeline. Convert the configured categorical columns to pandas `category` dtype before calling `predict`. If `use_log_target` is true, apply `np.expm1` to predictions.

The [`serving/`](serving/) Lambda already implements all of this (manifest-driven native + log-target handling), so Person 3 can deploy it directly or use it as a reference.

## Online inference deployment (`serving/`)

A container-image Lambda exposes `POST /predict` behind an API Gateway HTTP API.

- Loads `best_model.pkl` + `model_manifest.json` from S3 **once per container** (cached across warm invocations).
- Reuses `modeling.features.prepare_xy` so serve-time preprocessing is identical to training; reads `use_log_target` / `native_categorical_model` from the manifest and adapts automatically.
- **Security:** execution role only needs `s3:GetObject` on `artifacts/models/latest/*`; no keys in code.

```bash
# Build + push the inference image to ECR
./serving/deploy.sh us-east-1
# Then create a container-image Lambda from the printed URI, set
# USED_CARS_BUCKET, attach the read-only S3 role, and add an API Gateway
# POST /predict route. Example request body:
#   {"instances": [{"manufacturer":"toyota","year":2018,"odometer":45000, ...}]}
# Response: {"predictions": [23479.82], "model": "...", "run_id": "..."}
```

## Monitoring (requirement 5)

- **Logs:** every run emits single-line `STRUCTURED {...}` JSON (CloudWatch-friendly). On ECS/Lambda these ship to CloudWatch automatically; on EC2 they land in `/var/log/train.log` (add the CloudWatch agent to forward).
- **Metrics:** when `monitoring.push_metrics` is true and running against S3, training publishes the winner's `TestR2`, `TestRMSE`, `TestMAE`, and `CVR2Mean` to the CloudWatch namespace `UsedCarsML` (best-effort; never breaks training). Build an alarm on `TestR2` to catch model regressions.

## Presentation talking points

- **Data contract:** Person 1 cleaning (`MIN_PRICE`, `MAX_PRICE`, year/odometer caps) — ML does not re-filter price.
- **Reproducibility:** YAML config + fixed `random_seed`; versioned `runs/<run_id>/`.
- **Security:** No AWS keys in repo; EC2/ECS task role and read-only inference role only.
- **Cloud demo:** EC2 *or* ECS Fargate → EDA + train → S3 artifacts (*Model Training & Artifact Storage*), plus API Gateway + Lambda `/predict` (*Inference via exposed web application*).
- **Model story:** Ridge → Random Forest → one-hot XGBoost (early stopping) → native-categorical XGBoost; pick best by **cross-validated** R², fit on `log1p(price)`.

## Coordination checklist (Person 1)

- [ ] Bucket `mlds423-used-cars-project` created
- [ ] `processed/vehicles_clean.parquet` uploaded (~381k rows)
- [ ] Cleaning thresholds documented in team wiki

## Module reference

| Module | Command |
|--------|---------|
| EDA | `python -m modeling.eda` |
| Train | `python -m modeling.train` |
| Features | `from modeling.features import prepare_xy, build_preprocessor` |
| Inference Lambda | `serving/handler.py` (`./serving/deploy.sh`) |
| ECS training | `infra/Dockerfile.train` + `infra/ECS_TRAIN.md` |
| Notebook | `notebooks/used_cars_ml.ipynb` |
