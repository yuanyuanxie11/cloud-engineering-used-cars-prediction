# Used Cars Price Prediction

We built a **cloud-native machine learning pipeline** that estimates used-vehicle listing
prices from structured features (make, year, mileage, condition, location, and related
fields). The focus is **reliable AWS infrastructure**—data lake on S3, least-privilege IAM,
on-demand training compute, and serverless inference—not chasing state-of-the-art model
scores.

### What this project does

1. **Ingest & clean** — Raw Craigslist-style `vehicles.csv` lands in S3; a cleaning job
  produces validated Parquet and JSON quality reports.
2. **Train & version** — EDA and model training (Ridge, Random Forest, XGBoost) run on
  EC2 or ECS; the best model and metrics are stored under `artifacts/models/latest/`.
3. **Serve** — A container Lambda behind API Gateway exposes `POST /predict`; a static
  web form lets anyone submit a vehicle and get a price back in seconds.

Official evaluation numbers come only from training on the **real** processed dataset in
S3—not from local scratch files or the exploratory notebook.

### Try the live demo


|                   |                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| **Web UI**        | Run `[frontend/](frontend/)` locally (see [Run end-to-end](#run-end-to-end))                          |
| **API**           | `POST https://qodruw90xj.execute-api.us-east-1.amazonaws.com/predict` (JSON body; not a browser link) |
| **AWS resources** | Bucket `mlds423-used-cars-project`, region `us-east-1`                                                |


### Documentation

Role-specific and component guides (read this README first, then dive in as needed):


| Guide                                                | Topics                                                                      |
| ---------------------------------------------------- | --------------------------------------------------------------------------- |
| `[README_DATA_ENGINEER.md](README_DATA_ENGINEER.md)` | S3 layout, raw upload, cleaning rules, CLI and S3→Lambda event pipeline     |
| `[README_ML_ENGINEER.md](README_ML_ENGINEER.md)`     | EDA, model training (EC2/ECS), artifacts in S3, `/predict` serving, metrics |
| `[infra/README.md](infra/README.md)`                 | Architecture diagram, IAM, security, cost, Lambda/ECR deploy, CI/CD         |
| `[frontend/README.md](frontend/README.md)`           | Static web UI, local `http.server`, curl examples for the live API          |


Related runbooks (under `infra/`): `[EC2_LAUNCH.md](infra/EC2_LAUNCH.md)` · `[ECS_TRAIN.md](infra/ECS_TRAIN.md)`

---

## Repository structure

```
cloud-engineering-used-cars-prediction/
├── README.md                 # This file — start here
├── README_DATA_ENGINEER.md   # Data processing runbook
├── README_ML_ENGINEER.md     # Modeling runbook
├── requirements.txt          # Full runtime deps (local, train, CI)
├── requirements-dev.txt      # Ruff (lint / CI)
├── config.yaml               # ML features, models, S3 paths (source of truth)
│
├── pipeline/                 # Ingest & clean
│   ├── upload_raw.py         # Upload vehicles.csv → s3://.../raw/
│   ├── clean.py              # Clean CSV/Parquet → processed + JSON report
│   └── config.py             # Bucket prefixes, price/year thresholds
│
├── modeling/                 # EDA & training
│   ├── eda.py                # Plots + summary stats → S3 or local
│   ├── train.py              # Train models, pick best, write artifacts
│   └── features.py           # Shared preprocessing (train + serve)
│
├── lambda/                   # S3 event → clean Lambda
│   ├── handler.py
│   ├── Dockerfile
│   └── deploy.sh
│
├── serving/                  # Predict Lambda (container image)
│   ├── handler.py            # POST /predict — loads model from S3
│   ├── Dockerfile
│   ├── requirements.txt      # Slim deps (no matplotlib)
│   └── deploy.sh
│
├── frontend/                 # Static HTML form → API
│   ├── README.md             # Local UI + curl smoke test
│   └── index.html
│
├── infra/                    # AWS ops
│   ├── README.md             # Architecture, security, cost, deploy
│   ├── architecture.png      # Diagram for slides
│   ├── setup_bucket.sh       # Create bucket + prefixes
│   ├── ec2_train.sh          # EC2 user-data: EDA + train from S3
│   ├── Dockerfile.train      # ECS Fargate training image
│   ├── EC2_LAUNCH.md / ECS_TRAIN.md
│   └── iam_*.json            # Least-privilege policies (committed)
│
├── tests/                    # Unit + optional slow/live tests of each module and whole pipeline
├── scripts/make_sample_parquet.py
├── notebooks/used_cars_ml.ipynb
```

### S3 layout (shared bucket)


| Prefix                            | Contents                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------- |
| `raw/`                            | Original `vehicles.csv`                                                         |
| `processed/`                      | `vehicles_clean.parquet` (~381k rows after default rules)                       |
| `artifacts/eda/`                  | Training EDA plots + `summary_stats.json`                                       |
| `artifacts/models/latest/`        | `**best_model.pkl**`, `model_manifest.json`, `metrics.json` (inference handoff) |
| `artifacts/cleaning_reports/`     | JSON cleaning reports                                                           |
| `artifacts/models/runs/<run_id>/` | Versioned training runs                                                         |


## Architecture

Used Cars ML system architecture

---

## Prerequisites

- **Python 3.10+** (3.11 recommended for local dev; Lambdas use 3.12 images)
- **AWS CLI** configured (`aws sso login` or `AWS_PROFILE` — no long-lived keys in the repo)
- **Docker** (for Lambda container builds and ECS training image)
- **Dataset:** Craigslist-style `vehicles.csv` (team uses ~427k rows)

```bash
cd /path/to/cloud-engineering-used-cars-prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run end-to-end

Choose **local** (free sanity check) or **AWS** (submission demo). Steps build on each other.

### Path A — Local (no AWS charges)

Use this to verify code before cloud demo day.

**4.1 Sample or real data**

```bash
# Small synthetic parquet (if you lack the full CSV):
python scripts/make_sample_parquet.py --output ./out/vehicles_clean.parquet --rows 8000

# Or clean a local CSV:
python -m pipeline.clean \
  --input /path/to/vehicles.csv \
  --output ./out/vehicles_clean.parquet \
  --stats-output ./out/cleaning_report.json
```

**4.2 EDA + train**

```bash
python -m modeling.eda --config config.yaml \
  --local-input ./out/vehicles_clean.parquet \
  --local-output ./out/eda

python -m modeling.train --config config.yaml \
  --local-input ./out/vehicles_clean.parquet \
  --local-output ./out/models
```

Check `./out/models/latest/metrics.json` and `best_model.pkl`.

**4.3 Fast tests**

```bash
python -m unittest discover -s tests -v
# Full pipeline smoke (downloads nothing; uses local parquet):
RUN_SLOW_TESTS=1 python -m unittest tests.test_pipeline_e2e -v
```

---

### Path B — AWS (full cloud demo)

Set defaults once:

```bash
export BUCKET=mlds423-used-cars-project
export AWS_REGION=us-east-1
export AWS_PROFILE=your-sso-profile   # or rely on instance/task role
```

#### Step 1 — Bucket and raw upload

```bash
chmod +x infra/setup_bucket.sh
./infra/setup_bucket.sh "$BUCKET" "$AWS_REGION"

pip install -r requirements.txt
python -m pipeline.upload_raw \
  --file /path/to/vehicles.csv \
  --bucket "$BUCKET"
```

Verify: `aws s3 ls s3://$BUCKET/raw/vehicles.csv`

#### Step 2 — Clean → processed Parquet

**Option A — CLI (one-shot):**

```bash
python -m pipeline.clean \
  --s3-input "s3://$BUCKET/raw/vehicles.csv" \
  --s3-output "s3://$BUCKET/processed/vehicles_clean.parquet" \
  --stats-output "s3://$BUCKET/artifacts/cleaning_reports/latest.json"
```

**Option B — Event-driven (Week 9):** deploy `lambda/` (see `[README_DATA_ENGINEER.md](README_DATA_ENGINEER.md)`); uploading to `raw/` triggers clean automatically.

Verify: `aws s3 ls s3://$BUCKET/processed/vehicles_clean.parquet`

#### Step 3 — Train on EC2 or ECS

**EC2 (simplest for demo):** follow `[infra/EC2_LAUNCH.md](infra/EC2_LAUNCH.md)` — attach `UsedCarsMLTrainRole`, then on the instance:

```bash
export USED_CARS_BUCKET="$BUCKET"
export AWS_REGION="$AWS_REGION"
export TRAINING_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
sudo -E bash infra/ec2_train.sh   # EDA + train; writes to S3
```

**ECS (alternative):** `[infra/ECS_TRAIN.md](infra/ECS_TRAIN.md)` — build `infra/Dockerfile.train`, run Fargate task.

Verify official results:

```bash
aws s3 cp "s3://$BUCKET/artifacts/models/latest/metrics.json" -
aws s3 ls "s3://$BUCKET/artifacts/models/latest/"
```

Terminate the EC2 instance when done to avoid idle charges.

#### Step 4 — Deploy predict Lambda

```bash
./serving/deploy.sh "$AWS_REGION"
# In AWS Console: update Lambda mlds423-predict to the new ECR image
# Env: USED_CARS_BUCKET=$BUCKET
# Role: UsedCarsMLServingRole (iam_serving_policy.json)
```

Wire API Gateway HTTP API: `POST /predict` → Lambda. Details: `[infra/README.md](infra/README.md#deployment--cicd)`.

**Smoke test:**

```bash
curl -X POST "https://qodruw90xj.execute-api.us-east-1.amazonaws.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"manufacturer":"toyota","year":2018,"odometer":45000,"condition":"excellent","cylinders":"4 cylinders","fuel":"gas","transmission":"automatic","drive":"fwd","type":"sedan","paint_color":"white","state":"ca","title_status":"clean","region":"los angeles"}]}'
```

Expected: HTTP 200 with `predictions`, `model`, and `run_id`. **Warm up** one request ~30s before a live demo to avoid cold-start delay.

#### Step 5 — Frontend

```bash
cd frontend
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765 — form POSTs to the API URL in index.html
```

The UI only calls **predict**; it does not run clean, train, or deploy.

#### Step 6 — Monitoring

- **Logs:** `/aws/lambda/mlds423-predict`, `/aws/lambda/mlds423-clean-on-upload`, `/ecs/used-cars-train`, `/used-cars/ec2-train` (CloudWatch agent enabled on EC2)
- **Metrics:** CloudWatch namespace `UsedCarsML` after training (`TestR2`, `TestRMSE`, etc.)
- Alarms in the AWS Console when model r-squared is below 0.6

