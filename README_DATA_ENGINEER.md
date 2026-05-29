# Person 1: Data Engineer — Pipeline & Storage

This folder is your deliverable for MSiA 423 (Cloud Engineering). It covers **Week 4–5** (S3 + cleaning) and the **Week 9** stretch goal (S3 → Lambda).

## S3 layout

| Prefix | Purpose |
|--------|---------|
| `s3://used-cars-project/raw/` | Original `vehicles.csv` (immutable) |
| `s3://used-cars-project/processed/` | Cleaned Parquet for ML / EDA |
| `s3://used-cars-project/artifacts/` | Cleaning JSON reports, pipeline logs |

Create the bucket once (team agrees on name/region):

```bash
chmod +x infra/setup_bucket.sh
./infra/setup_bucket.sh used-cars-project us-east-1
```

## Dataset snapshot (your `vehicles.csv`)

| Metric | Value |
|--------|------:|
| Rows | 426,880 |
| Columns | 26 |
| Duplicate `id` | 0 |
| `price == 0` | 32,895 |
| `price >= $1B` | 9 |
| `price < $100` | 36,222 |
| `county` column | 100% null → **dropped** |

After default cleaning (`MIN_PRICE=100`, `MAX_PRICE=500000`, valid year, odometer cap), you get **381,063** rows (45,817 removed from 426,880).

## Week 4: Upload raw data

```bash
cd /Users/ericwuu/Downloads/used-cars-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export AWS_PROFILE=your-profile   # or use instance role on EC2

python -m pipeline.upload_raw \
  --file /Users/ericwuu/Downloads/vehicles.csv \
  --bucket used-cars-project
```

## Week 5: Clean and write processed data

**Local test (no AWS):**

```bash
python -m pipeline.clean \
  --input /Users/ericwuu/Downloads/vehicles.csv \
  --output ./out/vehicles_clean.parquet \
  --stats-output ./out/cleaning_report.json
```

**Cloud run:**

```bash
python -m pipeline.clean \
  --s3-input s3://used-cars-project/raw/vehicles.csv \
  --s3-output s3://used-cars-project/processed/vehicles_clean.parquet \
  --stats-output s3://used-cars-project/artifacts/cleaning_reports/latest.json
```

Share `processed/vehicles_clean.parquet` with Person 2 (EDA/training) and Person 3 (inference).

Person 2 uses the same default bucket name **`used-cars-project`** — see [`README_ML_ENGINEER.md`](README_ML_ENGINEER.md).

## Cleaning rules (document in team wiki)

1. Drop `county` (always null).
2. `drop_duplicates(subset=["id"])`.
3. **Price:** keep `100 <= price <= 500_000` and exclude `>= 1_000_000_000`.
4. **Year:** `1980 <= year <= current_year + 1`, non-null.
5. **Odometer:** if present, `0 <= odometer <= 1_000_000`.
6. Strip/lowercase key categoricals for consistent encoding.

Tune thresholds via env vars (`MIN_PRICE`, `MAX_PRICE`, etc.) — see `pipeline/config.py`.

## Week 9: Event-driven pipeline (deployment option)

1. Package `pipeline/` + `lambda/handler.py` + dependencies (Lambda layer with pandas/pyarrow, or container image).
2. Create Lambda with IAM: `s3:GetObject` on `raw/*`, `s3:PutObject` on `processed/*` and `artifacts/*`.
3. S3 event notification: `s3:ObjectCreated:*` on prefix `raw/`, suffix `.csv` → Lambda.
4. Upload a new CSV to `raw/` and verify Parquet + JSON report appear automatically.

**Lambda env vars:**

```
USED_CARS_BUCKET=used-cars-project
RAW_PREFIX=raw/
PROCESSED_PREFIX=processed/
ARTIFACTS_PREFIX=artifacts/
```

## Your presentation talking points

- **Lifecycle:** raw CSV → validated Parquet → consumed by training on EC2/ECS.
- **Idempotency:** same `id` deduped; reports versioned under `artifacts/cleaning_reports/`.
- **Security:** no keys in code; use IAM roles / `AWS_PROFILE` / Parameter Store.
- **Cloud requirement:** satisfies “Data stored in S3” (Week 4–5) and “S3 → Lambda” (Week 9).

## Handoff checklist

- [ ] Bucket created and documented in architecture diagram
- [ ] `raw/vehicles.csv` uploaded
- [ ] `processed/vehicles_clean.parquet` uploaded
- [ ] `artifacts/cleaning_reports/latest.json` with row counts
- [ ] README / thresholds shared with ML teammate
- [ ] (Week 9) Lambda demo recorded for final presentation
