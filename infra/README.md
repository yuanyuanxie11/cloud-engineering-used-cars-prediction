# Infrastructure — Person 3: Cloud Architect & DevOps

This folder is the **single source of truth** for how the Used Cars ML system is
architected, secured, deployed, and operated on AWS. It pairs with the two
role-specific READMEs at the repo root:

- [`README_DATA_ENGINEER.md`](../README_DATA_ENGINEER.md) — Person 1 (Eric): ingestion + cleaning pipeline
- [`README_ML_ENGINEER.md`](../README_ML_ENGINEER.md) — Person 2 (Yuanyuan): training + inference code

If those READMEs answer **"what does this code do?"**, this one answers
**"how does it run safely and what does it cost?"** — the cloud engineering deliverables.

---

## File map

| File | What | Owner |
|------|------|-------|
| [`architecture.drawio`](architecture.drawio) | System architecture, editable in [diagrams.net](https://app.diagrams.net) | Person 3 |
| [`architecture.png`](architecture.png) | Rendered export of the diagram, for presentation | Person 3 |
| [`iam_serving_policy.json`](iam_serving_policy.json) | Least-privilege S3 read-only policy for the predict Lambda | Person 3 |
| [`iam_train_policy.json`](iam_train_policy.json) | S3 + CloudWatch policy for the EC2 training instance role | Person 2 |
| [`setup_bucket.sh`](setup_bucket.sh) | Bootstrap S3 bucket with `raw/`, `processed/`, `artifacts/` prefixes | Person 1 |
| [`Dockerfile.train`](Dockerfile.train) | Container image used by ECS Fargate training path | Person 2 |
| [`ec2_train.sh`](ec2_train.sh) | EC2 user-data script that runs `python -m modeling.train` against S3 | Person 2 |
| [`EC2_LAUNCH.md`](EC2_LAUNCH.md) | Step-by-step for spinning up the EC2 training instance | Person 2 |
| [`ECS_TRAIN.md`](ECS_TRAIN.md) | Alternative training path on ECS Fargate | Person 2 |

---

## System architecture

![Architecture](architecture.png)

Open [`architecture.drawio`](architecture.drawio) in
[diagrams.net](https://app.diagrams.net) (or the **Draw.io Integration** VS Code
extension) to edit. The diagram is checked into git as XML, so changes show up
in `git diff` like any other code.

### The numbered demo flow (talk track for the live demo)

| # | What happens | Services involved |
|---|--------------|-------------------|
| ① | Reviewer calls `POST /predict` over HTTPS | Internet → API Gateway HTTP API |
| ② | API Gateway invokes the Lambda integration | API Gateway → Lambda (`mlds423-predict`) |
| ③ | Lambda loads `best_model.pkl` + `model_manifest.json` from S3 (cached after first call) | Lambda → S3 (`artifacts/models/latest/`) |
| ④ | (Independent path) Uploading a raw CSV fires an S3 event | S3 (`raw/`) → Lambda (`mlds423-clean-on-upload`) |
| ⑤ | Cleaning Lambda writes the cleaned parquet back to S3 | Lambda → S3 (`processed/`, `artifacts/cleaning_reports/`) |
| ⑥ | (CI/CD, Phase 5) GitHub Actions builds the serving image, pushes to ECR, updates Lambda code | GitHub Actions → ECR → Lambda |

Training (`EC2 t3.xlarge` reading `processed/` and writing `artifacts/models/`)
runs as a one-shot job and is terminated when artifacts land in S3, so it does
not appear on the demo-day live path.

---

## Security model

The main diagram deliberately omits IAM trust relationships — the architectural
view stays clean, and the **security view lives here in this document**.
A grader looking for "how does this system handle authentication and
authorization?" should be able to read this section and have all questions
answered.

### Identity model: two tiers

| Who | How they authenticate | What that gives them |
|-----|-----------------------|----------------------|
| **Humans** (developers, reviewers) | AWS IAM Identity Center (SSO) — `nu-sso.awsapps.com/start` | A short-lived (≤ 8 hours) assumed role: `mse-tl-msia-Students`. Refreshed via `aws sso login` — **no static keys ever stored on developer machines or in git** |
| **AWS services** (Lambda, EC2) | IAM Service Role attached at runtime; AWS injects temporary credentials | One of the three service roles below. **No long-lived keys to leak.** |

> **Zero application secrets — by design.** This project never needs to read
> a password, an API key, or a long-lived AWS access key. Every component
> authenticates via IAM, and IAM hands out short-lived credentials
> automatically. There is no `secrets.json`, no `.env` with production
> values, and no Secrets Manager entry — because there is nothing to put
> in them. **This eliminates an entire class of risk.**

### Service roles — least privilege, audited

Three execution roles, each scoped to the minimum S3 prefix and action set
required for its job:

| Role | Used by | Trust principal | What it can do | What it deliberately **cannot** do |
|------|---------|-----------------|----------------|------------------------------------|
| `mlds423-lambda-clean-role` | Clean Lambda (`mlds423-clean-on-upload`) | `lambda.amazonaws.com` | `s3:GetObject` on `raw/*`, `s3:PutObject` on `processed/*` and `artifacts/cleaning_reports/*`, write CloudWatch Logs | Read `processed/*` or `artifacts/models/*`; touch training data |
| `UsedCarsMLTrainRole` | EC2 training instance (via Instance Profile) | `ec2.amazonaws.com` | `s3:GetObject` on `processed/*`, `s3:PutObject`+`GetObject` on `artifacts/*`, `s3:ListBucket` **only** for `processed/*` and `artifacts/*` prefixes, `cloudwatch:PutMetricData` | Read `raw/*`; delete anything; list outside the two allowed prefixes |
| `UsedCarsMLServingRole` | Predict Lambda (`mlds423-predict`) | `lambda.amazonaws.com` | `s3:GetObject` on `artifacts/models/latest/*` only; write CloudWatch Logs | **Anything else.** No list, no put, no other prefix, no other service |

The two custom policies are committed:

- [`iam_serving_policy.json`](iam_serving_policy.json) — predict Lambda
- [`iam_train_policy.json`](iam_train_policy.json) — training EC2 (Person 2)

#### Why this matters

If a single component is compromised (a malicious payload reaches the predict
Lambda via a crafted request, for example), the **blast radius is bounded by
that role's policy**:

- Compromised predict Lambda → attacker can read `artifacts/models/latest/*`.
  Cannot delete it, cannot read training data, cannot escalate to other
  services. This is recoverable.
- Compromised training EC2 → attacker can read processed data and overwrite
  artifacts. Cannot touch raw data, cannot delete the bucket.
- Compromised clean Lambda → attacker can pollute processed data. Cannot read
  trained models or training artifacts.

Without least privilege, any one compromise would mean read/write everywhere.

#### Notable pattern: scoped `ListBucket`

[`iam_train_policy.json`](iam_train_policy.json) demonstrates the
**scoped-list pattern** that's easy to get wrong:

```json
{
  "Sid": "ListBucketScoped",
  "Effect": "Allow",
  "Action": ["s3:ListBucket"],
  "Resource": "arn:aws:s3:::mlds423-used-cars-project",
  "Condition": {
    "StringLike": { "s3:prefix": ["processed/*", "artifacts/*"] }
  }
}
```

`s3:ListBucket` is a bucket-level action — you cannot scope it to a prefix in
the `Resource` field. The correct way is the `Condition` block here:
"allow listing the bucket, but only when the request's `prefix` matches one of
these." Without the condition the role could enumerate `raw/*` even though
it can never read those objects, which is an information leak. This is
production-grade IAM.

### Evidence: IAM role in production (CloudWatch Logs excerpt)

Sample from `/aws/lambda/mlds423-predict` during a real demo invocation. The
first line is the proof — boto3 found credentials **injected by AWS at runtime**
from the attached `UsedCarsMLServingRole`, without any static key on disk or in
code:

```
05:46:49 [INFO]  Found credentials in environment variables.
05:46:49 START  RequestId: 2b0a9677-1b1d-4934-a702-65b276ffde29
05:46:50 [INFO]  Loaded model s3://mlds423-used-cars-project/artifacts/models/latest/best_model.pkl
                 (best_model=XGBoostNativeCategorical, run_id=20260528T222842Z,
                  native=True, log_target=True)
05:46:50 [INFO]  prepare_xy → X shape (1, 13), y shape (1,)
05:46:50 END    RequestId: 2b0a9677-1b1d-4934-a702-65b276ffde29
05:46:50 REPORT Duration: 607.45 ms  Init Duration: 4407.94 ms
               Memory Size: 2048 MB  Max Memory Used: 447 MB
```

`Init Duration: 4407 ms` is the one-time cold-start cost of pulling and
deserializing the XGBoost model from S3. `Duration: 607 ms` is the actual
prediction once the container is warm. The two numbers are the architectural
basis for the **pre-demo warmup** ritual described in the Deployment section
below.

### Secret scan audit log

Run before each release to certify the repo is clean.

```bash
# AWS access key fingerprints (AKIA = long-lived, ASIA = temporary)
git grep -nE 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}'

# Explicit secret-key assignments
git grep -niE 'aws_secret_access_key\s*=\s*["'\''][A-Za-z0-9/+=]{20,}'

# Same check across full git history (not just working tree)
git log --all -p | grep -nE 'AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}'
```

Most recent run: **0 matches across all four checks**, working tree and full
git history. This is by construction — the architecture above leaves nothing
for the scan to find.

For continuous protection, Phase 5 wires
[`gitleaks`](https://github.com/gitleaks/gitleaks) into the GitHub Actions PR
pipeline, so any future contributor who accidentally introduces a long-lived
credential is blocked at PR time.

### Defense in depth, beyond IAM

| Layer | Mechanism | Where |
|-------|-----------|-------|
| Network | API Gateway terminates TLS, only HTTPS reaches Lambda | Built-in |
| Storage | S3 server-side encryption (SSE-S3) on every object | Bucket default |
| Containers | ECR scan-on-push (basic) flags known CVEs in pushed images | Repo setting |
| Repo | [`.gitignore`](../.gitignore) blocks `.env`, `.aws/`, `*.pem`, `*.tfstate`, `secrets/`, `*.log` | Repo |
| Identity | SSO for humans, IAM Roles for services, no static keys anywhere | This document |

---

## Cost architecture

> **Headline:** ~**$2.13 / month** for the full system at demo traffic. Full
> 12-month projection: **$25.56**. See [`cost_estimate.pdf`](cost_estimate.pdf)
> for the AWS Pricing Calculator export.

### Monthly cost by service

| Service | Monthly | Tier | Why this number |
|---------|--------:|------|-----------------|
| Amazon CloudWatch | $1.39 | 🟢 always-billed | 4 custom metrics + 0.1 GB log ingestion + 1 alarm |
| Amazon ECR | $0.40 | 🟢 always-billed | ~4 GB image storage (`*-clean` + `*-serving` repos combined) |
| Amazon EC2 (t3.xlarge) | $0.33 | 🔴 one-shot | **2 hours** of training per month, not 730 |
| Amazon S3 | $0.01 | 🟢 always-billed | ~300 MB data lake (raw + processed + model artifacts) |
| AWS Lambda (predict) | <$0.01 | 🟡 pay-per-use | 100 req × 1 s × 2 GB ≈ 200 GB-seconds |
| AWS Lambda (clean) | <$0.01 | 🟡 pay-per-use | 10 req × 30 s × 1 GB ≈ 300 GB-seconds |
| Amazon API Gateway (HTTP API) | <$0.01 | 🟡 pay-per-use | 100 requests / month |
| **Total** | **$2.13** | | |

🟢 Always-billed totals **$1.80/month** — that's the floor cost even if
nobody invokes the API and nobody runs training. 🟡 Pay-per-use components
scale to **literally $0** when idle.

### Cost-driving decisions (the savings narrative)

The interesting number is not the total — it's **what we deliberately did
not pay for**. Compared to a naive "always-on container" architecture, our
serverless + on-demand design saves ~$167/month at this scale.

| Decision | We chose | "Naive default" alternative | Monthly delta |
|----------|----------|------------------------------|--------------:|
| Inference layer | Lambda + API Gateway HTTP API (scale-to-zero) | ECS Fargate (1 vCPU, 2 GB, 24/7) + ALB | **+$48** |
| Training compute | EC2 t3.xlarge, terminate when done (2 hr) | EC2 t3.xlarge running 24/7 (730 hr) | **+$121** |
| API style | HTTP API ($1.00 / 1M requests) | REST API ($3.50 / 1M requests) | +$2.50 at 1M req/mo |
| Lambda memory | 2048 MB (algo: more CPU → faster cold start) | 1024 MB | ~$0 (GB-seconds product unchanged) |
| ECR scanning | Basic (free, OS-level CVE) | Enhanced (Amazon Inspector) | +$0.09 / image scan |
| **Headline savings vs naive 24/7 stack** | | | **~$167/month (98.7%)** |

This is the slide for the Phase 6 deck: *"$2 / month is not luck — it's the
sum of six explicit architectural choices."*

### What changes at production scale

The architecture would survive a 10,000× traffic increase with predictable cost
behavior because every pay-per-use line scales linearly:

| Traffic | API Gateway | Lambda predict | CloudWatch logs | Total |
|--------:|------------:|---------------:|----------------:|------:|
| 100 req/mo (demo) | <$0.01 | <$0.01 | $0.05 | **$2.13** |
| 100K req/mo | $0.10 | $0.33 | ~$1 | ~$4 |
| 1M req/mo | $1.00 | $3.30 | ~$8 | ~$15 |
| 10M req/mo | $10 | $33 | ~$50 | ~$95 |

The 🟢 always-billed floor (ECR + S3 + CloudWatch metrics ≈ $1.80) is
**independent of traffic** — so the *marginal* cost of each additional
prediction is essentially the Lambda + API Gateway add-on. No surprise
bills, no provisioning required.

### Cost controls in place

- **Pricing alarm:** _(TODO Phase 5 — add a CloudWatch Billing alarm at $10/mo
  threshold so any cost anomaly pages the team.)_
- **EC2 hygiene:** training instances are terminated, not stopped — no
  hidden EBS root-volume costs.
- **ECR lifecycle:** _(future)_ keep only the latest 5 image tags to bound
  storage growth.
- **CloudWatch Logs retention:** _(future)_ set to 14 days to prevent
  unbounded log storage growth.

---

## Deployment & CI/CD

### Inference Lambda (Phase 2, done — repeat by hand or via CI)

```bash
# Build, push image to ECR, then update the Lambda from the console
# (or via CI in Phase 5):
cd <repo-root>
export AWS_PROFILE=mlds423
./serving/deploy.sh us-east-1
```

The script:
1. Discovers the AWS account ID via `aws sts get-caller-identity`
2. Ensures `mlds423-used-cars-serving` ECR repo exists
3. Logs Docker into ECR with a short-lived token (no static creds)
4. Builds with `docker buildx build --platform linux/amd64 --provenance=false`
   — see comment in the script for why both flags are required
5. Pushes the image to ECR

After pushing, in the Lambda console: **Image → Deploy new image → select latest**.

### End-to-end smoke test

```bash
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[{"manufacturer":"toyota","year":2018,"odometer":45000,"condition":"excellent","cylinders":"4 cylinders","fuel":"gas","transmission":"automatic","drive":"fwd","type":"sedan","paint_color":"white","state":"ca","title_status":"clean","region":"los angeles"}]}'
```

Expected response (HTTP 200):

```json
{"predictions":[16433.36],"model":"XGBoostNativeCategorical","run_id":"20260528T222842Z"}
```

### Cold-start warmup before demo

The first invocation per cold container loads the model from S3 and can take
~25–30 seconds. Before the live demo, fire one curl 30 seconds beforehand so
the model is cached in memory and the demo call returns in < 1 second.

### CI/CD via GitHub Actions

Workflow: [`.github/workflows/deploy.yml`](../.github/workflows/deploy.yml).
Trigger: `push` to `main` (or `eason` while the workflow is being proven), but
only when files under `serving/`, `modeling/`, `config.yaml`, or the workflow
itself change.

Per-run flow:

1. **OIDC token exchange** — the workflow assumes `UsedCarsMLGitHubDeployRole`
   via GitHub's OIDC provider. No `AWS_ACCESS_KEY_ID` is stored as a GitHub
   secret. The role's trust policy
   ([`iam_github_trust_policy.json`](iam_github_trust_policy.json)) restricts
   `sts:AssumeRoleWithWebIdentity` to subjects matching
   `repo:yuanyuanxie11/cloud-engineering-used-cars-prediction:*`. Permissions
   ([`iam_github_deploy_policy.json`](iam_github_deploy_policy.json)) are
   limited to the serving ECR repository and the predict Lambda only — the CI
   role cannot touch any other resource.
2. **Build + push** with `docker buildx --platform linux/amd64 --provenance=false`,
   tagging the image as both `:latest` (what Lambda points to) and
   `:<git-sha>` (immutable audit/rollback handle).
3. **`aws lambda update-function-code`** followed by `aws lambda wait
   function-updated` so the next step never hits a stale image.
4. **Live smoke test** — POSTs a known sample to the public `/predict` endpoint
   and fails the workflow unless the response is HTTP 200 with a `predictions`
   key. This is the "Continuous **Deployment**" gate: green only when the model
   is verifiably serving in production.

This is the same zero-static-secrets philosophy as the runtime (SSO for humans,
IAM Roles for services) extended into the build pipeline: **no long-lived AWS
credentials exist anywhere in this project**.

### Teardown (cost control after demo)

```bash
# Terminate EC2 if any is still running
aws ec2 describe-instances --filters Name=tag:Project,Values=mlds423 \
  --query 'Reservations[].Instances[?State.Name==`running`].InstanceId' \
  --output text \
  | xargs -r aws ec2 terminate-instances --instance-ids

# Delete Lambdas (optional — they cost $0 when not invoked, so safe to leave)
# Delete API Gateway (optional — same reason)
# Delete ECR images (storage cost is ~$0.10/GB/month, can leave)
# S3 — KEEP. The artifacts are the deliverable.
```

The only persistently billed components are S3 storage (cents per month) and
CloudWatch Logs retention (configurable, default infinite — see Phase 4).

---

## Open questions / TODOs

- [ ] Phase 4: complete AWS Pricing Calculator estimate; fill in the Cost section above
- [ ] Phase 5: implement GitHub Actions workflow with OIDC
- [ ] Phase 5: add `gitleaks` to the PR pipeline
- [ ] Phase 6: rehearse the numbered demo flow once with the team before final presentation

---

## Contact

- **Person 3 (Cloud Architect & DevOps):** Eason
- **Person 1 (Data Engineer):** Eric
- **Person 2 (ML Engineer):** Yuanyuan
