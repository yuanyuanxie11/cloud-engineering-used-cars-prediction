# ECS Fargate training (Model Training & Artifact Storage → S3)

This is the containerized, "no-SSH" alternative to [`ec2_train.sh`](ec2_train.sh).
It matches the rubric demo option **"Model Training and Artifact Storage (ECS → S3)"**
and ships logs to CloudWatch automatically.

## 1. Build & push the training image

```bash
REGION=us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO=mlds423-used-cars-train

aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$REGION"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest"
docker build --platform linux/amd64 -f infra/Dockerfile.train -t "$REPO:latest" .
docker tag "$REPO:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"
```

## 2. IAM roles

- **Task execution role** (`ecsTaskExecutionRole`): pull image from ECR + write
  CloudWatch logs (AWS-managed `AmazonECSTaskExecutionRolePolicy`).
- **Task role** (`UsedCarsMLTrainRole`): the app's own S3 access —
  `s3:GetObject` on `processed/*` and `s3:PutObject`/`s3:ListBucket` on
  `artifacts/*`. Plus, for monitoring, `cloudwatch:PutMetricData`.

## 3. Register the task definition

Save as `task-def.json` (replace `<ACCOUNT_ID>` and `<IMAGE_URI>`):

```json
{
  "family": "used-cars-train",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/UsedCarsMLTrainRole",
  "containerDefinitions": [
    {
      "name": "train",
      "image": "<IMAGE_URI>",
      "essential": true,
      "environment": [
        {"name": "USED_CARS_BUCKET", "value": "mlds423-used-cars-project"},
        {"name": "AWS_DEFAULT_REGION", "value": "us-east-1"},
        {"name": "TRAINING_RUN_ID", "value": "ecs-demo"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/used-cars-train",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "train",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
```

```bash
aws ecs register-task-definition --cli-input-json file://task-def.json --region "$REGION"
```

## 4. Run it (one-off task on Fargate)

```bash
aws ecs run-task \
  --cluster used-cars \
  --launch-type FARGATE \
  --task-definition used-cars-train \
  --network-configuration 'awsvpcConfiguration={subnets=[subnet-xxxx],securityGroups=[sg-xxxx],assignPublicIp=ENABLED}' \
  --region "$REGION"
```

(Create the cluster once with `aws ecs create-cluster --cluster-name used-cars`.)

## 5. Verify

- Logs: CloudWatch Logs group `/ecs/used-cars-train` (look for `STRUCTURED {...}` lines).
- Metrics: CloudWatch namespace `UsedCarsML` (`TestR2`, `TestRMSE`, `CVR2Mean`).
- Artifacts: `s3://mlds423-used-cars-project/artifacts/models/latest/best_model.pkl`.

## EC2 vs ECS

| Path | Script | Best for |
|------|--------|----------|
| EC2  | `infra/ec2_train.sh` | Quick demo, SSH access, GPU instance flexibility |
| ECS Fargate | this doc + `infra/Dockerfile.train` | Scalable, ephemeral, no SSH, auto CloudWatch logs |
