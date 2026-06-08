# Used Car Price Predictor Frontend

This is the Person 4 web interface for the used-car price prediction service.
It calls the deployed AWS inference API:

```text
https://qodruw90xj.execute-api.us-east-1.amazonaws.com/predict
```

The page is intentionally static: no separate web server or backend is required
for the demo. The prediction backend is AWS API Gateway + Lambda.

## Run Locally

```bash
cd frontend
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765
```

Fill the vehicle form and click `Predict Price`.

## AWS Smoke Test

Use this to verify the deployed inference API independently of the frontend:

```bash
curl -X POST "https://qodruw90xj.execute-api.us-east-1.amazonaws.com/predict" \
  -H "Content-Type: text/plain" \
  -d '{
    "instances": [{
      "manufacturer": "toyota",
      "year": 2018,
      "odometer": 45000,
      "condition": "excellent",
      "cylinders": "4 cylinders",
      "fuel": "gas",
      "transmission": "automatic",
      "drive": "fwd",
      "type": "sedan",
      "paint_color": "white",
      "state": "ca",
      "title_status": "clean",
      "region": "los angeles"
    }]
  }'
```

Expected response shape:

```json
{
  "predictions": [16433.36],
  "model": "XGBoostNativeCategorical",
  "run_id": "20260528T222842Z"
}
```
