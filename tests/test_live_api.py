"""Live integration test against deployed API Gateway."""

from __future__ import annotations

import json
import os
import unittest
import urllib.request

API_URL = os.environ.get(
    "PREDICT_API_URL",
    "https://qodruw90xj.execute-api.us-east-1.amazonaws.com/predict",
)
RUN_LIVE = os.environ.get("RUN_LIVE_TESTS", "").lower() in ("1", "true", "yes")

SAMPLE = {
    "instances": [
        {
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
            "region": "los angeles",
        }
    ]
}


@unittest.skipUnless(RUN_LIVE, "Set RUN_LIVE_TESTS=1 for live API test")
class LivePredictApiTest(unittest.TestCase):
    def test_predict_returns_200_and_predictions(self):
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(SAMPLE).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.assertEqual(resp.status, 200)
            body = json.loads(resp.read())
        self.assertIn("predictions", body)
        self.assertGreater(len(body["predictions"]), 0)


if __name__ == "__main__":
    unittest.main()
