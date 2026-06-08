"""
End-to-end pipeline test.

Chain:
  synthetic dirty DataFrame -> clean_vehicles -> parquet -> train -> predict

Slow (trains all configured models). Skipped by default. Run with:

  RUN_SLOW_TESTS=1 python -m unittest tests.test_pipeline_e2e -v
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from modeling.features import prepare_xy
from modeling.train import prepare_xgb_native_frame
from pipeline.clean import clean_vehicles

ROOT = Path(__file__).resolve().parents[1]
RUN_SLOW = os.environ.get("RUN_SLOW_TESTS", "").lower() in ("1", "true", "yes")


def _synthetic_dirty(n: int = 150) -> pd.DataFrame:
    """Build a small raw-like frame that survives cleaning rules."""
    rng = np.random.default_rng(42)
    year = rng.integers(2010, 2024, size=n)
    return pd.DataFrame(
        {
            "id": [f"e2e-{i}" for i in range(n)],
            "price": rng.integers(5_000, 45_000, size=n),
            "year": year,
            "odometer": rng.integers(10_000, 120_000, size=n),
            "county": [None] * n,
            "manufacturer": rng.choice(["toyota", "honda", "ford", "bmw"], size=n),
            "model": rng.choice(["camry", "civic", "f-150"], size=n),
            "condition": rng.choice(["good", "excellent", "fair"], size=n),
            "cylinders": rng.choice(["4 cylinders", "6 cylinders"], size=n),
            "fuel": rng.choice(["gas", "hybrid"], size=n),
            "transmission": rng.choice(["automatic", "manual"], size=n),
            "drive": rng.choice(["fwd", "rwd"], size=n),
            "type": rng.choice(["sedan", "suv"], size=n),
            "paint_color": rng.choice(["white", "black", "silver"], size=n),
            "state": rng.choice(["ca", "tx", "ny"], size=n),
            "title_status": rng.choice(["clean", "rebuilt"], size=n),
            "region": rng.choice(["los angeles", "dallas", "new york city"], size=n),
        }
    )


@unittest.skipUnless(RUN_SLOW, "Set RUN_SLOW_TESTS=1 to run the full pipeline end-to-end test")
class PipelineE2ETest(unittest.TestCase):
    def test_clean_train_predict_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            parquet = tmp_path / "cleaned.parquet"
            out_dir = tmp_path / "models"
            config_path = ROOT / "config.yaml"

            # Step 1: Data cleaning (Person 1 pipeline)
            dirty = _synthetic_dirty()
            cleaned, stats = clean_vehicles(dirty)
            self.assertGreater(stats["output_rows"], 50)
            cleaned.to_parquet(parquet, index=False)

            # Step 2: Model training (Person 2); writes artifacts under out_dir/latest/
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "modeling.train",
                    "--config",
                    str(config_path),
                    "--local-input",
                    str(parquet),
                    "--local-output",
                    str(out_dir),
                    "--run-id",
                    "e2e-test",
                ],
                check=True,
                cwd=ROOT,
            )

            latest = out_dir / "latest"
            model_path = latest / "best_model.pkl"
            manifest_path = latest / "model_manifest.json"
            self.assertTrue(model_path.is_file())
            self.assertTrue(manifest_path.is_file())

            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            with open(model_path, "rb") as fh:
                model = pickle.load(fh)

            # Step 3: Inference (same path as serving Lambda, but local)
            row = cleaned.iloc[[0]].copy()
            if manifest.get("native_categorical_model"):
                X, _, _ = prepare_xgb_native_frame(row, cfg)
                raw = model.predict(X)
            else:
                X, _ = prepare_xy(row, cfg)
                raw = model.predict(X)

            if manifest.get("use_log_target"):
                raw = np.expm1(raw)

            preds = np.asarray(raw).ravel()
            self.assertEqual(len(preds), 1)
            self.assertGreater(preds[0], 0)


if __name__ == "__main__":
    unittest.main()
