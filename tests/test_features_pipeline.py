"""Tests feature engineering against config.yaml"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml
from modeling.features import build_preprocessor, prepare_xy


class FeaturesPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        with open(root / "config.yaml", encoding="utf-8") as fh:
            cls.cfg = yaml.safe_load(fh)

    def test_prepare_xy_shapes_and_target(self):
        df = pd.DataFrame(
            {
                "id": ["x1"],
                "price": [20_000.0],
                "year": [2018],
                "odometer": [45_000],
                "manufacturer": ["toyota"],
                "condition": ["good"],
                "cylinders": ["4 cylinders"],
                "fuel": ["gas"],
                "transmission": ["automatic"],
                "drive": ["fwd"],
                "type": ["sedan"],
                "paint_color": ["white"],
                "state": ["ca"],
                "title_status": ["clean"],
                "region": ["los angeles"],
            }
        )
        X, y = prepare_xy(df, self.cfg)
        self.assertEqual(len(y), 1)
        self.assertNotIn("price", X.columns)
        self.assertIn("vehicle_age", X.columns)

    def test_preprocessor_fits_one_row(self):
        df = pd.DataFrame(
            {
                "price": [20_000.0],
                "year": [2018],
                "odometer": [45_000],
                "manufacturer": ["toyota"],
                "condition": ["good"],
                "cylinders": ["4 cylinders"],
                "fuel": ["gas"],
                "transmission": ["automatic"],
                "drive": ["fwd"],
                "type": ["sedan"],
                "paint_color": ["white"],
                "state": ["ca"],
                "title_status": ["clean"],
            }
        )
        X, _ = prepare_xy(df, self.cfg)
        pre, _ = build_preprocessor(X, self.cfg)
        out = pre.fit_transform(X)
        self.assertGreater(out.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
