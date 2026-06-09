"""Integration-style tests for pipeline.clean"""

from __future__ import annotations

import unittest

import pandas as pd
from pipeline.clean import clean_vehicles
from pipeline.config import MAX_PRICE, MIN_PRICE


def _dirty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "a", "b", "c", "d", "e"],
            "price": [0, 25_000, 25_000, 600_000, 2_000_000_000_000, 15_000],
            "year": [2015, 2015, 2010, 2018, 2016, 1970],
            "odometer": [50_000, 50_000, 10_000, 2_000_000, 80_000, 90_000],
            "county": [None] * 6,
            "manufacturer": [" Toyota ", "toyota", "honda", "ford", "bmw", "nissan"],
            "condition": ["good"] * 6,
        }
    )


class CleanPipelineTest(unittest.TestCase):
    def test_clean_removes_duplicates_and_bad_prices(self):
        cleaned, stats = clean_vehicles(_dirty_frame())

        self.assertIn("county", stats.get("dropped_columns", []))
        self.assertGreater(stats["duplicates_removed"], 0)
        self.assertGreater(stats["price_outliers_removed"], 0)
        self.assertEqual(stats["input_rows"], 6)
        self.assertLess(stats["output_rows"], stats["input_rows"])
        self.assertTrue((cleaned["price"] >= MIN_PRICE).all())
        self.assertTrue((cleaned["price"] <= MAX_PRICE).all())
        self.assertEqual(cleaned["id"].nunique(), len(cleaned))

    def test_manufacturer_normalized_lowercase(self):
        cleaned, _ = clean_vehicles(_dirty_frame())
        self.assertTrue(cleaned["manufacturer"].str.islower().all())


if __name__ == "__main__":
    unittest.main()
