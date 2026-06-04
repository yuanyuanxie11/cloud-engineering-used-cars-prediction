"""
Generate a small synthetic vehicles_clean.parquet for local ML dry-runs
when the full Craigslist CSV is not available.

Usage:
  python scripts/make_sample_parquet.py --output ./out/vehicles_clean.parquet --rows 8000
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ["salvage", "fair", "good", "excellent", "like new", "new"]
CYLINDERS = [
    "3 cylinders",
    "4 cylinders",
    "6 cylinders",
    "8 cylinders",
    "other",
]
MANUFACTURERS = ["ford", "toyota", "honda", "chevrolet", "bmw", "nissan"]
FUELS = ["gas", "diesel", "hybrid", "electric", "other"]
TRANS = ["automatic", "manual", "other"]
DRIVES = ["rwd", "fwd", "4wd"]
TYPES = ["sedan", "suv", "truck", "coupe", "other"]
COLORS = ["black", "white", "silver", "blue", "red", "other"]
STATES = ["ca", "tx", "fl", "ny", "il", "wa"]
TITLE = ["clean", "salvage", "rebuilt", "lien", "parts only"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./out/vehicles_clean.parquet")
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("make_sample_parquet")

    rng = np.random.default_rng(args.seed)
    n = args.rows
    current_year = datetime.now(timezone.utc).year

    year = rng.integers(2005, current_year + 1, size=n)
    age = current_year - year
    odometer = rng.integers(5_000, 180_000, size=n)
    base = 35_000 - 800 * age - 0.08 * odometer
    noise = rng.normal(0, 4_000, size=n)
    price = np.clip(base + noise, 100, 80_000)

    df = pd.DataFrame(
        {
            "id": [f"sample-{i}" for i in range(n)],
            "price": price,
            "year": year,
            "manufacturer": rng.choice(MANUFACTURERS, n),
            "model": rng.choice(["camry", "f-150", "civic", "3 series"], n),
            "condition": rng.choice(CONDITIONS, n),
            "cylinders": rng.choice(CYLINDERS, n),
            "fuel": rng.choice(FUELS, n),
            "odometer": odometer,
            "title_status": rng.choice(TITLE, n),
            "transmission": rng.choice(TRANS, n),
            "drive": rng.choice(DRIVES, n),
            "type": rng.choice(TYPES, n),
            "paint_color": rng.choice(COLORS, n),
            "state": rng.choice(STATES, n),
            "region": rng.choice(["bay area", "dallas", "miami"], n),
        }
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"Wrote {len(df):,} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
