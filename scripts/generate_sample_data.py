"""使用标准库生成来源明确的合成价格；不访问网络或真实行情。"""

from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "sample_data" / "sample_prices.csv"
SEED = 20261030
RETURN_PERIODS = 260


def sample_csv() -> bytes:
    """固定种子、固定参数；工作日历仅排除周末。"""
    rng = random.Random(SEED)
    prices = [100.0, 80.0, 120.0]
    daily_drifts = [0.0003, 0.00015, 0.00025]
    market_loadings = [0.008, 0.003, 0.006]
    idiosyncratic_scales = [0.006, 0.004, 0.005]
    current_date = date(2025, 1, 2)
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Date", "ETF_A", "ETF_B", "INDEX_C"])
    writer.writerow([current_date.isoformat(), *(f"{value:.4f}" for value in prices)])
    for _ in range(RETURN_PERIODS):
        current_date += timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        market_shock = rng.gauss(0.0, 1.0)
        for asset in range(len(prices)):
            log_return = (
                daily_drifts[asset]
                + market_loadings[asset] * market_shock
                + idiosyncratic_scales[asset] * rng.gauss(0.0, 1.0)
            )
            prices[asset] *= math.exp(log_return)
        writer.writerow([current_date.isoformat(), *(f"{value:.4f}" for value in prices)])
    return output.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic demo prices.")
    parser.add_argument("--check", action="store_true", help="Check CSV matches generated data")
    args = parser.parse_args()
    expected = sample_csv()
    if args.check:
        if not SAMPLE_FILE.is_file() or SAMPLE_FILE.read_bytes() != expected:
            print("Sample mismatch. Run python scripts/generate_sample_data.py to regenerate.")
            return 1
        print("Synthetic sample verified: 261 price observations, 3 fictional assets.")
        return 0
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_FILE.write_bytes(expected)
    print(f"Generated synthetic demo prices: {SAMPLE_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
