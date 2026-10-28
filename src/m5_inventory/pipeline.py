"""Command-line entry point for data loading, backtesting, and artifacts."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_m5_panel
from .forecast import run_backtest
from .paths import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR


def run(data_dir: Path, artifact_dir: Path, store_id: str, n_series: int, horizon: int) -> dict:
    print(f"Loading M5 data for {store_id}; selecting {n_series} items from training history...")
    panel = load_m5_panel(data_dir, store_id=store_id, n_series=n_series, horizon=horizon)
    print(f"Prepared {len(panel):,} item-day rows across {panel['item_id'].nunique():,} items.")
    result = run_backtest(panel, horizon=horizon)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    result.detail.to_csv(artifact_dir / "forecast_backtest.csv", index=False)
    result.inventory_plan.to_csv(artifact_dir / "inventory_plan.csv", index=False)
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result.metrics, file, indent=2)

    metric_table = pd.DataFrame(
        {
            "LightGBM": result.metrics["lightgbm"],
            "Seasonal naive": result.metrics["seasonal_naive"],
        }
    ).loc[["mae", "rmse", "wape", "bias"]]
    print(f"\nValidation: {result.metrics['validation_start']} to {result.metrics['validation_end']}")
    print(f"Training rows: {result.train_rows:,}")
    print(metric_table.to_string(float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved local outputs to {artifact_dir}")
    return result.metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--store-id", default="CA_1")
    parser.add_argument("--n-series", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=28)
    args = parser.parse_args()
    run(args.data_dir, args.artifact_dir, args.store_id, args.n_series, args.horizon)


if __name__ == "__main__":
    main()
