# M5 Retail Forecasting & Inventory Planner

An end-to-end retail demand forecasting and inventory decision-support project using the M5 Walmart dataset. A global LightGBM model is compared with a seasonal-naive baseline in a chronological 28-day backtest; the dashboard then turns forecasts into a transparent lead-time stock recommendation.

## What it answers

Can a global model improve daily unit forecasts for high-volume items at one store, and how might those forecasts inform a simple replenishment target?

## Measured result

The first run used the top 250 `CA_1` item series, selected using only the pre-validation sales window. It trained on 464,250 item-day rows and evaluated March 28–April 24, 2016.

| Metric | LightGBM | Seasonal naive |
|---|---:|---:|
| MAE (units/item/day) | 4.433 | 4.726 |
| RMSE | 6.576 | 7.481 |
| WAPE | 60.9% | 64.9% |
| Bias (forecast − actual) | +1.677 | +0.105 |

LightGBM reduced WAPE by 4.03 percentage points (about 6.2% relative) on this backtest. Its positive bias shows it tends to over-forecast, so the lower WAPE does not by itself establish that it is the better operational choice. The result is limited to one store, a top-selling-item subset, and one 28-day holdout. See [the results note](reports/results_summary.md) for interpretation and limitations.

## Method

- **Scope:** one configurable M5 store; top 250 item series by sales before the holdout.
- **Validation:** final 28 days of `sales_train_validation.csv` (`d_1886`–`d_1913` in the original files). Selection and model fitting use only earlier sales.
- **Forecast model:** global LightGBM Poisson regression with lag-1/7/28, shifted rolling means, item identity, calendar seasonality, event/SNAP flags, and weekly selling price.
- **Baseline:** recursive seasonal naive (`y[t] = y[t-7]`). Predictions replace unavailable future sales as the 28-day horizon advances.
- **Metrics:** MAE, RMSE, WAPE, and bias. WAPE is useful for a portfolio with many zero-sales days; bias makes over-forecasting visible.
- **Decision layer:** user-adjustable lead time, service level, and stock on hand. Safety stock uses recent daily-demand variability and a normal approximation. It is a transparent heuristic, not a cost-optimized policy.

The price table is treated as known for the forecast horizon, consistent with the M5 competition setup. This is an assumption a retailer would need to validate in its own planning process.

## Setup on Windows

Python 3.10 is recommended. Create the isolated environment and install the project:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, use the path to your Python 3.10 executable. On the development PC used for this run, Python 3.10 is at:

```powershell
& "C:\Users\hp\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv
```

All remaining commands call `.venv` directly. There is no need to activate it or register a Jupyter kernel.

## Data

1. Sign in to Kaggle and accept the rules on the [M5 Forecasting - Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
2. Download the competition files and extract them into `data/raw/`.
3. Keep the original filenames `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv`.

Raw data and generated forecasts are excluded from Git. Do not commit the Kaggle files.

## Run the model and dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m m5_inventory.pipeline --store-id CA_1 --n-series 250 --horizon 28
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The backtest writes `artifacts/forecast_backtest.csv`, `artifacts/metrics.json`, and `artifacts/inventory_plan.csv`. They remain local. If artifacts are absent, the dashboard shows a clearly labeled synthetic preview—not M5 data or model results.

For a quicker run, set `--n-series 50`. Other M5 stores can be selected with `--store-id`, such as `TX_1` or `WI_1`.

## Inventory assumptions and limits

The order-up-to estimate is forecast demand over the chosen lead time plus safety stock: `z × recent daily-demand standard deviation × sqrt(lead time)`. It assumes independent daily errors and ignores supplier minimums, pack sizes, capacity, order costs, and the causal effect of promotions. The backtest evaluates forecast accuracy, not realized profit or service-level improvement. Further work should add rolling-origin evaluation, error slices by item/category, and a cost-based inventory simulation.

## Repository layout

```text
app.py                  Streamlit forecast and inventory dashboard
src/m5_inventory/       data loading, features, backtest, inventory logic
reports/                measured summary and analysis
requirements.txt         editable install entry point
pyproject.toml           package metadata and dependencies
data/raw/               local Kaggle files (ignored)
artifacts/              local forecasts and metrics (ignored)
```

## Resume bullet

> Built a leakage-aware 28-day LightGBM backtest for 250 M5 item/store series; reduced WAPE from 64.9% to 60.9% versus a seasonal-naive baseline and surfaced forecast over-bias in an interactive replenishment planner.
