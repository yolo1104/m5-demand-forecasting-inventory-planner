"""Interactive M5 backtest and inventory decision dashboard."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from m5_inventory.inventory import recommend_stock
from m5_inventory.paths import DEFAULT_ARTIFACT_DIR


st.set_page_config(page_title="M5 Inventory Planner", page_icon="📦", layout="wide")


def _metrics(detail: pd.DataFrame) -> dict:
    scores = {}
    actual = detail["actual"].to_numpy(dtype=float)
    for label, column in (
        ("lightgbm", "lightgbm_forecast"),
        ("seasonal_naive", "seasonal_naive_forecast"),
    ):
        predicted = detail[column].to_numpy(dtype=float)
        error = predicted - actual
        total = float(np.abs(actual).sum())
        scores[label] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "wape": float(np.abs(error).sum() / total) if total else 0.0,
            "bias": float(error.mean()),
        }
    return scores


def _synthetic_preview() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic preview so the public app works without M5 data."""
    rng = np.random.default_rng(21)
    dates = pd.date_range("2025-01-01", periods=28, freq="D")
    pieces = []
    inventory_rows = []
    for number, base in enumerate((3.5, 8.0, 14.0, 22.0, 35.0), start=1):
        item = f"SYNTHETIC_ITEM_{number:03d}"
        seasonal = 1.0 + 0.18 * np.sin(np.arange(28) * 2 * np.pi / 7)
        actual = np.maximum(0, rng.poisson(base * seasonal)).astype(float)
        model = np.maximum(0, 0.88 * actual + 0.12 * base * seasonal)
        naive = np.r_[np.repeat(base, 7), actual[:-7]]
        pieces.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "date": dates,
                    "actual": actual,
                    "lightgbm_forecast": model,
                    "seasonal_naive_forecast": naive,
                }
            )
        )
        inventory_rows.append(
            {"item_id": item, "historical_daily_std_28d": float(np.std(actual, ddof=1))}
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(inventory_rows)


@st.cache_data
def _load_artifacts(artifact_dir: str):
    root = Path(artifact_dir)
    detail_path = root / "forecast_backtest.csv"
    metrics_path = root / "metrics.json"
    inventory_path = root / "inventory_plan.csv"
    if not (detail_path.exists() and metrics_path.exists() and inventory_path.exists()):
        return None
    detail = pd.read_csv(detail_path, parse_dates=["date"])
    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    inventory = pd.read_csv(inventory_path)
    return detail, metrics, inventory


st.title("M5 Retail Forecasting & Inventory Planner")
st.caption("A 28-day, leakage-aware backtest with a practical lead-time stock recommendation.")

loaded = _load_artifacts(str(DEFAULT_ARTIFACT_DIR))
if loaded is None:
    detail, inventory, = _synthetic_preview()
    metrics = _metrics(detail)
    is_demo = True
    st.info(
        "Showing a clearly labeled synthetic preview. Add the Kaggle M5 files under "
        "`data/raw/` and run `python -m m5_inventory.pipeline` to use real backtest outputs."
    )
else:
    detail, metrics, inventory = loaded
    is_demo = False
    st.success(
        f"Using M5 backtest artifacts for {metrics['store_id']} "
        f"({metrics['validation_start']} to {metrics['validation_end']})."
    )

items = sorted(detail["item_id"].astype(str).unique())
with st.sidebar:
    st.header("Inventory assumptions")
    selected_item = st.selectbox("Item", items)
    lead_time = st.slider("Lead time (days)", min_value=1, max_value=14, value=7)
    service_level = st.select_slider(
        "Target service level",
        options=[0.80, 0.90, 0.95, 0.98, 0.99],
        value=0.95,
        format_func=lambda value: f"{value:.0%}",
    )
    current_stock = st.number_input("Current on-hand stock (units)", min_value=0, value=0, step=1)

lightgbm_metrics = metrics["lightgbm"]
baseline_metrics = metrics["seasonal_naive"]
metric_cols = st.columns(4)
metric_cols[0].metric("LightGBM WAPE", f"{lightgbm_metrics['wape']:.1%}")
metric_cols[1].metric("Seasonal baseline WAPE", f"{baseline_metrics['wape']:.1%}")
metric_cols[2].metric("LightGBM MAE", f"{lightgbm_metrics['mae']:.2f} units/day")
metric_cols[3].metric("Validation horizon", f"{metrics.get('horizon_days', 28)} days")
st.caption("Forecast metrics are aggregated across the validation items and days.")

item_detail = detail.loc[detail["item_id"].astype(str).eq(selected_item)].sort_values("date")
chart = go.Figure()
chart.add_trace(go.Scatter(x=item_detail["date"], y=item_detail["actual"], name="Actual", mode="lines+markers"))
chart.add_trace(go.Scatter(x=item_detail["date"], y=item_detail["lightgbm_forecast"], name="LightGBM", mode="lines"))
chart.add_trace(go.Scatter(x=item_detail["date"], y=item_detail["seasonal_naive_forecast"], name="Seasonal naive", mode="lines"))
chart.update_layout(
    title=f"28-day forecast comparison · {selected_item}",
    xaxis_title="Date",
    yaxis_title="Units sold",
    legend_title="Series",
    hovermode="x unified",
    height=420,
    margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(chart, use_container_width=True)

inventory_row = inventory.loc[inventory["item_id"].astype(str).eq(selected_item)]
historical_std = (
    float(inventory_row["historical_daily_std_28d"].iloc[0]) if not inventory_row.empty else 0.0
)
recommendation = recommend_stock(
    item_detail["lightgbm_forecast"].to_numpy(dtype=float),
    historical_std,
    lead_time,
    service_level,
    int(current_stock),
)
st.subheader("Replenishment recommendation")
st.caption(
    "Order-up-to level = forecast demand over lead time + safety stock. "
    "Safety stock uses recent daily-demand variability and assumes independent daily errors."
)
stock_cols = st.columns(4)
stock_cols[0].metric("Lead-time demand", f"{recommendation['lead_time_demand']:.1f} units")
stock_cols[1].metric("Safety stock", f"{recommendation['safety_stock']:.1f} units")
stock_cols[2].metric("Order-up-to level", f"{recommendation['order_up_to']} units")
stock_cols[3].metric("Suggested replenishment", f"{recommendation['suggested_order']} units")

if is_demo:
    st.caption("The preview is synthetic and is not a model result or Walmart sales data.")
