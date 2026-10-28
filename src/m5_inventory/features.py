"""Leakage-safe lag features for daily item demand."""

import pandas as pd


FEATURE_COLUMNS = [
    "item_id",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "sell_price",
    "price_missing",
    "event_flag",
    "snap",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
]

LAG_FEATURES = ["lag_1", "lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"]


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add past-only lags and rolling means; no feature uses same-day sales."""
    result = panel.copy()
    grouped = result.groupby("item_id", observed=True, sort=False)["y"]
    for lag in (1, 7, 28):
        result[f"lag_{lag}"] = grouped.shift(lag).astype("float32")
    for window in (7, 28):
        result[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).mean()
        ).astype("float32")
    return result
