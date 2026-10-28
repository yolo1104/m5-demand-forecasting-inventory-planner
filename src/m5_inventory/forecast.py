"""Global LightGBM forecasting with recursive 28-day validation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import FEATURE_COLUMNS, LAG_FEATURES, add_lag_features
from .inventory import recommend_stock


@dataclass
class BacktestResult:
    detail: pd.DataFrame
    metrics: dict
    inventory_plan: pd.DataFrame
    train_rows: int


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / denominator) if denominator else 0.0,
        "bias": float(error.mean()),
    }


def run_backtest(
    panel: pd.DataFrame,
    horizon: int = 28,
    random_state: int = 42,
    default_lead_time: int = 7,
    default_service_level: float = 0.95,
) -> BacktestResult:
    """Fit through the pre-holdout period and forecast the final horizon."""
    n_days = int(panel["day_idx"].max()) + 1
    validation_start = n_days - horizon
    if validation_start <= 28:
        raise ValueError("At least 29 training days are required before validation.")

    featured = add_lag_features(panel)
    train = featured.loc[
        featured["day_idx"].lt(validation_start)
        & featured[LAG_FEATURES].notna().all(axis=1)
    ].copy()
    if train.empty:
        raise ValueError("No training rows remain after creating lag features.")

    model = LGBMRegressor(
        objective="poisson",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train["y"],
        categorical_feature=["item_id"],
    )

    item_order = list(panel["item_id"].cat.categories)
    actual_matrix = (
        panel.pivot(index="item_id", columns="day_idx", values="y")
        .reindex(index=item_order, columns=range(n_days))
        .to_numpy(dtype="float32")
    )
    model_matrix = actual_matrix.copy()
    naive_matrix = actual_matrix.copy()
    future_fields = [
        "date", "d", "sell_price", "price_missing", "event_flag", "snap",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_weekend",
    ]
    detail_parts = []

    for day_idx in range(validation_start, n_days):
        known = (
            panel.loc[panel["day_idx"].eq(day_idx), ["item_id", *future_fields]]
            .set_index("item_id")
            .reindex(item_order)
        )
        features = known[future_fields].copy()
        features.insert(
            0,
            "item_id",
            pd.Categorical(item_order, categories=panel["item_id"].cat.categories),
        )
        features["lag_1"] = model_matrix[:, day_idx - 1]
        features["lag_7"] = model_matrix[:, day_idx - 7]
        features["lag_28"] = model_matrix[:, day_idx - 28]
        features["rolling_mean_7"] = model_matrix[:, day_idx - 7 : day_idx].mean(axis=1)
        features["rolling_mean_28"] = model_matrix[:, day_idx - 28 : day_idx].mean(axis=1)
        model_prediction = np.clip(model.predict(features[FEATURE_COLUMNS]), 0, None).astype("float32")
        naive_prediction = np.clip(naive_matrix[:, day_idx - 7], 0, None).astype("float32")

        actual = actual_matrix[:, day_idx]
        detail_parts.append(
            pd.DataFrame(
                {
                    "item_id": item_order,
                    "date": known["date"].to_numpy(),
                    "d": known["d"].astype(str).to_numpy(),
                    "actual": actual,
                    "lightgbm_forecast": model_prediction,
                    "seasonal_naive_forecast": naive_prediction,
                }
            )
        )
        # Replace each target with its prediction before advancing the horizon.
        model_matrix[:, day_idx] = model_prediction
        naive_matrix[:, day_idx] = naive_prediction

    detail = pd.concat(detail_parts, ignore_index=True)
    metrics = {
        "store_id": str(panel["store_id"].iloc[0]),
        "n_series": int(panel["item_id"].nunique()),
        "horizon_days": int(horizon),
        "validation_start": str(detail["date"].min().date()),
        "validation_end": str(detail["date"].max().date()),
        "training_rows": int(len(train)),
        "lightgbm": _score(
            detail["actual"].to_numpy(), detail["lightgbm_forecast"].to_numpy()
        ),
        "seasonal_naive": _score(
            detail["actual"].to_numpy(), detail["seasonal_naive_forecast"].to_numpy()
        ),
    }

    lookback_start = max(0, validation_start - 28)
    history = actual_matrix[:, lookback_start:validation_start]
    history_std = history.std(axis=1, ddof=1) if history.shape[1] > 1 else np.zeros(len(item_order))
    lead_time = min(default_lead_time, horizon)
    lead_forecast = model_matrix[:, validation_start : validation_start + lead_time].sum(axis=1)
    recommendations = [
        recommend_stock(
            model_matrix[index, validation_start : validation_start + horizon],
            float(history_std[index]),
            lead_time,
            default_service_level,
            current_stock=0,
        )
        for index in range(len(item_order))
    ]
    inventory_plan = pd.DataFrame(
        {
            "item_id": item_order,
            "lead_time_days": lead_time,
            "historical_daily_std_28d": history_std,
            "forecast_lead_time_demand": lead_forecast,
            "safety_stock_95": [row["safety_stock"] for row in recommendations],
            "order_up_to_95": [row["order_up_to"] for row in recommendations],
        }
    )
    return BacktestResult(detail, metrics, inventory_plan, len(train))
