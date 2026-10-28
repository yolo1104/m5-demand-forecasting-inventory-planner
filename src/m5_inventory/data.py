"""Load and prepare a manageable M5 store-level item panel."""

from pathlib import Path

import numpy as np
import pandas as pd


def _check_inputs(data_dir: Path) -> tuple[Path, Path, Path]:
    sales_path = data_dir / "sales_train_validation.csv"
    calendar_path = data_dir / "calendar.csv"
    prices_path = data_dir / "sell_prices.csv"
    missing = [path.name for path in (sales_path, calendar_path, prices_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing M5 files in {data_dir}: {', '.join(missing)}. "
            "Place the Kaggle CSV files in data/raw/; see the README."
        )
    return sales_path, calendar_path, prices_path


def load_m5_panel(
    data_dir: Path,
    store_id: str = "CA_1",
    n_series: int = 250,
    horizon: int = 28,
) -> pd.DataFrame:
    """Return daily rows for the top training-period items in one store.

    Item selection uses only the pre-validation sales window. The validation
    window is retained as labels for the backtest, never for item selection.
    """
    data_dir = Path(data_dir)
    sales_path, calendar_path, prices_path = _check_inputs(data_dir)
    header = pd.read_csv(sales_path, nrows=0)
    day_columns = [column for column in header.columns if column.startswith("d_")]
    if len(day_columns) <= horizon:
        raise ValueError("The sales file is shorter than the requested validation horizon.")

    sales_dtypes = {column: "int16" for column in day_columns}
    sales = pd.read_csv(sales_path, dtype=sales_dtypes)
    store_sales = sales.loc[sales["store_id"].eq(store_id)].copy()
    if store_sales.empty:
        available = ", ".join(sorted(sales["store_id"].unique()))
        raise ValueError(f"Unknown store {store_id!r}. Available stores: {available}")

    training_days = day_columns[:-horizon]
    store_sales["_selection_volume"] = store_sales[training_days].sum(axis=1)
    selected = (
        store_sales.nlargest(min(n_series, len(store_sales)), "_selection_volume")
        .drop(columns="_selection_volume")
        .sort_values("item_id")
        .copy()
    )
    item_order = selected["item_id"].astype(str).tolist()
    state_id = str(selected["state_id"].iloc[0])

    id_columns = ["item_id", "cat_id", "dept_id", "store_id", "state_id"]
    panel = selected[id_columns + day_columns].melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="y",
    )
    panel["y"] = panel["y"].astype("float32")
    panel["day_idx"] = panel["d"].str[2:].astype("int16") - 1

    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    snap_column = f"snap_{state_id}"
    if snap_column not in calendar:
        raise ValueError(f"Expected SNAP calendar column {snap_column!r} was not found.")
    calendar["day_idx"] = calendar["d"].str[2:].astype("int16") - 1
    calendar["event_flag"] = (
        calendar["event_name_1"].notna() | calendar["event_name_2"].notna()
    ).astype("int8")
    calendar["snap"] = calendar[snap_column].astype("int8")
    calendar["dow"] = calendar["date"].dt.dayofweek.astype("int8")
    calendar["month"] = calendar["date"].dt.month.astype("int8")
    calendar["day_of_year"] = calendar["date"].dt.dayofyear.astype("int16")
    calendar["is_weekend"] = calendar["dow"].ge(5).astype("int8")
    calendar["dow_sin"] = np.sin(2 * np.pi * calendar["dow"] / 7).astype("float32")
    calendar["dow_cos"] = np.cos(2 * np.pi * calendar["dow"] / 7).astype("float32")
    calendar["month_sin"] = np.sin(2 * np.pi * (calendar["month"] - 1) / 12).astype("float32")
    calendar["month_cos"] = np.cos(2 * np.pi * (calendar["month"] - 1) / 12).astype("float32")
    calendar["doy_sin"] = np.sin(2 * np.pi * (calendar["day_of_year"] - 1) / 365.25).astype("float32")
    calendar["doy_cos"] = np.cos(2 * np.pi * (calendar["day_of_year"] - 1) / 365.25).astype("float32")

    calendar_columns = [
        "d", "day_idx", "date", "wm_yr_wk", "event_flag", "snap",
        "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_weekend",
    ]
    panel = panel.merge(
        calendar[calendar_columns], on=["d", "day_idx"], how="left", validate="many_to_one"
    )

    prices = pd.read_csv(
        prices_path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"wm_yr_wk": "int16", "sell_price": "float32"},
    )
    prices = prices.loc[
        prices["store_id"].eq(store_id) & prices["item_id"].isin(item_order)
    ]
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Price data contains duplicate store/item/week keys.")
    panel = panel.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    panel["price_missing"] = panel["sell_price"].isna().astype("int8")
    panel["sell_price"] = panel["sell_price"].fillna(0).astype("float32")
    panel["item_id"] = pd.Categorical(panel["item_id"], categories=item_order)
    panel = panel.sort_values(["item_id", "day_idx"], kind="stable").reset_index(drop=True)
    return panel
