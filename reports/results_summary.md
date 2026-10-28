# First backtest results

## Evaluation design

- Store: `CA_1`
- Series: 250 items, selected by training-period unit sales only
- Training targets: 464,250 item-days through `d_1885`
- Holdout: 28 days, `d_1886`–`d_1913` (2016-03-28 through 2016-04-24)
- Forecasting: recursive daily predictions; lag features for later holdout days use earlier model predictions rather than actual holdout sales

## Metrics

| Metric | Global LightGBM | Seasonal naive |
|---|---:|---:|
| MAE | 4.433 | 4.726 |
| RMSE | 6.576 | 7.481 |
| WAPE | 60.92% | 64.95% |
| Bias (forecast − actual) | +1.677 | +0.105 |

On this single backtest, LightGBM lowered WAPE by 4.03 percentage points, a relative reduction of about 6.2%. It also had lower MAE and RMSE. Its positive bias is material: average predictions were about 1.68 units too high per item/day. A planner should consider this alongside aggregate accuracy, particularly when excess stock is costly.

## Interpretation and caveats

- WAPE remains high because the portfolio contains many zero-sales days and is evaluated at item/day granularity.
- The model was tested on one store, a training-selected top-selling subset, and one 28-day period. This is an initial result, not evidence of performance across all M5 series or retailers.
- Weekly prices and calendar features are treated as known for the forecast horizon.
- The inventory recommendation is a normal-approximation order-up-to heuristic. It has not been evaluated with a full replenishment simulation or retailer-specific costs.

## Next analyses

1. Slice errors by item, department, demand volume, and zero-sales share.
2. Repeat the comparison at multiple rolling forecast origins.
3. Investigate and calibrate the positive forecast bias.
4. Add a clearly parameterized lost-sales/holding-cost simulation before claiming inventory value.
