"""Simple, explicit lead-time stock recommendations."""

import math
from statistics import NormalDist

import numpy as np


def recommend_stock(
    forecast: np.ndarray,
    historical_daily_std: float,
    lead_time_days: int,
    service_level: float,
    current_stock: int,
) -> dict[str, float | int]:
    """Return an order-up-to level using normal-demand safety stock.

    This is a planning heuristic, not a cost-optimal policy. It assumes daily
    demand variability is independent across lead-time days.
    """
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least one day.")
    if not 0 < service_level < 1:
        raise ValueError("Service level must be between zero and one.")
    if current_stock < 0:
        raise ValueError("Current stock cannot be negative.")

    lead_demand = float(np.clip(forecast[:lead_time_days], 0, None).sum())
    z_value = NormalDist().inv_cdf(service_level)
    safety_stock = max(0.0, z_value * max(0.0, historical_daily_std) * math.sqrt(lead_time_days))
    order_up_to = math.ceil(lead_demand + safety_stock)
    return {
        "lead_time_demand": lead_demand,
        "safety_stock": safety_stock,
        "order_up_to": order_up_to,
        "suggested_order": max(0, order_up_to - current_stock),
    }
