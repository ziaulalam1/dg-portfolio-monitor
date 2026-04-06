"""
Synthetic distributed generation portfolio data generator.
Produces daily power output readings for a mixed small/large asset fleet.
"""

from typing import Dict, List, Tuple

import numpy as np


def generate_portfolio(
    n_small: int = 5,
    n_large: int = 15,
    n_days: int = 365,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Generate a synthetic DG portfolio.

    Small assets (10-25 kW) are deliberately separated from large assets
    (60-100 kW) so the naive global detector is blind to small-asset anomalies.

    Returns:
        readings  -- shape (n_days, n_assets), daily output in kW
        asset_ids -- list of asset ID strings
        baselines -- shape (n_assets,), expected output per asset in kW
    """
    rng = np.random.default_rng(seed)

    small_baselines = rng.uniform(10, 25, n_small)
    large_baselines = rng.uniform(60, 100, n_large)
    baselines = np.concatenate([small_baselines, large_baselines])
    n_assets = n_small + n_large

    # Normal operating noise: 10% coefficient of variation per asset
    noise = rng.normal(0, 1, (n_days, n_assets)) * (baselines * 0.10)
    readings = np.clip(baselines + noise, 0.0, None)

    asset_ids = [f"asset_{i:03d}" for i in range(n_assets)]
    return readings, asset_ids, baselines


def inject_anomalies(
    readings: np.ndarray,
    asset_ids: List[str],
    baselines: np.ndarray,
    anomalies: List[Tuple[str, int]],
    drop_factor: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """
    Inject output-drop anomalies into the readings array.

    Each anomaly drops the named asset's output to drop_factor * baseline
    on the given day -- simulating a fault or outage.

    Returns modified readings (copy) and ground-truth dict {asset_id: [day, ...]}.
    """
    readings = readings.copy()
    ground_truth: Dict[str, List[int]] = {}

    for asset_id, day in anomalies:
        idx = asset_ids.index(asset_id)
        readings[day, idx] = baselines[idx] * drop_factor
        ground_truth.setdefault(asset_id, []).append(day)

    return readings, ground_truth
