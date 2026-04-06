"""
Anomaly detectors for distributed generation portfolios.

SegmentedDetector: per-asset z-score normalization.
NaiveDetector:     global z-score across all assets.

The key difference: a small asset (10 kW) dropping to 0.5 kW is a z-score
of ~-9 within its own history but only ~-2 globally. The naive detector
is blind to it; the segmented detector flags it immediately.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class DetectionResult:
    flagged: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def flagged_set(self) -> Set[Tuple[str, int]]:
        return {
            (asset_id, day)
            for asset_id, days in self.flagged.items()
            for day in days
        }


class SegmentedDetector:
    """Per-asset z-score detector. Each asset is normalized against its own history."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit_detect(self, readings: np.ndarray, asset_ids: List[str]) -> DetectionResult:
        """
        readings: shape (n_days, n_assets)
        Returns DetectionResult with all (asset_id, day) pairs exceeding threshold.
        """
        flagged: Dict[str, List[int]] = {}
        for i, asset_id in enumerate(asset_ids):
            col = readings[:, i]
            mean, std = col.mean(), col.std()
            if std == 0:
                continue
            z = (col - mean) / std
            days = list(np.where(np.abs(z) > self.threshold)[0])
            if days:
                flagged[asset_id] = days
        return DetectionResult(flagged=flagged)


class NaiveDetector:
    """Global z-score detector. Uses a single mean/std computed across all assets."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def fit_detect(self, readings: np.ndarray, asset_ids: List[str]) -> DetectionResult:
        """
        readings: shape (n_days, n_assets)
        Returns DetectionResult using global population statistics.
        """
        flat = readings.flatten()
        mean, std = flat.mean(), flat.std()
        flagged: Dict[str, List[int]] = {}
        if std == 0:
            return DetectionResult(flagged={})
        for i, asset_id in enumerate(asset_ids):
            col = readings[:, i]
            z = (col - mean) / std
            days = list(np.where(np.abs(z) > self.threshold)[0])
            if days:
                flagged[asset_id] = days
        return DetectionResult(flagged=flagged)
