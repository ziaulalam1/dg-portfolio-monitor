"""
Invariant tests for DG Portfolio Monitor.

Signal 1 (Correctness/reliability):
    The segmented detector must flag every injected anomaly -- recall = 1.0.
    A missed fault is a silent failure; false negatives are unacceptable.

Signal 2 (Measurement/evaluation):
    The naive detector must miss at least one injected anomaly on this dataset,
    confirming the benchmark's core claim that global normalization is blind to
    small-asset faults.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_portfolio, inject_anomalies
from monitor import DetectionResult, NaiveDetector, SegmentedDetector

ANOMALIES = [
    ("asset_000", 100),
    ("asset_001", 200),
    ("asset_002", 300),
]


@pytest.fixture(scope="module")
def portfolio():
    readings, asset_ids, baselines = generate_portfolio(
        n_small=5, n_large=15, n_days=365, seed=42
    )
    readings, ground_truth = inject_anomalies(
        readings, asset_ids, baselines, ANOMALIES
    )
    return readings, asset_ids, ground_truth


@pytest.fixture(scope="module")
def seg_result(portfolio):
    readings, asset_ids, _ = portfolio
    return SegmentedDetector(threshold=3.0).fit_detect(readings, asset_ids)


@pytest.fixture(scope="module")
def naive_result(portfolio):
    readings, asset_ids, _ = portfolio
    return NaiveDetector(threshold=3.0).fit_detect(readings, asset_ids)


# ── Signal 1: Correctness/reliability ────────────────────────────────────────

def test_segmented_catches_all_injected_anomalies(portfolio, seg_result):
    """Every injected fault must appear in segmented detector output. Recall = 1.0."""
    _, _, ground_truth = portfolio
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    missed = gt_set - seg_result.flagged_set
    assert missed == set(), f"Segmented detector missed injected anomalies: {missed}"


def test_segmented_recall_is_one(portfolio, seg_result):
    """Recall must be exactly 1.0 -- no injected anomaly goes undetected."""
    _, _, ground_truth = portfolio
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    tp = len(gt_set & seg_result.flagged_set)
    recall = tp / len(gt_set)
    assert recall == 1.0, f"Segmented recall={recall:.4f}, expected 1.0"


def test_segmented_anomaly_locations_correct(portfolio, seg_result):
    """Each injected (asset_id, day) pair is individually present in the result."""
    for asset_id, day in ANOMALIES:
        assert asset_id in seg_result.flagged, \
            f"asset {asset_id} not flagged at all"
        assert day in seg_result.flagged[asset_id], \
            f"asset {asset_id} day {day} not flagged"


# ── Signal 2: Measurement/evaluation ─────────────────────────────────────────

def test_naive_misses_small_asset_anomalies(portfolio, naive_result):
    """Naive global detector must miss at least one small-asset fault.
    This confirms the FPR benchmark claim is meaningful -- the detectors differ."""
    _, _, ground_truth = portfolio
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    naive_tp = len(gt_set & naive_result.flagged_set)
    assert naive_tp < len(gt_set), (
        "Naive detector caught all anomalies -- benchmark comparison is trivial. "
        "Check asset baseline ranges or anomaly drop factor."
    )


def test_naive_recall_is_zero(portfolio, naive_result):
    """Naive detector must catch zero injected faults on this mixed-scale fleet.
    The global std is dominated by baseline spread (10-100 kW range), making
    the effective threshold so wide that no reading -- including fault readings --
    ever exceeds it. The detector is completely inactive."""
    _, _, ground_truth = portfolio
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    naive_tp = len(gt_set & naive_result.flagged_set)
    assert naive_tp == 0, (
        f"Naive caught {naive_tp} anomaly(s) -- "
        "check asset baseline ranges, the global std should be too wide to trigger"
    )


def test_segmented_fpr_within_bounds(portfolio, seg_result):
    """Segmented FPR must be non-zero (detector is active) and below 1%."""
    _, asset_ids, ground_truth = portfolio
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    total_neg = 365 * len(asset_ids) - len(gt_set)
    seg_fp = len(seg_result.flagged_set - gt_set)
    fpr = seg_fp / total_neg
    assert fpr > 0, "Segmented FPR is exactly zero -- detector may not be working"
    assert fpr < 0.01, f"Segmented FPR {fpr:.4f} exceeds 1% -- threshold may be too low"


# ── DetectionResult unit tests ────────────────────────────────────────────────

def test_detection_result_flagged_set_empty():
    r = DetectionResult(flagged={})
    assert r.flagged_set == set()


def test_detection_result_flagged_set_correct():
    r = DetectionResult(flagged={"asset_001": [5, 10], "asset_002": [20]})
    assert r.flagged_set == {("asset_001", 5), ("asset_001", 10), ("asset_002", 20)}


def test_generate_portfolio_shape():
    from data import generate_portfolio
    readings, asset_ids, baselines = generate_portfolio(n_small=3, n_large=7, n_days=100)
    assert readings.shape == (100, 10)
    assert len(asset_ids) == 10
    assert len(baselines) == 10


def test_inject_anomalies_modifies_correct_cell():
    from data import generate_portfolio, inject_anomalies
    readings, asset_ids, baselines = generate_portfolio(n_small=2, n_large=3, n_days=50, seed=1)
    original_val = readings[10, 0]
    modified, gt = inject_anomalies(readings, asset_ids, baselines, [("asset_000", 10)])
    assert modified[10, 0] < original_val * 0.1
    assert "asset_000" in gt
    assert 10 in gt["asset_000"]


def test_inject_anomalies_does_not_mutate_original():
    from data import generate_portfolio, inject_anomalies
    readings, asset_ids, baselines = generate_portfolio(n_small=2, n_large=3, n_days=50, seed=1)
    original = readings.copy()
    inject_anomalies(readings, asset_ids, baselines, [("asset_000", 10)])
    np.testing.assert_array_equal(readings, original)


def test_readings_non_negative():
    from data import generate_portfolio
    readings, _, _ = generate_portfolio(seed=42)
    assert (readings >= 0).all(), "All readings must be non-negative (physical constraint)"
