"""
DG Portfolio Monitor benchmark.

Compares SegmentedDetector vs NaiveDetector on a synthetic fleet.
Measures false positive rate and recall on injected anomalies.

Usage:
    python benchmark.py

Outputs:
    reports/results.json       -- metrics for both detectors
    reports/fpr_comparison.png -- FPR + recall bar chart
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data import generate_portfolio, inject_anomalies
from monitor import DetectionResult, NaiveDetector, SegmentedDetector

REPORTS = Path(__file__).parent / "reports"

# Anomalies injected into small-baseline assets (asset_000 through asset_002).
# These assets have baselines of ~10-25 kW. A 95% output drop is a massive
# per-asset z-score but barely registers globally -- exactly what the naive
# detector misses.
ANOMALIES: List[Tuple[str, int]] = [
    ("asset_000", 100),
    ("asset_001", 200),
    ("asset_002", 300),
]


def compute_metrics(
    result: DetectionResult,
    ground_truth: Dict[str, List[int]],
    n_days: int,
    asset_ids: List[str],
) -> Dict:
    gt_set = {(a, d) for a, days in ground_truth.items() for d in days}
    det_set = result.flagged_set

    tp = len(gt_set & det_set)
    fn = len(gt_set - det_set)
    fp = len(det_set - gt_set)
    total_negatives = n_days * len(asset_ids) - len(gt_set)
    fpr = fp / total_negatives if total_negatives > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "fpr": round(fpr, 6),
        "recall": round(recall, 4),
    }


def main() -> None:
    n_days = 365
    readings, asset_ids, baselines = generate_portfolio(
        n_small=5, n_large=15, n_days=n_days, seed=42
    )
    readings, ground_truth = inject_anomalies(
        readings, asset_ids, baselines, ANOMALIES
    )

    seg = SegmentedDetector(threshold=3.0)
    naive = NaiveDetector(threshold=3.0)

    seg_result = seg.fit_detect(readings, asset_ids)
    naive_result = naive.fit_detect(readings, asset_ids)

    seg_m = compute_metrics(seg_result, ground_truth, n_days, asset_ids)
    naive_m = compute_metrics(naive_result, ground_truth, n_days, asset_ids)

    print("\n=== DG Portfolio Monitor Benchmark ===\n")
    print(f"  Assets          : {len(asset_ids)} ({5} small / {15} large)")
    print(f"  Days            : {n_days}")
    print(f"  Injected faults : {len(ANOMALIES)}")
    print(f"  Threshold       : z > 3.0\n")

    rows = [
        {"detector": "Segmented", **seg_m},
        {"detector": "Naive",     **naive_m},
    ]
    df = pd.DataFrame(rows).set_index("detector")
    print(df.to_string())
    print()

    if seg_m["recall"] == 1.0:
        print("  PASS  Segmented detector caught all injected anomalies")
    else:
        print("  FAIL  Segmented detector missed an injected anomaly")
        sys.exit(1)

    if naive_m["recall"] < seg_m["recall"]:
        missed = len(ANOMALIES) - naive_m["tp"]
        print(f"  OK    Naive detector missed {missed}/{len(ANOMALIES)} anomaly(s) -- expected")

    REPORTS.mkdir(exist_ok=True)

    report = {
        "seed": 42,
        "n_assets": len(asset_ids),
        "n_days": n_days,
        "threshold": 3.0,
        "injected_anomalies": len(ANOMALIES),
        "segmented": seg_m,
        "naive": naive_m,
    }
    results_path = REPORTS / "results.json"
    results_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report  -> {results_path}")

    _plot(seg_m, naive_m)
    print(f"  Chart   -> {REPORTS / 'fpr_comparison.png'}\n")


def _plot(seg: Dict, naive: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    colors = ["#3fb950", "#f85149"]

    ax1.bar(["Segmented", "Naive"], [seg["fpr"], naive["fpr"]], color=colors)
    ax1.set_title("False Positive Rate", color="#e6edf3")
    ax1.set_ylabel("FPR", color="#8b949e")
    ax1.set_ylim(0, max(naive["fpr"] * 1.4, 0.005))
    for i, v in enumerate([seg["fpr"], naive["fpr"]]):
        ax1.text(i, v + naive["fpr"] * 0.05, f"{v:.4f}",
                 ha="center", color="#e6edf3", fontsize=10)

    ax2.bar(["Segmented", "Naive"], [seg["recall"], naive["recall"]], color=colors)
    ax2.set_title("Recall (injected faults caught)", color="#e6edf3")
    ax2.set_ylabel("Recall", color="#8b949e")
    ax2.set_ylim(0, 1.15)
    for i, v in enumerate([seg["recall"], naive["recall"]]):
        ax2.text(i, v + 0.03, f"{v:.2f}", ha="center", color="#e6edf3", fontsize=10)

    fig.suptitle("Segmented vs Naive: DG Portfolio Anomaly Detection",
                 color="#e6edf3", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS / "fpr_comparison.png", dpi=120,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    main()
