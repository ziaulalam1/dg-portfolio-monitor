## DG Portfolio Monitor

Per-asset anomaly detection for distributed generation fleets. Compares segmented (per-asset z-score) vs naive (global z-score) detection on a synthetic mixed-scale portfolio.

**Demo**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py        # writes reports/results.json + reports/fpr_comparison.png
pytest tests/              # 11 tests
```

**Invariant**

Segmented detector recall = 1.0: every injected fault is flagged regardless of asset scale. Naive detector misses small-asset faults because global normalization is blind to within-asset deviations. Enforced by `test_segmented_catches_all_injected_anomalies`.
