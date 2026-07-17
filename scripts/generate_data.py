"""Generate and commit the synthetic dataset.

Writes three CSV tables into data/:
    ground_truth.csv        known labels and true parameters (scoring only)
    experiments_manifest.csv generation parameters and human-readable stories
    daily_observations.csv   the daily A/B readings the analysis consumes

Run directly, or via scripts/run_demo.py which calls generate() first.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, datagen, io_utils


def main() -> None:
    cfg = config.load_config()
    config.ensure_dirs()

    ground_truth, manifest, daily = datagen.generate(cfg)

    gt_path = os.path.join(config.DATA_DIR, "ground_truth.csv")
    mf_path = os.path.join(config.DATA_DIR, "experiments_manifest.csv")
    daily_path = os.path.join(config.DATA_DIR, "daily_observations.csv")

    io_utils.write_csv(gt_path, ground_truth, list(ground_truth[0].keys()))
    io_utils.write_csv(mf_path, manifest, list(manifest[0].keys()))
    io_utils.write_csv(daily_path, daily, list(daily[0].keys()))

    n_series = len(ground_truth)
    n_novelty = sum(r["is_novelty"] for r in ground_truth)
    cats = {}
    for r in ground_truth:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    print(f"[generate_data] wrote {n_series} experiment-metric series, {len(daily)} daily rows")
    print(f"[generate_data] categories: {cats}")
    print(f"[generate_data] novelty-positive series: {n_novelty}")
    print(f"[generate_data] files: {gt_path}, {mf_path}, {daily_path}")


if __name__ == "__main__":
    main()
