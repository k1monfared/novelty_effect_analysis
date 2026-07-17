"""Single entry point: reproduce the entire demonstration.

Steps:
  1. generate the synthetic data (fixed seed) and commit it to data/,
  2. run detection, duration guidance, and naive-vs-debiased estimation,
  3. write JSON and Markdown outputs to outputs/,
  4. render all figures to docs/images/.

Run:  python scripts/run_demo.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, datagen, io_utils, pipeline, reports, figures, explorer


def main() -> None:
    t0 = time.time()
    cfg = config.load_config()
    config.ensure_dirs()

    # 1. Data ---------------------------------------------------------------
    print("[1/4] generating synthetic data ...")
    ground_truth, manifest, daily = datagen.generate(cfg)
    io_utils.write_csv(os.path.join(config.DATA_DIR, "ground_truth.csv"),
                       ground_truth, list(ground_truth[0].keys()))
    io_utils.write_csv(os.path.join(config.DATA_DIR, "experiments_manifest.csv"),
                       manifest, list(manifest[0].keys()))
    io_utils.write_csv(os.path.join(config.DATA_DIR, "daily_observations.csv"),
                       daily, list(daily[0].keys()))
    print(f"      {len(ground_truth)} series, {len(daily)} daily rows written to data/")

    # 2. Analysis -----------------------------------------------------------
    print("[2/4] running detection, duration, and estimation ...")
    results = pipeline.analyze(cfg)

    # 3. Outputs ------------------------------------------------------------
    print("[3/4] writing outputs ...")
    json_paths = reports.write_json_outputs(results)
    md_paths = reports.write_markdown_outputs(results, cfg)
    for p in json_paths + md_paths:
        print(f"      wrote {os.path.relpath(p, config.ROOT)}")

    # 4. Figures ------------------------------------------------------------
    print("[4/4] rendering figures ...")
    fig_paths = figures.generate_all(cfg, results)
    for p in fig_paths:
        print(f"      wrote {os.path.relpath(p, config.ROOT)}")
    explorer_path = explorer.write_explorer(cfg)
    print(f"      wrote {os.path.relpath(explorer_path, config.ROOT)}")

    # Console headline ------------------------------------------------------
    b = results["detection"]["binary"]
    est = results["estimator"]
    nv_n = est["novelty_only"]["naive"]["mae"]
    nv_d = est["novelty_only"]["debiased"]["mae"]
    dur = results["duration"]
    print("")
    print("=" * 68)
    print("HEADLINE RESULTS (synthetic, reproducible)")
    print("=" * 68)
    print(f"Detection novelty flag : precision {b['precision']:.3f}  "
          f"recall {b['recall']:.3f}  F1 {b['f1']:.3f}")
    print(f"Category accuracy      : {results['detection']['category_accuracy']:.3f}")
    print(f"Naive MAE (novelty)    : {100*nv_n:.3f} pp")
    print(f"Debiased MAE (novelty) : {100*nv_d:.3f} pp  "
          f"({(1-nv_d/nv_n)*100:.0f}% lower error)")
    print(f"Duration validation    : {dur['n_validated']}/"
          f"{dur['n_recommendations_in_window']} in-window recommendations "
          f"reach tolerance")
    print("=" * 68)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
