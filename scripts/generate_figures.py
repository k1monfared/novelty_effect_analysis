"""Regenerate only the figures from committed data (no data regeneration).

Useful when tweaking figure styling. Requires that data/ already exists
(run scripts/run_demo.py or scripts/generate_data.py first).

Run:  python scripts/generate_figures.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, pipeline, figures


def main() -> None:
    cfg = config.load_config()
    config.ensure_dirs()
    results = pipeline.analyze(cfg)
    paths = figures.generate_all(cfg, results)
    for p in paths:
        print(f"wrote {os.path.relpath(p, config.ROOT)}")


if __name__ == "__main__":
    main()
