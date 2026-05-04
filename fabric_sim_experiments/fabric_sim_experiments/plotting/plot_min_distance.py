#!/usr/bin/env python3
"""
Standalone min-distance vs time plotter (shared utility / Exp 1 overlay).

Usage:
  python3 -m fabric_sim_experiments.plotting.plot_min_distance <run_dir>
      [--safety-margin 0.05] [--label "CBF on"]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300


def _read_distance_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, ds = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["t"]))
                ds.append(float(row["min_distance"]))
            except (ValueError, KeyError):
                pass
    return np.array(ts), np.array(ds)


def _save(fig, base: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix("." + ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Run directory (contains obstacle_distance.csv)")
    ap.add_argument("--safety-margin", type=float, default=0.05)
    ap.add_argument("--label", type=str, default="run")
    ap.add_argument("--out", type=str, default=None, help="Output base path (no extension)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    dist_csv = run_dir / "obstacle_distance.csv"
    if not dist_csv.exists():
        print(f"ERROR: {dist_csv} not found")
        return

    ts, ds = _read_distance_csv(dist_csv)
    if len(ts) == 0:
        print("ERROR: no data in obstacle_distance.csv")
        return

    ts = ts - ts[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts, ds, color="#1b7837", lw=1.8, label=args.label)
    ax.axhline(args.safety_margin, color="black", ls=":", lw=1.2,
               label=f"safety margin ({args.safety_margin} m)")
    ax.axhline(0.0, color="gray", ls="-", lw=0.7, alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("min obstacle distance (m)")
    ax.set_title(f"Obstacle min-distance — {args.label}")
    ax.legend()

    out_base = Path(args.out) if args.out else run_dir / "fig_min_distance"
    _save(fig, out_base)
    print(f"Saved: {out_base}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
