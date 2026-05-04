#!/usr/bin/env python3
"""
Plot Experiment 3 — minimum obstacle distance for leader-speed sweep.

Usage:
  python3 -m fabric_sim_experiments.plotting.plot_leader_speed_sweep \
      <runs/exp3_leader_speed_sweep> [--safety-margin 0.05]

Produces only:
  - fig_speed_min_distance_traces.{png,pdf,svg}
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
FIG_BASENAME = "fig_speed_min_distance_traces"
REMOVED_OUTPUTS = (
    "fig_speed_vs_min_distance.*",
    "fig_speed_follower_trajectories_3d.*",
)


def _save(fig, base: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix("." + ext), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _read_meta(run_dir: Path) -> dict:
    p = run_dir / "meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _is_complete_run_dir(run_dir: Path) -> bool:
    return (run_dir / "meta.json").exists() and (run_dir / "obstacle_distance.csv").exists()


def _read_distance_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts: list[float] = []
    ds: list[float] = []
    if not path.exists():
        return np.array(ts), np.array(ds)
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["t"]))
                ds.append(float(row["min_distance"]))
            except (ValueError, KeyError):
                pass
    return np.array(ts), np.array(ds)


def _cleanup_old_outputs(figs_dir: Path, report_dir: Path) -> None:
    for pattern in REMOVED_OUTPUTS:
        for path in figs_dir.glob(pattern):
            if path.exists() or path.is_symlink():
                path.unlink()
        for path in report_dir.glob(pattern):
            if path.exists() or path.is_symlink():
                path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", help="Root of exp3 runs, e.g. runs/exp3_leader_speed_sweep")
    ap.add_argument("--safety-margin", type=float, default=0.05)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    figs_dir = runs_dir / "figures"
    report_dir = runs_dir.parent / "report_figures"
    figs_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    _cleanup_old_outputs(figs_dir, report_dir)

    per_speed: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "speed": 0.0,
        "cbf_on": True,
        "dist_traces": [],
    })

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("figures", "report_figures"):
            continue
        if not _is_complete_run_dir(run_dir):
            continue

        meta = _read_meta(run_dir)
        run_params = meta.get("run_params", {})
        speed = float(run_params.get("leader_speed_y", run_params.get("leader_speed_x", 0.0)))
        cbf_on = bool(run_params.get("cbf_params", {}).get("enabled", True))
        if "cbf_on" in run_params:
            cbf_on = bool(run_params["cbf_on"])
        tag = str(run_params.get("speed_tag", f"{speed:.3f}"))
        if not cbf_on:
            tag = f"{tag} CBF off"

        ts, ds = _read_distance_csv(run_dir / "obstacle_distance.csv")
        if len(ts) == 0 or len(ds) == 0:
            continue

        group = per_speed[tag]
        group["speed"] = speed
        group["cbf_on"] = cbf_on
        group["dist_traces"].append((ts - ts[0], ds))

    if not per_speed:
        raise RuntimeError(f"No complete Exp3 runs with obstacle_distance.csv found under {runs_dir}")

    tags = sorted(
        per_speed.keys(),
        key=lambda tag: (float(per_speed[tag]["speed"]), 0 if per_speed[tag]["cbf_on"] else 1),
    )
    cmap = plt.colormaps["plasma"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(
        args.safety_margin,
        color="black",
        ls=":",
        lw=1.2,
        label=f"safety margin ({args.safety_margin} m)",
    )
    ax.axhline(0.0, color="gray", ls="-", lw=0.7, alpha=0.5)

    for i, tag in enumerate(tags):
        color = cmap(i / max(1, len(tags) - 1))
        traces = per_speed[tag]["dist_traces"]
        speed = float(per_speed[tag]["speed"])
        cbf_on = bool(per_speed[tag]["cbf_on"])
        base_label = f"{speed:g} m/s" if cbf_on else f"{speed:g} m/s CBF off"
        for j, (ts, ds) in enumerate(traces):
            ax.plot(
                ts,
                ds,
                color=color,
                lw=1.1,
                alpha=0.85,
                label=base_label if j == 0 else None,
            )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("min obstacle distance (m)")
    ax.set_title("Minimum obstacle distance for different leader speeds")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="major", alpha=0.2)
    _save(fig, figs_dir / FIG_BASENAME)

    for fig_file in sorted(figs_dir.glob(f"{FIG_BASENAME}.*")):
        link = report_dir / fig_file.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(fig_file.resolve())

    print(f"Figure written to {figs_dir}")


if __name__ == "__main__":
    main()
