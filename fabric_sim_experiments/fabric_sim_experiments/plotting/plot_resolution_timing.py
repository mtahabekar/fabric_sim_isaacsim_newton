#!/usr/bin/env python3
"""
Plot Experiment 2 — simulation performance vs cloth resolution.

Usage:
  python3 -m fabric_sim_experiments.plotting.plot_resolution_timing \
      <runs/exp2_resolution_sweep>

Reads each completed run's meta.json and sim_perf.csv.
Produces only:
  - fig_simulation_performance_vs_cloth_resolution.{png,pdf,svg}
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
FIG_BASENAME = "fig_simulation_performance_vs_cloth_resolution"
REMOVED_OUTPUTS = (
    "fig_resolution_fps.*",
    "fig_resolution_cbf_hz.*",
    "fig_resolution_solver_ms.*",
    "table_resolution_timing.csv",
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
    return (run_dir / "meta.json").exists() and (run_dir / "sim_perf.csv").exists()


def _read_sim_perf(run_dir: Path) -> tuple[list[float], list[float]]:
    fps_list: list[float] = []
    ms_list: list[float] = []
    p = run_dir / "sim_perf.csv"
    if not p.exists():
        return fps_list, ms_list
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                fps_list.append(float(row["sim_fps"]))
                ms_list.append(float(row["sim_step_ms"]))
            except (ValueError, KeyError):
                pass
    return fps_list, ms_list


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
    ap.add_argument("runs_dir", help="Root of exp2 runs, e.g. runs/exp2_resolution_sweep")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    figs_dir = runs_dir / "figures"
    report_dir = runs_dir.parent / "report_figures"
    figs_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    _cleanup_old_outputs(figs_dir, report_dir)

    points: list[dict[str, float | str]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in ("figures", "report_figures"):
            continue
        if not _is_complete_run_dir(run_dir):
            continue

        meta = _read_meta(run_dir)
        run_params = meta.get("run_params", {})
        n_particles = run_params.get("n_particles")
        tag = str(run_params.get("resolution_tag", "unknown"))
        if n_particles is None:
            isaac_args = run_params.get("isaac_args", {})
            grid_x = int(isaac_args.get("grid_x", 0))
            grid_y = int(isaac_args.get("grid_y", 0))
            n_particles = (grid_x + 1) * (grid_y + 1) if grid_x and grid_y else 0

        fps_vals, ms_vals = _read_sim_perf(run_dir)
        if not fps_vals or not ms_vals:
            continue

        points.append({
            "tag": tag,
            "n_particles": float(n_particles),
            "fps": statistics.median(fps_vals),
            "step_ms": statistics.median(ms_vals),
        })

    if not points:
        raise RuntimeError(f"No complete Exp2 runs with sim_perf.csv found under {runs_dir}")

    points.sort(key=lambda p: float(p["n_particles"]))
    x = [float(p["n_particles"]) for p in points]
    fps = [float(p["fps"]) for p in points]
    step_ms = [float(p["step_ms"]) for p in points]
    labels = [str(p["tag"]) for p in points]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    line1 = ax1.plot(
        x,
        fps,
        marker="o",
        color="#1f78b4",
        lw=1.8,
        label="Sim FPS",
    )
    line2 = ax2.plot(
        x,
        step_ms,
        marker="s",
        color="#e31a1c",
        lw=1.8,
        ls="--",
        label="Sim step time",
    )

    ax1.set_xscale("log")
    ax1.set_xlabel("Number of cloth particles")
    ax1.set_ylabel("Simulation FPS", color="#1f78b4")
    ax2.set_ylabel("Simulation step time (ms)", color="#e31a1c")
    ax1.set_title("Simulation performance vs Cloth resolution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    lines = line1 + line2
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.grid(True, which="both", axis="x", alpha=0.2)
    ax1.grid(True, which="major", axis="y", alpha=0.2)

    base = figs_dir / FIG_BASENAME
    _save(fig, base)

    for fig_file in sorted(figs_dir.glob(f"{FIG_BASENAME}.*")):
        link = report_dir / fig_file.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(fig_file.resolve())

    print(f"Figure written to {figs_dir}")


if __name__ == "__main__":
    main()
