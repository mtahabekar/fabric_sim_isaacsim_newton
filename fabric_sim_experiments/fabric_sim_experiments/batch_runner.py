#!/usr/bin/env python3
"""
Iterates over all cells in a sweep config YAML and calls ExperimentRunner for each.
Writes a top-level summary.csv with per-run scalar KPIs.

Run via:
  ros2 run fabric_sim_experiments batch_runner \
      --ros-args -p config:=<path/to/exp.yaml> [--headless]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node

from fabric_sim_experiments.experiment_runner import (
    ExperimentRunner,
    _load_yaml,
    generate_figures_for_experiment,
)
from fabric_sim_experiments.corner_naming import CORNER_KEYS, row_xyz


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

def _kpis_from_run_dir(run_dir: Path) -> dict[str, Any]:
    kpis: dict[str, Any] = {}

    # --- obstacle_distance.csv ---
    dist_csv = run_dir / "obstacle_distance.csv"
    if dist_csv.exists():
        import csv as _csv
        min_dist = float("inf")
        n_violations = 0
        with open(dist_csv) as f:
            for row in _csv.DictReader(f):
                try:
                    md = float(row["min_distance"])
                    min_dist = min(min_dist, md)
                    if md < 0.0:
                        n_violations += 1
                except (ValueError, KeyError):
                    pass
        kpis["min_obstacle_dist_global_min"] = min_dist if min_dist < float("inf") else None
        kpis["n_violation_steps"] = n_violations

    # --- sim_perf.csv ---
    perf_csv = run_dir / "sim_perf.csv"
    if perf_csv.exists():
        import csv as _csv
        fps_vals, ms_vals = [], []
        with open(perf_csv) as f:
            for row in _csv.DictReader(f):
                try:
                    fps_vals.append(float(row["sim_fps"]))
                    ms_vals.append(float(row["sim_step_ms"]))
                except (ValueError, KeyError):
                    pass
        if fps_vals:
            kpis["mean_sim_fps"] = sum(fps_vals) / len(fps_vals)
        if ms_vals:
            kpis["mean_sim_step_ms"] = sum(ms_vals) / len(ms_vals)

    # --- cbf_debug.csv ---
    debug_csv = run_dir / "cbf_debug.csv"
    if debug_csv.exists():
        import csv as _csv
        solve_ms = []
        active_steps = 0
        stretch_h_min = float("inf")
        stretch_h_max = float("-inf")
        with open(debug_csv) as f:
            for row in _csv.DictReader(f):
                try:
                    solve_ms.append(float(row["compute_time_ms"]))
                    if int(row.get("active", 0)):
                        active_steps += 1
                    h_str = row.get("stretch_h_values", "")
                    if h_str:
                        for v in h_str.split("|"):
                            if v:
                                hv = float(v)
                                stretch_h_min = min(stretch_h_min, hv)
                                stretch_h_max = max(stretch_h_max, hv)
                except (ValueError, KeyError):
                    pass
        if solve_ms:
            kpis["mean_cbf_solve_ms"] = sum(solve_ms) / len(solve_ms)
            kpis["cbf_solver_max_ms"] = max(solve_ms)
        kpis["n_cbf_active_steps"] = active_steps
        if stretch_h_min < float("inf"):
            kpis["min_stretch_h"] = stretch_h_min
        if stretch_h_max > float("-inf"):
            kpis["max_stretch_h"] = stretch_h_max

    # --- corners.csv ---
    corners_csv = run_dir / "corners.csv"
    if corners_csv.exists():
        import csv as _csv
        max_z = {c: float("-inf") for c in CORNER_KEYS}
        with open(corners_csv) as f:
            for row in _csv.DictReader(f):
                for c in CORNER_KEYS:
                    try:
                        _, _, z = row_xyz(row, c)
                        max_z[c] = max(max_z[c], z)
                    except (ValueError, KeyError):
                        pass
        for c, v in max_z.items():
            if v > float("-inf"):
                kpis[f"max_{c}_z"] = v

    return kpis


# ---------------------------------------------------------------------------
# Batch runner node
# ---------------------------------------------------------------------------

class BatchRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("batch_runner")
        self.declare_parameter("config", "")
        self.declare_parameter("runs_root", "runs")
        self.declare_parameter("headless", False)
        self.declare_parameter("settle_time", 5.0)
        self.declare_parameter("generate_figures", True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BatchRunnerNode()

    config_path_str = node.get_parameter("config").value
    runs_root = Path(node.get_parameter("runs_root").value).expanduser()
    headless = bool(node.get_parameter("headless").value)
    settle_time = float(node.get_parameter("settle_time").value)
    generate_figures = bool(node.get_parameter("generate_figures").value)

    node.destroy_node()
    rclpy.shutdown()

    if not config_path_str:
        print("ERROR: 'config' parameter is required", file=sys.stderr)
        sys.exit(1)

    config_path = Path(config_path_str).expanduser()
    cfg = _load_yaml(config_path)
    sweep: list[dict] = cfg.get("sweep", [{}])
    exp_name: str = cfg.get("exp_name", config_path.stem)
    n_repeats: int = int(cfg.get("n_repeats", 1))

    summary_rows: list[dict] = []

    for cell_idx, run_params in enumerate(sweep):
        for rep in range(n_repeats):
            label = f"cell_{cell_idx:03d}_rep_{rep:02d}"
            rp = dict(run_params)
            rp["cell_idx"] = cell_idx
            rp["rep"] = rep

            print(f"\n[batch] {label}: {rp}")
            runner = ExperimentRunner(
                config_path=config_path,
                run_params=rp,
                runs_root=runs_root,
                exp_name=exp_name,
                headless=headless,
                settle_time=settle_time,
                generate_figures=False,
            )
            try:
                run_dir = runner.run()
            except Exception as exc:
                print(f"[batch] ERROR in {label}: {exc} — skipping to next run")
                continue

            kpis = _kpis_from_run_dir(run_dir)
            row = {"run_id": runner.run_id, "exp_name": exp_name, **rp, **kpis}
            summary_rows.append(row)

    # Write summary CSV
    summary_path = runs_root / exp_name / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        all_keys = list(dict.fromkeys(k for r in summary_rows for k in r))
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[batch] Summary written to {summary_path}")

    if generate_figures and summary_rows:
        generate_figures_for_experiment(exp_name, runs_root / exp_name, cfg)

    print(f"\n[batch] All {len(summary_rows)} run(s) complete.")


if __name__ == "__main__":
    main()
