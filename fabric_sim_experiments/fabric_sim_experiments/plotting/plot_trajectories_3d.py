#!/usr/bin/env python3
"""
Plot 3D corner trajectories for Experiment 1 (CBF on vs off).

Usage:
  python3 -m fabric_sim_experiments.plotting.plot_trajectories_3d <runs/exp1_cbf_on_off>
      [--safety-margin 0.05] [--table-hx 0.22] [--table-hy 0.22] [--table-hz 0.16]

Discovers run directories, groups them by (z_tag, y_tag, cbf_on), and for
each group produces:
  - fig_traj3d_z{z}_y{y}.{png,pdf,svg}       — overlaid leader/follower trajectories
  - fig_min_distance_z{z}_y{y}.{png,pdf,svg} — min-distance vs time
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from fabric_sim_experiments.corner_naming import (
    CORNER_COLORS,
    CORNER_KEYS,
    CORNER_SHORT_LABELS,
    row_xyz,
)

DPI = 300


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_corners(path: Path) -> dict[str, np.ndarray]:
    data: dict[str, list] = {c: [] for c in CORNER_KEYS}
    t: list[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t.append(float(row["t"]))
                for c in CORNER_KEYS:
                    data[c].append(list(row_xyz(row, c)))
            except (ValueError, KeyError):
                pass
    return {c: np.array(v) for c, v in data.items()} | {"t": np.array(t)}


def _read_distances(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, ds = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(float(row["t"]))
                ds.append(float(row["min_distance"]))
            except (ValueError, KeyError):
                pass
    return np.array(ts), np.array(ds)


def _read_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _is_complete_run_dir(run_dir: Path) -> bool:
    return (
        (run_dir / "meta.json").exists()
        and (run_dir / "corners.csv").exists()
        and (run_dir / "obstacle_distance.csv").exists()
    )


# ---------------------------------------------------------------------------
# Box drawing helpers
# ---------------------------------------------------------------------------

def _box_faces(center, half):
    cx, cy, cz = center
    hx, hy, hz = half
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    faces = [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
    ]
    return faces


def _add_table_3d(ax, hx, hy, hz, alpha=0.15):
    faces = _box_faces([0, 0, hz], [hx, hy, hz])
    collection = Poly3DCollection(faces, alpha=alpha, facecolor="gray", edgecolor="darkgray", linewidth=0.4)
    ax.add_collection3d(collection)


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, base: Path) -> None:
    for ext in ("png", "pdf", "svg"):
        p = base.with_suffix("." + ext)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _normalise_t(t: np.ndarray) -> np.ndarray:
    if len(t) == 0:
        return t
    return t - t[0]


def plot_traj3d(
    cbf_on_corners: dict,
    cbf_off_corners: dict,
    out_base: Path,
    title: str,
    hx: float, hy: float, hz: float,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for c in CORNER_KEYS:
        col = CORNER_COLORS[c]
        label = CORNER_SHORT_LABELS[c]
        if cbf_on_corners and c in cbf_on_corners and len(cbf_on_corners[c]) > 0:
            pts = cbf_on_corners[c]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=1.8, label=f"{label} CBF on")
        if cbf_off_corners and c in cbf_off_corners and len(cbf_off_corners[c]) > 0:
            pts = cbf_off_corners[c]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=1.2, ls="--", alpha=0.7, label=f"{label} CBF off")

    _add_table_3d(ax, hx, hy, hz)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    _save(fig, out_base)


def plot_min_distance(
    on_t: np.ndarray, on_d: np.ndarray,
    off_t: np.ndarray, off_d: np.ndarray,
    out_base: Path,
    title: str,
    safety_margin: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    if len(on_t) > 0:
        ax.plot(_normalise_t(on_t), on_d, color="#1b7837", lw=1.8, label="CBF on")
    if len(off_t) > 0:
        ax.plot(_normalise_t(off_t), off_d, color="#a50026", lw=1.2, ls="--", label="CBF off")

    ax.axhline(safety_margin, color="black", ls=":", lw=1.2, label=f"safety margin ({safety_margin} m)")
    ax.axhline(0.0, color="gray", ls="-", lw=0.7, alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("min obstacle distance (m)")
    ax.set_title(title)
    ax.legend()
    _save(fig, out_base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", help="Root of experiment runs, e.g. runs/exp1_cbf_on_off")
    ap.add_argument("--safety-margin", type=float, default=0.05)
    ap.add_argument("--table-hx", type=float, default=0.22)
    ap.add_argument("--table-hy", type=float, default=0.22)
    ap.add_argument("--table-hz", type=float, default=0.16)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    figs_dir = runs_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    # Group runs by (z_tag, y_tag, cbf_on). Older x-sweep metadata is accepted.
    groups: dict[tuple, Path] = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "figures":
            continue
        if not _is_complete_run_dir(run_dir):
            continue
        meta = _read_meta(run_dir)
        rp = meta.get("run_params", {})
        z_tag = str(rp.get("cloth_start_z_tag", rp.get("cloth_start_z", "unk")))
        y_tag = str(
            rp.get(
                "cloth_start_y_tag",
                rp.get("cloth_start_y", rp.get("cloth_start_x_tag", rp.get("cloth_start_x", "unk"))),
            )
        )
        cbf_on = bool(rp.get("cbf_on", True))
        key = (z_tag, y_tag, cbf_on)
        # Take last run for this key (latest run_id lexicographically)
        if key not in groups or run_dir.name > groups[key].name:
            groups[key] = run_dir

    # Collect unique (z, y) cells
    cells: set[tuple[str, str]] = {(k[0], k[1]) for k in groups}

    for z_tag, y_tag in sorted(cells):
        on_dir = groups.get((z_tag, y_tag, True))
        off_dir = groups.get((z_tag, y_tag, False))

        on_corners = _read_corners(on_dir / "corners.csv") if on_dir else {}
        off_corners = _read_corners(off_dir / "corners.csv") if off_dir else {}
        on_t, on_d = _read_distances(on_dir / "obstacle_distance.csv") if on_dir else (np.array([]), np.array([]))
        off_t, off_d = _read_distances(off_dir / "obstacle_distance.csv") if off_dir else (np.array([]), np.array([]))

        tag = f"z{z_tag}_y{y_tag}".replace("-", "m").replace(".", "p")
        # 3D trajectory
        plot_traj3d(
            on_corners, off_corners,
            figs_dir / f"fig_traj3d_{tag}",
            "3D corner trajectories",
            args.table_hx, args.table_hy, args.table_hz,
        )

        # min distance vs time
        plot_min_distance(
            on_t, on_d, off_t, off_d,
            figs_dir / f"fig_min_distance_{tag}",
            "Minimum distance to the obstacle",
            args.safety_margin,
        )

    # Report figure output directory
    report_dir = runs_dir.parent / "report_figures"
    report_dir.mkdir(exist_ok=True)
    for fig_file in sorted(figs_dir.glob("*")):
        link = report_dir / fig_file.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(fig_file.resolve())

    print(f"Figures written to {figs_dir}")
    print(f"Report symlinks in {report_dir}")


if __name__ == "__main__":
    main()
