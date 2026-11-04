#!/usr/bin/env python3
"""Experiment 1: corner trajectories and capsule clearance, CBF on vs off.

    python3 -m fabric_sim_experiments.plotting.plot_trajectories_3d <runs/exp1_cbf_on_off>
        [--safety-margin 0.05] [--table-hx 0.22] [--table-hy 0.22] [--table-hz 0.16]

Runs are grouped by their `scenario` tag; the latest CBF-on and CBF-off run of
each scenario are overlaid (solid vs dashed). Per scenario it writes
figures/fig_traj3d_<scenario> and figures/fig_min_distance_<scenario>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from fabric_sim_experiments.corner_naming import CORNER_COLORS, CORNER_KEYS, CORNER_SHORT_LABELS
from fabric_sim_experiments.plotting.common import clearance_trace, columns, margin_lines, plt, runs, save


def read_corners(path: Path) -> dict[str, np.ndarray]:
    keys = [f'{c}_{a}' for c in CORNER_KEYS for a in 'xyz']
    data = columns(path, *keys)
    return {c: np.column_stack(data[3*i:3*i+3]) for i, c in enumerate(CORNER_KEYS)}


def draw_table(ax, hx, hy, hz):
    x0, x1, y0, y1, z0, z1 = -hx, hx, -hy, hy, 0., 2*hz
    faces = [[[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]] for z in (z0, z1)]
    faces += [[[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]] for y in (y0, y1)]
    faces += [[[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]] for x in (x0, x1)]
    ax.add_collection3d(Poly3DCollection(faces, alpha=.15, facecolor='gray', edgecolor='darkgray', linewidth=.4))


def draw_capsule(ax, hx, hy, hz, meridians=12):
    """Above-ground part of the CBF envelope: cylinder plus centered spherical top."""
    radius, top = np.hypot(hx, hy), np.linalg.norm((hx, hy, hz))
    angle = np.linspace(np.arctan2(radius, hz), 0, 30)
    rho = np.concatenate(([radius, radius], top*np.sin(angle)))
    z = np.concatenate(([0., 2*hz], hz+top*np.cos(angle)))
    for phi in np.linspace(0, 2*np.pi, meridians, endpoint=False):
        ax.plot(rho*np.cos(phi), rho*np.sin(phi), z, color='#1f78b4', lw=.5, alpha=.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs_dir', type=Path)
    ap.add_argument('--safety-margin', type=float, default=0.05)
    ap.add_argument('--table-hx', type=float, default=0.22)
    ap.add_argument('--table-hy', type=float, default=0.22)
    ap.add_argument('--table-hz', type=float, default=0.16)
    args = ap.parse_args()
    latest: dict[tuple[str, bool], Path] = {}
    for run_dir, meta in runs(args.runs_dir, 'corners.csv', 'cbf_debug.csv'):
        params = meta.get('run_params', {})
        latest[(str(params.get('scenario', 'default')), bool(params.get('cbf_on', True)))] = run_dir
    if not latest:
        raise SystemExit(f'No complete runs under {args.runs_dir}')
    figures, report = args.runs_dir/'figures', args.runs_dir.parent/'report_figures'
    for scenario in sorted({s for s, _ in latest}):
        on, off = latest.get((scenario, True)), latest.get((scenario, False))
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        for run_dir, style, tag in ((on, dict(lw=1.8), 'CBF on'), (off, dict(lw=1.2, ls='--', alpha=.7), 'CBF off')):
            if run_dir is None:
                continue
            for corner, points in read_corners(run_dir/'corners.csv').items():
                if len(points):
                    ax.plot(*points.T, color=CORNER_COLORS[corner], label=f'{CORNER_SHORT_LABELS[corner]} {tag}',
                            **style)
        draw_table(ax, args.table_hx, args.table_hy, args.table_hz)
        draw_capsule(ax, args.table_hx, args.table_hy, args.table_hz)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('z (m)')
        ax.set_title(f'Corner trajectories — {scenario}')
        ax.legend(fontsize=7, ncol=2, loc='upper left')
        save(fig, figures/f'fig_traj3d_{scenario}', report)

        fig, ax = plt.subplots(figsize=(8, 4))
        for run_dir, style, tag in ((on, dict(color='#1b7837', lw=1.8), 'CBF on'),
                                    (off, dict(color='#a50026', lw=1.2, ls='--'), 'CBF off')):
            if run_dir is not None:
                t, d = clearance_trace(run_dir)
                ax.plot(t, d, label=tag, **style)
        margin_lines(ax, args.safety_margin)
        ax.set_xlabel('simulation time (s)')
        ax.set_ylabel('min capsule clearance (m)')
        ax.set_title(f'Minimum clearance to the capsule — {scenario}')
        ax.legend()
        save(fig, figures/f'fig_min_distance_{scenario}', report)
    print(f'Figures written to {figures}; report symlinks in {report}')


if __name__ == '__main__':
    main()
