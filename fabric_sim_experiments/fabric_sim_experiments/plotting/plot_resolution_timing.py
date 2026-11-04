#!/usr/bin/env python3
"""Experiment 2: simulation rate and per-frame cost vs cloth resolution.

    python3 -m fabric_sim_experiments.plotting.plot_resolution_timing <runs/exp2_resolution_sweep>

Reads meta.json and sim_timing.csv of each run (median over the run). Writes
figures/fig_simulation_performance_vs_cloth_resolution.{png,pdf,svg}; the
frame time includes the sensitivity CBF update, shown separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fabric_sim_experiments.plotting.common import columns, plt, runs, save

FIG_BASENAME = 'fig_simulation_performance_vs_cloth_resolution'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs_dir', type=Path)
    args = ap.parse_args()
    points = []
    for run_dir, meta in runs(args.runs_dir, 'sim_timing.csv'):
        params = meta.get('run_params', {})
        grid = params.get('isaac_args', {})
        particles = params.get('n_particles') or (int(grid.get('grid_x', 0))+1)*(int(grid.get('grid_y', 0))+1)
        fps, frame_ms, cbf_ms = columns(run_dir/'sim_timing.csv', 'fps', 'frame_ms', 'cbf_ms')
        active = cbf_ms > 0  # frames after the CBF warm-up
        if active.any():
            points.append((float(particles), str(params.get('resolution_tag', particles)),
                           np.median(fps[active]), np.median(frame_ms[active]), np.median(cbf_ms[active])))
    if not points:
        raise SystemExit(f'No complete runs with sim_timing.csv under {args.runs_dir}')
    points.sort()
    x, labels, fps, frame_ms, cbf_ms = zip(*points)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    lines = ax1.plot(x, fps, marker='o', color='#1f78b4', lw=1.8, label='Simulation FPS')
    lines += ax2.plot(x, frame_ms, marker='s', color='#e31a1c', lw=1.8, ls='--', label='Frame time')
    lines += ax2.plot(x, cbf_ms, marker='^', color='#33a02c', lw=1.5, ls=':', label='CBF update')
    ax1.set_xscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha='right')
    ax1.set_xlabel('Cloth resolution (particles, log scale)')
    ax1.set_ylabel('Simulation FPS', color='#1f78b4')
    ax2.set_ylabel('Time per frame (ms)', color='#e31a1c')
    ax1.set_title('Simulation performance vs cloth resolution (sensitivity CBF in the loop)')
    ax1.legend(lines, [line.get_label() for line in lines], loc='best')
    ax1.grid(True, alpha=.2)
    save(fig, args.runs_dir/'figures'/FIG_BASENAME, args.runs_dir.parent/'report_figures')
    print(f'Figure written to {args.runs_dir/"figures"}')


if __name__ == '__main__':
    main()
