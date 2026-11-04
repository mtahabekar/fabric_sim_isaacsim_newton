#!/usr/bin/env python3
"""Experiment 3: minimum capsule clearance vs time for each leader speed.

    python3 -m fabric_sim_experiments.plotting.plot_leader_speed_sweep <runs/exp3_leader_speed_sweep>

Writes figures/fig_speed_min_distance_traces.{png,pdf,svg}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from fabric_sim_experiments.plotting.common import clearance_trace, margin_lines, plt, runs, save

FIG_BASENAME = 'fig_speed_min_distance_traces'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs_dir', type=Path)
    ap.add_argument('--safety-margin', type=float, default=0.05)
    args = ap.parse_args()
    groups: dict[tuple[float, bool], list] = defaultdict(list)
    for run_dir, meta in runs(args.runs_dir, 'cbf_debug.csv'):
        params = meta.get('run_params', {})
        speed = float(np.linalg.norm(params.get('leader_velocity', (0, 0, 0))))
        t, d = clearance_trace(run_dir)
        if len(t):
            groups[(speed, bool(params.get('cbf_on', True)))].append((t, d))
    if not groups:
        raise SystemExit(f'No complete runs with cbf_debug.csv under {args.runs_dir}')
    keys = sorted(groups, key=lambda k: (k[0], not k[1]))
    cmap = plt.colormaps['plasma']
    fig, ax = plt.subplots(figsize=(9, 5))
    margin_lines(ax, args.safety_margin)
    for i, key in enumerate(keys):
        speed, cbf_on = key
        color = cmap(i/max(1, len(keys)-1))
        label = f'{speed:g} m/s' + ('' if cbf_on else ' (CBF off)')
        for j, (t, d) in enumerate(groups[key]):
            ax.plot(t, d, color=color, lw=1.1, alpha=.85, ls='-' if cbf_on else '--',
                    label=label if j == 0 else None)
    ax.set_xlabel('simulation time (s)')
    ax.set_ylabel('min capsule clearance (m)')
    ax.set_title('Minimum capsule clearance for different leader speeds')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=.2)
    save(fig, args.runs_dir/'figures'/FIG_BASENAME, args.runs_dir.parent/'report_figures')
    print(f'Figure written to {args.runs_dir/"figures"}')


if __name__ == '__main__':
    main()
