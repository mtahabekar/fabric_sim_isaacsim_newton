#!/usr/bin/env python3
"""Minimum capsule clearance vs time for one run.

    python3 -m fabric_sim_experiments.plotting.plot_min_distance <run_dir> [--safety-margin 0.05]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fabric_sim_experiments.plotting.common import clearance_trace, margin_lines, plt, save


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run_dir', type=Path, help='Run directory containing cbf_debug.csv')
    ap.add_argument('--safety-margin', type=float, default=0.05)
    ap.add_argument('--label', default='run')
    ap.add_argument('--out', type=Path, help='Output base path without extension')
    args = ap.parse_args()
    t, d = clearance_trace(args.run_dir)
    if not len(t):
        raise SystemExit(f'No active-frame data in {args.run_dir/"cbf_debug.csv"}')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, d, color='#1b7837', lw=1.8, label=args.label)
    margin_lines(ax, args.safety_margin)
    ax.set_xlabel('simulation time (s)')
    ax.set_ylabel('min capsule clearance (m)')
    ax.set_title(f'Minimum clearance to the capsule — {args.label}')
    ax.legend()
    out = args.out or args.run_dir/'fig_min_distance'
    save(fig, out)
    print(f'Saved {out}.{{png,pdf,svg}}')


if __name__ == '__main__':
    main()
