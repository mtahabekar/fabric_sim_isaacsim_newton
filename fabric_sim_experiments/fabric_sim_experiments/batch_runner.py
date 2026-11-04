#!/usr/bin/env python3
"""Run every sweep cell of an experiment YAML (n_repeats each) and write summary.csv.

    ros2 run fabric_sim_experiments batch_runner --ros-args -p config:=<exp.yaml> [-p headless:=true]
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

from fabric_sim_experiments.corner_naming import CORNER_KEYS, row_xyz
from fabric_sim_experiments.experiment_runner import (ExperimentRunner, generate_figures_for_experiment,
                                                      read_parameters)
from fabric_sim_bringup.sim_command import load_yaml


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as stream:
        return list(csv.DictReader(stream))


def _floats(rows: list[dict[str, str]], key: str) -> np.ndarray:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(out)


def kpis_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """Scalar results of one run from its CSVs (frames after the CBF warm-up)."""
    kpis: dict[str, Any] = {}
    debug = [r for r in _rows(run_dir/'cbf_debug.csv') if r.get('warmup') in ('False', 'false', '0')]
    if debug:
        clearance, h = _floats(debug, 'min_clearance'), _floats(debug, 'h_min')
        tension, cbf_ms = _floats(debug, 'max_tension'), _floats(debug, 'controller_ms')
        statuses = [r['status'] for r in debug]
        kpis.update(frames=len(debug), min_clearance=float(clearance.min()), min_barrier=float(h.min()),
                    frames_inside_margin=int((h < 0).sum()), frames_touching_capsule=int((clearance < 0).sum()),
                    frames_relaxed=sum(s.startswith('relaxed') for s in statuses),
                    statuses=json.dumps({s: statuses.count(s) for s in sorted(set(statuses))}))
        if len(tension):
            kpis['max_tension'] = float(tension.max())
        if len(cbf_ms):
            kpis.update(cbf_ms_mean=float(cbf_ms.mean()), cbf_ms_p95=float(np.percentile(cbf_ms, 95)),
                        cbf_ms_max=float(cbf_ms.max()))
    timing = _rows(run_dir/'sim_timing.csv')
    if timing:
        kpis.update(sim_fps_median=float(np.median(_floats(timing, 'fps'))),
                    frame_ms_median=float(np.median(_floats(timing, 'frame_ms'))))
    corners = _rows(run_dir/'corners.csv')
    for corner in CORNER_KEYS:
        z = [row_xyz(row, corner)[2] for row in corners]
        if z:
            kpis[f'max_{corner}_z'] = max(z)
    return kpis


def main(args=None) -> None:
    p = read_parameters('batch_runner', dict(config='', runs_root='runs', headless=False, settle_time=3.0,
                                             generate_figures=True), args)
    if not p['config']:
        raise SystemExit("The 'config' parameter is required")
    config_path = Path(p['config']).expanduser()
    exp = load_yaml(config_path)
    sweep, exp_name = exp.get('sweep') or [{}], exp.get('exp_name', config_path.stem)
    runs_root = Path(p['runs_root']).expanduser()
    try:
        summary, margin = [], 0.05
        for cell_idx, cell in enumerate(sweep):
            for rep in range(int(exp.get('n_repeats', 1))):
                params = {**cell, 'cell_idx': cell_idx, 'rep': rep}
                print(f'\n[batch] cell {cell_idx} rep {rep}: {params}')
                runner = ExperimentRunner(config_path, params, runs_root, exp_name, headless=bool(p['headless']),
                                          settle_time=float(p['settle_time']), generate_figures=False)
                try:
                    run_dir = runner.run()
                except Exception as error:
                    print(f'[batch] ERROR in cell {cell_idx} rep {rep}: {error}; continuing')
                    continue
                margin = json.loads((run_dir/'cbf_config.json').read_text()).get('margin', margin)
                summary.append({'run_id': runner.run_id, 'exp_name': exp_name,
                                **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in params.items()},
                                **kpis_from_run_dir(run_dir)})
        if summary:
            path = runs_root/exp_name/'summary.csv'
            keys = list(dict.fromkeys(k for row in summary for k in row))
            with open(path, 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(summary)
            print(f'\n[batch] Summary written to {path}')
            if p['generate_figures']:
                generate_figures_for_experiment(exp_name, runs_root/exp_name, margin)
        print(f'[batch] {len(summary)} run(s) complete.')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
