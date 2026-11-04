"""Shared I/O for the experiment plotters (CSV only; no ROS or simulator)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 300


def read_meta(run_dir: Path) -> dict:
    path = run_dir/'meta.json'
    return json.loads(path.read_text()) if path.exists() else {}


def runs(runs_dir: Path, *required: str) -> Iterator[tuple[Path, dict]]:
    """Complete run directories (meta.json plus `required` files), oldest first."""
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if (run_dir/'meta.json').exists() and all((run_dir/name).exists() for name in required):
            yield run_dir, read_meta(run_dir)


def columns(path: Path, *keys: str, where=None) -> list[np.ndarray]:
    """Float columns of a CSV; rows failing `where(row)` or holding non-numbers are skipped."""
    out: list[list[float]] = [[] for _ in keys]
    with open(path) as stream:
        for row in csv.DictReader(stream):
            if where is not None and not where(row):
                continue
            try:
                values = [float(row[k]) for k in keys]
            except (KeyError, TypeError, ValueError):
                continue
            for column, value in zip(out, values):
                column.append(value)
    return [np.asarray(c) for c in out]


def clearance_trace(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(time from the first recorded active frame [s], min capsule clearance [m])."""
    t, d = columns(run_dir/'cbf_debug.csv', 'sim_time', 'min_clearance',
                   where=lambda r: r.get('warmup') in ('False', 'false', '0'))
    return (t-t[0], d) if len(t) else (t, d)


def save(fig: plt.Figure, base: Path, report_dir: Path | None = None) -> None:
    """Write png/pdf/svg; optionally symlink them into a shared report directory."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        path = base.with_suffix('.'+ext)
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        if report_dir is not None:
            report_dir.mkdir(parents=True, exist_ok=True)
            link = report_dir/path.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(path.resolve())
    plt.close(fig)


def margin_lines(ax, margin: float) -> None:
    ax.axhline(margin, color='black', ls=':', lw=1.2, label=f'safety margin ({margin:g} m)')
    ax.axhline(0.0, color='gray', ls='-', lw=0.7, alpha=0.5)
