"""Figures for SENSITIVITY_CBF_REPORT.md from benchmark/validation JSON.

Needs matplotlib (not required by the controller).
    python -m capsule_cbf.plot_report --results capsule_cbf/results --output capsule_cbf/figures
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reference categorical slots 1-3 (validated all-pairs), text and surface tokens.
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
FRAME_DT = 1/60


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def records(path):
    return json.loads(Path(path).read_text())['records']


def save(fig, out, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out/name, dpi=160)
    plt.close(fig)


def sensitivity_figure(results, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    runs = {'it10': json.loads((results/'validation/sens_affine_it10.json').read_text()),
            'it40': json.loads((results/'validation/sens_affine_it40.json').read_text())}
    for ax, state in zip(axes, ('settled', 'descending_0.1mps')):
        series = [('Legacy bilinear heuristic', runs['it10'], 'bilinear_heuristic', MUTED, '--'),
                  ('This model vs default plant (10/16 it.)', runs['it10'], 'coarse14_affine_stride2', ORANGE, '-'),
                  ('This model vs converged plant (40/64 it.)', runs['it40'], 'coarse14_affine_stride2', BLUE, '-')]
        for label, run, model, color, dash in series:
            report = next(r for r in run['reports'] if r['state'] == state)['models'][model]
            frames = report.get('output_frames', list(range(1, len(report['clearance_gradient_cosine_median'])+1)))
            ax.plot(frames, report['clearance_gradient_cosine_median'], dash, color=color, linewidth=2,
                    marker='o', markersize=4, label=label)
        style(ax, 'Prediction horizon (frames, 60 Hz)', 'Median cosine of ∂h/∂u' if ax is axes[0] else '')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_title('Settled cloth' if state == 'settled' else 'Leaders descending 0.1 m/s',
                     color=INK, fontsize=10, loc='left')
        ax.set_ylim(-.05, 1.05)
    axes[1].legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
    save(fig, out, 'sensitivity_direction.png')


def clearance_figure(results, out):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for label, path, color in (('No filter', results/'closed_loop/main/descend_nominal.json', MUTED),
                               ('Legacy ray-basin filter', results/'closed_loop/main_legacy/descend_ray_basins.json', ORANGE),
                               ('Sensitivity CBF', results/'closed_loop/main/descend_sensitivity.json', BLUE)):
        r = records(path)
        t = FRAME_DT*np.arange(len(r))
        ax.plot(t, [100*x['min_barrier'] for x in r], color=color, linewidth=2, label=label)
    ax.axhline(0, color=INK, linewidth=1, linestyle='--')
    ax.axhline(-5, color=MUTED, linewidth=1, linestyle=':')
    ax.text(.02, .5, 'margin boundary h = 0', color=INK, fontsize=8, va='bottom')
    ax.text(.02, -4.6, 'capsule surface', color=MUTED, fontsize=8, va='bottom')
    style(ax, 'Time (s); leaders descend 0.1 m/s for 3.33 s, then hold', 'Min. barrier h = d − margin (cm)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    save(fig, out, 'descend_clearance.png')


def tension_figure(results, out, limit):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for label, path, color in (('No filter', results/'closed_loop/main/pull_nominal.json', MUTED),
                               ('Sensitivity CBF', results/'closed_loop/main/pull_sensitivity.json', BLUE)):
        r = records(path)
        t = FRAME_DT*np.arange(len(r))
        ax.plot(t, [max(x['tension']) for x in r], color=color, linewidth=2, label=label)
    ax.axhline(limit, color=INK, linewidth=1, linestyle='--')
    ax.text(.02, limit*1.08, f'tension limit {limit:g} N', color=INK, fontsize=8)
    ax.set_yscale('log')
    style(ax, 'Time (s); task command pulls followers at 0.05 m/s', 'Max. gripper tension (N, log)')
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left', bbox_to_anchor=(0, .9))
    save(fig, out, 'pull_tension.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tension-limit', type=float, default=20.)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sensitivity_figure(args.results, args.output)
    clearance_figure(args.results, args.output)
    tension_figure(args.results, args.output, args.tension_limit)


if __name__ == '__main__':
    main()
