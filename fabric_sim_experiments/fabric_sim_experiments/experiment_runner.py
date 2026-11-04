#!/usr/bin/env python3
"""Run one experiment cell end to end: Isaac Sim (+ sensitivity CBF) -> recorder -> leader driver.

    ros2 run fabric_sim_experiments experiment_runner --ros-args \
        -p config:=<exp.yaml> [-p run_index:=1] [-p run_all_cells:=true] [-p headless:=true]

Each experiment YAML names a `base_config` (the bringup full_sim.yaml) whose
simulator arguments and CBF settings it overrides, first for the whole
experiment (`isaac_args`, `cbf_overrides`), then per sweep cell (`isaac_args`,
`cbf_params`). Outputs go to <runs_root>/<exp_name>/<run_id>/; the merged CBF
JSON and the exact simulator command are stored there for reproduction.
batch_runner runs the full sweep.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import rclpy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String

from fabric_sim_bringup.sim_command import (SIM_ENV, cbf_config_path, find_repo_root, isaac_command,
                                            load_yaml, resolve)

DEBUG_TOPIC = '/fabric/capsule_cbf/debug'
CORNERS_TOPIC = '/fabric/sim/corners'


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return f'{stamp}_{hashlib.sha1(f"{stamp}{time.time_ns()}".encode()).hexdigest()[:6]}'


def _popen(cmd: list[str], *, cwd: str | None = None, env: dict | None = None) -> subprocess.Popen:
    """Start a child in its own process group so cleanup reaches its descendants."""
    return subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)


def _kill(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, timeout), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _ros2_run(package: str, executable: str, params: dict[str, Any]) -> list[str]:
    cmd = ['ros2', 'run', package, executable, '--ros-args']
    for key, value in params.items():
        cmd += ['-p', f'{key}:={json.dumps(value) if isinstance(value, (list, bool)) else value}']
    return cmd


def experiment_settings(config_path: Path) -> tuple[dict, Path, dict]:
    """Experiment YAML, repository root and its merged base (bringup) config."""
    exp = load_yaml(config_path)
    root = find_repo_root(config_path)
    base = load_yaml(resolve(exp.get('base_config', 'fabric_sim_bringup/config/full_sim.yaml'), root))
    return exp, root, base


def cbf_margin(cbf_json: Path) -> float:
    return float(json.loads(cbf_json.read_text()).get('margin', 0.05))


def generate_figures_for_experiment(exp_name: str, runs_dir: Path, margin: float) -> None:
    """Regenerate the figures of an experiment from its CSVs."""
    plotters = {'exp1': ('plot_trajectories_3d', ['--safety-margin', str(margin)]),
                'exp2': ('plot_resolution_timing', []),
                'exp3': ('plot_leader_speed_sweep', ['--safety-margin', str(margin)])}
    entry = plotters.get(exp_name[:4])
    if entry is None:
        print(f'[runner] No automatic plotter for {exp_name}.')
        return
    module, extra = entry
    cmd = [sys.executable, '-m', f'fabric_sim_experiments.plotting.{module}', str(runs_dir), *extra]
    print(f'[runner] Generating figures for {exp_name} ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(result.stderr.rstrip(), file=sys.stderr)
        print(f'[runner] WARNING: figure generation failed ({result.returncode})', file=sys.stderr)


class _Waiter:
    """Small rclpy helper: wait for messages and publisher counts."""

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('experiment_runner_waiter')

    def message(self, msg_type: type, topic: str, accept: Callable[[Any], bool], timeout: float,
                alive: Callable[[], None]) -> bool:
        found = []
        sub = self.node.create_subscription(msg_type, topic, lambda m: accept(m) and found.append(m), 10)
        deadline = time.monotonic()+timeout
        try:
            while not found and time.monotonic() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.2)
                alive()
        finally:
            self.node.destroy_subscription(sub)
        return bool(found)

    def no_publishers(self, topic: str, timeout: float) -> bool:
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self.node.count_publishers(topic) == 0:
                return True
            rclpy.spin_once(self.node, timeout_sec=0.5)
        return False

    def close(self) -> None:
        self.node.destroy_node()


def _active(msg: String) -> bool:
    """A CBF debug message published after the warm-up frames."""
    try:
        return json.loads(msg.data).get('warmup') is False
    except ValueError:
        return False


class ExperimentRunner:
    def __init__(self, config_path: Path, run_params: dict, runs_root: Path, exp_name: str,
                 headless: bool = False, settle_time: float = 3.0, isaac_ready_timeout: float = 300.0,
                 generate_figures: bool = True) -> None:
        self.config_path = config_path
        self.exp, self.root, self.base = experiment_settings(config_path)
        self.run_params = run_params
        self.runs_root, self.exp_name = runs_root, exp_name
        self.headless, self.settle_time = headless, settle_time
        self.isaac_ready_timeout = isaac_ready_timeout
        self.generate_figures = generate_figures
        self.run_id = _run_id()
        self.run_dir = (runs_root/exp_name/self.run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._procs: list[subprocess.Popen] = []

    def _alive(self, ignore: set[int] = frozenset()) -> None:
        for proc in self._procs:
            if proc.pid not in ignore and proc.poll() is not None:
                raise RuntimeError(f'child exited with code {proc.returncode}: {" ".join(map(str, proc.args))}')

    def _commands(self) -> tuple[list[str], str, Path, dict]:
        arguments = {**(self.exp.get('isaac_args') or {}), **self.run_params.get('isaac_args', {})}
        if self.headless:
            arguments['headless'] = True
        cbf_overrides = {**(self.exp.get('cbf_overrides') or {}), **self.run_params.get('cbf_params', {})}
        cbf_json = cbf_config_path(self.base, self.root, self.run_dir, overrides=cbf_overrides)
        if cbf_json.parent != self.run_dir:  # keep a copy of the settings with every run
            (self.run_dir/'cbf_config.json').write_text(cbf_json.read_text())
            cbf_json = self.run_dir/'cbf_config.json'
        command, cwd = isaac_command(self.base, self.root, cbf_json, argument_overrides=arguments)
        velocity = [float(v) for v in self.run_params.get('leader_velocity', (0.0, 0.0, 0.0))]
        leader = dict(linear_speed_x=velocity[0], linear_speed_y=velocity[1], linear_speed_z=velocity[2],
                      motion_distance=float(self.run_params.get('leader_motion_distance', 0.4)),
                      start_delay=float(self.run_params.get('leader_start_delay', 0.0)))
        return command, cwd, cbf_json, leader

    def run(self) -> Path:
        command, cwd, cbf_json, leader = self._commands()
        (self.run_dir/'meta.json').write_text(json.dumps(dict(
            run_id=self.run_id, exp_name=self.exp_name, run_params=self.run_params,
            config_file=str(self.config_path), headless=self.headless, settle_time=self.settle_time,
            isaac_command=command, leader_driver=leader, created_utc=datetime.now(timezone.utc).isoformat()),
            indent=2))
        banner = self.run_params.get('banner')
        print(f"\n{'='*70}\n  {self.exp_name} | run {self.run_id}\n  params: {self.run_params}\n"
              f"  output: {self.run_dir}\n{'='*70}")
        if banner:
            print(f'>>> {banner}: take screenshots now (t = 0 %)')

        waiter = _Waiter()
        try:
            if not waiter.no_publishers(CORNERS_TOPIC, 10.0):
                raise RuntimeError(f'{CORNERS_TOPIC} is already published: stop the running simulator first')
            print('[runner] Starting Isaac Sim ...')
            self._procs.append(_popen(command, cwd=cwd, env={**os.environ, **SIM_ENV}))
            if not waiter.message(PoseArray, CORNERS_TOPIC, lambda m: True, self.isaac_ready_timeout, self._alive):
                raise RuntimeError(f'Isaac Sim did not publish {CORNERS_TOPIC} in time')
            print('[runner] Waiting for the CBF warm-up to finish ...')
            if not waiter.message(String, DEBUG_TOPIC, _active, 120.0, self._alive):
                raise RuntimeError(f'{DEBUG_TOPIC} did not report the end of warm-up in time')

            print('[runner] Starting data recorder ...')
            self._procs.append(_popen(_ros2_run('fabric_sim_experiments', 'data_recorder',
                                                {'run_dir': str(self.run_dir)})))
            time.sleep(2.0)
            self._alive()
            print('[runner] Starting leader driver ...')
            driver = _popen(_ros2_run('fabric_sim_experiments', 'leader_driver', leader))
            self._procs.append(driver)
            while driver.poll() is None:
                self._alive(ignore={driver.pid})
                time.sleep(0.5)
            if driver.returncode != 0:
                raise RuntimeError(f'leader_driver exited with code {driver.returncode}')
            print(f'[runner] Leader motion finished; recording {self.settle_time} s more ...')
            time.sleep(self.settle_time)
        finally:
            if banner:
                print(f'>>> {banner}: done (run {self.run_id})')
            print('[runner] Shutting down ...')
            for proc in reversed(self._procs):
                _kill(proc)
            if not waiter.no_publishers(CORNERS_TOPIC, 60.0):
                print(f'[runner] WARNING: {CORNERS_TOPIC} still has publishers')
            waiter.close()

        self._check_outputs()
        if self.generate_figures:
            generate_figures_for_experiment(self.exp_name, self.runs_root/self.exp_name, cbf_margin(cbf_json))
        print(f'[runner] Done: {self.run_dir}')
        return self.run_dir

    def _check_outputs(self) -> None:
        for name in ('corners.csv', 'cbf_debug.csv', 'sim_timing.csv', 'cmd_vels.csv', 'meta.json',
                     'cbf_config.json'):
            path = self.run_dir/name
            size = path.stat().st_size if path.exists() else 0
            print(f"[runner] {'OK' if size else 'WARNING: missing or empty'}: {path} ({size} bytes)")


def select_cells(sweep: list[dict], run_index: int, run_all: bool) -> list[dict]:
    if run_all:
        return sweep
    if not 0 <= run_index < len(sweep):
        raise SystemExit(f'run_index={run_index} outside 0..{len(sweep)-1}')
    return [sweep[run_index]]


def read_parameters(node_name: str, defaults: dict[str, Any], args=None) -> dict[str, Any]:
    rclpy.init(args=args)
    node = rclpy.create_node(node_name)
    values = {k: node.declare_parameter(k, v).value for k, v in defaults.items()}
    node.destroy_node()
    return values


def main(args=None) -> None:
    p = read_parameters('experiment_runner', dict(config='', run_index=0, run_all_cells=False, runs_root='runs',
                                                  headless=False, settle_time=3.0, generate_figures=True), args)
    if not p['config']:
        raise SystemExit("The 'config' parameter is required")
    config_path = Path(p['config']).expanduser()
    exp = load_yaml(config_path)
    sweep = exp.get('sweep') or [{}]
    exp_name = exp.get('exp_name', config_path.stem)
    cells = select_cells(sweep, int(p['run_index']), bool(p['run_all_cells']))
    if len(cells) < len(sweep):
        print(f'[runner] Running sweep cell {p["run_index"]} of {len(sweep)}; use batch_runner for all cells.')
    try:
        for cell in cells:
            ExperimentRunner(config_path, dict(cell), Path(p['runs_root']).expanduser(), exp_name,
                             headless=bool(p['headless']), settle_time=float(p['settle_time']),
                             generate_figures=bool(p['generate_figures'])).run()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
