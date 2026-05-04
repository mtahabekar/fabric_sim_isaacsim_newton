#!/usr/bin/env python3
"""
Orchestrates a single experiment run end-to-end:
  IsaacSim → obstacle_distance → cbf_controller → data_recorder → leader_driver

Run via:
  ros2 run fabric_sim_experiments experiment_runner \
      --ros-args -p config:=<path/to/exp.yaml>

By default this runs one sweep cell (index 0). Use batch_runner for the full
sweep, or pass -p run_all_cells:=true when you intentionally want that here.
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
from typing import Any

import yaml

import rclpy
from rclpy.node import Node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(ts.encode()).hexdigest()[:6]
    return f"{ts}_{h}"


def _topic_publisher_count(topic: str) -> int:
    """Return the number of active publishers on a topic (0 if unknown)."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", topic],
            capture_output=True,
            timeout=5.0,
        )
        for line in result.stdout.decode().splitlines():
            if "Publisher count:" in line:
                return int(line.split(":")[-1].strip())
    except Exception:
        pass
    return 0


def _wait_for_topic(topic: str, timeout: float = 120.0) -> bool:
    """Block until at least one publisher is live on the topic."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _topic_publisher_count(topic) > 0:
            # Also confirm a message actually arrives (publisher may be mid-init)
            try:
                result = subprocess.run(
                    ["ros2", "topic", "echo", "--once", "--no-daemon", topic],
                    capture_output=True,
                    timeout=8.0,
                )
                if result.returncode == 0:
                    return True
            except subprocess.TimeoutExpired:
                pass
        time.sleep(1.0)
    return False


def _wait_for_topic_message(
    topic: str,
    timeout: float = 120.0,
    required_text: list[str] | None = None,
) -> bool:
    """Block until a topic publishes a message, optionally matching text.

    This intentionally shells out to the ROS 2 CLI so the orchestrator does not
    need to import every message type that it only uses for readiness checks.
    """
    required_text = required_text or []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _topic_publisher_count(topic) > 0:
            try:
                result = subprocess.run(
                    ["ros2", "topic", "echo", "--once", "--no-daemon", topic],
                    capture_output=True,
                    timeout=min(8.0, max(1.0, deadline - time.monotonic())),
                )
            except subprocess.TimeoutExpired:
                result = None

            if result is not None and result.returncode == 0:
                text = result.stdout.decode(errors="replace")
                if all(needle in text for needle in required_text):
                    return True
        time.sleep(1.0)
    return False


def _wait_for_topic_gone(topic: str, timeout: float = 60.0) -> bool:
    """Block until the topic has zero publishers (previous run fully dead)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _topic_publisher_count(topic) == 0:
            return True
        time.sleep(2.0)
    print(f"[runner] WARNING: {topic} still has publishers after {timeout} s")
    return False


def _popen(cmd: list[str], *, cwd: str | None = None) -> subprocess.Popen:
    """Start a child in its own process group so cleanup reaches descendants."""
    return subprocess.Popen(cmd, cwd=cwd, start_new_session=True)


def _kill(proc: subprocess.Popen, timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            proc.kill()
        proc.wait(timeout=5.0)


def _bool_flag(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _leader_stop_params(cfg: dict, run_params: dict) -> dict:
    """Build leader-driver params for fixed-distance motion."""
    if "leader_speed_y" in run_params:
        axis = "y"
    elif "leader_speed_z" in run_params:
        axis = "z"
    else:
        axis = str(run_params.get("leader_motion_axis", "x")).strip().lower()

    speed = float(run_params.get("leader_speed", run_params.get(f"leader_speed_{axis}", 0.05)))
    params = {
        "linear_speed_x": 0.0,
        "linear_speed_y": 0.0,
        "linear_speed_z": 0.0,
        "motion_distance": run_params.get("leader_motion_distance", 1.4),
        "start_delay": run_params.get("leader_start_delay", 0.0),
    }
    if axis in ("x", "y", "z"):
        params[f"linear_speed_{axis}"] = speed
    else:
        params["linear_speed_x"] = speed
    return params


def _build_isaac_cmd(
    isaac_cfg: dict,
    run_params: dict,
    headless: bool = False,
) -> list[str]:
    python_sh = str(isaac_cfg["python_sh"])
    script = str(isaac_cfg["script"])
    base_args: dict = dict(isaac_cfg.get("arguments", {}) or {})
    base_args.update(run_params.get("isaac_args", {}))

    BOOLEAN_FLAGS = {
        "disable_cuda_graph",
        "external_cbf_enabled",
    }

    cmd = [python_sh, script]
    if headless:
        cmd += ["--headless"]

    for key, value in base_args.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if key in BOOLEAN_FLAGS:
            if _bool_flag(value):
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    for extra in (isaac_cfg.get("extra_args") or []):
        cmd.append(str(extra))

    return cmd


def _ros2_run(package: str, executable: str, params: dict) -> list[str]:
    cmd = ["ros2", "run", package, executable, "--ros-args"]
    for k, v in params.items():
        cmd += ["-p", f"{k}:={v}"]
    return cmd


def generate_figures_for_experiment(
    exp_name: str,
    runs_dir: Path,
    cfg: dict,
) -> None:
    """Generate report figures from CSVs for experiments with plotters."""
    obstacle_params = dict((cfg.get("obstacle_distance") or {}).get("parameters", {}) or {})
    safety_margin = str(obstacle_params.get("safety_margin", 0.05))
    table_hx = str(obstacle_params.get("table_hx", 0.22))
    table_hy = str(obstacle_params.get("table_hy", 0.22))
    table_hz = str(obstacle_params.get("table_hz", 0.16))

    if exp_name.startswith("exp1"):
        cmd = [
            sys.executable,
            "-m",
            "fabric_sim_experiments.plotting.plot_trajectories_3d",
            str(runs_dir),
            "--safety-margin",
            safety_margin,
            "--table-hx",
            table_hx,
            "--table-hy",
            table_hy,
            "--table-hz",
            table_hz,
        ]
    elif exp_name.startswith("exp2"):
        cmd = [
            sys.executable,
            "-m",
            "fabric_sim_experiments.plotting.plot_resolution_timing",
            str(runs_dir),
        ]
    elif exp_name.startswith("exp3"):
        cmd = [
            sys.executable,
            "-m",
            "fabric_sim_experiments.plotting.plot_leader_speed_sweep",
            str(runs_dir),
            "--safety-margin",
            safety_margin,
        ]
    else:
        print(f"[runner] No automatic plotter configured for {exp_name}.")
        return

    print(f"[runner] Generating figures for {exp_name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        print(
            f"[runner] WARNING: figure generation failed with return code {result.returncode}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    def __init__(
        self,
        config_path: Path,
        run_params: dict,
        runs_root: Path,
        exp_name: str,
        headless: bool = False,
        settle_time: float = 5.0,
        isaac_ready_timeout: float = 120.0,
        cbf_ready_timeout: float = 30.0,
        generate_figures: bool = True,
    ) -> None:
        self.config = _load_yaml(config_path)
        self.run_params = run_params
        self.runs_root = runs_root
        self.exp_name = exp_name
        self.headless = headless
        self.settle_time = settle_time
        self.isaac_ready_timeout = isaac_ready_timeout
        self.cbf_ready_timeout = cbf_ready_timeout
        self.config_path = config_path
        self.generate_figures = generate_figures

        self.run_id = _run_id()
        self.run_dir = runs_root / exp_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._procs: list[subprocess.Popen] = []

    # ------------------------------------------------------------------
    def _check_child_processes(
        self,
        context: str,
        ignore_pids: set[int] | None = None,
    ) -> None:
        ignore_pids = ignore_pids or set()
        for proc in self._procs:
            if proc.pid in ignore_pids:
                continue
            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"child process exited during {context} with return code {rc}: "
                    f"{' '.join(str(x) for x in proc.args)}"
                )

    def _write_meta(self) -> None:
        meta = {
            "run_id": self.run_id,
            "exp_name": self.exp_name,
            "run_params": self.run_params,
            "config_file": str(self.config_path),
            "headless": self.headless,
            "settle_time": self.settle_time,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def run(self) -> Path:
        cfg = self.config
        isaac_cfg = cfg.get("isaac_sim", {})
        cbf_params: dict = dict((cfg.get("cbf_controller") or {}).get("parameters", {}) or {})
        obstacle_params: dict = dict((cfg.get("obstacle_distance") or {}).get("parameters", {}) or {})

        # Apply run-specific overrides
        cbf_params.update(self.run_params.get("cbf_params", {}))
        obstacle_params.update(self.run_params.get("obstacle_params", {}))

        leader_params = _leader_stop_params(cfg, self.run_params)

        recorder_params = {
            "run_dir": str(self.run_dir),
        }
        self._write_meta()

        # Banner for qualitative experiments (Exp 4 / 5) — one per param value
        mu_tag = self.run_params.get("mu_tag")
        stiffness_tag = self.run_params.get("stiffness_tag")
        if mu_tag is not None:
            print(f"\n{'#'*70}")
            print(f"### EXP4 FRICTION SWEEP  contact_mu = {mu_tag}  ###")
            print(f"### Run ID: {self.run_id}")
            print(f"{'#'*70}\n")
            print(">>> TAKE SCREENSHOTS NOW at t=0%  (before starting motion)")
        elif stiffness_tag is not None:
            print(f"\n{'#'*70}")
            print(f"### EXP5 STIFFNESS SWEEP  preset = {stiffness_tag}  ###")
            print(f"### Run ID: {self.run_id}")
            print(f"{'#'*70}\n")
            print(">>> TAKE SCREENSHOTS NOW at t=0%  (before starting motion)")
        else:
            print(f"\n{'='*60}")
            print(f"  Run: {self.run_id}  |  exp: {self.exp_name}")
            print(f"  Params: {self.run_params}")
            print(f"  Output: {self.run_dir}")
            print(f"{'='*60}\n")

        isaac_cmd = _build_isaac_cmd(isaac_cfg, self.run_params, self.headless)
        cwd = str(isaac_cfg.get("working_directory", "."))

        # Make stale/orphaned simulators visible before launching another one.
        print("[runner] Checking for stale /fabric/sim/corners publishers ...")
        if not _wait_for_topic_gone("/fabric/sim/corners", timeout=10.0):
            raise RuntimeError(
                "stale /fabric/sim/corners publisher detected before this run. "
                "Stop any existing IsaacSim/fabric simulation before running experiments."
            )

        try:
            # 1. Start IsaacSim
            print("[runner] Starting IsaacSim...")
            self._procs.append(_popen(isaac_cmd, cwd=cwd))

            # 2. Wait for /fabric/sim/corners
            print("[runner] Waiting for /fabric/sim/corners ...")
            if not _wait_for_topic("/fabric/sim/corners", timeout=self.isaac_ready_timeout):
                self._check_child_processes("IsaacSim startup")
                raise RuntimeError("IsaacSim did not publish /fabric/sim/corners in time")
            self._check_child_processes("IsaacSim startup")
            print("[runner] IsaacSim ready.")

            # 3. Start obstacle_distance node
            print("[runner] Starting obstacle_distance node...")
            self._procs.append(_popen(
                _ros2_run("fabric_obstacle_distance", "obstacle_distance_node.py", obstacle_params)
            ))
            time.sleep(1.0)
            self._check_child_processes("obstacle_distance startup")
            print("[runner] Waiting for valid obstacle distance ...")
            if not _wait_for_topic_message(
                "/fabric/obstacle_distance",
                timeout=30.0,
                required_text=["valid: true"],
            ):
                self._check_child_processes("obstacle_distance startup")
                raise RuntimeError("obstacle_distance did not publish a valid message in time")
            self._check_child_processes("obstacle_distance startup")

            # 4. Start CBF controller
            print("[runner] Starting CBF controller...")
            self._procs.append(_popen(
                _ros2_run("fabric_cbf_controller", "fabric_cbf_controller_node.py", cbf_params)
            ))
            time.sleep(1.0)
            self._check_child_processes("CBF startup")

            # Wait for CBF to consume a valid distance message before the leader moves.
            print("[runner] Waiting for valid CBF debug message...")
            if not _wait_for_topic_message(
                "/fabric/cbf/debug",
                timeout=self.cbf_ready_timeout,
                required_text=["valid_distance: true"],
            ):
                self._check_child_processes("CBF startup")
                raise RuntimeError("CBF debug did not report valid_distance=true in time")
            self._check_child_processes("CBF startup")

            # 5. Start data recorder
            print("[runner] Starting data recorder...")
            self._procs.append(_popen(
                _ros2_run("fabric_sim_experiments", "data_recorder", recorder_params)
            ))
            time.sleep(1.0)
            self._check_child_processes("data recorder startup")

            # 6. Start leader driver and wait for it to finish
            print("[runner] Starting leader driver...")
            leader_cmd = _ros2_run("fabric_sim_experiments", "leader_driver", leader_params)
            leader_proc = _popen(leader_cmd)
            self._procs.append(leader_proc)
            while leader_proc.poll() is None:
                self._check_child_processes("leader motion", ignore_pids={leader_proc.pid})
                time.sleep(1.0)
            if leader_proc.returncode != 0:
                raise RuntimeError(
                    f"leader_driver exited with return code {leader_proc.returncode}: "
                    f"{' '.join(leader_cmd)}"
                )
            print(f"[runner] Leader driver finished. Settling for {self.settle_time} s ...")
            time.sleep(self.settle_time)

        finally:
            # End-of-run banner for qualitative experiments
            mu_tag = self.run_params.get("mu_tag")
            stiffness_tag = self.run_params.get("stiffness_tag")
            if mu_tag is not None:
                print(f"\n{'#'*70}")
                print(f"### EXP4 DONE  contact_mu = {mu_tag}  run_id = {self.run_id}")
                print(f"{'#'*70}\n")
            elif stiffness_tag is not None:
                print(f"\n{'#'*70}")
                print(f"### EXP5 DONE  preset = {stiffness_tag}  run_id = {self.run_id}")
                print(f"{'#'*70}\n")
            print("[runner] Shutting down all processes...")
            for p in reversed(self._procs):
                _kill(p)
            # Wait until IsaacSim has fully released its ROS topics before
            # returning, so the next run does not see stale publishers.
            print("[runner] Waiting for /fabric/sim/corners to disappear ...")
            _wait_for_topic_gone("/fabric/sim/corners", timeout=60.0)
            print("[runner] Cleanup complete.")

        # Validate outputs
        self._check_outputs()
        if self.generate_figures:
            generate_figures_for_experiment(
                self.exp_name,
                self.runs_root / self.exp_name,
                self.config,
            )
        print(f"[runner] Done. Output: {self.run_dir}")
        return self.run_dir

    def _check_outputs(self) -> None:
        expected = ["corners.csv", "obstacle_distance.csv", "cbf_debug.csv",
                    "sim_perf.csv", "cmd_vels.csv", "meta.json"]
        for fname in expected:
            fpath = self.run_dir / fname
            if not fpath.exists():
                print(f"[runner] WARNING: expected output not found: {fpath}")
            elif fpath.stat().st_size == 0:
                print(f"[runner] WARNING: output file is empty: {fpath}")
            else:
                print(f"[runner] OK: {fpath} ({fpath.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# ROS 2 node wrapper (for `ros2 run fabric_sim_experiments experiment_runner`)
# ---------------------------------------------------------------------------

class ExperimentRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("experiment_runner")
        self.declare_parameter("config", "")
        self.declare_parameter("run_only_first_cell", True)
        self.declare_parameter("run_all_cells", False)
        self.declare_parameter("run_index", 0)
        self.declare_parameter("runs_root", "runs")
        self.declare_parameter("headless", False)
        self.declare_parameter("settle_time", 5.0)
        self.declare_parameter("generate_figures", True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExperimentRunnerNode()

    config_path_str = node.get_parameter("config").value
    run_only_first = bool(node.get_parameter("run_only_first_cell").value)
    run_all_cells = bool(node.get_parameter("run_all_cells").value)
    run_index = int(node.get_parameter("run_index").value)
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

    if not sweep:
        sweep = [{}]

    if run_all_cells or not run_only_first:
        selected_runs = sweep
        print(f"[runner] Running all {len(selected_runs)} sweep cell(s).")
    else:
        if run_index < 0 or run_index >= len(sweep):
            print(
                f"ERROR: run_index={run_index} is outside the sweep range "
                f"0..{len(sweep) - 1}",
                file=sys.stderr,
            )
            sys.exit(1)
        selected_runs = [sweep[run_index]]
        print(
            f"[runner] Running one sweep cell by default: index {run_index} "
            f"of {len(sweep)}. Use batch_runner or -p run_all_cells:=true "
            "for the full sweep."
        )

    for run_params in selected_runs:
        runner = ExperimentRunner(
            config_path=config_path,
            run_params=dict(run_params),
            runs_root=runs_root,
            exp_name=exp_name,
            headless=headless,
            settle_time=settle_time,
            generate_figures=generate_figures,
        )
        runner.run()


if __name__ == "__main__":
    main()
