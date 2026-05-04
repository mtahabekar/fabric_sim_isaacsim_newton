#!/usr/bin/env python3
"""Records leader/follower poses, distances, CBF debug, sim perf, and cmd_vels to CSV."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import PoseArray, TwistStamped
from std_msgs.msg import Float32MultiArray

from fabric_obstacle_distance.msg import ObstacleDistance
from fabric_cbf_controller.msg import CbfDebug
from fabric_sim_experiments.corner_naming import CORNER_KEYS, corner_columns


class _CsvWriter:
    """Thread-safe buffered CSV writer with periodic flush."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
        self._writer.writeheader()
        self._path = path

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.flush()
            self._fh.close()


class DataRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("data_recorder_node")

        self.declare_parameter("run_dir", "runs/default")
        self.declare_parameter("meta", "{}")
        self.declare_parameter("flush_period", 1.0)

        run_dir = Path(self.get_parameter("run_dir").value).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            meta_str = self.get_parameter("meta").value
            try:
                meta = json.loads(meta_str)
            except Exception:
                meta = {}
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        self._writers: dict[str, _CsvWriter] = {
            "corners": _CsvWriter(
                run_dir / "corners.csv",
                corner_columns(),
            ),
            "obstacle_distance": _CsvWriter(
                run_dir / "obstacle_distance.csv",
                [
                    "t",
                    "valid",
                    "min_distance",
                    "h",
                    "safety_margin",
                    "compute_time_ms",
                    "closest_idx",
                    "normal_x",
                    "normal_y",
                    "normal_z",
                ],
            ),
            "cbf_debug": _CsvWriter(
                run_dir / "cbf_debug.csv",
                ["t", "enabled", "active", "valid_distance", "nominal_ready", "mode",
                 "solver_status", "h", "min_distance", "safety_margin", "alpha_h",
                 "lhs_nominal", "lhs_safe", "rhs", "compute_time_ms",
                 "stretch_active", "stretch_pair_names", "stretch_h_values",
                 "stretch_distances", "stretch_d_max_values",
                 "nominal_vx_follower_left", "nominal_vy_follower_left", "nominal_vz_follower_left",
                 "nominal_vx_follower_right", "nominal_vy_follower_right", "nominal_vz_follower_right",
                 "safe_vx_follower_left", "safe_vy_follower_left", "safe_vz_follower_left",
                 "safe_vx_follower_right", "safe_vy_follower_right", "safe_vz_follower_right"],
            ),
            "sim_perf": _CsvWriter(
                run_dir / "sim_perf.csv",
                ["t", "sim_step_ms", "sim_fps"],
            ),
            "cmd_vels": _CsvWriter(
                run_dir / "cmd_vels.csv",
                ["t", "role", "topic", "vx", "vy", "vz"],
            ),
        }

        # One callback group per subscription so callbacks never block each other.
        cg_corners = MutuallyExclusiveCallbackGroup()
        cg_dist = MutuallyExclusiveCallbackGroup()
        cg_cbf = MutuallyExclusiveCallbackGroup()
        cg_perf = MutuallyExclusiveCallbackGroup()
        cmd_topics = (
            "/fabric/leader_left/cmd_vel",
            "/fabric/leader_right/cmd_vel",
            "/fabric/follower_left/cbf_cmd_vel",
            "/fabric/follower_right/cbf_cmd_vel",
            "/fabric/leader_left/cbf_input_cmd_vel",
            "/fabric/leader_right/cbf_input_cmd_vel",
            "/fabric/follower_left/applied_cmd_vel",
            "/fabric/follower_right/applied_cmd_vel",
        )

        self.create_subscription(
            PoseArray, "/fabric/sim/corners", self._corners_cb, 50,
            callback_group=cg_corners,
        )
        self.create_subscription(
            ObstacleDistance, "/fabric/obstacle_distance", self._dist_cb, 50,
            callback_group=cg_dist,
        )
        self.create_subscription(
            CbfDebug, "/fabric/cbf/debug", self._cbf_debug_cb, 50,
            callback_group=cg_cbf,
        )
        self.create_subscription(
            Float32MultiArray, "/fabric/sim/perf", self._perf_cb, 50,
            callback_group=cg_perf,
        )
        for topic in cmd_topics:
            self.create_subscription(
                TwistStamped,
                topic,
                lambda msg, t=topic: self._cmd_vel_cb(msg, t),
                50,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        flush_period = float(self.get_parameter("flush_period").value)
        self.create_timer(flush_period, self._flush_all)

        self.get_logger().info(f"DataRecorder writing to {run_dir}")

    # ------------------------------------------------------------------
    def _t(self) -> float:
        return time.monotonic()

    def _corners_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) < 4:
            return
        p = msg.poses
        row = {"t": self._t()}
        for idx, corner in enumerate(CORNER_KEYS):
            row[f"{corner}_x"] = p[idx].position.x
            row[f"{corner}_y"] = p[idx].position.y
            row[f"{corner}_z"] = p[idx].position.z
        self._writers["corners"].write(row)

    def _dist_cb(self, msg: ObstacleDistance) -> None:
        row = {
            "t": self._t(),
            "valid": int(getattr(msg, "valid", True)),
            "min_distance": msg.min_distance,
            "h": msg.h,
            "safety_margin": getattr(msg, "safety_margin", 0.0),
            "compute_time_ms": getattr(msg, "compute_time_ms", 0.0),
            "closest_idx": msg.closest_particle_index,
            "normal_x": msg.normal.x,
            "normal_y": msg.normal.y,
            "normal_z": msg.normal.z,
        }
        self._writers["obstacle_distance"].write(row)

    def _cbf_debug_cb(self, msg: CbfDebug) -> None:
        nom = list(msg.nominal_velocity) + [0.0] * 6
        safe = list(msg.safe_velocity) + [0.0] * 6
        row = {
            "t": self._t(),
            "enabled": int(msg.enabled),
            "active": int(msg.active),
            "valid_distance": int(msg.valid_distance),
            "nominal_ready": int(msg.nominal_ready),
            "mode": msg.mode,
            "solver_status": msg.solver_status,
            "h": msg.h,
            "min_distance": msg.min_distance,
            "safety_margin": msg.safety_margin,
            "alpha_h": msg.alpha_h,
            "lhs_nominal": msg.lhs_nominal,
            "lhs_safe": msg.lhs_safe,
            "rhs": msg.rhs,
            "compute_time_ms": msg.compute_time_ms,
            "stretch_active": int(msg.stretch_active),
            "stretch_pair_names": "|".join(msg.stretch_pair_names),
            "stretch_h_values": "|".join(f"{v:.6f}" for v in msg.stretch_h_values),
            "stretch_distances": "|".join(f"{v:.6f}" for v in msg.stretch_distances),
            "stretch_d_max_values": "|".join(f"{v:.6f}" for v in msg.stretch_d_max_values),
            "nominal_vx_follower_left": nom[0],
            "nominal_vy_follower_left": nom[1],
            "nominal_vz_follower_left": nom[2],
            "nominal_vx_follower_right": nom[3],
            "nominal_vy_follower_right": nom[4],
            "nominal_vz_follower_right": nom[5],
            "safe_vx_follower_left": safe[0],
            "safe_vy_follower_left": safe[1],
            "safe_vz_follower_left": safe[2],
            "safe_vx_follower_right": safe[3],
            "safe_vy_follower_right": safe[4],
            "safe_vz_follower_right": safe[5],
        }
        self._writers["cbf_debug"].write(row)

    def _perf_cb(self, msg: Float32MultiArray) -> None:
        data = list(msg.data) + [0.0, 0.0]
        row = {"t": self._t(), "sim_step_ms": data[0], "sim_fps": data[1]}
        self._writers["sim_perf"].write(row)

    def _cmd_vel_cb(self, msg: TwistStamped, topic: str) -> None:
        role = {
            "/fabric/leader_left/cmd_vel": "leader_left_cmd_vel",
            "/fabric/leader_right/cmd_vel": "leader_right_cmd_vel",
            "/fabric/follower_left/cbf_cmd_vel": "follower_left_cbf_cmd_vel",
            "/fabric/follower_right/cbf_cmd_vel": "follower_right_cbf_cmd_vel",
            "/fabric/leader_left/cbf_input_cmd_vel": "leader_left_cbf_input_cmd_vel",
            "/fabric/leader_right/cbf_input_cmd_vel": "leader_right_cbf_input_cmd_vel",
            "/fabric/follower_left/applied_cmd_vel": "follower_left_applied_cmd_vel",
            "/fabric/follower_right/applied_cmd_vel": "follower_right_applied_cmd_vel",
        }.get(topic, topic)
        row = {
            "t": self._t(),
            "role": role,
            "topic": topic,
            "vx": msg.twist.linear.x,
            "vy": msg.twist.linear.y,
            "vz": msg.twist.linear.z,
        }
        self._writers["cmd_vels"].write(row)

    def _flush_all(self) -> None:
        for w in self._writers.values():
            w.flush()

    def destroy_node(self) -> None:
        for w in self._writers.values():
            try:
                w.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataRecorderNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
