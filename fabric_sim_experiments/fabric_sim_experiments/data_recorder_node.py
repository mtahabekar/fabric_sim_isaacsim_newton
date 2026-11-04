#!/usr/bin/env python3
"""Record corners, sensitivity-CBF diagnostics, simulator timing and commands to CSV.

Files in <run_dir>: corners.csv, cbf_debug.csv, sim_timing.csv, cmd_vels.csv.
"""
from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, TwistStamped
from std_msgs.msg import Float64MultiArray, String

from fabric_sim_experiments.corner_naming import CORNER_KEYS, corner_columns

FOLLOWERS = ('follower_left', 'follower_right')
# Column -> key in the JSON published on /fabric/capsule_cbf/debug.
DEBUG_SCALARS = ('frame', 'sim_time', 'status', 'mode', 'warmup', 'min_clearance', 'h_min', 'candidate_count',
                 'total_rows', 'correction', 'slack_soft', 'slack_hard', 'max_violation', 'controller_ms')
DEBUG_COLUMNS = (['t', *DEBUG_SCALARS, 'max_tension', *(f'tension_{c}' for c in CORNER_KEYS), 'max_strain_ks']
                 + [f'{kind}_v{axis}_{f}' for kind in ('nominal', 'safe') for f in FOLLOWERS for axis in 'xyz'])
CMD_TOPICS = {'/fabric/leader_left/cmd_vel': 'leader_left_cmd_vel',
              '/fabric/leader_right/cmd_vel': 'leader_right_cmd_vel',
              '/fabric/follower_left/applied_cmd_vel': 'follower_left_applied_cmd_vel',
              '/fabric/follower_right/applied_cmd_vel': 'follower_right_applied_cmd_vel'}


def debug_row(t: float, report: dict) -> dict:
    """Flatten one CBF debug report into a cbf_debug.csv row."""
    row = {'t': t, **{key: report.get(key) for key in DEBUG_SCALARS}}
    tension = report.get('tension') or []
    for corner, value in zip(CORNER_KEYS, tension):
        row[f'tension_{corner}'] = value
    row['max_tension'] = max(tension) if tension else None
    strain = report.get('strain_ks') or []
    row['max_strain_ks'] = max(strain) if strain else None
    for kind, key in (('nominal', 'nominal_velocity'), ('safe', 'follower_velocity')):
        for follower, velocity in zip(FOLLOWERS, report.get(key) or []):
            for axis, value in zip('xyz', velocity):
                row[f'{kind}_v{axis}_{follower}'] = value
    return row


class _CsvWriter:
    """Thread-safe CSV writer, flushed periodically."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self._lock = threading.Lock()
        self._fh = open(path, 'w', newline='', buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction='ignore')
        self._writer.writeheader()

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


class DataRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__('data_recorder_node')
        self.declare_parameter('run_dir', 'runs/default')
        self.declare_parameter('flush_period', 1.0)
        run_dir = Path(self.get_parameter('run_dir').value).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._writers = {
            'corners': _CsvWriter(run_dir/'corners.csv', corner_columns()),
            'cbf_debug': _CsvWriter(run_dir/'cbf_debug.csv', DEBUG_COLUMNS),
            'sim_timing': _CsvWriter(run_dir/'sim_timing.csv', ['t', 'sim_time', 'frame_ms', 'fps', 'cbf_ms']),
            'cmd_vels': _CsvWriter(run_dir/'cmd_vels.csv', ['t', 'role', 'topic', 'vx', 'vy', 'vz']),
        }
        group = MutuallyExclusiveCallbackGroup
        self.create_subscription(PoseArray, '/fabric/sim/corners', self._corners_cb, 50, callback_group=group())
        self.create_subscription(String, '/fabric/capsule_cbf/debug', self._debug_cb, 50, callback_group=group())
        self.create_subscription(Float64MultiArray, '/fabric/sim/timing', self._timing_cb, 50,
                                 callback_group=group())
        for topic in CMD_TOPICS:
            self.create_subscription(TwistStamped, topic, lambda msg, t=topic: self._cmd_cb(msg, t), 50,
                                     callback_group=group())
        self.create_timer(float(self.get_parameter('flush_period').value), self._flush_all)
        self.get_logger().info(f'DataRecorder writing to {run_dir}')

    @staticmethod
    def _t() -> float:
        return time.monotonic()

    def _corners_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) < 4:
            return
        row = {'t': self._t()}
        for pose, corner in zip(msg.poses, CORNER_KEYS):
            row.update({f'{corner}_x': pose.position.x, f'{corner}_y': pose.position.y,
                        f'{corner}_z': pose.position.z})
        self._writers['corners'].write(row)

    def _debug_cb(self, msg: String) -> None:
        try:
            report = json.loads(msg.data)
        except ValueError:
            self.get_logger().warning('Malformed CBF debug message skipped')
            return
        self._writers['cbf_debug'].write(debug_row(self._t(), report))

    def _timing_cb(self, msg: Float64MultiArray) -> None:
        data = list(msg.data)+[0.0]*4
        self._writers['sim_timing'].write(dict(t=self._t(), sim_time=data[0], frame_ms=data[1], fps=data[2],
                                               cbf_ms=data[3]))

    def _cmd_cb(self, msg: TwistStamped, topic: str) -> None:
        v = msg.twist.linear
        self._writers['cmd_vels'].write(dict(t=self._t(), role=CMD_TOPICS[topic], topic=topic,
                                             vx=v.x, vy=v.y, vz=v.z))

    def _flush_all(self) -> None:
        for writer in self._writers.values():
            writer.flush()

    def destroy_node(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataRecorderNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
