#!/usr/bin/env python3
"""Programmatic leader driver for automated experiment runs.

Publishes a constant leader velocity until the average leader position has
advanced by ``motion_distance`` metres along the commanded direction.
"""

from __future__ import annotations

import sys
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import PoseArray, TwistStamped


class LeaderDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("leader_driver_node")

        self.declare_parameter("linear_speed_x", 0.05)
        self.declare_parameter("linear_speed_y", 0.0)
        self.declare_parameter("linear_speed_z", 0.0)
        self.declare_parameter("motion_distance", 1.4)
        self.declare_parameter("start_delay", 0.0)
        self.declare_parameter("frame_id", "world")
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("subscriber_wait_timeout", 30.0)
        self.declare_parameter("corner_wait_timeout", 30.0)
        self.declare_parameter("min_initial_corner_samples", 3)

        self._speed_x = float(self.get_parameter("linear_speed_x").value)
        self._speed_y = float(self.get_parameter("linear_speed_y").value)
        self._speed_z = float(self.get_parameter("linear_speed_z").value)
        self._motion_distance = float(self.get_parameter("motion_distance").value)
        self._start_delay = float(self.get_parameter("start_delay").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._subscriber_wait_timeout = float(
            self.get_parameter("subscriber_wait_timeout").value
        )
        self._corner_wait_timeout = float(self.get_parameter("corner_wait_timeout").value)
        self._min_initial_corner_samples = max(
            1,
            int(self.get_parameter("min_initial_corner_samples").value),
        )

        self._initial_leader_position: tuple[float, float, float] | None = None
        self._current_leader_position: tuple[float, float, float] | None = None
        self._motion_unit = self._compute_motion_unit()
        self._corner_samples = 0
        self._started = False
        self._done = False
        self._failed = False
        self._start_time: float | None = None
        self._node_start_time = time.monotonic()
        self._logged_waiting_for_subscribers = False
        self._logged_subscribers_ready = False

        self._pub_leader_left = self.create_publisher(
            TwistStamped, "/fabric/leader_left/cmd_vel", 10
        )
        self._pub_leader_right = self.create_publisher(
            TwistStamped, "/fabric/leader_right/cmd_vel", 10
        )

        self._sub_corners = self.create_subscription(
            PoseArray, "/fabric/sim/corners", self._corners_cb, 50
        )

        rate = float(self.get_parameter("publish_rate").value)
        self._timer = self.create_timer(1.0 / rate, self._timer_cb)

        self.get_logger().info(
            f"LeaderDriver: speed=({self._speed_x}, {self._speed_y}, {self._speed_z}) m/s, "
            f"distance={self._motion_distance} m, start_delay={self._start_delay} s"
        )

    # ------------------------------------------------------------------
    def _corners_cb(self, msg: PoseArray) -> None:
        if len(msg.poses) < 4:
            return
        # PoseArray order: follower_left, follower_right, leader_left, leader_right.
        leader_left = msg.poses[2].position
        leader_right = msg.poses[3].position
        avg_position = (
            0.5 * (leader_left.x + leader_right.x),
            0.5 * (leader_left.y + leader_right.y),
            0.5 * (leader_left.z + leader_right.z),
        )
        self._current_leader_position = avg_position
        self._corner_samples += 1

        if (
            self._initial_leader_position is None
            and self._corner_samples >= self._min_initial_corner_samples
        ):
            self._initial_leader_position = avg_position
            self.get_logger().info(
                "LeaderDriver: initial leader position = "
                f"({avg_position[0]:.4f}, {avg_position[1]:.4f}, {avg_position[2]:.4f}) m"
            )

    def _timer_cb(self) -> None:
        if self._done:
            return

        now = time.monotonic()

        if not self._command_subscribers_ready():
            self._publish_vel(0.0, 0.0, 0.0)
            elapsed = now - self._node_start_time
            if not self._logged_waiting_for_subscribers:
                self.get_logger().info(
                    "LeaderDriver: waiting for IsaacSim subscribers on leader cmd_vel topics"
                )
                self._logged_waiting_for_subscribers = True
            if (
                self._subscriber_wait_timeout > 0.0
                and elapsed > self._subscriber_wait_timeout
            ):
                self.get_logger().error(
                    "LeaderDriver: timed out waiting for subscribers on "
                    "leader command topics (/fabric/leader_left/cmd_vel and "
                    "/fabric/leader_right/cmd_vel)"
                )
                self._failed = True
                self._done = True
            return

        if not self._logged_subscribers_ready:
            self.get_logger().info("LeaderDriver: IsaacSim command subscribers are ready")
            self._logged_subscribers_ready = True

        # Honour start_delay: wait until stable corners data has arrived and delay elapsed
        if self._initial_leader_position is None:
            self._publish_vel(0.0, 0.0, 0.0)
            elapsed = now - self._node_start_time
            if self._corner_wait_timeout > 0.0 and elapsed > self._corner_wait_timeout:
                self.get_logger().error(
                    "LeaderDriver: timed out waiting for stable /fabric/sim/corners samples"
                )
                self._failed = True
                self._done = True
            return

        if not self._started:
            if self._start_time is None:
                self._start_time = now
            if now - self._start_time < self._start_delay:
                self._publish_vel(0.0, 0.0, 0.0)
                return
            self._started = True
            self.get_logger().info("LeaderDriver: starting motion")

        if self._current_leader_position is not None:
            travelled = self._projected_travel_distance()
            if travelled >= self._motion_distance:
                self._publish_stop_burst()
                self._done = True
                self.get_logger().info(
                    f"LeaderDriver: motion complete (travelled {travelled:.4f} m). Stopping."
                )
                return

        self._publish_vel(self._speed_x, self._speed_y, self._speed_z)

    def _command_subscribers_ready(self) -> bool:
        return (
            self._pub_leader_left.get_subscription_count() > 0
            and self._pub_leader_right.get_subscription_count() > 0
        )

    def _compute_motion_unit(self) -> tuple[float, float, float]:
        norm = math.sqrt(self._speed_x**2 + self._speed_y**2 + self._speed_z**2)
        if norm <= 1.0e-12:
            return (1.0, 0.0, 0.0)
        return (self._speed_x / norm, self._speed_y / norm, self._speed_z / norm)

    def _projected_travel_distance(self) -> float:
        if self._initial_leader_position is None or self._current_leader_position is None:
            return 0.0
        delta = tuple(
            self._current_leader_position[i] - self._initial_leader_position[i]
            for i in range(3)
        )
        return sum(delta[i] * self._motion_unit[i] for i in range(3))

    def _publish_vel(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        self._pub_leader_left.publish(msg)
        self._pub_leader_right.publish(msg)

    def _publish_stop_burst(self) -> None:
        for _ in range(3):
            self._publish_vel(0.0, 0.0, 0.0)
            time.sleep(0.02)

    @property
    def motion_done(self) -> bool:
        return self._done

    @property
    def failed(self) -> bool:
        return self._failed


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LeaderDriverNode()
    exit_code = 0
    try:
        while rclpy.ok() and not node.motion_done:
            rclpy.spin_once(node, timeout_sec=0.05)
        if node.failed:
            exit_code = 1
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node._publish_stop_burst()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
