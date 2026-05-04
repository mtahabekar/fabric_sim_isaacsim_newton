#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ROS2 keyboard teleop node for fabric leader velocity commands.

Publishes constant velocity commands while keys are held; zeros when released.
Uses pynput for reliable key-press/release detection in a background thread.

Key mappings
------------
  Leader left  (topic: /fabric/leader_left/cmd_vel)
    w / s  =>  +x / -x
    d / a  =>  +y / -y
    e / q  =>  +z / -z

  Leader right  (topic: /fabric/leader_right/cmd_vel)
    i / k  =>  +x / -x
    l / j  =>  +y / -y
    o / u  =>  +z / -z

Both leaders can be controlled simultaneously. Opposing keys on the same axis cancel.
Use Ctrl+C to quit — do NOT use q as a quit key (q is the leader-left -z command).

Dependencies
------------
  pip install pynput

Run
---
  python3 fabric_corner_keyboard_teleop.py
  # with custom speed:
  ros2 run <pkg> fabric_corner_keyboard_teleop --ros-args -p linear_speed:=0.08
"""

import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import numpy as np

try:
    from pynput import keyboard as pynput_keyboard
except ImportError as exc:
    raise SystemExit(
        "pynput is required for keyboard teleop.  Install it with:  pip install pynput"
    ) from exc


# Maps each key character to (leader_slot, axis, sign).
#   leader_slot 0 = leader_left, 1 = leader_right
#   axis        0 = x,        1 = y,      2 = z
_KEY_MAP: dict[str, tuple[int, int, float]] = {
    # leader_left
    "w": (0, 0, +1.0),
    "s": (0, 0, -1.0),
    "d": (0, 1, +1.0),
    "a": (0, 1, -1.0),
    "e": (0, 2, +1.0),
    "q": (0, 2, -1.0),
    # leader_right
    "i": (1, 0, +1.0),
    "k": (1, 0, -1.0),
    "l": (1, 1, +1.0),
    "j": (1, 1, -1.0),
    "o": (1, 2, +1.0),
    "u": (1, 2, -1.0),
}


class FabricCornerTeleop(Node):
    def __init__(self) -> None:
        super().__init__("fabric_corner_keyboard_teleop")

        self.declare_parameter("linear_speed", 0.05)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "world")

        self._speed = float(self.get_parameter("linear_speed").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self._frame_id = str(self.get_parameter("frame_id").value)

        self._pub_leader_left = self.create_publisher(
            TwistStamped, "/fabric/leader_left/cmd_vel", 10
        )
        self._pub_leader_right = self.create_publisher(
            TwistStamped, "/fabric/leader_right/cmd_vel", 10
        )

        self._pressed: set[str] = set()
        self._lock = threading.Lock()

        self._timer = self.create_timer(1.0 / publish_rate, self._timer_cb)

        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

        self.get_logger().info(
            f"Keyboard teleop ready  speed={self._speed} m/s  rate={publish_rate} Hz\n"
            "  leader_left : w/s=±x  d/a=±y  e/q=±z\n"
            "  leader_right: i/k=±x  l/j=±y  o/u=±z\n"
            "  Ctrl+C to quit."
        )

    # ----- Key event callbacks (run in pynput listener thread) -------------------

    def _on_press(self, key) -> None:
        try:
            ch = key.char.lower()
        except AttributeError:
            return  # special key (Shift, Ctrl, …) — ignore
        if ch in _KEY_MAP:
            with self._lock:
                self._pressed.add(ch)

    def _on_release(self, key) -> None:
        try:
            ch = key.char.lower()
        except AttributeError:
            return
        with self._lock:
            self._pressed.discard(ch)

    # ----- Velocity computation --------------------------------------------------

    def _compute_corner_vels(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (leader_left_vel, leader_right_vel), each shape (3,)."""
        with self._lock:
            keys = frozenset(self._pressed)

        vels = np.zeros((2, 3), dtype=np.float32)
        for ch, (slot, axis, sign) in _KEY_MAP.items():
            if ch in keys:
                vels[slot, axis] += sign * self._speed

        return vels[0], vels[1]

    # ----- ROS publish -----------------------------------------------------------

    def _make_twist(self, vel: np.ndarray) -> TwistStamped:
        msg = TwistStamped()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vel[0])
        msg.twist.linear.y = float(vel[1])
        msg.twist.linear.z = float(vel[2])
        # angular fields remain zero — only linear velocity commands are used
        return msg

    def _timer_cb(self) -> None:
        leader_left_vel, leader_right_vel = self._compute_corner_vels()
        self._pub_leader_left.publish(self._make_twist(leader_left_vel))
        self._pub_leader_right.publish(self._make_twist(leader_right_vel))

    # ----- Cleanup ---------------------------------------------------------------

    def shutdown(self) -> None:
        self._listener.stop()


def main() -> None:
    rclpy.init()
    node = FabricCornerTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
