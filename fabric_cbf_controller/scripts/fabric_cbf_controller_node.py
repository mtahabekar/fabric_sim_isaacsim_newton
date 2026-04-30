#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

import rclpy
from fabric_cbf_controller.msg import CbfDebug
from fabric_obstacle_distance.msg import ObstacleDistance
from geometry_msgs.msg import Point, PoseArray, TwistStamped, Vector3
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


JACOBIAN_MODES = ("geometric_weighted", "normal_equal", "normal_left_right_x")
STALE_POLICIES = ("nominal", "zero", "hold-last")
INFEASIBLE_POLICIES = ("nominal", "zero", "hold-last", "project_bounds")
FORMATION_REFERENCE_MODES = ("initial",)


@dataclass(frozen=True)
class ProjectionResult:
    velocity: np.ndarray
    solver_status: str


class FabricCbfControllerNode(Node):
    """External CBF velocity filter for the two bottom fabric corners."""

    def __init__(self) -> None:
        super().__init__("fabric_cbf_controller_node")
        self._declare_parameters()

        self._top_left_input = np.zeros(3, dtype=np.float64)
        self._top_right_input = np.zeros(3, dtype=np.float64)
        self._top_left_input_time: float | None = None
        self._top_right_input_time: float | None = None

        self._distance_msg: ObstacleDistance | None = None
        self._distance_time: float | None = None
        self._corners_msg: PoseArray | None = None
        self._corners_time: float | None = None

        self._last_safe_velocity = np.zeros((2, 3), dtype=np.float64)
        self._last_debug_log_time = 0.0
        self._formation_reference_ready = False
        self._initial_bottom_left_offset = np.zeros(3, dtype=np.float64)
        self._initial_bottom_right_offset = np.zeros(3, dtype=np.float64)
        self._last_formation_error = np.zeros((2, 3), dtype=np.float64)
        self._last_recovery_velocity = np.zeros((2, 3), dtype=np.float64)
        self._warned_unsupported_formation_reference_mode = False

        self._create_ros_interfaces()

        control_rate_hz = self._float_param("control_rate_hz", 60.0, minimum=1.0e-3)
        self._timer = self.create_timer(1.0 / control_rate_hz, self._control_timer_callback)

        self.get_logger().info(
            "Fabric CBF controller ready: %s, %s -> %s, %s"
            % (
                self._str_param("top_left_cbf_input_topic", "/fabric/top_left/cbf_input_cmd_vel"),
                self._str_param("top_right_cbf_input_topic", "/fabric/top_right/cbf_input_cmd_vel"),
                self._str_param("bottom_left_cbf_topic", "/fabric/bottom_left/cbf_cmd_vel"),
                self._str_param("bottom_right_cbf_topic", "/fabric/bottom_right/cbf_cmd_vel"),
            )
        )

    def _declare_parameters(self) -> None:
        declarations = {
            "enabled": True,
            "control_rate_hz": 60.0,
            "frame_id": "world",
            "top_left_cbf_input_topic": "/fabric/top_left/cbf_input_cmd_vel",
            "top_right_cbf_input_topic": "/fabric/top_right/cbf_input_cmd_vel",
            "obstacle_distance_topic": "/fabric/obstacle_distance",
            "fabric_corners_topic": "/fabric/sim/corners",
            "bottom_left_cbf_topic": "/fabric/bottom_left/cbf_cmd_vel",
            "bottom_right_cbf_topic": "/fabric/bottom_right/cbf_cmd_vel",
            "debug_topic": "/fabric/cbf/debug",
            "input_timeout": 0.25,
            "distance_timeout": 0.25,
            "corners_timeout": 0.50,
            "activation_distance": 0.0,
            "alpha_gain_near": 1.0,
            "alpha_gain_far": 1.0,
            "alpha_switch_h": 0.05,
            "max_speed": 0.10,
            "output_filter_beta": 0.0,
            "weight_x": 1.0,
            "weight_y": 1.0,
            "weight_z": 1.0,
            "jacobian_mode": "geometric_weighted",
            "min_jacobian_norm": 1.0e-6,
            "cloth_half_width_x": 0.42,
            "left_right_weight_power": 1.0,
            "min_corner_weight": 0.0,
            "formation_recovery_enabled": True,
            "formation_recovery_gain": 0.5,
            "formation_recovery_max_speed": 0.03,
            "formation_recovery_deadband": 0.003,
            "formation_reference_mode": "initial",
            "stale_policy": "nominal",
            "infeasible_policy": "project_bounds",
            "publish_debug": True,
            "debug_log_enabled": True,
            "debug_log_period": 1.0,
        }

        for name, default_value in declarations.items():
            self.declare_parameter(name, default_value)

    def _create_ros_interfaces(self) -> None:
        self._top_left_input_sub = self.create_subscription(
            TwistStamped,
            self._str_param("top_left_cbf_input_topic", "/fabric/top_left/cbf_input_cmd_vel"),
            self._top_left_input_callback,
            10,
        )
        self._top_right_input_sub = self.create_subscription(
            TwistStamped,
            self._str_param("top_right_cbf_input_topic", "/fabric/top_right/cbf_input_cmd_vel"),
            self._top_right_input_callback,
            10,
        )
        self._distance_sub = self.create_subscription(
            ObstacleDistance,
            self._str_param("obstacle_distance_topic", "/fabric/obstacle_distance"),
            self._distance_callback,
            10,
        )
        self._corners_sub = self.create_subscription(
            PoseArray,
            self._str_param("fabric_corners_topic", "/fabric/sim/corners"),
            self._corners_callback,
            10,
        )

        self._bottom_left_cbf_pub = self.create_publisher(
            TwistStamped,
            self._str_param("bottom_left_cbf_topic", "/fabric/bottom_left/cbf_cmd_vel"),
            10,
        )
        self._bottom_right_cbf_pub = self.create_publisher(
            TwistStamped,
            self._str_param("bottom_right_cbf_topic", "/fabric/bottom_right/cbf_cmd_vel"),
            10,
        )
        self._debug_pub = self.create_publisher(
            CbfDebug,
            self._str_param("debug_topic", "/fabric/cbf/debug"),
            10,
        )

    def _top_left_input_callback(self, msg: TwistStamped) -> None:
        self._top_left_input = self._twist_linear_to_array(msg)
        self._top_left_input_time = time.monotonic()

    def _top_right_input_callback(self, msg: TwistStamped) -> None:
        self._top_right_input = self._twist_linear_to_array(msg)
        self._top_right_input_time = time.monotonic()

    def _distance_callback(self, msg: ObstacleDistance) -> None:
        self._distance_msg = msg
        self._distance_time = time.monotonic()

    def _corners_callback(self, msg: PoseArray) -> None:
        self._corners_msg = msg
        self._corners_time = time.monotonic()
        self._capture_formation_reference(msg)

    def _control_timer_callback(self) -> None:
        now_monotonic = time.monotonic()
        enabled = self._bool_param("enabled", True)
        input_ready = self._input_is_ready(now_monotonic)
        valid_distance = self._distance_is_valid(now_monotonic)
        nominal = self._current_nominal_velocity()

        distance_msg = self._distance_msg
        corners_msg = self._fresh_corners(now_monotonic)

        jacobian = np.zeros(6, dtype=np.float64)
        normal = np.zeros(3, dtype=np.float64)
        h = self._distance_float(distance_msg, "h", 0.0)
        min_distance = self._distance_float(distance_msg, "min_distance", 0.0)
        safety_margin = self._distance_float(distance_msg, "safety_margin", 0.0)
        alpha_h = 0.0
        rhs = 0.0
        active = False
        mode = "fallback_input"
        solver_status = "invalid_input"

        if valid_distance and distance_msg is not None:
            alpha_h = self.compute_alpha(h)
            rhs = -alpha_h
            jacobian, normal = self.compute_jacobian(distance_msg, corners_msg)

        if not enabled:
            if input_ready:
                safe = nominal
                solver_status = "nominal_feasible"
            else:
                safe = self._apply_stale_policy(nominal)
            mode = "disabled"
        elif not input_ready:
            safe = self._apply_stale_policy(nominal)
            mode = "stale_input"
        elif not valid_distance:
            safe = nominal
            mode = "invalid_distance"
        else:
            activation_distance = self._float_param("activation_distance", 0.0)
            if h > activation_distance:
                safe = nominal
                active = False
                mode = "inactive"
                solver_status = "nominal_feasible"
            else:
                active = True
                mode = "active"
                weights = self._control_weights()
                max_speed = self._float_param("max_speed", 0.10, minimum=0.0)
                min_jacobian_norm = self._float_param(
                    "min_jacobian_norm", 1.0e-6, minimum=0.0
                )
                result = self.solve_projection_qp(
                    nominal,
                    jacobian,
                    rhs,
                    weights,
                    max_speed,
                    min_jacobian_norm,
                )
                safe = result.velocity
                solver_status = result.solver_status

        safe = self._finite_vector(safe, 6)
        safe = self._filter_safe_velocity(safe)
        command_published = self._publish_safe_velocity(safe)

        lhs_nominal = float(np.dot(jacobian, self._finite_vector(nominal, 6)))
        lhs_safe = float(np.dot(jacobian, safe))
        self._publish_debug(
            enabled=enabled,
            active=active,
            valid_distance=valid_distance,
            nominal_ready=input_ready,
            command_published=command_published,
            mode=mode,
            solver_status=solver_status,
            obstacle_id=str(distance_msg.obstacle_id) if distance_msg is not None else "",
            h=h,
            min_distance=min_distance,
            safety_margin=safety_margin,
            alpha_h=alpha_h,
            lhs_nominal=lhs_nominal,
            lhs_safe=lhs_safe,
            rhs=rhs,
            jacobian=jacobian,
            nominal_velocity=nominal,
            safe_velocity=safe,
            normal=normal,
            distance_msg=distance_msg,
        )

        self._maybe_log_debug(
            mode=mode,
            solver_status=solver_status,
            h=h,
            alpha_h=alpha_h,
            jacobian=jacobian,
            nominal=nominal,
            safe=safe,
            lhs_nominal=lhs_nominal,
            lhs_safe=lhs_safe,
            rhs=rhs,
        )

    def compute_alpha(self, h: float) -> float:
        h = self._finite_scalar(h, 0.0)
        alpha_gain_near = self._float_param("alpha_gain_near", 1.0)
        alpha_gain_far = self._float_param("alpha_gain_far", 1.0)
        alpha_switch_h = self._float_param("alpha_switch_h", 0.05)

        if h <= alpha_switch_h:
            return float(alpha_gain_near * h)
        return float(alpha_gain_near * alpha_switch_h + alpha_gain_far * (h - alpha_switch_h))

    def compute_jacobian(
        self,
        distance_msg: ObstacleDistance,
        corners_msg: PoseArray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        min_jacobian_norm = self._float_param("min_jacobian_norm", 1.0e-6, minimum=0.0)
        raw_normal = self._vector3_to_array(distance_msg.normal)
        normal_norm = float(np.linalg.norm(raw_normal))
        if (not math.isfinite(normal_norm)) or normal_norm <= min_jacobian_norm:
            return np.zeros(6, dtype=np.float64), self._finite_vector(raw_normal, 3)

        normal = raw_normal / normal_norm
        mode = self._choice_param("jacobian_mode", "geometric_weighted", JACOBIAN_MODES)
        w_left, w_right = self._compute_left_right_weights(mode, distance_msg, corners_msg)
        w_left, w_right = self._apply_min_corner_weight(w_left, w_right)

        jacobian = np.concatenate((w_left * normal, w_right * normal)).astype(
            np.float64,
            copy=False,
        )
        jacobian = self._finite_vector(jacobian, 6)
        if float(np.linalg.norm(jacobian)) <= min_jacobian_norm:
            jacobian[:] = 0.0

        return jacobian, self._finite_vector(normal, 3)

    def solve_projection_qp(
        self,
        nominal: np.ndarray,
        jacobian: np.ndarray,
        rhs: float,
        weights: np.ndarray,
        max_speed: float,
        min_jacobian_norm: float,
    ) -> ProjectionResult:
        nominal = self._finite_vector(nominal, 6)
        jacobian = self._finite_vector(jacobian, 6)
        weights = self._positive_weights(weights)
        rhs = self._finite_scalar(rhs, 0.0)
        max_speed = max(0.0, self._finite_scalar(max_speed, 0.0))

        nominal_clipped = np.clip(nominal, -max_speed, max_speed)
        tolerance = 1.0e-10

        if float(np.dot(jacobian, nominal_clipped)) >= rhs - tolerance:
            return ProjectionResult(nominal_clipped, "nominal_feasible")

        if float(np.linalg.norm(jacobian)) < min_jacobian_norm:
            fallback = self._apply_infeasible_policy(
                "zero_jacobian",
                nominal_clipped,
                max_speed,
            )
            return ProjectionResult(fallback, "zero_jacobian")

        u_best = nominal_clipped.copy()
        positive = jacobian > 1.0e-12
        negative = jacobian < -1.0e-12
        u_best[positive] = max_speed
        u_best[negative] = -max_speed
        if float(np.dot(jacobian, u_best)) < rhs - tolerance:
            fallback = self._apply_infeasible_policy(
                "infeasible_velocity_bounds",
                nominal_clipped,
                max_speed,
                u_best,
            )
            return ProjectionResult(fallback, "infeasible_velocity_bounds")

        inv_weight_sq = 1.0 / np.maximum(weights * weights, 1.0e-12)

        def project_for_lambda(lam: float) -> np.ndarray:
            return np.clip(
                nominal_clipped + lam * jacobian * inv_weight_sq,
                -max_speed,
                max_speed,
            )

        hi = 1.0
        for _ in range(120):
            if float(np.dot(jacobian, project_for_lambda(hi))) >= rhs - tolerance:
                break
            hi *= 2.0
        else:
            return ProjectionResult(self._finite_vector(u_best, 6), "projected")

        lo = 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(np.dot(jacobian, project_for_lambda(mid))) >= rhs:
                hi = mid
            else:
                lo = mid

        safe = self._finite_vector(project_for_lambda(hi), 6)
        return ProjectionResult(safe, "projected")

    def _compute_left_right_weights(
        self,
        mode: str,
        distance_msg: ObstacleDistance,
        corners_msg: PoseArray | None,
    ) -> tuple[float, float]:
        if mode == "normal_equal":
            return 0.5, 0.5

        if mode == "geometric_weighted" and corners_msg is not None:
            closest = self._point_to_array(distance_msg.closest_particle)
            left = self._point_to_array(corners_msg.poses[0].position)
            right = self._point_to_array(corners_msg.poses[1].position)
            d_left = float(np.linalg.norm(closest - left))
            d_right = float(np.linalg.norm(closest - right))
            power = self._float_param("left_right_weight_power", 1.0, minimum=0.0)
            eps = 1.0e-9
            raw_left = 1.0 / ((d_left + eps) ** power)
            raw_right = 1.0 / ((d_right + eps) ** power)
            return self._normalize_weights(raw_left, raw_right)

        return self._weights_from_closest_x(distance_msg, corners_msg)

    def _weights_from_closest_x(
        self,
        distance_msg: ObstacleDistance,
        corners_msg: PoseArray | None,
    ) -> tuple[float, float]:
        closest_x = self._finite_scalar(distance_msg.closest_particle.x, 0.0)
        if corners_msg is not None:
            x_left = self._finite_scalar(corners_msg.poses[0].position.x, 0.0)
            x_right = self._finite_scalar(corners_msg.poses[1].position.x, 0.0)
        else:
            half_width = self._float_param("cloth_half_width_x", 0.42, minimum=1.0e-9)
            x_left = -half_width
            x_right = half_width

        span = x_right - x_left
        if abs(span) <= 1.0e-12:
            return 0.5, 0.5

        t = (closest_x - x_left) / span
        t = min(1.0, max(0.0, t))
        return 1.0 - t, t

    def _apply_min_corner_weight(self, w_left: float, w_right: float) -> tuple[float, float]:
        min_corner_weight = self._float_param("min_corner_weight", 0.0, minimum=0.0)
        return self._normalize_weights(
            max(w_left, min_corner_weight),
            max(w_right, min_corner_weight),
        )

    @staticmethod
    def _normalize_weights(w_left: float, w_right: float) -> tuple[float, float]:
        if not math.isfinite(w_left):
            w_left = 0.5
        if not math.isfinite(w_right):
            w_right = 0.5
        total = w_left + w_right
        if total <= 0.0 or not math.isfinite(total):
            return 0.5, 0.5
        return float(w_left / total), float(w_right / total)

    def _apply_stale_policy(self, nominal: np.ndarray) -> np.ndarray:
        policy = self._choice_param("stale_policy", "nominal", STALE_POLICIES)
        max_speed = self._float_param("max_speed", 0.10, minimum=0.0)
        if policy == "zero":
            return np.zeros(6, dtype=np.float64)
        if policy == "hold-last":
            return np.clip(self._last_safe_velocity.reshape(6), -max_speed, max_speed)
        return np.clip(self._finite_vector(nominal, 6), -max_speed, max_speed)

    def _apply_infeasible_policy(
        self,
        reason: str,
        nominal_clipped: np.ndarray,
        max_speed: float,
        u_best: np.ndarray | None = None,
    ) -> np.ndarray:
        policy = self._choice_param(
            "infeasible_policy",
            "project_bounds",
            INFEASIBLE_POLICIES,
        )
        if policy == "nominal":
            return self._finite_vector(nominal_clipped, 6)
        if policy == "zero":
            return np.zeros(6, dtype=np.float64)
        if policy == "hold-last":
            return np.clip(self._last_safe_velocity.reshape(6), -max_speed, max_speed)
        if reason == "infeasible_velocity_bounds" and u_best is not None:
            return self._finite_vector(u_best, 6)
        return np.clip(self._last_safe_velocity.reshape(6), -max_speed, max_speed)

    def _publish_safe_velocity(self, safe_flat: np.ndarray) -> bool:
        safe = self._finite_vector(safe_flat, 6).reshape(2, 3)

        stamp = self.get_clock().now().to_msg()
        frame_id = self._str_param("frame_id", "world")
        self._bottom_left_cbf_pub.publish(self._make_twist_stamped(safe[0], stamp, frame_id))
        self._bottom_right_cbf_pub.publish(self._make_twist_stamped(safe[1], stamp, frame_id))
        return True

    def _filter_safe_velocity(self, safe_flat: np.ndarray) -> np.ndarray:
        safe = self._finite_vector(safe_flat, 6).reshape(2, 3)
        beta = self._float_param("output_filter_beta", 0.0)
        beta = min(0.99, max(0.0, beta))
        if beta > 0.0:
            safe = beta * self._last_safe_velocity + (1.0 - beta) * safe
        self._last_safe_velocity = safe.copy()
        return safe.reshape(6)

    def _publish_debug(
        self,
        *,
        enabled: bool,
        active: bool,
        valid_distance: bool,
        nominal_ready: bool,
        command_published: bool,
        mode: str,
        solver_status: str,
        obstacle_id: str,
        h: float,
        min_distance: float,
        safety_margin: float,
        alpha_h: float,
        lhs_nominal: float,
        lhs_safe: float,
        rhs: float,
        jacobian: np.ndarray,
        nominal_velocity: np.ndarray,
        safe_velocity: np.ndarray,
        normal: np.ndarray,
        distance_msg: ObstacleDistance | None,
    ) -> None:
        if not self._bool_param("publish_debug", True):
            return

        msg = CbfDebug()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._str_param("frame_id", "world")
        msg.enabled = bool(enabled)
        msg.active = bool(active)
        msg.valid_distance = bool(valid_distance)
        msg.nominal_ready = bool(nominal_ready)
        msg.command_published = bool(command_published)
        msg.mode = mode
        msg.solver_status = solver_status
        msg.obstacle_id = obstacle_id
        msg.h = self._finite_scalar(h, 0.0)
        msg.min_distance = self._finite_scalar(min_distance, 0.0)
        msg.safety_margin = self._finite_scalar(safety_margin, 0.0)
        msg.alpha_h = self._finite_scalar(alpha_h, 0.0)
        msg.lhs_nominal = self._finite_scalar(lhs_nominal, 0.0)
        msg.lhs_safe = self._finite_scalar(lhs_safe, 0.0)
        msg.rhs = self._finite_scalar(rhs, 0.0)
        msg.jacobian = self._finite_vector(jacobian, 6).tolist()
        msg.nominal_velocity = self._finite_vector(nominal_velocity, 6).tolist()
        msg.safe_velocity = self._finite_vector(safe_velocity, 6).tolist()
        msg.correction = (
            self._finite_vector(safe_velocity, 6) - self._finite_vector(nominal_velocity, 6)
        ).tolist()
        msg.normal = self._make_vector3(normal)

        if distance_msg is None:
            msg.closest_particle = Point()
            msg.closest_obstacle_point = Point()
        else:
            msg.closest_particle = self._make_point(
                self._point_to_array(distance_msg.closest_particle)
            )
            msg.closest_obstacle_point = self._make_point(
                self._point_to_array(distance_msg.closest_obstacle_point)
            )

        self._debug_pub.publish(msg)

    def _maybe_log_debug(
        self,
        *,
        mode: str,
        solver_status: str,
        h: float,
        alpha_h: float,
        jacobian: np.ndarray,
        nominal: np.ndarray,
        safe: np.ndarray,
        lhs_nominal: float,
        lhs_safe: float,
        rhs: float,
    ) -> None:
        if not self._bool_param("debug_log_enabled", True):
            return
        period = self._float_param("debug_log_period", 1.0, minimum=0.0)
        now = time.monotonic()
        if now - self._last_debug_log_time < period:
            return
        self._last_debug_log_time = now

        self.get_logger().info(
            "CBF mode=%s status=%s h=%.4f alpha=%.4f lhs_nom=%.4f "
            "lhs_safe=%.4f rhs=%.4f J=%s nominal=%s safe=%s "
            "formation_error=%s recovery=%s"
            % (
                mode,
                solver_status,
                self._finite_scalar(h, 0.0),
                self._finite_scalar(alpha_h, 0.0),
                self._finite_scalar(lhs_nominal, 0.0),
                self._finite_scalar(lhs_safe, 0.0),
                self._finite_scalar(rhs, 0.0),
                self._array_str(jacobian),
                self._array_str(nominal),
                self._array_str(safe),
                self._array_str(self._last_formation_error),
                self._array_str(self._last_recovery_velocity),
            )
        )

    def _input_is_ready(self, now_monotonic: float) -> bool:
        timeout = self._float_param("input_timeout", 0.25, minimum=0.0)
        left_ready = (
            self._top_left_input_time is not None
            and now_monotonic - self._top_left_input_time <= timeout
            and np.isfinite(self._top_left_input).all()
        )
        right_ready = (
            self._top_right_input_time is not None
            and now_monotonic - self._top_right_input_time <= timeout
            and np.isfinite(self._top_right_input).all()
        )
        return bool(left_ready and right_ready)

    def _distance_is_valid(self, now_monotonic: float) -> bool:
        timeout = self._float_param("distance_timeout", 0.25, minimum=0.0)
        msg = self._distance_msg
        if msg is None or self._distance_time is None:
            return False
        if now_monotonic - self._distance_time > timeout:
            return False
        required = np.array(
            [
                msg.h,
                msg.min_distance,
                msg.safety_margin,
                msg.closest_particle.x,
                msg.closest_particle.y,
                msg.closest_particle.z,
                msg.closest_obstacle_point.x,
                msg.closest_obstacle_point.y,
                msg.closest_obstacle_point.z,
                msg.normal.x,
                msg.normal.y,
                msg.normal.z,
            ],
            dtype=np.float64,
        )
        return bool(msg.valid and np.isfinite(required).all())

    def _fresh_corners(self, now_monotonic: float) -> PoseArray | None:
        timeout = self._float_param("corners_timeout", 0.50, minimum=0.0)
        msg = self._corners_msg
        if msg is None or self._corners_time is None:
            return None
        if now_monotonic - self._corners_time > timeout or len(msg.poses) < 2:
            return None
        left = self._point_to_array(msg.poses[0].position)
        right = self._point_to_array(msg.poses[1].position)
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            return None
        return msg

    def _fresh_formation_corners(self, now_monotonic: float) -> PoseArray | None:
        timeout = self._float_param("corners_timeout", 0.50, minimum=0.0)
        msg = self._corners_msg
        if msg is None or self._corners_time is None:
            return None
        if now_monotonic - self._corners_time > timeout or len(msg.poses) < 4:
            return None
        if self._formation_corner_positions(msg) is None:
            return None
        return msg

    def _current_nominal_velocity(self) -> np.ndarray:
        base = self._finite_vector(
            np.concatenate((self._top_left_input, self._top_right_input)),
            6,
        )
        self._last_formation_error[:] = 0.0
        self._last_recovery_velocity[:] = 0.0

        if not self._bool_param("formation_recovery_enabled", True):
            return base
        if not self._formation_reference_ready:
            return base

        corners_msg = self._fresh_formation_corners(time.monotonic())
        if corners_msg is None:
            return base

        positions = self._formation_corner_positions(corners_msg)
        if positions is None:
            return base
        bottom_left, bottom_right, top_left, top_right = positions

        desired_bottom_left = top_left + self._initial_bottom_left_offset
        desired_bottom_right = top_right + self._initial_bottom_right_offset
        deadband = self._float_param("formation_recovery_deadband", 0.003, minimum=0.0)
        err_left = self._apply_position_deadband(desired_bottom_left - bottom_left, deadband)
        err_right = self._apply_position_deadband(desired_bottom_right - bottom_right, deadband)

        gain = self._float_param("formation_recovery_gain", 0.5, minimum=0.0)
        recovery = np.vstack((gain * err_left, gain * err_right))
        recovery = self._limit_per_corner_norm(
            recovery,
            self._float_param("formation_recovery_max_speed", 0.03, minimum=0.0),
        )

        self._last_formation_error = np.vstack((err_left, err_right))
        self._last_recovery_velocity = recovery.copy()
        return self._finite_vector(base + recovery.reshape(6), 6)

    def _capture_formation_reference(self, msg: PoseArray) -> None:
        if self._formation_reference_ready or len(msg.poses) < 4:
            return

        self._formation_reference_mode()
        positions = self._formation_corner_positions(msg)
        if positions is None:
            return
        bottom_left, bottom_right, top_left, top_right = positions

        self._initial_bottom_left_offset = bottom_left - top_left
        self._initial_bottom_right_offset = bottom_right - top_right
        self._formation_reference_ready = True
        self.get_logger().info(
            "Formation reference captured: bottom-left/top-left offset=%s, "
            "bottom-right/top-right offset=%s"
            % (
                self._array_str(self._initial_bottom_left_offset),
                self._array_str(self._initial_bottom_right_offset),
            )
        )

    def _formation_reference_mode(self) -> str:
        mode = self._str_param("formation_reference_mode", "initial")
        if mode in FORMATION_REFERENCE_MODES:
            return mode
        if not self._warned_unsupported_formation_reference_mode:
            self.get_logger().warn(
                "Unsupported formation_reference_mode=%r; falling back to 'initial'"
                % mode
            )
            self._warned_unsupported_formation_reference_mode = True
        return "initial"

    def _formation_corner_positions(
        self,
        msg: PoseArray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if len(msg.poses) < 4:
            return None
        positions = (
            self._point_to_array(msg.poses[0].position),
            self._point_to_array(msg.poses[1].position),
            self._point_to_array(msg.poses[2].position),
            self._point_to_array(msg.poses[3].position),
        )
        if not all(np.isfinite(position).all() for position in positions):
            return None
        return positions

    def _limit_per_corner_norm(self, velocities: np.ndarray, max_norm: float) -> np.ndarray:
        velocities = np.asarray(velocities, dtype=np.float64).reshape(2, 3)
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        if (not math.isfinite(max_norm)) or max_norm <= 0.0:
            return np.zeros_like(velocities)

        limited = velocities.copy()
        for idx in range(limited.shape[0]):
            norm = float(np.linalg.norm(limited[idx]))
            if math.isfinite(norm) and norm > max_norm:
                limited[idx] *= max_norm / norm
        return limited

    def _apply_position_deadband(self, error: np.ndarray, deadband: float) -> np.ndarray:
        error = self._finite_vector(error, 3)
        deadband = max(0.0, self._finite_scalar(deadband, 0.0))
        if float(np.linalg.norm(error)) < deadband:
            return np.zeros(3, dtype=np.float64)
        return error

    def _control_weights(self) -> np.ndarray:
        weights = np.array(
            [
                self._float_param("weight_x", 1.0),
                self._float_param("weight_y", 1.0),
                self._float_param("weight_z", 1.0),
                self._float_param("weight_x", 1.0),
                self._float_param("weight_y", 1.0),
                self._float_param("weight_z", 1.0),
            ],
            dtype=np.float64,
        )
        return self._positive_weights(weights)

    @staticmethod
    def _positive_weights(weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64).reshape(6)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = np.where(weights > 0.0, weights, 1.0)
        return weights

    @staticmethod
    def _twist_linear_to_array(msg: TwistStamped) -> np.ndarray:
        return np.array(
            [
                float(msg.twist.linear.x),
                float(msg.twist.linear.y),
                float(msg.twist.linear.z),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _point_to_array(point: Point) -> np.ndarray:
        return np.array([float(point.x), float(point.y), float(point.z)], dtype=np.float64)

    @staticmethod
    def _vector3_to_array(vector: Vector3) -> np.ndarray:
        return np.array([float(vector.x), float(vector.y), float(vector.z)], dtype=np.float64)

    def _make_twist_stamped(self, velocity: np.ndarray, stamp, frame_id: str) -> TwistStamped:
        velocity = self._finite_vector(velocity, 3)
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(velocity[0])
        msg.twist.linear.y = float(velocity[1])
        msg.twist.linear.z = float(velocity[2])
        return msg

    def _make_vector3(self, values: np.ndarray) -> Vector3:
        values = self._finite_vector(values, 3)
        msg = Vector3()
        msg.x = float(values[0])
        msg.y = float(values[1])
        msg.z = float(values[2])
        return msg

    def _make_point(self, values: np.ndarray) -> Point:
        values = self._finite_vector(values, 3)
        msg = Point()
        msg.x = float(values[0])
        msg.y = float(values[1])
        msg.z = float(values[2])
        return msg

    def _distance_float(
        self,
        msg: ObstacleDistance | None,
        field_name: str,
        default_value: float,
    ) -> float:
        if msg is None:
            return default_value
        return self._finite_scalar(getattr(msg, field_name), default_value)

    @staticmethod
    def _finite_vector(values: np.ndarray, size: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(size)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _finite_scalar(value: float, default_value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default_value
        if not math.isfinite(value):
            return default_value
        return value

    def _bool_param(self, name: str, default_value: bool) -> bool:
        try:
            return bool(self.get_parameter(name).value)
        except Exception:
            return default_value

    def _float_param(
        self,
        name: str,
        default_value: float,
        minimum: float | None = None,
    ) -> float:
        try:
            value = float(self.get_parameter(name).value)
        except Exception:
            value = default_value
        if not math.isfinite(value):
            value = default_value
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _str_param(self, name: str, default_value: str) -> str:
        try:
            value = str(self.get_parameter(name).value)
        except Exception:
            return default_value
        return value if value else default_value

    def _choice_param(self, name: str, default_value: str, allowed_values: tuple[str, ...]) -> str:
        value = self._str_param(name, default_value)
        return value if value in allowed_values else default_value

    @staticmethod
    def _array_str(values: np.ndarray) -> str:
        return np.array2string(
            np.asarray(values, dtype=np.float64),
            precision=3,
            suppress_small=True,
            separator=", ",
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FabricCbfControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
