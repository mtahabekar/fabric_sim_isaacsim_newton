#!/usr/bin/env python3

from dataclasses import dataclass
import math
import time

import numpy as np

import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from fabric_obstacle_distance.msg import ObstacleDistance


@dataclass(frozen=True)
class BoxConfig:
    obstacle_id: str
    center: np.ndarray
    half_extents: np.ndarray
    particle_radius: float
    safety_margin: float


@dataclass(frozen=True)
class ClosestDistance:
    min_distance: float
    h: float
    closest_particle_index: int
    closest_particle: np.ndarray
    closest_obstacle_point: np.ndarray
    normal: np.ndarray


class ObstacleDistanceNode(Node):
    def __init__(self):
        super().__init__("obstacle_distance_node")

        self.declare_parameter("fabric_state_topic", "/fabric/sim/state")
        self.declare_parameter("distance_topic", "/fabric/obstacle_distance")
        self.declare_parameter("obstacle_id", "table")
        self.declare_parameter("table_center_x", 0.0)
        self.declare_parameter("table_center_y", 0.0)
        self.declare_parameter("table_hx", 0.22)
        self.declare_parameter("table_hy", 0.22)
        self.declare_parameter("table_hz", 0.16)
        self.declare_parameter("particle_radius", 0.008)
        self.declare_parameter("safety_margin", 0.10)
        self.declare_parameter("publish_debug_log", False)
        self.declare_parameter("debug_log_period", 1.0)

        fabric_state_topic = self.get_parameter("fabric_state_topic").value
        distance_topic = self.get_parameter("distance_topic").value

        self._publisher = self.create_publisher(ObstacleDistance, distance_topic, 10)
        self._subscription = self.create_subscription(
            PointCloud2,
            fabric_state_topic,
            self._point_cloud_callback,
            10,
        )
        self._last_debug_log_time = 0.0
        self._last_error_log_time = 0.0

        self.get_logger().info(
            f"Publishing obstacle distances from {fabric_state_topic} to {distance_topic}"
        )

    def _point_cloud_callback(self, msg: PointCloud2) -> None:
        try:
            config = self._read_box_config()
            points, source_indices = self._read_xyz_points(msg)
            if points.shape[0] == 0:
                out_msg = self._make_invalid_message(msg, config.obstacle_id, config.safety_margin)
            else:
                closest = self._compute_closest_distance(points, source_indices, config)
                out_msg = self._make_valid_message(msg, config, closest)
        except Exception as exc:  # Malformed PointCloud2 messages should not kill the node.
            obstacle_id = self.get_parameter("obstacle_id").value
            safety_margin = float(self.get_parameter("safety_margin").value)
            out_msg = self._make_invalid_message(msg, obstacle_id, safety_margin)
            self._log_parse_or_config_error(exc)

        self._publisher.publish(out_msg)
        self._maybe_log_debug(out_msg)

    def _read_box_config(self) -> BoxConfig:
        obstacle_id = str(self.get_parameter("obstacle_id").value)
        table_center_x = float(self.get_parameter("table_center_x").value)
        table_center_y = float(self.get_parameter("table_center_y").value)
        table_hx = float(self.get_parameter("table_hx").value)
        table_hy = float(self.get_parameter("table_hy").value)
        table_hz = float(self.get_parameter("table_hz").value)
        particle_radius = float(self.get_parameter("particle_radius").value)
        safety_margin = float(self.get_parameter("safety_margin").value)

        center = np.array([table_center_x, table_center_y, table_hz], dtype=np.float64)
        half_extents = np.array([table_hx, table_hy, table_hz], dtype=np.float64)

        if not np.isfinite(center).all():
            raise ValueError("box center parameters must be finite")
        if not np.isfinite(half_extents).all() or np.any(half_extents <= 0.0):
            raise ValueError("table_hx, table_hy, and table_hz must be finite and positive")
        if not math.isfinite(particle_radius) or particle_radius < 0.0:
            raise ValueError("particle_radius must be finite and non-negative")
        if not math.isfinite(safety_margin):
            raise ValueError("safety_margin must be finite")

        return BoxConfig(
            obstacle_id=obstacle_id,
            center=center,
            half_extents=half_extents,
            particle_radius=particle_radius,
            safety_margin=safety_margin,
        )

    def _read_xyz_points(self, msg: PointCloud2) -> tuple[np.ndarray, np.ndarray]:
        available_fields = {field.name for field in msg.fields}
        missing_fields = {"x", "y", "z"} - available_fields
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"PointCloud2 is missing required fields: {missing}")

        read_result = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points = self._read_points_result_to_array(read_result)
        if points.shape[0] == 0:
            return points, np.empty((0,), dtype=np.int32)

        finite_mask = np.isfinite(points).all(axis=1)
        points = points[finite_mask]
        if points.shape[0] == 0:
            return points, np.empty((0,), dtype=np.int32)

        return points, self._read_source_indices(msg, points.shape[0])

    def _read_source_indices(self, msg: PointCloud2, expected_count: int) -> np.ndarray:
        read_result = point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=False,
        )
        all_points = self._read_points_result_to_array(read_result)
        if all_points.shape[0] == 0:
            return np.arange(expected_count, dtype=np.int32)

        finite_mask = np.isfinite(all_points).all(axis=1)
        source_indices = np.flatnonzero(finite_mask).astype(np.int32, copy=False)
        if source_indices.shape[0] != expected_count:
            return np.arange(expected_count, dtype=np.int32)

        return source_indices

    @staticmethod
    def _read_points_result_to_array(read_result) -> np.ndarray:
        if isinstance(read_result, np.ndarray):
            if read_result.dtype.names is not None:
                points = np.column_stack(
                    (
                        np.asarray(read_result["x"], dtype=np.float64).reshape(-1),
                        np.asarray(read_result["y"], dtype=np.float64).reshape(-1),
                        np.asarray(read_result["z"], dtype=np.float64).reshape(-1),
                    )
                )
            else:
                array = np.asarray(read_result, dtype=np.float64)
                if array.size == 0:
                    return np.empty((0, 3), dtype=np.float64)
                if array.ndim == 1:
                    if array.shape[0] != 3:
                        raise ValueError("PointCloud2 read_points returned rows without xyz triples")
                    points = array.reshape(1, 3)
                else:
                    points = array.reshape(-1, array.shape[-1])
                    if points.shape[1] < 3:
                        raise ValueError("PointCloud2 read_points returned fewer than three columns")
                    points = points[:, :3]
        else:
            rows = [(float(point[0]), float(point[1]), float(point[2])) for point in read_result]
            if not rows:
                return np.empty((0, 3), dtype=np.float64)
            points = np.asarray(rows, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("PointCloud2 xyz data could not be converted to shape (N, 3)")

        return points.astype(np.float64, copy=False)

    def _compute_closest_distance(
        self,
        points: np.ndarray,
        source_indices: np.ndarray,
        config: BoxConfig,
    ) -> ClosestDistance:
        local = points - config.center
        q = np.abs(local) - config.half_extents

        outside = np.maximum(q, 0.0)
        outside_norm = np.linalg.norm(outside, axis=1)
        inside_dist = np.minimum(np.max(q, axis=1), 0.0)

        signed_distance_to_box = outside_norm + inside_dist
        signed_particle_distance = signed_distance_to_box - config.particle_radius

        closest_index = int(np.argmin(signed_particle_distance))
        closest_particle = points[closest_index]
        closest_obstacle_point, normal = self._closest_obstacle_point_and_normal(
            closest_particle,
            config.center,
            config.half_extents,
        )
        min_distance = float(signed_particle_distance[closest_index])

        return ClosestDistance(
            min_distance=min_distance,
            h=float(min_distance - config.safety_margin),
            closest_particle_index=int(source_indices[closest_index]),
            closest_particle=closest_particle,
            closest_obstacle_point=closest_obstacle_point,
            normal=normal,
        )

    def _closest_obstacle_point_and_normal(
        self,
        point: np.ndarray,
        center: np.ndarray,
        half_extents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        box_min = center - half_extents
        box_max = center + half_extents
        q = np.abs(point - center) - half_extents
        eps = 1.0e-12

        if np.any(q > 0.0):
            closest_point = np.minimum(np.maximum(point, box_min), box_max)
            normal_vec = point - closest_point
            norm = float(np.linalg.norm(normal_vec))
            if norm > eps:
                return closest_point, normal_vec / norm

        return self._nearest_face_projection(point, box_min, box_max)

    @staticmethod
    def _nearest_face_projection(
        point: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distances = np.array(
            [
                point[0] - box_min[0],
                box_max[0] - point[0],
                point[1] - box_min[1],
                box_max[1] - point[1],
                point[2] - box_min[2],
                box_max[2] - point[2],
            ],
            dtype=np.float64,
        )
        face_index = int(np.argmin(distances))
        axis = face_index // 2
        use_max_face = (face_index % 2) == 1

        closest_point = np.array(point, dtype=np.float64, copy=True)
        normal = np.zeros(3, dtype=np.float64)

        if use_max_face:
            closest_point[axis] = box_max[axis]
            normal[axis] = 1.0
        else:
            closest_point[axis] = box_min[axis]
            normal[axis] = -1.0

        return closest_point, normal

    @staticmethod
    def _make_valid_message(
        msg: PointCloud2,
        config: BoxConfig,
        closest: ClosestDistance,
    ) -> ObstacleDistance:
        out_msg = ObstacleDistance()
        out_msg.header = msg.header
        out_msg.valid = True
        out_msg.obstacle_id = config.obstacle_id
        out_msg.min_distance = closest.min_distance
        out_msg.h = closest.h
        out_msg.safety_margin = config.safety_margin
        out_msg.closest_particle_index = closest.closest_particle_index
        out_msg.closest_particle = Point(
            x=float(closest.closest_particle[0]),
            y=float(closest.closest_particle[1]),
            z=float(closest.closest_particle[2]),
        )
        out_msg.closest_obstacle_point = Point(
            x=float(closest.closest_obstacle_point[0]),
            y=float(closest.closest_obstacle_point[1]),
            z=float(closest.closest_obstacle_point[2]),
        )
        out_msg.normal = Vector3(
            x=float(closest.normal[0]),
            y=float(closest.normal[1]),
            z=float(closest.normal[2]),
        )
        return out_msg

    @staticmethod
    def _make_invalid_message(
        msg: PointCloud2,
        obstacle_id: str,
        safety_margin: float,
    ) -> ObstacleDistance:
        out_msg = ObstacleDistance()
        out_msg.header = msg.header
        out_msg.valid = False
        out_msg.obstacle_id = obstacle_id
        out_msg.min_distance = float("inf")
        out_msg.h = float("inf")
        out_msg.safety_margin = safety_margin
        out_msg.closest_particle_index = -1
        out_msg.closest_particle = Point()
        out_msg.closest_obstacle_point = Point()
        out_msg.normal = Vector3()
        return out_msg

    def _maybe_log_debug(self, out_msg: ObstacleDistance) -> None:
        if not bool(self.get_parameter("publish_debug_log").value):
            return

        period = max(float(self.get_parameter("debug_log_period").value), 0.0)
        now = time.monotonic()
        if now - self._last_debug_log_time < period:
            return

        self._last_debug_log_time = now
        closest = out_msg.closest_particle
        normal = out_msg.normal
        self.get_logger().info(
            "obstacle_id=%s min_distance=%.6f h=%.6f closest_particle_index=%d "
            "closest_particle=(%.6f, %.6f, %.6f) normal=(%.6f, %.6f, %.6f)"
            % (
                out_msg.obstacle_id,
                out_msg.min_distance,
                out_msg.h,
                out_msg.closest_particle_index,
                closest.x,
                closest.y,
                closest.z,
                normal.x,
                normal.y,
                normal.z,
            )
        )

    def _log_parse_or_config_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_log_time < 1.0:
            return

        self._last_error_log_time = now
        self.get_logger().warn(f"Publishing invalid obstacle distance: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ObstacleDistanceNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
