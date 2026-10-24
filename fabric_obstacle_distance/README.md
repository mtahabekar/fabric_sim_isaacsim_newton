# fabric_obstacle_distance

`fabric_obstacle_distance` publishes signed distance information between IsaacSim fabric particles and a known axis-aligned table obstacle. It subscribes to the IsaacSim fabric point cloud, computes the closest particle-surface distance to the table box, and publishes a structured `ObstacleDistance` message for a future CBF node.

The node expects `/fabric/sim/state` to be a `sensor_msgs/msg/PointCloud2` with `x`, `y`, and `z` float fields.

The simulator also publishes `vx`, `vy`, and `vz` in world m/s for every particle.
This node selects XYZ by name and ignores the extra fields. Consumers must use
the message's `fields` and `point_step`; the full-state rows are 24 bytes.

## Build

```bash
cd /home/taha/ros2_colcon_ws
colcon build --packages-select fabric_obstacle_distance
source install/setup.bash
```

## Run

```bash
ros2 run fabric_obstacle_distance obstacle_distance_node.py
```

## Run With Parameters

```bash
ros2 run fabric_obstacle_distance obstacle_distance_node.py --ros-args \
  -p table_hx:=0.22 \
  -p table_hy:=0.22 \
  -p table_hz:=0.16 \
  -p particle_radius:=0.008 \
  -p safety_margin:=0.10 \
  -p publish_debug_log:=true
```

## Topics

- Subscribes: `/fabric/sim/state` (`sensor_msgs/msg/PointCloud2`)
- Publishes: `/fabric/obstacle_distance` (`fabric_obstacle_distance/msg/ObstacleDistance`)

## Obstacle Convention

The table box center is `[table_center_x, table_center_y, table_hz]` with half extents `[table_hx, table_hy, table_hz]`. With the default IsaacSim convention, the table extends from `z = 0` to `z = 2 * table_hz`.
