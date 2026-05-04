# Fabric IsaacSim Newton CBF Simulation

This repository contains a ROS 2 + Isaac Sim fabric manipulation stack. It runs
a Newton Style3D cloth simulation in Isaac Sim, publishes the simulated fabric
state over ROS 2, computes distance from the fabric to a table obstacle, and
uses an external control barrier function (CBF) node to filter follower
velocity commands before they are applied back in the simulator.

The main workflow is:

```text
keyboard teleop
  -> leader velocity commands
  -> Isaac Sim / Newton cloth simulation
  -> simulated fabric state and corner poses
  -> obstacle distance node
  -> CBF controller
  -> safe follower velocity commands
  -> Isaac Sim applies leader and follower commands
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `fabric_newton_isaacsim_standalone.py` | Standalone Isaac Sim application. Builds the cloth, table, gripper patches, Newton solver, USD visuals, and ROS 2 bridge. |
| `fabric_sim_bringup/` | Launch package for starting the full stack from one configuration file. |
| `fabric_sim_bringup/config/full_sim.yaml` | Central configuration for launch flags, simulator CLI arguments, node startup delays, and ROS 2 node parameters. |
| `fabric_teleop/` | Keyboard teleop package that publishes leader-left and leader-right fabric velocity commands. |
| `fabric_obstacle_distance/` | ROS 2 node and message for signed distance from fabric particles to the table obstacle. |
| `fabric_cbf_controller/` | ROS 2 node and debug message for the external CBF velocity filter. |

## What Each Part Does

### Isaac Sim / Newton Standalone

`fabric_newton_isaacsim_standalone.py` is the simulator entry point. It:

- starts Isaac Sim and enables the Isaac ROS 2 bridge extension;
- creates a USD scene with a ground plane, a table box, a cloth mesh, and
  visual markers for the fabric corner targets and actual corner positions;
- builds a Newton Style3D cloth grid with configurable resolution, material
  stiffness, damping, particle radius, contact settings, spawn center, height,
  and solver iterations;
- selects square gripper patches around the four cloth corners and hard-pins
  those particles to moving corner targets;
- subscribes to leader teleop commands on `/fabric/leader_left/cmd_vel` and
  `/fabric/leader_right/cmd_vel`;
- republishes those leader commands as CBF inputs on
  `/fabric/leader_left/cbf_input_cmd_vel` and
  `/fabric/leader_right/cbf_input_cmd_vel`;
- optionally subscribes to external CBF follower commands on
  `/fabric/follower_left/cbf_cmd_vel` and `/fabric/follower_right/cbf_cmd_vel`;
- applies the leader commands directly to the leader gripper targets and applies
  either CBF-safe, nominal, zero, or held follower commands depending on the CBF
  settings and command freshness;
- publishes simulated corner poses on `/fabric/sim/corners` and fabric particle
  positions on `/fabric/sim/state`;
- publishes lightweight frame timing on `/fabric/sim/timing`.

The simulator accepts CLI arguments for cloth size, material parameters,
contact parameters, ROS frame IDs, command timeouts, and CBF behavior. In the
full launch path these arguments are generated from
`fabric_sim_bringup/config/full_sim.yaml`. The default spawn remains centered at
`cloth_center_x: 0.0`, `cloth_center_y: 0.0`, and `cloth_start_z: 0.72`.

### Bringup Package

`fabric_sim_bringup` provides the one-command launch path for the whole stack.
The launch file:

- reads `config/full_sim.yaml`;
- starts the Isaac Sim standalone process first;
- starts the obstacle distance node, CBF controller, and teleop node after the
  configured delays;
- passes each package's parameter block from the YAML file into the matching
  ROS 2 node;
- shuts down the rest of the launch when the Isaac Sim process exits.

The YAML file has four main sections:

- `launch`: which processes to start, startup delays, and shutdown behavior.
- `isaac_sim`: Isaac Sim working directory, `python.sh`, simulator script, and
  simulator CLI arguments.
- `obstacle_distance`: parameters for the signed-distance node.
- `cbf_controller`: parameters for the CBF filter.
- `teleop`: keyboard teleop speed, publish rate, and frame ID.

Keys in `isaac_sim.arguments` map from snake case to CLI flags. For example,
`external_cbf_enabled: true` becomes `--external-cbf-enabled`, and
`cbf_stale_policy: nominal` becomes `--cbf-stale-policy nominal`.

### Keyboard Teleop

`fabric_teleop/fabric_teleop/fabric_corner_keyboard_teleop.py` publishes
`geometry_msgs/msg/TwistStamped` velocity commands for the two fabric leaders.
It publishes continuously while keys are held and sends zeros when
keys are released.

| Role | Keys | Axes |
| --- | --- | --- |
| Leader left | `w` / `s` | `+x` / `-x` |
| Leader left | `d` / `a` | `+y` / `-y` |
| Leader left | `e` / `q` | `+z` / `-z` |
| Leader right | `i` / `k` | `+x` / `-x` |
| Leader right | `l` / `j` | `+y` / `-y` |
| Leader right | `o` / `u` | `+z` / `-z` |

Use `Ctrl+C` to stop the launch. The `q` key is a movement key, not a quit key.

### Obstacle Distance Node

`fabric_obstacle_distance/scripts/obstacle_distance_node.py` subscribes to
`/fabric/sim/state`, which is a `sensor_msgs/msg/PointCloud2` containing fabric
particle `x`, `y`, and `z` fields.

For every point cloud message, it:

- reads the configured table box center and half extents;
- computes the signed particle distance to the axis-aligned table box;
- subtracts the particle radius;
- finds the closest fabric particle;
- computes the closest point on the obstacle and the outward normal;
- publishes `fabric_obstacle_distance/msg/ObstacleDistance` on
  `/fabric/obstacle_distance`.

The barrier value published by this node is:

```text
h = min_distance - safety_margin
```

Positive `h` means the closest particle is outside the configured safety
margin. Negative `h` means the fabric has entered that margin.

The distance message also includes `compute_time_ms` for experiment timing
analysis.

### CBF Controller

`fabric_cbf_controller/scripts/fabric_cbf_controller_node.py` is the external
velocity filter for the two fabric followers. It subscribes to:

- `/fabric/leader_left/cbf_input_cmd_vel`
- `/fabric/leader_right/cbf_input_cmd_vel`
- `/fabric/obstacle_distance`
- `/fabric/sim/corners`

It publishes:

- `/fabric/follower_left/cbf_cmd_vel`
- `/fabric/follower_right/cbf_cmd_vel`
- `/fabric/cbf/debug`

The controller first builds a nominal follower command from the leader
commands. If formation recovery is enabled, it also tries to preserve the
initial follower-to-leader offsets captured from the first valid
`/fabric/sim/corners` message.

When the obstacle barrier is active, the controller enforces the CBF constraint:

```text
J * u >= -alpha(h)
```

where:

- `u` is the six-dimensional follower command vector:
  `[follower_left_xyz, follower_right_xyz]`;
- `h` comes from the obstacle distance node;
- `alpha(h)` is a configurable piecewise-linear gain;
- `J` is a lightweight geometric Jacobian based on the obstacle normal and
  estimated left/right follower influence weights.

If the nominal command already satisfies the constraint, it is passed through
after speed clipping. If not, the controller projects it to the closest safe
velocity under configurable axis weights and speed bounds. If the constraint is
not feasible, the configured infeasible policy is used.

The debug message includes mode, solver status, barrier values, nominal and safe
velocities, correction, Jacobian, normal, and closest-particle information.
It also includes `compute_time_ms` for controller timing analysis.

## ROS 2 Topics

The primary topic names use the leader/follower convention. The simulator also
keeps historical `top_*` and `bottom_*` aliases enabled for backward
compatibility.

| Topic | Type | Producer | Consumer |
| --- | --- | --- | --- |
| `/fabric/leader_left/cmd_vel` | `geometry_msgs/msg/TwistStamped` | teleop | Isaac Sim |
| `/fabric/leader_right/cmd_vel` | `geometry_msgs/msg/TwistStamped` | teleop | Isaac Sim |
| `/fabric/leader_left/cbf_input_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | CBF controller |
| `/fabric/leader_right/cbf_input_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | CBF controller |
| `/fabric/sim/state` | `sensor_msgs/msg/PointCloud2` | Isaac Sim | obstacle distance |
| `/fabric/sim/corners` | `geometry_msgs/msg/PoseArray` | Isaac Sim | CBF controller |
| `/fabric/obstacle_distance` | `fabric_obstacle_distance/msg/ObstacleDistance` | obstacle distance | CBF controller |
| `/fabric/follower_left/cbf_cmd_vel` | `geometry_msgs/msg/TwistStamped` | CBF controller | Isaac Sim |
| `/fabric/follower_right/cbf_cmd_vel` | `geometry_msgs/msg/TwistStamped` | CBF controller | Isaac Sim |
| `/fabric/follower_left/applied_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | debugging/monitoring |
| `/fabric/follower_right/applied_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | debugging/monitoring |
| `/robot_left/ee_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | downstream robot bridge/debugging |
| `/robot_right/ee_cmd_vel` | `geometry_msgs/msg/TwistStamped` | Isaac Sim | downstream robot bridge/debugging |
| `/fabric/cbf/debug` | `fabric_cbf_controller/msg/CbfDebug` | CBF controller | debugging/monitoring |
| `/fabric/sim/timing` | `std_msgs/msg/Float64MultiArray` | Isaac Sim | experiment logging |

## Build

This repository is intended to live inside a ROS 2 colcon workspace under
`src/`. From the base directory of this repository, build the parent workspace:

```bash
source /opt/ros/jazzy/setup.bash
cd ../..
colcon build --symlink-install
source install/setup.bash
```

If you only changed Python files and built with `--symlink-install`, you usually
do not need to rebuild. If you changed message definitions, package metadata,
launch files, or installed config files, rebuild the workspace.

## Run

From the base directory of this repository:

```bash
./fabric_sim_bringup/scripts/launch_full_sim.sh
```

To run with a custom YAML config:

```bash
./fabric_sim_bringup/scripts/launch_full_sim.sh path/to/custom_full_sim.yaml
```

The helper script sources ROS 2 Jazzy, sources the parent workspace install, and
then runs:

```bash
ros2 launch fabric_sim_bringup full_sim.launch.py
```

## Configuration Notes

Before running on a new machine, check
`fabric_sim_bringup/config/full_sim.yaml`:

- `isaac_sim.working_directory` should point to the Isaac Sim build/release
  directory.
- `isaac_sim.python_sh` should point to Isaac Sim's `python.sh`.
- `isaac_sim.script` should point to this repository's
  `fabric_newton_isaacsim_standalone.py`.
- Startup delays in `launch` may need tuning if ROS nodes start before Isaac Sim
  has created its ROS topics.
- Table dimensions, particle radius, and safety margin should stay consistent
  between the simulator and obstacle distance node.

## Useful Debug Commands

After sourcing the workspace, these commands are useful in another terminal:

```bash
ros2 topic list
ros2 topic echo /fabric/obstacle_distance
ros2 topic echo /fabric/cbf/debug
ros2 topic echo /fabric/follower_left/applied_cmd_vel  # follower-left applied command
```

The CBF node also has periodic log output controlled by
`cbf_controller.parameters.debug_log_enabled` and
`cbf_controller.parameters.debug_log_period` in `full_sim.yaml`.

## Common Issues

- `Workspace install/setup.bash was not found`: build the parent colcon
  workspace first.
- Isaac Sim does not start: verify `isaac_sim.working_directory` and
  `isaac_sim.python_sh` in `full_sim.yaml`.
- ROS nodes start before simulator topics exist: increase
  `obstacle_start_delay`, `cbf_start_delay`, and `teleop_start_delay`.
- Keyboard input does not move the fabric: make sure the launch terminal has
  keyboard focus and that `pynput` is installed.
- CBF commands are stale: inspect `/fabric/cbf/debug`, the configured CBF
  command timeout, and the Isaac Sim `cbf_stale_policy`.
