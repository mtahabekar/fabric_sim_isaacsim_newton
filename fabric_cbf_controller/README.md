# fabric_cbf_controller

ROS2 Python CBF controller for the IsaacSim fabric manipulation setup.

This package listens to leader CBF input velocity commands from IsaacSim,
adds a slow formation recovery term to produce nominal follower commands,
filters them through an obstacle-avoidance control barrier function, and
publishes final follower velocity commands back to IsaacSim. It also
publishes a debug message with the active mode, barrier values, approximate
Jacobian, nominal/safe velocities, and solver status.

## Architecture

The primary ROS topic strings use the leader/follower naming convention. The
simulator keeps historical `top_*`/`bottom_*` aliases enabled for compatibility.

- IsaacSim receives teleop commands on:
  - `/fabric/leader_left/cmd_vel`
  - `/fabric/leader_right/cmd_vel`
- IsaacSim republishes those leader velocities as CBF inputs on:
  - `/fabric/leader_left/cbf_input_cmd_vel`
  - `/fabric/leader_right/cbf_input_cmd_vel`
- The CBF node computes nominal follower commands from each leader input
  velocity plus a slow formation recovery term:
  - leader-left input + recovery -> desired follower-left velocity
  - leader-right input + recovery -> desired follower-right velocity
- IsaacSim publishes fabric corner poses on `/fabric/sim/corners`.
- `fabric_obstacle_distance` publishes obstacle distance information on
  `/fabric/obstacle_distance`.
- `fabric_cbf_controller` publishes final follower commands on:
  - `/fabric/follower_left/cbf_cmd_vel`
  - `/fabric/follower_right/cbf_cmd_vel`
- Debug output is published on `/fabric/cbf/debug`.

The formation recovery term tries to restore the initial follower-to-leader offsets
captured from the first valid `/fabric/sim/corners` message. When the CBF is
inactive or the barrier is already satisfied, the controller publishes the
recovered nominal velocity. When the CBF is active, it modifies that recovered
nominal velocity to satisfy the obstacle CBF constraint as much as possible.

The current Jacobian estimator is a lightweight geometric approximation:

```text
J = [w_left * n, w_right * n]
```

where `n` is the obstacle-to-fabric normal and `w_left`, `w_right` approximate
how much each follower influences the closest particle. This is modular so
a future finite-difference or rollout-based estimator can replace it without
changing the rest of the controller.

## Build

```bash
cd /home/taha/ros2_colcon_ws
colcon build --packages-select fabric_obstacle_distance fabric_cbf_controller
source install/setup.bash
```

## Run

Start the obstacle distance node:

```bash
ros2 run fabric_obstacle_distance obstacle_distance_node.py --ros-args -p publish_debug_log:=true
```

Start the CBF controller:

```bash
ros2 run fabric_cbf_controller fabric_cbf_controller_node.py --ros-args \
  -p enabled:=true \
  -p jacobian_mode:=normal_equal \
  -p alpha_gain_near:=1.0 \
  -p max_speed:=0.10 \
  -p infeasible_policy:=hold-last \
  -p output_filter_beta:=0.5 \
  -p formation_recovery_enabled:=true \
  -p formation_recovery_gain:=0.5 \
  -p formation_recovery_max_speed:=0.03 \
  -p formation_recovery_deadband:=0.003 \
  -p debug_log_enabled:=true
```

Run IsaacSim with the external CBF filter enabled:

```bash
python3 standalone_controller_codex.py --external-cbf-enabled --cbf-stale-policy nominal
```

In this architecture, IsaacSim's `--cbf-stale-policy nominal` means "use the
leader velocities directly for the followers if CBF output is stale."

## Stretch Avoidance

The controller enforces stretch-avoidance CBF constraints that prevent the
fabric from being overstretched between its hold-points. The barrier function
for a pair (i, j) is:

```
h_stretch_ij = d_max_ij - ||x_j - x_i||
```

Positive h means the fabric is within the safe distance limit; negative means
it is already overstretched.

### ROS Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stretch_enabled` | bool | `True` | Enable all stretch constraints |
| `stretch_alpha_gain` | float | `1.0` | Class-K gain α(h) = gain × h |
| `stretch_velocity_filter_beta` | float | `0.8` | Low-pass filter for corner velocity estimates (0 = no filter, 0.99 = heavy) |
| `d_max_bl_tl` | float | `0.85` m | Max allowed distance follower-left↔leader-left. Parameter name is legacy. |
| `d_max_br_tr` | float | `0.85` m | Max allowed distance follower-right↔leader-right. Parameter name is legacy. |
| `d_max_bl_br` | float | `0.85` m | Max allowed distance follower-left↔follower-right. Parameter name is legacy. |
| `d_max_bl_tr_enabled` | bool | `False` | Enable follower-left↔leader-right diagonal. Parameter name is legacy. |
| `d_max_bl_tr` | float | `1.2` m | Max allowed distance follower-left↔leader-right. Parameter name is legacy. |
| `d_max_br_tl_enabled` | bool | `False` | Enable follower-right↔leader-left diagonal. Parameter name is legacy. |
| `d_max_br_tl` | float | `1.2` m | Max allowed distance follower-right↔leader-left. Parameter name is legacy. |

### Corner topology

Corner indices on `/fabric/sim/corners` (PoseArray):
- `[0]` = follower-left — controllable
- `[1]` = follower-right — controllable
- `[2]` = leader-left — passive (velocity estimated)
- `[3]` = leader-right — passive (velocity estimated)

Active pairs by default: follower-left↔leader-left,
follower-right↔leader-right, follower-left↔follower-right (two rows).

### Enable/disable pairs at runtime

```bash
# Disable all stretch constraints
ros2 param set /fabric_cbf_controller_node stretch_enabled false

# Enable the left-to-right diagonal
ros2 param set /fabric_cbf_controller_node d_max_bl_tr_enabled true

# Change the left-edge limit
ros2 param set /fabric_cbf_controller_node d_max_bl_tl 0.90
```

### Expected behavior

When the leaders are pulled far apart, the stretch constraints resist and
eventually prevent the followers from
following. When a pair is overstretched (h < 0) the RHS becomes positive,
forcing the followers to converge toward the limit.

### Monitoring

```bash
ros2 topic echo /fabric/cbf/debug
```

Watch `stretch_h_values` approach 0 as the fabric nears its limit and
`stretch_active` flip to `True` when at least one constraint is active.
`stretch_pair_names` lists which pairs are currently monitored and their
order matches `stretch_h_values`, `stretch_distances`, and
`stretch_d_max_values`.

### Running tests

```bash
cd /home/taha/ros2_colcon_ws/src/fabric_sim_isaacsim_newton
python3 -m pytest fabric_cbf_controller/test/test_stretch_cbf.py -v
```

## Notes

The controller filters both obstacle-avoidance and stretch-avoidance CBF
constraints for follower-left and follower-right velocity commands. When both are
active simultaneously the combined QP is solved via iterative weighted
half-plane projection. It does not run internal IsaacSim rollouts and does not
implement proximity CBFs.
