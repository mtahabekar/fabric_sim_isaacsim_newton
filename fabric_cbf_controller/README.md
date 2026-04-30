# fabric_cbf_controller

ROS2 Python CBF controller for the IsaacSim fabric manipulation setup.

This package listens to top-corner CBF input velocity commands from IsaacSim,
adds a slow formation recovery term to produce nominal bottom-corner commands,
filters them through an obstacle-avoidance control barrier function, and
publishes final bottom-corner velocity commands back to IsaacSim. It also
publishes a debug message with the active mode, barrier values, approximate
Jacobian, nominal/safe velocities, and solver status.

## Architecture

- IsaacSim receives teleop commands on:
  - `/fabric/top_left/cmd_vel`
  - `/fabric/top_right/cmd_vel`
- IsaacSim republishes those top velocities as CBF inputs on:
  - `/fabric/top_left/cbf_input_cmd_vel`
  - `/fabric/top_right/cbf_input_cmd_vel`
- The CBF node computes nominal bottom-corner commands from each top input
  velocity plus a slow formation recovery term:
  - top-left input + recovery -> desired bottom-left velocity
  - top-right input + recovery -> desired bottom-right velocity
- IsaacSim publishes fabric corner poses on `/fabric/sim/corners`.
- `fabric_obstacle_distance` publishes obstacle distance information on
  `/fabric/obstacle_distance`.
- `fabric_cbf_controller` publishes final bottom-corner commands on:
  - `/fabric/bottom_left/cbf_cmd_vel`
  - `/fabric/bottom_right/cbf_cmd_vel`
- Debug output is published on `/fabric/cbf/debug`.

The formation recovery term tries to restore the initial bottom-to-top offsets
captured from the first valid `/fabric/sim/corners` message. When the CBF is
inactive or the barrier is already satisfied, the controller publishes the
recovered nominal velocity. When the CBF is active, it modifies that recovered
nominal velocity to satisfy the obstacle CBF constraint as much as possible.

The current Jacobian estimator is a lightweight geometric approximation:

```text
J = [w_left * n, w_right * n]
```

where `n` is the obstacle-to-fabric normal and `w_left`, `w_right` approximate
how much each bottom corner influences the closest particle. This is modular so
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
top velocities directly for the bottom corners if CBF output is stale."

## Notes

This first implementation only filters obstacle-avoidance CBF constraints for
bottom-left and bottom-right velocity commands. It does not run internal
IsaacSim rollouts, it does not implement overstretching CBFs, and it does not
implement proximity CBFs.
