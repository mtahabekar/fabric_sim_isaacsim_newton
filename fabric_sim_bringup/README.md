# fabric_sim_bringup

Bringup package for launching the full IsaacSim fabric CBF simulation stack from
one command.

## What It Starts

- IsaacSim standalone simulation:
  `/home/taha/ros2_colcon_ws/src/fabric_sim_isaacsim_newton/fabric_newton_isaacsim_standalone.py`
- `fabric_obstacle_distance`:
  `obstacle_distance_node.py`
- `fabric_cbf_controller`:
  `fabric_cbf_controller_node.py`
- `fabric_teleop`:
  `fabric_corner_keyboard_teleop`

The launch file starts IsaacSim first, then starts the ROS nodes after the
delays configured in `config/full_sim.yaml`.

## Configure

Edit:

```bash
/home/taha/ros2_colcon_ws/src/fabric_sim_isaacsim_newton/fabric_sim_bringup/config/full_sim.yaml
```

The `isaac_sim.arguments` section maps directly to simulator command-line
arguments. For example:

```yaml
external_cbf_enabled: true
cbf_stale_policy: nominal
```

becomes:

```bash
--external-cbf-enabled --cbf-stale-policy nominal
```

The `obstacle_distance.parameters`, `cbf_controller.parameters`, and
`teleop.parameters` sections are passed as ROS 2 node parameters.

## Build

```bash
cd /home/taha/ros2_colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run

Either run through ROS 2 launch:

```bash
ros2 launch fabric_sim_bringup full_sim.launch.py
```

or use the helper script, which sources ROS and the workspace for you:

```bash
/home/taha/ros2_colcon_ws/src/fabric_sim_isaacsim_newton/fabric_sim_bringup/scripts/launch_full_sim.sh
```

Use `Ctrl+C` in the launch terminal to stop the whole stack.
