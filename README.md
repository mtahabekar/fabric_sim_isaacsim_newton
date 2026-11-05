# Fabric simulation with a sensitivity control barrier function

Collaborative cloth manipulation in Isaac Sim with a Newton Style3D cloth.
Four grippers hold the corners of a 0.84 m x 0.84 m cloth. Two **leaders**
(a human or a planner) are commanded over ROS 2; the two **followers** are
commanded by a safety filter, the **sensitivity CBF**, that keeps the cloth
5 cm clear of an obstacle while limiting gripper tension and fabric strain.

Every 60 Hz frame the filter measures the full cloth (3,249 particles),
predicts it 0.33 s ahead with a linearized 17 x 17 cloth model together with
the exact sensitivity of every predicted point to the follower velocities, and
solves one small QP: obstacle clearance, gripper tension (20 N) and fabric
strain (2 %) as multi-step discrete-time barrier constraints, with stretch
taking priority over clearance. A typical update takes 13-16 ms on the CPU.
The obstacle (the table) is represented by an enclosing capsule.
[`capsule_cbf/SENSITIVITY_CBF_REPORT.md`](capsule_cbf/SENSITIVITY_CBF_REPORT.md)
has the formulation, validation and results.

```text
keyboard teleop / experiment driver ──> /fabric/leader_{left,right}/cmd_vel
                                              │
Isaac Sim + Newton cloth (fabric_newton_isaacsim.py), every frame:
    measure cloth ─> sensitivity CBF (capsule_cbf) ─> follower commands
    ─> advance cloth with prescribed gripper motion ─> publish state + diagnostics
```

## Repository layout

| Path | Contents |
| --- | --- |
| `fabric_newton_isaacsim.py` | The simulator: Isaac Sim scene, Newton cloth, gripper patches, ROS 2 I/O, sensitivity CBF in the frame loop, headless smoke mode. |
| `capsule_cbf/` | The sensitivity CBF as a pure NumPy/SciPy library, its configuration, headless Newton plant, benchmark, validation scripts, tests, report, results and figures. See [its README](capsule_cbf/README.md). |
| `fabric_sim_bringup/` | ROS 2 package: `full_sim.launch.py` starts the simulator and keyboard teleop from one YAML file. |
| `fabric_teleop/` | ROS 2 package: keyboard teleop for the two leaders. |
| `fabric_sim_experiments/` | ROS 2 package: scripted Isaac Sim experiments (CBF on/off, resolution, leader speed, friction, stiffness), CSV recording and figures. |

## Requirements

- Isaac Sim 6.2 with its bundled Newton 1.6.1 and Warp 1.17 (default path
  `/media/taha/ExtermeLinux/isaacsim6.2/python.sh`; override with the
  `ISAAC_PYTHON_SH` environment variable or `isaac_python:=...`).
- ROS 2 Jazzy and `pynput` for the keyboard teleop.
- For headless tests and benchmarks without Kit: Python 3.12 with
  `newton==1.6.1 warp-lang==1.17.0 numpy scipy` (plus `matplotlib` for
  figures), or Isaac's own interpreter and libraries (see
  [capsule_cbf/README.md](capsule_cbf/README.md#tests-and-tools)). Newton
  older than 1.6 is not supported: its Style3D bending force differs (see
  the Configuration section) and 1.0.x crashes.

## Build and run

This repository lives in `src/` of a colcon workspace.

```bash
cd ~/ros2_colcon_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select fabric_teleop fabric_sim_bringup fabric_sim_experiments
source install/setup.bash

ros2 launch fabric_sim_bringup full_sim.launch.py                # window + keyboard teleop
ros2 launch fabric_sim_bringup full_sim.launch.py headless:=true  # no window
ros2 launch fabric_sim_bringup full_sim.launch.py dry_run:=true   # print the simulator command
./src/fabric_sim_isaacsim_newton/fabric_sim_bringup/scripts/launch_full_sim.sh  # sources ROS for you
```

`--symlink-install` is expected: the launch file finds this repository (the
simulator script and `capsule_cbf/`) through the installed, symlinked config.
Otherwise set `FABRIC_SIM_ROOT` to the repository path.

Keyboard teleop (`pynput`; keep the launch terminal focused). Use `Ctrl+C`
to stop; `q` is a movement key.

| Leader | Keys | Axes |
| --- | --- | --- |
| left | `w`/`s`, `d`/`a`, `e`/`q` | +x/-x, +y/-y, +z/-z |
| right | `i`/`k`, `l`/`j`, `o`/`u` | +x/-x, +y/-y, +z/-z |

For the first 90 frames (1.5 s) the cloth settles while the followers simply
track the leaders; the filter then takes over (`warmup_frames`).

### Headless smoke run (no ROS launch needed)

```bash
cd src/fabric_sim_isaacsim_newton
/media/taha/ExtermeLinux/isaacsim6.2/python.sh fabric_newton_isaacsim.py --headless \
    --cloth-start-z 0.85 --smoke-frames 260 --smoke-leader-velocity 0 0 -0.1 --smoke-output /tmp/smoke
```

Leaders hold still for 60 frames, then descend at 0.1 m/s; the run ends with
gripper and ground checks and writes `diagnostics.json` (clearance, tension,
status counts, CBF timing), `scene.usda` and, when a viewport exists,
`viewport.png`. `--help` lists every simulator flag.

## Configuration

`fabric_sim_bringup/config/full_sim.yaml` is the single place to configure a
run: simulator flags (cloth size, resolution, `areal_density`, material,
contact, solver iterations, `headless`), the CBF settings file
(`capsule_cbf/config.json`) with optional per-launch `overrides`
(e.g. `{controller: none}` for an unfiltered run), and teleop parameters.
Every CBF setting is documented in `capsule_cbf/settings.py`. The capsule is
built from the table half extents, so obstacle and filter always agree.

Newton 1.6's Style3D solver ignores `tri_ka`, `tri_kd`, `edge_kd` and
`contact_radius`; they are kept as model parameters only. `contact_margin`
is passed to Newton as `soft_contact_gap` (detection distance on top of the
particle radius).

Newton 1.6 changed Style3D bending to the Bergou et al. (2006) quadratic
model, force `-(3 k / A) w wᵀ x`, which is linear in `edge_ke`; 1.2.1 applied
`(k / A)² w wᵀ` with a rest area 2/3 as large. With the default `edge_ke` on
the 56x56 grid, bending is therefore about 7x softer than in runs made with
Isaac Sim 6.0 / Newton 1.2.1. Stretch, contact and gravity are unchanged.

## ROS 2 interface

| Topic | Type | Direction |
| --- | --- | --- |
| `/fabric/leader_left/cmd_vel`, `/fabric/leader_right/cmd_vel` | `geometry_msgs/TwistStamped` | in: leader velocities (zero after `cmd_timeout`) |
| `/fabric/follower_left/applied_cmd_vel`, `/fabric/follower_right/applied_cmd_vel` | `geometry_msgs/TwistStamped` | out: follower commands applied this frame |
| `/robot_left/ee_cmd_vel`, `/robot_right/ee_cmd_vel` | `geometry_msgs/TwistStamped` | out: the same follower commands for a robot bridge |
| `/fabric/sim/corners` | `geometry_msgs/PoseArray` | out: corners in order follower_left, follower_right, leader_left, leader_right |
| `/fabric/sim/state` | `sensor_msgs/PointCloud2` | out: every cloth particle, fields `x y z vx vy vz` (every `state_publish_every` frames) |
| `/fabric/sim/timing` | `std_msgs/Float64MultiArray` | out: `[sim_time_s, frame_ms, fps, cbf_ms]` |
| `/fabric/capsule_cbf/debug` | `std_msgs/String` (JSON) | out: per-frame CBF diagnostics |

The debug JSON includes `status` (`warmup`, `feasible`, `relaxed_soft`
= clearance rows relaxed, `relaxed_hard` = stretch rows relaxed,
`solver_failure`, `unfiltered`), `min_clearance` (m to the capsule),
`h_min` (clearance minus margin), `tension` (N per gripper), `strain_ks`,
`candidate_count`, `correction` (m/s from nominal), `controller_ms` and the
nominal and applied follower velocities.

```bash
ros2 topic echo /fabric/capsule_cbf/debug --full-length
ros2 topic pub -r 30 /fabric/leader_left/cmd_vel geometry_msgs/TwistStamped "{twist: {linear: {z: -0.05}}}"
```

In the viewport: blue wireframe = capsule envelope (visual only, not a
collision shape); orange points and segments = cloth points with active
clearance rows and their closest capsule points (5 Hz); blue/green spheres =
follower/leader grippers.

## Tests, benchmark, experiments

```bash
# Unit tests (46; Newton-dependent ones skip without Newton):
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m unittest discover -s capsule_cbf/tests -t .
# Launch/config tests (ROS sourced):
python3 -m pytest fabric_sim_bringup/test
# Closed-loop benchmark on the headless Newton plant, with and without the filter:
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 python -m capsule_cbf.benchmark --output /tmp/bench
# Isaac Sim experiments (see fabric_sim_experiments/README.md):
ros2 run fabric_sim_experiments batch_runner --ros-args -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp1_cbf_on_off.yaml
```

## History

The repository previously contained an external ROS 2 CBF node with a
geometric Jacobian (`fabric_cbf_controller`), a table-distance node
(`fabric_obstacle_distance`), and the earlier capsule filters (bilinear GID
basins, Newton-rollout predictor with ray basins and solver-tangent
derivatives). They were removed when the sensitivity CBF became the method;
all of them remain in the git history (commit `67c3efb`).
