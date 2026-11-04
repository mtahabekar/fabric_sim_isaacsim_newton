# fabric_sim_bringup

One-command launch of the Isaac Sim fabric simulation with the sensitivity CBF
(`fabric_newton_isaacsim.py` in the repository root) and the keyboard teleop.

```bash
ros2 launch fabric_sim_bringup full_sim.launch.py                       # window + teleop
ros2 launch fabric_sim_bringup full_sim.launch.py headless:=true        # no window
ros2 launch fabric_sim_bringup full_sim.launch.py isaac_python:=/path/to/isaacsim/python.sh
ros2 launch fabric_sim_bringup full_sim.launch.py config_file:=/path/to/my_sim.yaml
ros2 launch fabric_sim_bringup full_sim.launch.py dry_run:=true         # print the command, start nothing
scripts/launch_full_sim.sh [my_sim.yaml] [launch arguments]             # sources ROS 2 and the workspace first
```

`Ctrl+C` stops everything; closing the Isaac window also stops the teleop
(`shutdown_on_isaac_exit`).

## config/full_sim.yaml

| Section | Meaning |
| --- | --- |
| `launch` | `start_isaac`, `start_teleop`, `teleop_start_delay`, `shutdown_on_isaac_exit` |
| `isaac_sim` | `python_sh` (Isaac Sim's python.sh), `script`, optional `working_directory`, `arguments` (simulator flags), `extra_args` |
| `cbf` | `config`: CBF settings JSON (default `capsule_cbf/config.json`); `overrides`: per-launch changes, e.g. `{controller: none}` |
| `teleop` | `parameters` of `fabric_corner_keyboard_teleop` |

`arguments` keys become flags (`cloth_start_z: 0.85` -> `--cloth-start-z 0.85`;
`headless: true` -> `--headless`). `python.sh fabric_newton_isaacsim.py --help`
lists them all. The Isaac interpreter is chosen by the `isaac_python` launch
argument, then `ISAAC_PYTHON_SH`, then `isaac_sim.python_sh`. With overrides,
the merged CBF JSON is written to `/tmp/fabric_sim_launch_<pid>/cbf_config.json`.

Relative paths are resolved against the repository root, found from the real
path of the config file (or `FABRIC_SIM_ROOT`), so build with
`colcon build --symlink-install`; with it, YAML edits need no rebuild.

`fabric_sim_bringup.sim_command` builds the simulator command and is shared
with `fabric_sim_experiments`. Tests: `python3 -m pytest test` (ROS sourced).
