# fabric_sim_experiments

Scripted Isaac Sim experiments with the sensitivity CBF in the loop: each run
starts the simulator, waits for the CBF warm-up, records ROS 2 topics to CSV
while a leader driver moves both leaders, shuts everything down and plots.
Inspired by Aksoy and Wen, "Collaborative Manipulation of Deformable Objects
with Predictive Obstacle Avoidance", ICRA 2024 (Fig. 8-9, Table I analogues).

```bash
cd ~/ros2_colcon_ws && colcon build --symlink-install --packages-select fabric_sim_bringup fabric_sim_experiments
source install/setup.bash
CFG=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config

ros2 run fabric_sim_experiments experiment_runner --ros-args -p config:=$CFG/exp1_cbf_on_off.yaml            # cell 0
ros2 run fabric_sim_experiments experiment_runner --ros-args -p config:=$CFG/exp1_cbf_on_off.yaml -p run_index:=1
ros2 run fabric_sim_experiments batch_runner --ros-args -p config:=$CFG/exp1_cbf_on_off.yaml -p headless:=true
ros2 launch fabric_sim_experiments single_run.launch.py config:=$CFG/exp3_leader_speed_sweep.yaml run_all_cells:=true
```

Parameters: `config`, `runs_root` (default `runs`, relative to the current
directory), `headless`, `settle_time` (s recorded after the motion),
`generate_figures`; `experiment_runner` also takes `run_index` / `run_all_cells`.

## Experiments

| Config | Question | Scenario | CBF |
| --- | --- | --- | --- |
| `exp1_cbf_on_off` | trajectories and clearance with vs without the filter | cloth at 0.85 m, leaders descend 0.33 m at 0.1 m/s | on / off |
| `exp2_resolution_sweep` | simulation rate and CBF cost vs cloth resolution (20x20 to 128x128) | as Exp. 1 | on |
| `exp3_leader_speed_sweep` | clearance vs leader speed (0.05-0.40 m/s) | as Exp. 1 | on, one off baseline |
| `exp4_friction_sweep` | contact friction (screenshots) | cloth beside the table, dragged across it at 0.05 m/s | off |
| `exp5_stiffness_sweep` | stretch/bending stiffness (screenshots) | as Exp. 4 | off |

"CBF off" is `controller: none`: the followers receive the same nominal
command unfiltered, and clearance, tension and strain are still recorded.
Exp. 4-5 drag the cloth over the table, i.e. into the capsule envelope, so
they run unfiltered. Validated range: the filter is validated at 0.1 m/s on
the 56x56 cloth; at 0.2-0.4 m/s and at 20x20 the stretch limits become
infeasible and clearance yields (see `capsule_cbf/README.md`). Exp. 2's
coarse cells and Exp. 3's fast cells therefore show these limits.

## Experiment YAML

```yaml
exp_name: exp1_cbf_on_off
n_repeats: 1
base_config: fabric_sim_bringup/config/full_sim.yaml   # simulator + CBF defaults
isaac_args: {cloth_start_z: 0.85}                      # simulator flag overrides, every cell
cbf_overrides: {}                                      # CBF JSON overrides, every cell
sweep:                                                 # one entry per cell
  - scenario: descend                                  # tags read by the plotters
    cbf_on: true
    leader_velocity: [0.0, 0.0, -0.10]                 # m/s, both leaders
    leader_motion_distance: 0.33                       # m along the velocity, from /fabric/sim/corners
    leader_start_delay: 1.0                            # s after the warm-up
    isaac_args: {}                                     # per-cell simulator overrides
    cbf_params: {controller: sensitivity}              # per-cell CBF overrides
    banner: "..."                                      # optional: printed for manual screenshots
```

## Outputs (`<runs_root>/<exp_name>/<run_id>/`)

| File | Contents |
| --- | --- |
| `meta.json` | run parameters, exact simulator command, leader driver parameters |
| `cbf_config.json` | the merged CBF settings the simulator used |
| `corners.csv` | `t`, `<corner>_x/y/z` for follower_left, follower_right, leader_left, leader_right |
| `cbf_debug.csv` | per frame: `sim_time, status, mode, warmup, min_clearance, h_min, candidate_count, total_rows, correction, slack_soft, slack_hard, controller_ms, max_tension, tension_<corner>, max_strain_ks`, nominal and applied follower velocities |
| `sim_timing.csv` | `sim_time, frame_ms, fps, cbf_ms` |
| `cmd_vels.csv` | leader commands and applied follower commands |

`batch_runner` adds `<exp_name>/summary.csv` (minimum clearance and barrier,
frames inside the margin and touching the capsule, relaxed frames, max
tension, CBF time, FPS, max corner heights). Figures (png/pdf/svg) go to
`<exp_name>/figures/` and are symlinked into `<runs_root>/report_figures/`:

```bash
python3 -m fabric_sim_experiments.plotting.plot_trajectories_3d runs/exp1_cbf_on_off   # fig_traj3d_*, fig_min_distance_*
python3 -m fabric_sim_experiments.plotting.plot_resolution_timing runs/exp2_resolution_sweep
python3 -m fabric_sim_experiments.plotting.plot_leader_speed_sweep runs/exp3_leader_speed_sweep
python3 -m fabric_sim_experiments.plotting.plot_min_distance runs/<exp>/<run_id>
```

The runner refuses to start while another simulator publishes
`/fabric/sim/corners`, and stops every child process group (SIGINT, then
SIGTERM, then SIGKILL) after each run.
