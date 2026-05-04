# fabric_sim_experiments

Reproducible experiment suite for the IsaacSim + Newton fabric CBF stack.

Inspired by Aksoy & Wen, "Collaborative Manipulation of Deformable Objects with
Predictive Obstacle Avoidance," ICRA 2024, adapted to the 2-leader / 2-follower
setup with collision **and** stretch CBFs.

---

## Prerequisites

```bash
cd /home/taha/ros2_colcon_ws
colcon build --symlink-install --packages-select fabric_sim_experiments
source install/setup.bash
```

---

## Package layout

```
fabric_sim_experiments/
├── fabric_sim_experiments/
│   ├── data_recorder_node.py    # records to CSVs during a live run
│   ├── leader_driver_node.py    # programmatic leader velocity driver
│   ├── experiment_runner.py     # orchestrates one run end-to-end
│   ├── batch_runner.py          # sweeps over a config YAML
│   └── plotting/
│       ├── plot_trajectories_3d.py    # Exp 1 figures
│       ├── plot_min_distance.py       # standalone distance plotter
│       ├── plot_resolution_timing.py  # Exp 2 performance figure
│       └── plot_leader_speed_sweep.py # Exp 3 distance figure
├── config/
│   ├── exp1_cbf_on_off.yaml
│   ├── exp2_resolution_sweep.yaml
│   ├── exp3_leader_speed_sweep.yaml
│   ├── exp4_friction_sweep.yaml
│   └── exp5_stiffness_sweep.yaml
└── launch/
    └── single_run.launch.py
```

---

## Running experiments

### Quick sanity check — single cell, CBF on

```bash
ros2 run fabric_sim_experiments experiment_runner \
    --ros-args \
    -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp1_cbf_on_off.yaml
```

Produces CSVs under `runs/exp1_cbf_on_off/<run_id>/`.
Figures are generated automatically under `runs/exp1_cbf_on_off/figures/` and
symlinked into `runs/report_figures/`. With only the first sanity cell, the Exp.
1 trajectory plots contain the CBF-on run only; run the CBF-off cell or the full
batch to get solid-vs-dashed overlays.
`experiment_runner` intentionally runs one sweep cell by default; use
`batch_runner` for the full sweep, or pass `-p run_all_cells:=true` only when
you want all cells from `experiment_runner`.

### Experiment 1 — CBF on vs off, full sweep (12 cells)

```bash
ros2 run fabric_sim_experiments batch_runner \
    --ros-args -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp1_cbf_on_off.yaml
```

After the sweep finishes, final overlay figures are regenerated automatically.

### Experiment 2 — Resolution sweep (headless, 5 repeats × 6 resolutions)

```bash
ros2 run fabric_sim_experiments batch_runner \
    --ros-args \
    -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp2_resolution_sweep.yaml \
    -p headless:=true
```

### Experiment 3 — Leader-speed sweep (headless)

```bash
ros2 run fabric_sim_experiments batch_runner \
    --ros-args \
    -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp3_leader_speed_sweep.yaml \
    -p headless:=true
```

### Experiment 4 — Friction sweep (windowed, for manual screenshots)

```bash
ros2 run fabric_sim_experiments batch_runner \
    --ros-args -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp4_friction_sweep.yaml
```

The runner prints a `===` banner at the start and end of each `contact_mu` value.
Take IsaacSim viewport screenshots at t = 0 %, 25 %, 50 %, 75 %, 100 % of motion.

### Experiment 5 — Stiffness sweep (windowed)

```bash
ros2 run fabric_sim_experiments batch_runner \
    --ros-args -p config:=$(ros2 pkg prefix fabric_sim_experiments)/share/fabric_sim_experiments/config/exp5_stiffness_sweep.yaml
```

---

## Generating figures from CSVs (no simulator required)

### Experiment 1 — 3D trajectories and min-distance

```bash
python3 -m fabric_sim_experiments.plotting.plot_trajectories_3d runs/exp1_cbf_on_off
```

Figures are written to `runs/exp1_cbf_on_off/figures/` and symlinked into
`runs/report_figures/`.
The plotters skip incomplete/interrupted run directories.

Key outputs per (z, x) cell:
- `fig_traj3d_*.{png,pdf,svg}` — overlaid solid (CBF on) vs dashed (CBF off) leader/follower trajectories with translucent table box
- `fig_min_distance_*.{png,pdf,svg}` — min obstacle distance vs time with safety-margin reference line

### Experiment 2 — Resolution timing

```bash
python3 -m fabric_sim_experiments.plotting.plot_resolution_timing runs/exp2_resolution_sweep
```

Outputs:
- `fig_simulation_performance_vs_cloth_resolution.*` — simulation FPS and simulation step time vs cloth particle count

### Experiment 3 — Speed sweep

```bash
python3 -m fabric_sim_experiments.plotting.plot_leader_speed_sweep runs/exp3_leader_speed_sweep
```

Outputs:
- `fig_speed_min_distance_traces.*` — minimum obstacle distance vs time for different leader speeds

---

## CSV column reference

| File | Key columns |
|---|---|
| `corners.csv` | `t, follower_left_x/y/z, follower_right_x/y/z, leader_left_x/y/z, leader_right_x/y/z` |
| `obstacle_distance.csv` | `t, min_distance, h, closest_idx, normal_x/y/z` |
| `cbf_debug.csv` | `t, mode, solver_status, h, min_distance, compute_time_ms, stretch_pair_names (pipe-sep), stretch_h_values (pipe-sep), ...` |
| `sim_perf.csv` | `t, sim_step_ms, sim_fps` |
| `cmd_vels.csv` | `t, role, topic, vx, vy, vz` |
| `meta.json` | full run config (sweep params, CBF on/off, timestamps) |

**Corner ordering in `corners.csv`** matches the IsaacSim node's canonical order:
index 0 = follower_left, 1 = follower_right, 2 = leader_left, 3 = leader_right.
Older run files with `BL/BR/TL/TR` headers are still accepted by the plotters.

**`h` in `obstacle_distance.csv`** = `min_distance − safety_margin` (positive = safe).
**`stretch_h_values`** = `d_max − current_dist` for each active pair; h < 0 means violation.
The safety margin reference line in all figures uses **0.05 m** by default (pass `--safety-margin` to override).

---

## Reproducing the ICRA 2024 analogues

| Paper figure | This package |
|---|---|
| Fig. 8–9 (fabric trajectories) | `plot_trajectories_3d` — 3D corner trajectory and min-distance figures |
| Resolution performance figure | `plot_resolution_timing` → `fig_simulation_performance_vs_cloth_resolution.*` |

---

## Notes

- All spawned processes (IsaacSim, nodes) are SIGTERM'd then SIGKILL'd after each run; no orphaned instances.
- Run IDs have the form `YYYYMMDD_HHMMSS_<sha1[:6]>`; reruns never overwrite previous results.
- Exp 2 & 3 should be run with `-p headless:=true` for clean timing data (no render overhead).
- Exp 4 & 5 are intentionally windowed — take IsaacSim viewport screenshots manually.
