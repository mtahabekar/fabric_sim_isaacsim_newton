# Capsule interaction CBF experiment

This is a separate, experimental run path. No existing simulator, controller,
message, or launch source is modified. The new model-construction mixin removes
the legacy `_precompute(builder)` call because current Newton precomputes in the
solver constructor from the finalized model. The existing full-stack launcher still
runs the original method.

## Run

From the repository root, using the existing built ROS workspace:

```bash
python3 capsule_cbf/launch_capsule_sim.py
```

The default runtime is `/media/taha/ExtermeLinux/isaacsim/python.sh`. Override it
with `--isaac-python` or `ISAAC_PYTHON_SH`. The runtime must contain Newton Style3D,
NumPy, SciPy, Isaac Sim and its ROS bridge, as required by the original simulator
plus the new SciPy optimizer. The host launcher needs PyYAML. The original
`full_sim.yaml` retains its previous runtime path; the new launcher overrides it
with the location supplied for this experiment.

The launcher copies the original YAML in memory, selects
`fabric_newton_isaacsim_capsule.py`, disables the old distance and CBF nodes,
retains keyboard teleop, and invokes the existing bringup with a temporary YAML.
It sets the new experiment's initial cloth height to 1.05 m (override with
`--cloth-start-z`) to allow startup sag above the larger capsule. It does not
edit the original config. No new ROS package/message build is needed
if the existing bringup and teleop packages are already installed.

```bash
# Inspect the complete resolved launch configuration without starting anything:
python3 capsule_cbf/launch_capsule_sim.py --dry-run
# Use another original simulator configuration and/or capsule tuning:
python3 capsule_cbf/launch_capsule_sim.py --base-config path/to/full_sim.yaml \
  --capsule-config capsule_cbf/config.json --isaac-python /path/to/python.sh
# Direct simulator entry point (teleop must be started separately):
/path/to/python.sh fabric_newton_isaacsim_capsule.py --capsule-config capsule_cbf/config.json
# Bounded simulator validation (source ROS and workspace first):
source /opt/ros/jazzy/setup.bash
source ../../install/setup.bash
/media/taha/ExtermeLinux/isaacsim/python.sh fabric_newton_isaacsim_capsule.py \
  --capsule-smoke-frames 120 --capsule-smoke-output /tmp/capsule-smoke
# Simulation-only approach test: 60 settling frames, then downward leader motion:
/media/taha/ExtermeLinux/isaacsim/python.sh fabric_newton_isaacsim_capsule.py \
  --capsule-smoke-frames 360 --capsule-smoke-leader-z -0.05 \
  --capsule-smoke-output /tmp/capsule-driven
# CPU checks:
python3 -m unittest discover -s capsule_cbf/tests -v
```

The new filter runs synchronously each simulation frame, using current particles
and the same leader commands applied that frame. This avoids building a new
multi-point ROS message and consuming stale distance/command pairs during the
first experiment. Leader teleop, applied follower velocity topics, cloth state,
corner state, and simulator timing retain their existing interfaces. External
CBF command subscriptions are not used by this entry point.

Diagnostics are JSON in `std_msgs/String` on `/fabric/capsule_cbf/debug`:

```bash
ros2 topic echo /fabric/capsule_cbf/debug
ros2 topic echo /fabric/follower_left/applied_cmd_vel
```

They include minimum capsule clearance, minimum barrier, occupied cells, active
basin rows, total rows, correction magnitude, solver status and maximum inequality
violation. `recovery_not_safe` means the simultaneous constraints could not be
satisfied or the solver failed: the output is a speed-bounded least-squares
violation recovery, NOT a safe command. Leaders remain under user control; a
follower-only filter cannot prevent a leader from driving its pinned patch into
the obstacle. Logs summarize every 60 simulation frames.

## Paper and adaptation

Reference: Yiqing Fan and John T. Wen, *Perception-Driven Control Barrier Functions
via Geometric Interaction Decomposition*, supplied PDF, Sections II-III and V.
The paper protects a moving manipulator capsule against scene points. Here the
capsule is fixed around the obstacle, and the fabric points move. The paper's
atlas/local-minimum seeds, distance-persistent basin supports, Gaussian local
row averaging, and multi-row filter inspire this implementation. It is not an
exact reproduction or a proof of cloth safety. In particular, the paper itself
states that averaged rows are not exact derivatives of its minimum barriers.

For the current box, capsule center is `(0, 0, table_hz)`, the centerline extends
from `z=0` to `z=2*table_hz`, and radius is `sqrt(table_hx^2 + table_hy^2)`.
Every box point lies inside or on this capsule. Defaults give radius 0.311127 m
and top z=0.631127 m, compared with the box top z=0.32 m. Particle radius 0.008 m
and margin 0.05 m are subtracted separately: the capsule itself is not inflated
again. The original flat cloth at z=0.72 has minimum barrier about 0.03087 m.
A GPU smoke run confirmed that startup sag crosses this boundary within ten
frames; the followers saturate and report `recovery_not_safe`. The new entry
point and launcher therefore default to z=1.05, rather than claiming that the
original spawn is a viable safe initial state for this enclosing capsule.
The substantial extra volume is the cost of this enclosing vertical capsule.
A rounded-box atlas or a union of smaller enclosing primitives would be a less
conservative follow-up if this capsule blocks useful motions.

For fabric point p, let a be its closest point on the capsule centerline:

```
d(p) = norm(p-a) - capsule_radius - particle_radius
h(p) = d(p) - margin
n(p) = (p-a) / norm(p-a)
v(p) ~= w_FL*u_FL + w_FR*u_FR + w_LL*v_LL + w_LR*v_LR + residual
A(p) = [w_FL*n(p)^T, w_FR*n(p)^T]
A(p)*u >= -alpha*h(p) - n(p)^T*(w_LL*v_LL + w_LR*v_LR + residual)
```

Material-coordinate bilinear weights stay attached to particle identities,
rather than changing as the fabric folds. Hard-pinned patch weights are replaced
with exact one-hot corner influence. The residual is the latest finite-difference
particle velocity minus the previous four-corner velocity prediction. It accounts
approximately for sagging and ongoing dynamics, but is not a Newton sensitivity
Jacobian and is not bounded robustly. At startup it is zero.

Each particle maps to the nearest cell center at its analytic capsule projection.
The fixed atlas uses azimuth and meridian arc length over both caps and cylinder;
the seam is periodic. Every occupied cell retains its minimum signed-clearance
particle. Empty cells are invisible. Seeds use first-visible directional searches;
support expansion and connected components use distance persistence. Basin
barriers use the closest seed; local Gaussian row AND drift averages produce
one active row per basin. All retained samples participate in decomposition;
constraints activate when h <= activation (default 0.12 m).

The default `cell_guards: true` also constrains every active cell representative.
This is a deliberate revision of the paper to check a safer baseline before
relying exclusively on smoothed rows with an approximate cloth model. Set false
only to compare the compact basin-only experiment. Even guards omit non-retained
particles in each cell, and particle sampling does not cover triangle interiors.
It is still possible to miss a cloth face crossing, or a different point with a
more adverse velocity in the same cell. Analytic clearance is monitored over ALL
particles regardless of the active constraint set.

The filter minimizes squared change from leader tracking plus initial-offset
formation recovery, subject to simultaneous obstacle rows, two leader-follower
stretch rows, one coupled follower-follower stretch row, and Euclidean follower
speed bounds. Stretch limits are initial edge lengths times `stretch_ratio`.
SLSQP solves the convex problem; feasibility is checked after numerical speed
limiting. There is no post-filter smoothing or ordinary stale-command fallback
that would overwrite a computed correction.

## Visualization

`/World/CapsuleCBF` contains USD geometry only. It is never added to Newton's
collision model and has no physical collision schema:

- Blue wireframe: enclosing capsule with meridian/ring boundaries.
- Blue dots: fixed capsule cell centers.
- Orange dots: retained fabric detections within the activation range.
- Orange segments: analytic capsule surface to those fabric detections.
- Red dots: closest seed for each active basin.

Markers correspond to the pre-step geometry used to calculate that frame's
command, so they can lag the rendered post-step cloth by one simulation step.
The lower capsule cap extends below the ground and can be occluded by it.

## Implementation and validation checklist

- [x] Trace current launch -> simulator -> distance node -> CBF -> follower path.
- [x] Read paper geometry, basin construction, filtering, and stated limitations.
- [x] Preserve original source and provide a separate simulator/launch path.
- [x] Build an enclosing nonphysical capsule and fixed surface atlas.
- [x] Compute signed distances, normals, occupied-cell samples, and basin seeds.
- [x] Replace a single 1x6 avoidance row with Kx6 rows and leader/drift terms.
- [x] Add simultaneous cell guards, stretch constraints, and speed bounds.
- [x] Apply filtered commands to follower targets within the simulation step.
- [x] Add capsule, cells, detections, seed, and distance-segment USD visuals.
- [x] Add residual/status/correction diagnostics and bounded infeasible recovery.
- [x] Test enclosure, analytic gradients, material Jacobian derivative, atlas seam,
      sparse cells, independent interactions, multiple-row feasibility, infeasible
      recovery, actual command correction, and an idealized closed-loop plant.
- [x] Run Isaac Sim and visually inspect the capsule, cells, and moving detections.
      A 10-frame GPU run with CUDA graphs disabled completed and its viewport
      was inspected. It also exposed the original-height startup violation.
- [x] Validate revised-height settling with normal CUDA graph capture: 120 GPU
      frames, minimum observed capsule clearance 0.20634 m, zero recovery frames.
- [x] Run a driven GPU approach: 360 frames, 60 settling frames then both leaders
      descending at 0.05 m/s. Observed minimum capsule clearance was 0.05562 m
      against the 0.05 m target; peak command correction norm was 0.40683 m/s.
      17 frames used bounded recovery. The final frame was feasible with 70 rows.
      These recoveries are a material limitation despite the positive clearance.
- [ ] Compare legacy, basin-only, and guarded runs on identical leader trajectories:
      actual box and capsule clearance, violated rows, correction, command variation,
      stretch, solver latency, and fallback frequency.
- [ ] Validate the cloth response Jacobian against small follower perturbations in
      Newton, including sagging, folds, and contact. Replace bilinear influence with
      identified or differentiated sensitivities if prediction error is excessive.
- [ ] Evaluate face-interior coverage, per-cell omitted samples, atlas resolution,
      temporal basin persistence, and robust/discrete-time margins before treating
      this as a reliable safety mechanism.

The synthetic closed-loop test is an ideal bilinear velocity plant, not the
Newton solver. The GPU smoke test checks Isaac integration and rendering separately. Broad
closed-loop safety and real-time performance are not established by these tests.

Validation artifacts from this session: `/tmp/capsule-smoke-high/` (settling)
and `/tmp/capsule-smoke-driven/` (descent), each with `viewport.png`,
`capsule_scene.usda`, and `diagnostics.json`. Observed run minima use the particle
states sampled for control; they are not continuous-time clearance certificates.
