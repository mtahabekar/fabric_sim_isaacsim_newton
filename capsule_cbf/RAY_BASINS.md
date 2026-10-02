# Ray-selected capsule basins

## What changed and why

The previous yellow markers were all cloth particles with clearance below
`margin + activation` (17 cm). The full predictor could add clearance rows for
those particles at every substep, alongside four strain barriers per triangle.
Strain recovery can reduce tension by lowering a follower, even with stationary
leaders. The formation feedback also previously corrected vertical offsets.

The defaults now use ray detection and one representative per basin.
`strain_enabled: false` removes strain rows **and triangle strain observation
kernels**. Newton's actual material forces remain: disabling a controller policy
constraint does not change the cloth constitutive model. The old corner stretch
proxy is also gated by this setting in bilinear comparison mode.
`vertical_formation_recovery: false` removes vertical formation feedback.
Followers track leader vertical velocity and retain planar formation feedback;
obstacle recovery can still modify all six follower velocity components.
`max_speed: 0` retains unlimited absolute follower commands. Linearization trust
bounds restrict each optimizer step's change in displacement, not absolute speed.

## 1. Nearest hit for each capsule cell

Let c be the capsule center and s_j its j-th surface cell center. Cast

r_j(t) = c + t (s_j-c)/||s_j-c||, t >= 0.

The live cloth triangle mesh owns a Warp BVH. Each frame we refit the BVH to the
canonical live position buffer and cast all rays on the GPU. The first positive
intersection is retained. Rays missing the cloth yield no detection; multiple
cloth layers along a ray yield only the nearest layer. This is a ray intersection,
not an approximate nearest particle lookup. The simulator's obstacle geometry
is excluded from the query, since the origin is inside the enclosing obstacle.

For hit p, project to the capsule axis segment:

a(p) = (c_x, c_y, clamp(p_z, c_z-L, c_z+L)),
d(p) = ||p-a(p)|| - R - r_particle.

Retain only d(p) <= margin, including penetrations. The 5 cm margin measures
clearance beyond the capsule and the cloth collision radius. `activation: 0.12`
is not used in ray mode. Distance filtering uses true capsule clearance, not the
length along an oblique ray. Yellow markers and segments show these retained
hits and their nearest capsule surface points.

## 2. k-nearest-neighbor grouping

Build a CPU cKDTree over the retained hit positions, at most 432 samples.
For each sample, inspect its four nearest other neighbors. Connect a neighbor
only within `basin_neighbor_radius: 0.08` meters; take connected components of
this undirected graph. The radius prevents distant patches from being joined
merely to fill the requested neighbor count. Both settings are configurable.

Each component retains the sample with minimum signed clearance. We do not
average positions or constraint rows. A connected component can form a long
chain, so its other samples are not necessarily all within 8 cm of the chosen
representative. Red markers show representatives; only these supply barriers.

## 3. Track a material location through the prediction

A representative stores its triangle face and barycentric weights (w0,w1,w2).
In each isolated Newton substep, observe

p_b(q) = w0 q_i0 + w1 q_i1 + w2 q_i2.

The support is frozen throughout the six finite differences, optimization and
nonlinear checks in that control frame. Recasting a different ray independently
for each perturbation would change which material location is being
differentiated, introducing discontinuities unrelated to that location's motion.
Selection is repeated from the live mesh on the next control frame.

Predictions still solve the entire coupled cloth. Only observing a few material
locations does not make it valid to simulate those locations independently:
neighbors transmit forces, and the implicit solver couples the mesh. Predictor
states, contact buffers, collision pipeline and solver workspaces remain isolated
from the live simulation. GPU observation storage has fixed atlas-size capacity;
a device-side count changes active supports without recapturing CUDA graphs.
All-particle clearance is available as a diagnostic, not an optimizer constraint.

## 4. Exactly one barrier per basin

For current representative barrier h_b = d(p_b)-margin and substep time t_s,
define a tightened discrete CBF target

T_sb = exp(-alpha t_s) h_b + (1-exp(-alpha t_s)) eta,
eta = clearance_tightening = 0.0001 m.

The nonlinear Newton rollout gives h_sb(u), with leader commands held fixed.
Aggregate all substeps into **one** residual per basin:

F_b(u) = min_s [h_sb(u)-T_sb].

F_b(u) >= 0 therefore checks the selected material point at every substep.
Forward perturbations in all six follower directions estimate J_b. Solve

min_u 0.5 ||u-u_nom||²,
subject to J_b u >= -F_b(u0) + J_b u0.

The objective retains planar formation and leader velocity tracking. Divide
rows by 1 cm for numerical scaling. Apply the existing local displacement trust
bounds and validate the resulting command in another full nonlinear rollout.
The substep minimum is nonsmooth when its minimizing time changes; the nonlinear
check is essential. Use at most the configured two linearization iterations.
If none passes, return the best checked recovery and report `prediction_not_safe`.
Sensitivities are refreshed for each necessary linearization; we do not reuse
cached derivatives across changing barycentric supports.

With no active basin, return nominal immediately without any Newton prediction.
With basins but a nominal rollout satisfying F >= 0, use that command after one
rollout. Hard recovery can need 15 rollouts (baseline + 2*(6 directions + check)).
`total_rows` counts optimizer rows; `basin_count` includes selected barriers even
when nominal passes and no optimizer solve is needed.

A passed barrier check is not equivalent to absolute nonpenetration: when h_b
starts negative, the CBF target permits negative h while recovering. Diagnostics
separate all-particle initial clearance and predicted representative clearance.

## GPU verification, 2026-10-02

RTX 5070 Laptop GPU, full 56x56 cloth cells: 3,249 particles, 6,272 triangles,
eight substeps. Timings are control plus live physics, excluding rendering,
ROS and independent validation rollouts.

| Scenario | Result |
| --- | --- |
| Idle, initial z=1.05 m, 40 frames | Exactly zero follower commands; no active basins; about 25 ms median/p95 |
| Close start z=0.68 m, 20 frames, leaders descend after frame 4 | 23–75 qualifying hits, 1–4 basins, at most 4 optimizer rows; 394 ms median, 431 ms p95 |
| Close-start nonlinear selected barrier checks | Passed 8/20 frames; minimum full-mesh signed clearance reached -0.0648 m |
| Live versus isolated predicted positions | Maximum error below 1.3e-6 m in both runs |
| Visible Isaac smoke, 40 idle frames | Completed, cloth sagged, zero follower motion, no recovery frames |
| Visible Isaac close-start smoke, 20 frames | Completed; active detections were reduced to basins; 10 recovery frames, transient penetration, no active hits by the final frame |

Reports: `/tmp/capsule-ray-idle.json`, `/tmp/capsule-ray-active.json`,
`/tmp/capsule-ray-gui/diagnostics.json` and `viewport.png`; active GUI artifacts
are in `/tmp/capsule-ray-gui-active/`.
The close-start test deliberately begins inside the safety margin. It is a
recovery stress test, not evidence of an invariant safe set.

## Limits of the requested sparsification

Rays can miss geometry between cells and hide deeper folded layers. A basin's
least-clearance point now need not be its most dangerous future point. Removing
all other samples removes their safety guarantees; checking selected supports
against Newton does not check the entire fabric. Starting predictions only when
a hit reaches the margin also sacrifices advance warning. These limitations
explain why the global clearance diagnostic can disagree with a passed selected
barrier check. Strain limits are disabled, so material damage protection is not
provided by this controller configuration.

The row reduction eliminates the large optimization workload, but full Newton
rollouts still dominate active recovery time. This configuration is not real-time
at the tested full resolution when corrections are needed. Reducing rollout cost
or changing the control/model update rate requires separate validation.
