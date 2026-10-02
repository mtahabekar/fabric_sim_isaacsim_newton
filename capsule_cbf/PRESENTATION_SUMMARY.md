# Capsule CBF — presentation summary

## Slide 1 — Goal and control inputs

Two teleoperated leaders move the fabric. Two followers are adjusted to preserve
selected fabric–capsule clearance while staying close to leader-following motion.

- Cloth state: \(x=(q,v)\).
- Known leader velocities: \(\ell\in\mathbb R^6\).
- Optimized follower velocities: \(u\in\mathbb R^6\).
- Capsule is a virtual envelope; physical contacts use the actual box/ground.

\[
u_{\mathrm{nom},i}=\ell_i+K P_{xy}\big(g_{L_i}+\Delta_i^0-g_{F_i}\big),
\qquad P_{xy}=\operatorname{diag}(1,1,0),\quad K=0.5.
\]

No absolute follower speed bound. Strain constraints and vertical formation
recovery are disabled.

## Slide 2 — Sparse detection and marker meanings

Cast 432 GPU rays from the capsule center through its surface cell centers:

\[
r_j(t)=c+t\frac{s_j-c}{\|s_j-c\|},\qquad t\ge0.
\]

Keep the closest cloth triangle hit when \(d(p)\le m\), \(m=0.05\) m.

- Blue: capsule cells/wireframe.
- Yellow/orange: qualifying ray hits, not individual optimizer constraints.
- Red: minimum-clearance representative of each basin.

Group hits using 4-nearest-neighbor edges within 8 cm; basins are graph connected
components. Select one minimum-clearance hit per basin.

## Slide 3 — Clearance barrier

For capsule center \(c\), radius \(R\), and centerline half-length \(L\):

\[
a(p)=\big(c_x,c_y,\operatorname{clip}(p_z,c_z-L,c_z+L)\big),
\qquad h(p)=\|p-a(p)\|-R-r_p-m.
\]

\(h\ge0\): selected point meets the margin. \(h<0\): margin violation.

A red point is attached to a triangle using fixed barycentric weights:

\[
p_b(q)=w_0q_{i_0}+w_1q_{i_1}+w_2q_{i_2},\qquad \sum_iw_i=1.
\]

The same material point is tracked across all predictions in a control frame.

## Slide 4 — Isolated Newton prediction and one barrier per basin

Clone the current state and use the same full-cloth Newton Style3D stepping as
live physics. Predict one frame, with prescribed gripper motion at every substep.
Hypothetical rollouts have private states, contacts and solver workspaces.

\[
\widehat{x}_s=\Phi_s(x_k,u,\ell_k),\qquad
T_{sb}=e^{-\alpha t_s}h_{b,k}+(1-e^{-\alpha t_s})\eta,
\]

\[
F_b(u)=\min_s\left[h\big(p_b(\widehat q_s(u))\big)-T_{sb}\right]\ge0.
\]

\(\alpha=2\,\mathrm{s}^{-1}\), \(\eta=0.0001\) m. The minimum combines every
substep into **one inequality per basin**. This is a finite-step approximation
of the exponential CBF condition, not an exact continuous-time certificate.

## Slide 5 — Sensitivity, optimization and validation

Estimate the effect of all six follower velocity directions by finite differences:

\[
J_{b,j}\approx\frac{F_b(u^{(r)}+\varepsilon e_j)-F_b(u^{(r)})}{\varepsilon},
\qquad \varepsilon=0.02\ \mathrm{m/s}.
\]

\[
\begin{aligned}
\min_u\;&\tfrac12\|u-u_{\mathrm{nom}}\|^2\\
\text{s.t. }&J_bu\ge J_bu^{(r)}-F_b(u^{(r)}),\quad\forall b,\\
&|u_j-u_j^{(r)}|\le\frac{0.005\ \mathrm m}{\Delta t}.
\end{aligned}
\]

Predict again with the candidate and check the nonlinear basin barriers. Allow
at most two linearizations. If no candidate passes, report `prediction_not_safe`
and apply the best checked recovery. No active basins: apply nominal without a
prediction. The local trust bound is not an absolute speed cap.

## Slide 6 — Why it changed and what it achieves

| Before | Current |
|---|---|
| Particle velocities approximated from corner weights | Full-cloth Newton prediction and six-direction sensitivities |
| Later full predictor: nearby-particle/substep rows plus triangle strain rows | One obstacle row per ray-selected basin; strain rows disabled |
| Vertical formation and strain recovery could lower idle followers | Vertical nominal velocity matches the leader; planar feedback remains |
| Large optimization workload and frozen-looking GUI | Small optimization problem; UI events processed between predictions |

Full-resolution GPU tests: 3,249 particles, 6,272 triangles, eight substeps.

- Idle: zero follower commands, about 25 ms control + physics.
- Close-start recovery: 23–75 hits reduced to 1–4 basins; about 394 ms median.
- Prediction matches live stepping to within approximately 1.3 micrometers in
  these tests; this validates consistency with the simulator, not real fabric.
- Sparse rays and one representative can miss unsafe cloth regions. Starting
  only at the margin gives no advance warning. Strain damage protection is off.
- Close-start recovery penetrated the capsule and passed selected barrier checks
  in only 8/20 frames. Active recovery is not yet real-time at full resolution.

**Presentation claim:** “A sparse, Newton-predictive capsule barrier filter that
reduces obstacle constraints to one per detected basin, with nonlinear validation
of selected material points.” Do not claim certified whole-fabric safety.
