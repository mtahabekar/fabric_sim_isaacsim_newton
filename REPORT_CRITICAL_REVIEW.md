# Critical review: OARBot simulation as the physical world for fabric estimation

Reviewed: `/home/taha/Downloads/OARBot Simulation as the Real World Fabric State Estimation in the Loop.pdf`, 14 pages. Review date: 2026-11-05. Page references below refer to the PDF's printed page numbers.

This is a review, not an implementation of the PDF's proposed actions. The original PDF and simulation/estimator source were not modified. Source inspection and small numerical checks support the findings; no new coupled-physics or estimator experiment was run.

## Overall assessment

Keep the architecture: one physical robot/cloth world, ROS sensor interfaces, an independent estimator, and a private truth recorder. Begin with oracle hands and visual observations. A perfect no-slip attachment is a useful declared approximation for this task. Full hand/tag perception can remain optional because the requested experiment assumes the leader grips are already detected and holding.

**Do not implement the force-coupling recipe literally yet.** It leaves the stiffness sign and reduction undefined, assumes force-model compatibility that the estimator does not have, and proposes taring in a state that can remove the signal being estimated. Camera placement and depth validity also need correction before the first sensor benchmark.

The biggest risk is attributing error to the estimator when it actually comes from an inconsistent mechanical interface, measurement model, time base, or observation geometry. Deliberate model mismatch is valuable only after a consistent baseline identifies these effects.

## 1. The proposed stiffness correction is not a complete stable coupling method

**Pages 4-5; high priority.** The report proposes `F + K*(x_frame - x_tool)` and identifies `ClothMechanics.resultant_jacobian()` as the available patch stiffness. That function returns `dF/dq` with shape `(3, N, 3)`, not a ready-to-use positive 3×3 stiffness. [Force and Jacobian implementation](capsule_cbf/cloth_model.py:136).

Define the boundary-motion map before reducing it. For a translating patch with frozen free cloth, its local force derivative is obtained by summing the Jacobian columns associated with that patch. The relaxed or dynamic response also includes free-cloth motion and cross-grip coupling. It is not generally the same local block. A static condensed stiffness involves a Schur complement; a dynamic interface additionally depends on mass, damping and the exchange interval.

The sign matters. For the scalar example `F(x) = -k*x`, the returned derivative is `J = -k`. With `dx = x_tool - x_frame`, the correct linear prediction is `F_new = F_old + J*dx`. Substituting that returned `J` into the report's `K` expression produces the opposite correction. At `k=1000 N/m` and `dx=1 mm`, it predicts +1 N instead of -1 N. The report's expression can be restoring if `K=-J` is a properly defined positive stiffness, but that definition is missing.

The installed shear derivative is also not necessarily symmetric. Symmetrizing or projecting it to a positive matrix for stabilization changes the derivative; that can be a useful numerical approximation, but must be distinguished from an exact physical interface stiffness.

Reevaluating this spring-like correction before each robot step still gives an explicitly sampled external force. It is not automatically an implicit Robin coupling merely because either subsolver has an implicit integrator. Increasing arm gains and seeing no oscillation does not prove stable or accurate load transfer.

**Change:** define signs, reference state, interface motion map, and evaluation interval; verify the derivative with finite differences. Start with smaller synchronized exchanges and a documented staggered approximation. If necessary, move to a genuinely implicit interface or iterations that restore both solvers to the beginning of the same interval before every trial. Monitor interface work and step refinement, not only visible oscillation. Newton's experimental coupling framework provides relevant interface and effective-mass concepts, but is not a verified drop-in replacement for this PhysX/Style3D integration. [Newton coupled solvers](https://newton-physics.github.io/newton/1.6.1/concepts/coupling.html).

## 2. Taring after attachment can erase the absolute cloth load

**Page 6; high priority.** Closing the fingers and then taring the loaded wrist can absorb the steady cloth load into the bias. Subsequent measurements represent changes from that loaded state, whereas the estimator predicts absolute robot-on-sheet forces.

**Change:** determine bias and downstream gravity compensation in an unloaded configuration before attaching the fabric. If the hardware workflow intentionally uses a loaded tare, record its reference load and use the same reference convention in the observation model. Keep sensor bias, gravity compensation and any hardware zero command distinct.

Closing the fingers should activate a declared attachment lifecycle without teleporting the sheet. Closure alone is not evidence of a successful frictional grasp when contact is not simulated.

## 3. The estimator does not currently enforce the reported slow-motion restriction

**Page 7; factual correction and high priority for force fusion.** The PDF says the estimator already restricts force use to slow motion. In the inspected code, `forces_at()` checks force enablement, sample age and TF availability, then rotates/sign-converts the force. It does not gate wrist acceleration or angular motion. `max_surface_speed_mps` is an observation-skew uncertainty setting, not a force-validity gate. [Force ingestion](../oarbots/src/fabric_state_estimator/fabric_state_estimator/node.py:143).

Low speed alone would not solve the problem: a slowly moving wrist can still have substantial acceleration during starts and stops. A hypothetical 1.8 kg downstream assembly accelerating at 0.2 m/s² generates 0.36 N, comparable to the nominal sheet's symmetric static vertical contribution per grip. Angular acceleration and the offset COM add further terms.

**Change:** initially use force observations only in a measured quasi-static regime, with criteria based on acceleration/rotation and observed compensation residuals. Otherwise add a compatible dynamic observation model or treat uncompensated dynamics as quantified measurement error. Keeping a hardware imperfection in the simulated sensor is useful for a robustness test; silently assimilating it as accurate cloth force is not.

## 4. The finite-patch plant and point-grip force observation are not equivalent

**Pages 3-5 and 12; high priority.** The report recognizes a finer plant and a coarse estimator as useful mismatch, but does not establish their nominal force compatibility. The plant clamps nine particles per corner. The estimator's `reactions(q, loads)` predicts a quasi-static force at a single pinned vertex, including that vertex's mass and a distributed residual load; it has no grip-acceleration term. [Estimator reaction model](../oarbots/src/fabric_state_estimator/fabric_state_estimator/newton_backend.py:127).

Sharing membrane formulas does not make the resulting forces match by construction: patch size, local mass treatment, mesh, bending, and deformation all differ. Conversely, making the plant dynamic solely to make its observations harder is not a substitute for defining the intended physical experiment.

**Change:** before material/noise perturbations, measure point-versus-patch discrepancy on static and slow held-out cases. Either extend the estimator to a physical patch of consistent size in metres, or explicitly validate the point approximation and its uncertainty. Test force fusion on/off on the same sensor recording. Do this before the PDF's Phase 5 adds several other error sources.

## 5. The quoted overhead-camera pose omits the floor-plane correction

**Page 8; numerical correction.** The quoted approximately `(1.50, -0.98, 2.57) m` matches `T_oc` from the stored camera calibration. The legacy broadcaster also composes the fitted floor-plane transform:

```text
R_camera_floor = R_co * R_op
t_camera_floor = T_co + R_co * T_op
T_floor_camera = inverse(T_camera_floor)
```

Recomputing the broadcaster's composition from the cited files gives:

```yaml
floor_from_rgb_camera:
  translation_m: [1.474690199, -1.073286028, 2.610532424]
  quaternion_xyzw: [0.897421770, -0.072991318, 0.056661619, -0.431388334]
```

The difference is about **0.10568 m and 2.1883 degrees**, with an optical-axis depression of about 38.42 degrees rather than the quoted 40.6 degrees. This is the legacy calibrated RGB frame, not automatically the camera housing or depth frame. Sources: the vendored `camera_undistorted_1536p.yml`, `camera_aruco_1536p.yml`, and [legacy broadcaster](/media/taha/ExtermeLinux/oarbot_simulation/vendor/Assistive-Robotics/src/tf_broadcasters/src/tf_overhead_camera_aruco_broadcaster.py:478).

If an uncorrected object/tag frame is intentionally chosen as the simulation world, that can be valid, but do not call it the legacy floor/map frame without the corresponding registration. Recompute camera coverage using the selected convention and RGB/depth extrinsics.

## 6. RGB coverage does not establish usable overhead depth

**Pages 8-9; high priority.** The proposed tag-grid placement puts the sheet several metres away. WFOV 2×2 binned mode has a documented nominal range of 0.25-2.88 m under specified reflectivity/illumination conditions. Out-of-range returns are not guaranteed. [Microsoft Azure Kinect hardware specification](https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/hardware-specification).

As a check using the report's own uncorrected camera transform and proposed cloth centre `(4.06, 1.83, 1.05)`, the four flat sheet corners have RGB optical-axis depths of approximately **3.10-3.86 m**. Native depth-frame coverage must be checked separately, but this already conflicts with the current perception default `max_depth_m=3.0`. [Camera adapter and depth configuration](../oarbots/src/fabric_perception/fabric_perception/node.py:63).

**Change:** select placement based on valid cloth observations, not merely whether all base tags appear in RGB. With anchored bases, seeing every robot localization tag is not required for the first estimator benchmark. Move/reorient the cell, select and validate an appropriate depth mode, or declare overhead RGB-only coverage for that configuration. Do not extend a software cutoff and describe that as fixing the physical camera's range.

Also render depth at its native resolution/FOV before registration to RGB. A dense perfect 2048×1536 depth render is not equivalent to registered 512×512 WFOV measurements. Preserve invalid pixels, differing optical origins, occlusion boundaries, axial-depth units and supported encodings. Visible dark color alone does not determine NIR return; depth failures require a measured or explicitly assumed model. [Azure Kinect depth principles and invalidation](https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/depth-camera).

## 7. Frame averaging alone does not guarantee momentum-consistent transfer

**Pages 5-6; claim needs qualification.** `resultant + m_patch*(g-a_patch)` is a reasonable contact-free starting balance when patch mass and acceleration are consistently defined. It is not automatically the exact reaction of a discrete constrained solver. Contact at the patch, other external forces, pin overwrites, solver residuals, and velocity filtering need an accounting convention.

The code's `0.998` velocity multiplier removes momentum even though it is not stored as a viscous force. It may legitimately contribute no direct damping term on overwritten pins, but its effect cannot be ignored in whole-sheet momentum or energy validation. Likewise, a stiffness correction applied only on the robot side changes the transferred load relative to the cloth's recovered reaction.

**Change:** recover substep reactions consistently with the discrete update, and integrate impulse over matching intervals: `J=sum(F_k*dt_k)`. State which interval a published F/T sample represents. Validate the robot/cloth impulse and work budgets, explicitly including numerical dissipation, leader support and environmental forces. Log stabilizer work separately. Do not claim conservation merely because eight force samples were averaged.

There is an additional numerical-model mismatch: the plant uses 480 substeps/s, while the estimator example permits 240. A multiplier of 0.998 gives velocity retention `0.998^480≈0.383` versus `0.998^240≈0.618` over one second if other effects are omitted. This is not a full cloth trajectory prediction, but it demonstrates different effective damping. Match or document that rate before treating residuals as material uncertainty. [Plant stepping](capsule_cbf/newton_runtime.py:77), [estimator stepping](../oarbots/src/fabric_state_estimator/fabric_state_estimator/newton_backend.py:143).

## 8. Transfer the patch wrench, even if the estimator initially uses only force

**Page 3; avoid an unnecessary approximation.** Applying a resultant at the centroid reproduces the force and the centroid lever-arm torque. It omits the patch's independent couple. That couple can affect wrist readings and tool orientation even if the current estimator discards torque.

**Change:** sum nodal reactions into both force and moment about a defined grasp origin, rotate/shift the wrench to the actual loaded body, and apply it downstream of the measurement joint. Resolve the actual rigid body corresponding to the named end-effector frame; a TF/virtual frame is not necessarily the dynamic body to load. A point force approximation is acceptable only if its moment residual is measured and below the task's tolerance.

Map the joint-reaction convention into the device raw frame before reusing hardware axis flips. Frame names being present in the asset do not establish sign or wrench origin. Verify known offset loads in multiple orientations; a 2 N transverse load at 0.1 m should yield the corresponding 0.2 N·m moment. Isaac's incoming-joint API reports parent/child joint spatial forces, which must be interpreted as such. [Isaac joint sensor documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics_articulation_force.html).

## 9. Do not stiffen the arm merely to make the coupling behave

**Pages 4 and 7; validation requirement.** The reported 150 N·m/rad gains and gravity sag identify the present simulated response, not the real robot's effective Cartesian impedance. Vendor controller stiffness, damping, gravity compensation, effort limits, configuration and servo mode all affect that response.

**Change:** calibrate or sweep a declared nominal actuator model separately from the coupling algorithm. Do not tune gains until a cloth benchmark passes and then call them realistic. If exact measurements are deferred, retain provisional values and report sensitivity. When adding the documented 0.44 kg camera, account for its COM and inertia, mounting bracket/cables and all existing downstream masses; do not just add a point mass at an arbitrary camera frame.

The correct F/T compensation should pass an unloaded rotation test before any cloth load is used. Extracting a shared compensation function is a sound idea, but a hardware-node split is not necessary for the first visual baseline. Preserve hardware behavior with replay tests before changing that driver interface.

## 10. Time, rendering and evaluation need explicit contracts

**Pages 2 and 9-12.** `/clock` and `use_sim_time` are necessary, but the current estimator also uses `monotonic()` for camera liveness, reorder waiting, forecasting throttling and diagnostics. A slower simulation or shared-GPU stall can therefore alter workload or status without changing physical time. [Estimator orchestration](../oarbots/src/fabric_state_estimator/fabric_state_estimator/node.py:160).

Keep measurement age and physical scheduling in simulation time; keep wall-time service watchdogs separate and identified. Define pause/reset behavior, including clearing measurement histories, TF caches and observer state on a backward clock jump. Stamps must identify acquisition time rather than completion of rendering/processing. A frame-averaged wrench and an instantaneous image also need a documented temporal relationship.

The existing evaluator requires identical timestamps, material indices and array shapes; it does not resample automatically. [Evaluator contract](../oarbots/src/fabric_state_estimator/fabric_state_estimator/metrics.py:9). Export truth at each estimate/forecast target time and interpolate the truth mesh into the estimator's material coordinates. Choose estimator resolution for accuracy/cost rather than just for integer alignment with the fine grid. Keep actual capture skew and forecast lead time in the record.

Score dropped or unpublished intervals, not only successful estimates. Include all-surface/occluded error, uncertainty coverage, force residuals, inference availability, forecast error, wall-time latency and achieved real-time factor. Label offline smoother results separately from causal online estimates. Remapping truth topics is good hygiene, but also verify subscriptions and keep truth out of correspondence generation and tuning on held-out runs.

Treat camera/grip calibration offsets as shared systematic errors, typically fixed over a run, rather than independent noise on every depth sample. Test correlated depth holes, edge artifacts and temporal bias as well as white noise. Decide how multi-camera acquisition/exposure timing and any depth-sensor interference will be measured before claiming hardware sensor fidelity.

## 11. State estimation running does not mean estimated-state robot control is tested

**Pages 2 and 12; define the intended claim.** The architecture routes sensors into the estimator and truth into scoring; it does not show estimated cloth state driving the CBF. The inspected robot runner calls the plant's full-state CBF when `route_cbf` is enabled.

This is appropriate for an observer benchmark. It does not validate obstacle avoidance or tension limits under estimator error, latency, stale data or uncertainty.

**Change:** label the initial result as an observer running beside an oracle-controlled plant. If estimated-state control is a goal, add a separate later mode with one command owner, achievable loaded robot motion, state-age handling and uncertainty-aware margins. Do not reuse the oracle CBF's prior safety claims for that new mode. Start contact-free: the estimator predictor explicitly has no obstacle or self-contact.

## 12. Narrow the rejected alternatives and avoid scope growth

**Page 5.** Rejecting naive explicit stiff particle springs for this step size is reasonable. Rejecting all compliant/implicit attachment methods is broader than that evidence supports. The cited near-10 N/m limit is not universal; for an isolated damped symplectic point update at the stated mass and step size, the linear stability limit is roughly 40 N/m before safety/accuracy factors. The coupled cloth operator changes the analysis again. This does not make an explicit spring grasp a good recommendation; it makes the numerical limit an assumption to state and test.

Elastic ties also do not implement frictional slip by themselves. Slip requires a constitutive/contact or release law. General finger contact is unnecessary for the current ideal no-slip benchmark, but the estimator's point-grip assumption is not evidence that a real grasp never slips.

The requested ideal detected-hand assumption permits a much smaller first experiment than full glove/cuff/tag rendering and detection. Keep that as an optional perception extension, rather than allowing its FOV requirements to compromise cloth sensing. Exact instance masks, if used, are an oracle perception mode; do not silently count them as real robot exclusion performance.

## 13. Update the snapshot assumptions

The PDF uses fabric `0c4df2a` and estimator `6f17744`. During this review the local fabric and vendored cloth were `249e940`, estimator `169b42a`, and the robot repository had active uncommitted work on top of `aa78814`. Its source manifest already points to the newer fabric/estimator revisions. Do not overwrite that concurrent work to reproduce the older PDF literally.

Dry carbon appearance is already implemented with local shader/maps and unchanged uncalibrated mechanics. The PDF's instruction to replace a pink sheet is historical now. The old source did not establish optical translucency merely by rendering both faces; two-sided rendering and transmission are different settings.

Pin the complete source and configuration set for each experiment, including working-tree changes if they are part of a run. Updating the cloth dependency must include its appearance helper and assets, not just the main simulator script. Visual realism does not make the existing mechanics or Azure measurements calibrated.

Keep reusable patch kinematics and reaction recovery in the cloth library, with robot-specific body loading and sensor adapters in the robot scene. The PDF's insistence that the fabric repository remain unchanged should not force copies of mechanical formulas or separate stepping implementations into the runner subclass. A small reusable cloth interface is cleaner than accumulating parallel physics implementations.

## Revised implementation sequence

1. **Sensor contract and observer baseline:** corrected provisional camera registration with valid depth coverage; one time authority; robot TF/joints; existing carbon appearance; direct assumed leader hand states; force fusion off. Start with static hang, then slow asymmetric motion clear of all obstacles. Record bags for repeatable replay.
2. **Measurement/model compatibility:** nominal mass/material meaning, explicit damping mismatch, physical grip identity, point-versus-patch approximation, and causal time/material truth alignment. Record visual-only estimator results and availability on held-out motions.
3. **Independent force validation:** first apply known loads to the robot without cloth. Separately validate fine-cloth patch reactions and moments, including patch mass and discrete dissipation. Establish unloaded bias/compensation and raw-to-calibrated sign/origin conventions.
4. **Mechanical coupling:** connect the validated parts; specify lag, wrench reduction and any stabilizer. Require finite-difference sign checks, exchange-step refinement, bounded interface work residuals and sensible loaded servo behavior. Re-run existing regressions, but do not require unchanged trajectories after adding real loads.
5. **Force-informed inference:** make its force observation compatible with the chosen grasp/dynamic regime, then enable it on the same recordings used for vision-only comparison. Only after this baseline add material, calibration and sensor perturbations separately and in selected combinations.
6. **Optional extensions:** rendered glove detection, moving bases/localization, slip, table/self-contact, estimated-state CBF control, and eventual hardware validation. Each changes the scope and needs its own acceptance criteria.

## Open questions to resolve

- Is success an online observer benchmark, force-assisted estimation, or closed-loop control driven by estimated cloth state? These are three different claims.
- Which physical motions, shears and contacts are inside the intended validated domain? Are tool rotations and clamp moments relevant?
- What interface lag and force/shape error are acceptable? What refinement and energy residuals would reject the proposed coupler?
- Will inference use point grips as a validated approximation or the same physical clamp size as the plant?
- Is calibrated F/T absolute or referenced to a loaded tare? What acceleration/rotation regime permits its use?
- Which overhead depth mode and registration are intended, and does the entire motion stay in its usable range/FOV?
- Which nominal motor gains, masses/COMs, cloth parameters and optical values are provisional, and which will be measured later?
- How will unavailable estimates, occlusion, bias, shared calibration error and combined failures be scored, rather than excluded?

Real-material calibration can wait as requested. Correct force signs, consistent measurement conventions, valid camera geometry and an honest evaluation contract should not wait: otherwise later calibration can fit around software inconsistencies instead of material behavior.
