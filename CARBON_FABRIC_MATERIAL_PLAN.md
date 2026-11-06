# Dry woven carbon fabric: material and appearance plan

Target confirmed by the user: **dry woven carbon fabric without resin**. Prepared 2026-11-05. This plan extends the [robot–fabric coupling report](FABRIC_ROBOT_COUPLING_REPORT.md).

**Implementation status:** dry carbon twill is now the default appearance, with original local maps, a two-direction MDL shader, a USD Preview Surface fallback, stable rest-space UVs and configurable yarn pitch. See [material assets](assets/materials/dry_carbon_twill/README.md). Existing mass and mechanical settings were retained; real-material calibration, constitutive-model validation and sensor calibration remain future work.

## Recommended starting specification

Use a single dry, balanced **3K carbon, 2×2 twill, nominally 200 g/m²** as a documented reference. This is a starting product specification, not an assertion about the user's eventual physical specimen. Record the actual supplier, product/lot, sizing or binder, weave, orientation and cut dimensions before physical validation.

One supplier datasheet gives 200 g/m² ±4%, 5 warp ends/cm and 5 weft picks/cm, and dry thickness 0.32 mm ±0.05 mm. Those counts imply approximately 2 mm yarn spacing. The 0.84 m square simulated sheet would weigh 0.14112 kg, matching the existing plant's nominal `areal_density: 0.2`. These construction values are a reasonable initial visual scale and mass specification; the datasheet does not supply a calibrated draping model. [Compositesplaza CAWFINTW200-120 datasheet](https://compositesplaza.com/wp-content/uploads/product-downloads/technical_data_sheet_twill_woven_filament_fabric_ind._carbon_200g-m2_120_cm_cp_0.pdf).

Do not use the listed single-fiber tensile modulus directly as a cloth solver coefficient. Fiber modulus, tow response, dry fabric membrane response and cured laminate stiffness are different quantities. Likewise, fabric bending is not reliably obtained by inserting that modulus and a nominal thickness into an isotropic solid-plate formula.

## Useful assets

| Candidate | Verified availability | Appropriate role |
| --- | --- | --- |
| [ShareTextures Carbon Fiber](https://www.sharetextures.com/textures/fabric/carbon-fiber) | Publisher lists a seamless texture, PBR maps and CC0 | Quick appearance prototype; inspect maps and remove any resin-like coating assumptions |
| Installed ANYmal carbon image | An 800×800 RGBA image exists and was visually inspected at `/media/taha/ExtermeLinux/isaacsim_assets/Assets/Isaac/6.1/Isaac/Samples/Mujoco_Menagerie/anybotics_anymal_b/assets/carbon_uv_texture.png` | Local placeholder; it contains a repeating weave with baked directional shading. It is not a complete PBR or mechanical material, and redistribution terms were not checked |
| [NVIDIA vMaterials](https://developer.nvidia.com/vmaterials) | NVIDIA provides an MDL appearance library | Compatible material-authoring ecosystem; a specific dry-carbon material was not verified in this search |
| [NVIDIA PhysicalAI-SimReady-Materials](https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials) | USD/MaterialX/OpenPBR materials with MIT-0 licensing | Alternative appearance pipeline; a specific dry-carbon entry was not verified. SimReady metadata does not provide a dry-textile constitutive model |

**Preferred production asset:** a small, local, procedural weave shader or an authored set of consistent albedo/normal/roughness maps, parameterized by actual weave construction. A prebuilt rigid carbon panel mesh is unnecessary: retain the deforming cloth mesh and bind the appearance material to it. If photographs of the selected fabric become available, use them to fit color, yarn spacing and highlight behavior.

The local placeholder and generic downloaded maps should not be treated as measured reflectance. A convincing image can still have incorrect weave scale, reflectivity or fixed lighting baked into its color.

## Rendering changes in this repository

The sheet is created in `FabricSimulation._create_visuals()` in [fabric_newton_isaacsim.py](fabric_newton_isaacsim.py). It now uses [fabric_appearance.py](fabric_appearance.py) to bind the carbon material. The appearance design and later validation targets are:

1. Author stable material UV coordinates from the rest sheet, aligned with the physical warp/weft axes. Do not use camera or world-projected texture coordinates. The pattern must remain attached to material points during stretch, shear and bending.
2. Bind a USD/MDL material with charcoal yarn color, alternating warp/weft structure, small yarn-scale normal relief and directional highlights. Dry carbon should not inherit the clear resin surface of a polished automotive panel.
3. Use the reference yarn spacing in metres. For an ideal 2×2 twill, a four-yarn weave repeat is about 8 mm at 2 mm spacing. Keep weave repeat size distinct from individual yarn width and from image resolution.
4. Use a normal/bump shader for yarn detail. The current 15 mm mechanical cells cannot resolve 2 mm yarn geometry. Keep that detail out of the physical mesh initially. Add a separately refined render mesh only if camera-visible silhouettes or wrinkles require it, with interpolation from the mechanical sheet.
5. Keep warp/weft highlight directions attached to the deformed material. One fixed global anisotropy direction is insufficient, especially under shear. A more complete shader uses the two deformed yarn directions and blends their responses according to the over/under weave pattern.
6. Verify both faces. A two-sided sheet needs consistent back-face normals and weave behavior; a solid resin-panel appearance is inappropriate. Avoid artificial opacity holes or strong displacement unless the specimen has camera-resolvable gaps or relief.
7. Test lighting and filtering at actual overhead and wrist camera resolutions. Fine repeated weave can produce moiré, temporal shimmer and unreliable image features. Use appropriate texture filtering and compare short moving sequences, not only close-up screenshots.
8. Hide debug spheres, capsule lines and ground-truth overlays from sensor renders. Adjust the existing perception segmentation using these camera images rather than relying on the former pink/red appearance.

Use a conservative isotropic PBR material plus texture maps as the initial portable implementation if a custom MDL shader is too costly. Treat it as an approximation and compare it with the directional model before deciding whether more shader complexity improves the actual camera test. MDL is an appearance definition supported by Omniverse, independent of Newton's mechanical material. [Omniverse materials documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html).

## Mechanical realism

For this manipulation task, use a homogenized directional cloth/shell model. Individual filaments are unnecessary. A useful model needs strong tension response along yarns, compliant yarn rotation during initial shear, increasing resistance as the weave compacts, and separately calibrated bending. The desired model complexity depends on the shear and curvature reached by the experiments.

The installed Style3D stretch kernel has constant warp/weft/shear coefficients and a shear term based on the normalized yarn-direction dot product. Its quadratic bending model provides a baseline. It does not provide an independently fitted shear-locking curve, crimp response, permanent yarn sliding, or hysteresis merely by changing those constants. Use it first for moderate-deformation hanging and manipulation, then check its error against measurements. Large shear and wrinkle onset may require extending the constitutive law or using a textile-specific shell model.

Research on twill carbon forming demonstrates that tensile, shear, out-of-plane bending, in-plane bending and torsion can all matter to reproducing wrinkling. A constant-coefficient cloth model should therefore be validated over the operating range rather than called a fully realistic woven-carbon model. [Harrison et al., fabric characterization and forming study](https://eprints.gla.ac.uk/131416/).

| Quantity | Measurement or identification | Simulation use |
| --- | --- | --- |
| Areal mass | Weigh a known-area sample | Mass per particle; start at 0.2 kg/m² for the reference product |
| Warp/weft extension | Strip tests in each direction, recording force versus strain | Fit membrane response; report force per specimen width rather than importing fiber modulus |
| In-plane shear | Bias-extension or picture-frame test, recording load and yarn angle | Fit shear curve and identify where constant shear stiffness ceases to be adequate |
| Bending in both directions | Cantilever or overhang tests; fit full observed shapes | Fit directional bending independently of membrane response |
| Dissipation | Gentle oscillation decay and cyclic loading/unloading | Fit effective damping and determine whether hysteresis is important |
| Table/finger friction | Pull or slide tests against the actual contact surfaces | Set surface-pair friction, accounting for direction/load dependence if observed |
| Grasp behavior | Pull tests with the actual jaws and clamp force | Patch extent, slip threshold, effective compliance and any damage limits |

For a first calibration pass, prioritize mass, two-direction bending, shear, and a four-corner static sag/pull test with F/T recordings. Use separate trajectories to validate the fit. A single static sag shape cannot uniquely determine all material parameters. Select a constitutive model based on observed residuals, not visual preference.

Do not simply increase every stiffness because carbon fibers are stiff. A dry woven sheet can change shape mainly by yarn rotation and bending while yarn extension stays small. Fitting a large shear stiffness can suppress the very draping behavior being studied. Current `tri_kd` and `edge_kd` are unused by this installed Style3D version; changing those values does not calibrate dissipation.

The current `particle_radius: 0.008` and `contact_margin: 0.012` are numerical collision settings, not a measured 0.32 mm sheet thickness. Review contact offset, penetration and stability explicitly for table or clamp-contact experiments. Do not assume setting the radius to half the physical thickness will remain numerically robust on a 15 mm grid.

If the force law changes, update and revalidate all dependent implementations: plant force extraction, `capsule_cbf/cloth_model.py`, the reduced predictor and sensitivities, and the estimator's mechanical/force-observation model. Preserve an independently discretized estimator and test intentional parameter mismatch after establishing a consistent nominal baseline.

## Sensor realism is a third calibration problem

Realistic visible shading does not make rendered depth a realistic Azure Kinect measurement. Time-of-flight return depends on near-infrared reflectance, geometry and multipath; visible blackness alone is not a reliable depth-validity model. Azure Kinect measurements have shown material-dependent precision and reflective-surface artifacts. [Azure Kinect experimental evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7827245/).

Capture the actual dry fabric with the intended Azure camera at several distances and incidence angles. Fit depth variance, bias and dropout behavior from those observations. Until then, report results under ideal depth and a separately labelled assumed noise/dropout model. Avoid claiming a physically calibrated carbon-fabric depth model from a shader alone.

Use real or realistically rendered lighting variations for the RGB pipeline. Test specular highlights, partial occlusion and repeated-weave feature ambiguity. Do not add visually distinctive marks that the real fabric lacks simply to make tracking easier; if markers are part of the intended hardware experiment, model them explicitly in both cases.

## Implementation order and acceptance

1. Implemented: the default `dry_carbon_twill` appearance and stable material UVs, preserving the existing uncalibrated physics. Verify the final overhead and wrist-camera configurations when integrating the estimator; those sensor scenarios are separate from appearance checks.
2. Add a product-specific mechanical profile with provenance, units and measured curves. Retain separate rendering and mechanics settings so appearance changes cannot silently alter force response.
3. Fit and validate the current solver on hanging, bending and slow pull/shear cases. Compare shape, force, yarn angle and energy/damping across mesh/time-step refinement.
4. If large-shear or wrinkle errors remain, implement nonlinear textile mechanics and revalidate force extraction, controller derivatives and estimator prediction. Keep the simpler model when it meets the task's measured accuracy requirements.
5. Add measured camera degradation and evaluate estimation on held-out lighting, motion and material uncertainty. Compare visible and occluded shape errors and force residuals, not screenshot appearance alone.

This yields three separate claims to validate: **appearance matches the fabric**, **mechanical response matches the fabric**, and **sensor outputs match the camera observing that fabric**. Existing visual assets help with the first; physical tests and camera recordings are needed for the other two.
