# Toothwear Studio

**Experimental — not for research use. Simulated wear has not been validated.**

[Open Toothwear Studio](https://adamrountrey.github.io/toothwear-studio/)

A browser application for exploring mutual toothrow wear during closure and optional abrasion from vegetation sliding between the teeth. This standalone repository contains the toothwear application only.

## Use

1. Choose **Build 6 × 6 crown rows** (two single-crown STLs, approximately 61 MB), **Load example toothrows** (two PLY files, approximately 96 MB), or import your own aligned pair of closed PLY/STL surfaces. Imported files are processed in your browser; they are not uploaded. For single-crown imports, choose that option under **Mesh contents**.
2. Open the **Setup** tab. Choose **Whole upper row** or **Whole lower row** and use the Move/Rotate handles. For crown arrays, choose **Each upper crown** or **Each lower crown**: rotating one crown applies the same rotation about each of the six individual centers, leaving the base fixed. Choose **Pitch**, **Roll**, or **Yaw** in the **Crown rotation ring** selector above the model, then drag the ring. These modes use the same rules as the sliders under **Prepare geometry**; editing a slider selects its matching ring. **Free X/Y/Z** restores unconstrained source-axis rotation. Click a crown or select its tooth number to place the handle. Check tooth spacing, then choose a column pitch and prepare the wear grid.
3. The **Run** tab opens with **Jaw motion** at the top. Choose **Close jaw**, set a stopping distance, and run or step through closure. Both rows wear where they contact. To abrade without closing the jaw, choose **Hold jaw fixed**, set **Stop at total pull distance** and **Pull increment**, configure the vegetation band below, then choose **Run vegetation**. Pause, step, and resume preserve the held pose; the total includes previous pulls. Switching back to closure preserves worn surfaces. For chipping-enabled presets, **Rebuild for fixed-jaw vegetation** turns chipping off and starts from the unworn setup, as stated beside the button.
4. Open **Display & export** for shading, row visibility, resulting PLY surfaces, and run measurements. Wear-model selection and computation details are expandable groups in **Run**.

Fast geometric wear is the default. Repeated bite cycles remain experimental. These approximations are not calibrated biological wear models and must not be used as scientific evidence.

## Crown angles and saved setups

Pitch and roll describe the estimated crown long axis relative to the occlusal plane: **90° is perpendicular; 0° is parallel**. Pitch is the along-row projection and roll is the across-row projection, with separate lean-direction selectors. The long axis joins cross-section area centers at 20% and 80% of the unworn source crown height. Whole-row rotation is included; the reference plane remains perpendicular to the Y closure direction. The sliders support 0–90°, including horizontal poses.

Pitch, roll, and yaw each have a numeric degree input beside the slider. Type a value, then press **Enter**, **Tab**, or click away to apply it; **Escape** cancels typing. Pitch and roll accept 0–90° and yaw accepts −180–180°, including decimals. Empty or out-of-range entries show an error without moving the crowns. Typed values use the same rotation behavior, lean directions, row linking, and matching gizmo ring as the sliders. Slider and gizmo edits update the numeric readouts.

**Yaw is axial spin about the tooth's own estimated long axis. At pitch = roll = 90°, yaw 0° aligns the cutting-edge heading with the projected toothrow.** Positive yaw uses upper −L and lower +L, so both turn in the same spatial direction when upright (right-handed about world +Y). Changing yaw leaves pitch, roll, and true 3D inclination unchanged. Changing crown pitch, roll, or their lean directions preserves yaw. Its −180–180° value uses a swing–twist decomposition of the local crown rotation, with a fixed offset calibrated from the upright, edge-aligned pose. That offset is fixed for a source and whole-row pose: crown tilts do not re-measure the top-view edge heading as yaw. Whole-row tilts can change the calibration because the upright reference and projected jawline change relative to the source. Pitch/roll edits move the long axis in the jaw/occlusal frame, then retain its axial twist: roll tilts it about the projected jawline and pitch about the across-row axis. Those two projected inclinations can affect one another. The zero reference is undefined at an exact 180° reversal of the source long axis, a vertical jawline, or a missing transverse edge direction; the named controls reject those exceptional poses. Use Free X/Y/Z to leave them. The free local-X/Y/Z gizmo remains unconstrained and can change all three readings.

**Cutting-edge heading along the jaw** is a separate readout from the middle crown section viewed from above. It is not the yaw control and can change during axial spin. Its zero aligns the projected edge with the jawline; it may differ from the multi-section angle used to name matrix presets. Saved poses and matrix presets are unchanged, so a preset named for 13° edge obliquity need not display 13° axial yaw. These are geometric proxies, not annotated anatomical measurements.

**Same upper/lower pitch, roll & yaw** is an optional checkbox above the angle sliders in **Setup → Prepare geometry**. It starts off. Enabling it copies the selected row's displayed angles and lean directions to the other row; subsequent slider, named-ring, or Free X/Y/Z edits on either row update both. Each row retains its existing tip-up/tip-down orientation, normally upper down and lower up. The matching uses each crown's measured long axis and each row's pose, so the underlying source X/Y/Z rotations usually differ. The table shows both rows' resulting angles. Whole-row rotations also update the opposing crowns to keep the displayed angles matched; translations and row bases remain independent. Disabling the link leaves both poses as they are. Spacing checks still run after either crown pose changes. Run JSON records `setup.crowns.linkAngles`; older reports and freshly loaded models/presets default to unlinked. Reset restores the link setting recorded at the last successful preparation. Restoring a linked report uses its upper angles for both rows.

Run JSON also records `setup.crowns.yawConvention: "upright-edge-aligned-v1"`. Reports linked under the older source-zero convention restore their exact stored poses with linking turned off; the status message explains this. Re-enable the link to match the newly calibrated angles. Saved rigid poses and matrix presets are not reoriented by this change; their displayed yaw can differ from older versions.

Tooth spacing is checked after crown rotations. If crowns intersect or separation is not certified, use **Space teeth to clear** before preparation. This checks crowns within each row, excluding their shared support base.

While rotating a crown array or a whole row of crown copies, the selected tooth carries a direction marker. X/Y/Z are its original mesh-local axes, gold **L** is the estimated long axis used for pitch/roll and the axis of yaw spin, and pink **E** is the estimated cutting-edge direction. These source directions are fixed when the original crown is loaded and rotate with the tooth; they are not re-estimated during rotation or wear. Under axial yaw, L stays fixed while the crown and E turn around it. The arrows share the crown pivot for clarity and do not trace anatomical edges. The gray patch shows the fixed world-XZ occlusal reference orientation; its display position follows the selected tooth, and it is not a fitted contact surface. Whole-row rotation changes the projected jawline direction but never tilts this reference plane. The named crown ring uses the selected slider's axis: pitch about the across-row direction, roll about the projected jawline, and yaw about L. A tilted L need not be perpendicular to the pitch/roll axes, so one named ring is shown at a time. Pitch/roll drags preserve axial yaw, cross upright by changing lean direction, and stop at horizontal like their sliders. The source X/Y/Z arrows are ring axes only in **Free X/Y/Z** mode. **Done** hides the markers.

Direct links load and prepare fresh unworn setups; click **Run wear** to begin:

- [Y13-S083](https://adamrountrey.github.io/toothwear-studio/?preset=y13-s083)
- [S128](https://adamrountrey.github.io/toothwear-studio/?preset=s128)
- [Non-imbricated crown arrays](https://adamrountrey.github.io/toothwear-studio/?preset=non-imbricated)

Fast contact supports a Gaussian footprint specified in model units (`Contact spread σ`, zero for exact contact). It reaches three sigma sideways, up to a 16-cell radius, while exact overlap removal still prevents interpenetration. Start with pitch 0.5 and sigma 0.75; smaller pitch or wider footprints cost more.

`Thin-wall chipping` is optional and off at zero. With vegetation off, it trims narrow caps only to nearby tooth-contact wear depth. Width is measured along grid X and Z, making this an orientation- and resolution-dependent geometric approximation, not a fracture calculation. Chipping runs on fixed closure ticks, creates no subdivisions, and its volume is reported separately from tooth-contact abrasion. A threshold of 1.5 at pitch 0.5 is a useful experimental comparison. Rebuild after changing either control. These fast calculations run on the CPU worker; GPU vegetation features remain separate.

The app uses WebGPU where supported, with a CPU fallback. Computation runs locally in the browser. Large meshes, finer grids, and vegetation bundles can increase memory use and processing time.

Crown arrays start with six teeth spaced along X, a half-pitch upper stagger, and rectangular bases. Both whole-row poses and linked crown rotations are recorded in run JSON and retained by Reset. The supplied crown filenames are opposite to their physical Y placement; the higher crown is assigned upper, and **Swap** remains available. Large tilts can expose the embedded crown ends or separate crowns from their base; these controls do not model gums or tooth sockets.

## Develop

Use Node.js 22 and pnpm 10.11.0:

```sh
npm install --global pnpm@10.11.0
pnpm install --frozen-lockfile
pnpm dev
```

```sh
pnpm test
pnpm build
pnpm preview
```

The production build is written to `dist/`. It contains one app entry point, its worker/assets, and the example row and crown models. Relative asset URLs support GitHub Pages repository paths.

## Publish

GitHub Pages uses the **GitHub Actions** publishing source. Pushing to `main` runs the tests, builds the app, and deploys `dist/` through `.github/workflows/pages.yml`.

Example meshes are fetched when their load/build button is selected. A saved-setup link also downloads the two crown models automatically (approximately 61 MB). The default page does not download meshes until requested.
