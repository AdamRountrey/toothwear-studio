# Toothwear Studio

**Experimental — not for research use. Simulated wear has not been validated.**

[Open Toothwear Studio](https://adamrountrey.github.io/toothwear-studio/)

A browser application for exploring mutual toothrow wear during closure and optional abrasion from vegetation sliding between the teeth. This standalone repository contains the toothwear application only.

## Use

1. Choose **Build 6 × 6 crown rows** (two single-crown STLs, approximately 61 MB), **Load example toothrows** (two PLY files, approximately 96 MB), or import your own aligned pair of closed PLY/STL surfaces. Imported files are processed in your browser; they are not uploaded. For single-crown imports, choose that option under **Mesh contents**.
2. Open the **Setup** tab. Choose **Whole upper row** or **Whole lower row** and use the Move/Rotate handles. For crown arrays, choose **Each upper crown** or **Each lower crown**: rotating one crown applies the same rotation about each of the six individual centers, leaving the base fixed. Pitch, roll, and yaw sliders are available under **Prepare geometry**. Click a crown or select its tooth number to place the handle. Check tooth spacing, then choose a column pitch and prepare the wear grid.
3. The **Run** tab opens with **Jaw motion** at the top. Choose **Close jaw**, set a stopping distance, and run or step through closure. Both rows wear where they contact. To abrade without closing the jaw, choose **Hold jaw fixed**, set **Stop at total pull distance** and **Pull increment**, configure the vegetation band below, then choose **Run vegetation**. Pause, step, and resume preserve the held pose; the total includes previous pulls. Switching back to closure preserves worn surfaces. For chipping-enabled presets, **Rebuild for fixed-jaw vegetation** turns chipping off and starts from the unworn setup, as stated beside the button.
4. Open **Display & export** for shading, row visibility, resulting PLY surfaces, and run measurements. Wear-model selection and computation details are expandable groups in **Run**.

Fast geometric wear is the default. Repeated bite cycles remain experimental. These approximations are not calibrated biological wear models and must not be used as scientific evidence.

## Crown angles and saved setups

Pitch and roll describe the estimated crown long axis relative to the occlusal plane: **90° is perpendicular; 0° is parallel**. Pitch is the along-row projection and roll is the across-row projection, with separate lean-direction selectors. The long axis joins cross-section area centers at 20% and 80% of the unworn source crown height. Whole-row rotation is included; the reference plane remains perpendicular to the Y closure direction. The sliders support 1–90°; use the gizmo for horizontal or inverted poses.

Yaw controls an estimated cutting-edge heading from the middle crown section, viewed from above. Zero aligns that edge with the jawline. This geometric estimate can differ from the multi-section angle used to name the matrix presets. Each slider holds the other two displayed angles fixed; impossible combinations are rejected without deforming the crowns. These are geometric proxies, not annotated anatomical measurements.

Tooth spacing is checked after crown rotations. If crowns intersect or separation is not certified, use **Space teeth to clear** before preparation. This checks crowns within each row, excluding their shared support base.

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
