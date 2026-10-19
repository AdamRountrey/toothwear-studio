# Toothwear Studio

**Experimental — not for research use. Simulated wear has not been validated.**

[Open Toothwear Studio](https://adamrountrey.github.io/toothwear-studio/)

A browser application for exploring mutual toothrow wear during closure and optional abrasion from vegetation sliding between the teeth. This standalone repository contains the toothwear application only.

## Use

1. Choose **Load example toothrows** (two PLY files, approximately 96 MB), or import your own aligned pair of closed PLY/STL surfaces. Imported files are processed in your browser; they are not uploaded.
2. Position the upper row with the Move/Rotate handles, choose a column pitch, and prepare the wear grid.
3. Set a stopping distance and run or step through closure. Both rows wear where they contact. Optional Gaussian contact spread and vegetation abrasion are available.
4. Export the resulting PLY surfaces and run measurements.

Fast geometric wear is the default. Repeated bite cycles remain experimental. These approximations are not calibrated biological wear models and must not be used as scientific evidence.

The app uses WebGPU where supported, with a CPU fallback. Computation runs locally in the browser. Large meshes, finer grids, and vegetation bundles can increase memory use and processing time.

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

The production build is written to `dist/`. It contains one app entry point, its worker/assets, and the two example models. Relative asset URLs support GitHub Pages repository paths.

## Publish

GitHub Pages uses the **GitHub Actions** publishing source. Pushing to `main` runs the tests, builds the app, and deploys `dist/` through `.github/workflows/pages.yml`.

The two example models are fetched only when **Load example toothrows** is selected; they are not downloaded when the page first opens.
