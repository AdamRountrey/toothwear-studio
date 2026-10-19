import { Matrix4 } from "three";
import { createDefaultBiteCycleSettings, evaluateBiteCycleCPU, type BiteCycleEvaluation, type BiteCycleEvaluationInput, type BiteCycleSurface } from "./biteCycles";
import type { VegetationContact } from "./slidingVegetation";
import { createGpuBiteCycles } from "./gpuBiteCycles";

/** Small deterministic contact quadrature cases; the larger plane also tests
 * pressure reduction across multiple workgroups without using mesh rendering. */
export function biteCycleEvaluationFixtures(includeLarge = false): Array<{ name: string; input: BiteCycleEvaluationInput }> {
  const plane = (nx: number, nz: number): BiteCycleEvaluationInput => {
    const grid = { originX: 0, originZ: 0, spacing: 1, nx, nz }, cells = nx * nz;
    const faces: VegetationContact[] = [];
    for (const row of ["upper", "lower"] as const) for (let cell = 0; cell < cells; cell++) {
      const x = cell % nx, z = Math.floor(cell / nx), y = row === "upper" ? 1 : 0;
      faces.push({ cell, row, normal: [0, row === "upper" ? -1 : 1, 0], minX: x, maxX: x + 1,
        minY: y, maxY: y, minZ: z, maxZ: z + 1 });
    }
    const records = new Float32Array(faces.length * 16);
    faces.forEach((face, i) => records.set([face.minX, face.minY, face.minZ, face.row === "upper" ? 0 : 1,
      face.maxX, face.maxY, face.maxZ, 1, (face.minX + face.maxX) / 2, face.minY,
      (face.minZ + face.maxZ) / 2, face.cell, ...face.normal, 0], i * 16));
    const surface: BiteCycleSurface = { faces, records,
      offsets: Uint32Array.from({ length: faces.length + 1 }, (_, i) => i), indices: Uint32Array.from({ length: faces.length }, (_, i) => i) };
    const settings = createDefaultBiteCycleSettings({ min: [0, 0, 0], max: [nx, 2, nz] }, 1);
    settings.contactReach = 2; settings.biteForce = cells * 30; settings.wearCoefficientUpper = 0.001; settings.wearCoefficientLower = 0.002;
    const upper = new Matrix4().makeTranslation(0.2, -0.1, 0.1);
    return { surface, grid, settings, load: settings.biteForce,
      upperMatrix: new Float32Array(upper.elements), previousUpperMatrix: new Float32Array(new Matrix4().elements), inverseUpperMatrix: new Float32Array(upper.clone().invert().elements) };
  };
  const fixtures = [{ name: "parallel faces with two wear coefficients", input: plane(3, 2) }, { name: "multiple GPU pressure reduction groups", input: plane(20, 20) }];
  const noSlip = plane(2, 2); noSlip.previousUpperMatrix = noSlip.upperMatrix.slice(); fixtures.push({ name: "pressure without sliding gives zero wear", input: noSlip });
  const noLoad = plane(2, 2); noLoad.load = 0; fixtures.push({ name: "zero load gives zero wear", input: noLoad });
  const separated = plane(2, 2), far = new Matrix4().makeTranslation(0.1, 5, 0);
  separated.upperMatrix = new Float32Array(far.elements); separated.inverseUpperMatrix = new Float32Array(far.clone().invert().elements);
  fixtures.push({ name: "outside compliance reach gives no contact", input: separated });
  const pureClosure = plane(2, 2), down = new Matrix4().makeTranslation(0, -0.1, 0);
  pureClosure.upperMatrix = new Float32Array(down.elements); pureClosure.inverseUpperMatrix = new Float32Array(down.clone().invert().elements);
  fixtures.push({ name: "pure normal closure has no sliding wear", input: pureClosure });
  const tilted = plane(3, 3), tilt = new Matrix4().makeRotationZ(0.08);
  tilted.upperMatrix = new Float32Array(tilt.elements); tilted.inverseUpperMatrix = new Float32Array(tilt.clone().invert().elements);
  fixtures.push({ name: "rotated hinge normals and tangential motion", input: tilted });
  const limited = plane(2, 2); limited.load = 1e9; fixtures.push({ name: "force remains bounded by compliance reach", input: limited });
  const side = plane(2, 2);
  for (let i = 0; i < side.surface.faces.length; i++) {
    const face = side.surface.faces[i], row = face.row === "upper" ? 0 : 1, x = face.minX + (row === 0 ? 0.75 : 0.25);
    face.minX = x; face.maxX = x; face.minY = 0; face.maxY = 1; face.normal = [row === 0 ? -1 : 1, 0, 0];
    side.surface.records.set([face.minX, face.minY, face.minZ, row, face.maxX, face.maxY, face.maxZ, 1,
      x, 0.5, (face.minZ + face.maxZ) / 2, face.cell, ...face.normal, 0], i * 16);
  }
  fixtures.push({ name: "side-wall contacts receive tangential wear", input: side });
  const hidden = plane(1, 1), boxFaces: VegetationContact[] = [];
  for (const row of ["upper", "lower"] as const) {
    const y0 = row === "upper" ? 2 : 0, y1 = y0 + 1;
    for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
      const min = [0, y0, 0], max = [1, y1, 1], normal: [number, number, number] = [0, 0, 0];
      normal[axis] = sign;
      if (sign < 0) max[axis] = min[axis]; else min[axis] = max[axis];
      boxFaces.push({ row, cell: 0, normal, minX: min[0], minY: min[1], minZ: min[2], maxX: max[0], maxY: max[1], maxZ: max[2] });
    }
  }
  const boxRecords = new Float32Array(boxFaces.length * 16);
  boxFaces.forEach((face, id) => boxRecords.set([face.minX, face.minY, face.minZ, face.row === "upper" ? 0 : 1,
    face.maxX, face.maxY, face.maxZ, 1, (face.minX + face.maxX) / 2, (face.minY + face.maxY) / 2,
    (face.minZ + face.maxZ) / 2, 0, ...face.normal, 0], id * 16));
  hidden.surface = { faces: boxFaces, records: boxRecords, offsets: new Uint32Array([0, 6, 12]), indices: Uint32Array.from({ length: 12 }, (_, i) => i) };
  hidden.load = 1; hidden.settings.contactStiffness = 100;
  hidden.upperMatrix = new Float32Array(new Matrix4().elements); hidden.inverseUpperMatrix = hidden.upperMatrix.slice();
  hidden.previousUpperMatrix = new Float32Array(new Matrix4().makeTranslation(-0.1, 0, 0).elements);
  fixtures.push({ name: "nearer cap blocks false opposed side penetration", input: hidden });
  const tall = plane(1, 2);
  tall.surface.faces.forEach((face, id) => {
    const upper = face.row === "upper", isTall = upper ? face.cell === 1 : face.cell === 0;
    face.minX = face.maxX = upper ? 0.75 : 0.25;
    face.minY = isTall ? -12 : 0; face.maxY = isTall ? 2 : 1;
    face.normal = [upper ? -1 : 1, 0, 0];
    tall.surface.records.set([face.minX, face.minY, face.minZ, upper ? 0 : 1,
      face.maxX, face.maxY, face.maxZ, face.maxY - face.minY,
      face.minX, (face.minY + face.maxY) / 2, (face.minZ + face.maxZ) / 2, face.cell, ...face.normal, 0], id * 16);
  });
  // Short faces on both rows slide against tall opponents whose centres lie
  // more than contactReach away in Y. Filtering centres alone loses contact.
  fixtures.push({ name: "Y-bin query: tall rectangles contact outside their centre band", input: tall });
  const mismatch = plane(1, 1), translated = new Matrix4().makeTranslation(0.2, 0, 0.1);
  mismatch.upperMatrix = new Float32Array(translated.elements);
  mismatch.inverseUpperMatrix = new Float32Array(translated.clone().invert().elements);
  // Deliberately inconsistent but finite: the CPU clamps this local query onto
  // the upper cap and still accepts its near world-space point. Bounds must
  // include the composition residual rather than assume exact inverse matrices.
  mismatch.inverseUpperMatrix[13] += 5;
  fixtures.push({ name: "Y-bin query: mismatched inverse retains world-space contacts", input: mismatch });
  const scaled = plane(1, 1), affine = new Matrix4().makeTranslation(0.2, 0, 0.1).multiply(new Matrix4().makeScale(1, 0.25, 1));
  scaled.surface.faces[0].minY = scaled.surface.faces[0].maxY = 6;
  for (const word of [1, 5, 9]) scaled.surface.records[word] = 6;
  scaled.upperMatrix = new Float32Array(affine.elements);
  scaled.inverseUpperMatrix = new Float32Array(affine.clone().invert().elements);
  // Six local Y units become 1.5 world units, inside the two-unit reach.
  // The inverse row norm must expand the local search window accordingly.
  fixtures.push({ name: "Y-bin query: affine Y scaling expands the local search band", input: scaled });
  // Beyond 256 reduction groups, with a partial final group. Keep this optional
  // so ordinary backend unit tests do not allocate the large browser smoke case.
  if (includeLarge) {
    // All faces remain in reach. Exercise both sides of the single-workgroup
    // compliance threshold, including partially filled contact groups.
    for (const [nx, nz] of [[89, 23], [64, 32], [683, 3]])
      fixtures.push({ name: `compact solve boundary: ${2 * nx * nz} contacts`, input: plane(nx, nz) });
    fixtures.push({ name: "compact reduction beyond 256 groups and a partial tail", input: plane(257, 129) });
  }
  return fixtures;
}

export async function runGpuBiteCycleParity(device: GPUDevice) {
  const gpu = await createGpuBiteCycles(device), results: Array<Record<string, string | number>> = [];
  try {
    const fixtures = biteCycleEvaluationFixtures(true), shared = fixtures[0].input;
    const withPose = (matrix: Matrix4): BiteCycleEvaluationInput => ({ ...shared,
      upperMatrix: new Float32Array(matrix.elements), inverseUpperMatrix: new Float32Array(matrix.clone().invert().elements) });
    const shifted = withPose(new Matrix4().makeTranslation(0.4, -0.15, 0.2));
    shifted.previousUpperMatrix = shared.upperMatrix; shifted.load *= 0.6;
    // Every phase below reuses the SAME immutable surface, after the much larger
    // fixture has occupied the workspace. Empty phases must clear stale contacts,
    // doses and diagnostics; later loaded phases must still recover their result.
    fixtures.push(
      { name: "reused workspace: loaded after surface shrinks", input: shared },
      { name: "reused workspace: separation gives no contact", input: withPose(new Matrix4().makeTranslation(0.1, 5, 0)) },
      { name: "reused workspace: changed pose and load after separation", input: shifted },
      { name: "reused workspace: zero load gives zero wear", input: { ...shared, load: 0 } },
      { name: "reused workspace: pressure without sliding gives zero wear", input: { ...shared, previousUpperMatrix: shared.upperMatrix } },
      { name: "reused workspace: original loaded phase restored", input: shared },
    );
    for (const { name, input } of fixtures) {
      const cpu = evaluateBiteCycleCPU(input), dense = await gpu.evaluate(input), denseMs = gpu.lastRun?.totalMs ?? 0;
      const sparse = await gpu.evaluate(input, { sparse: true });
      const near = (label: string, a: number, b: number, absolute = 0.0004) => {
        if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) > absolute + Math.max(Math.abs(a), Math.abs(b)) * 0.0003)
          throw new Error(`${name}: ${label}: reference ${a}, GPU ${b}`);
      };
      const expand = (value: BiteCycleEvaluation): Float32Array => {
        if (!value.faceIndices || value.faceIndices.length !== value.depths.length)
          throw new Error(`${name}: sparse indices and doses must have matching lengths.`);
        const expanded = new Float32Array(cpu.depths.length), seen = new Set<number>();
        value.faceIndices.forEach((face, at) => {
          if (face >= expanded.length || seen.has(face)) throw new Error(`${name}: invalid or duplicated sparse face ${face}.`);
          seen.add(face); expanded[face] = value.depths[at];
        });
        return expanded;
      };
      for (const key of ["contactArea", "supportedLoad", "peakPressure", "maxSlip", "compliantApproach"] as const) {
        near(`dense ${key}`, cpu[key], dense[key]); near(`sparse ${key}`, cpu[key], sparse[key]);
        near(`dense/sparse ${key}`, dense[key], sparse[key]);
      }
      if (dense.faceIndices || cpu.depths.length !== dense.depths.length) throw new Error(`${name}: expected full dense doses.`);
      const sparseDepths = expand(sparse);
      let largestDepthError = 0, wornFaces = 0;
      cpu.depths.forEach((value, index) => {
        near(`dense face ${index} depth`, value, dense.depths[index], 0.0000001);
        near(`sparse face ${index} depth`, value, sparseDepths[index], 0.0000001);
        near(`dense/sparse face ${index} depth`, dense.depths[index], sparseDepths[index], 0.0000001);
        largestDepthError = Math.max(largestDepthError, Math.abs(value - dense.depths[index]), Math.abs(value - sparseDepths[index]));
        if (dense.depths[index] > 0) wornFaces++;
      });
      if ((name.includes("zero wear") || name.includes("no sliding wear") || name.includes("no contact"))
        && (wornFaces !== 0 || !sparse.depths.every(depth => depth === 0)))
        throw new Error(`${name}: unexpectedly retained wear in a dense or sparse result.`);
      if (name.includes("no contact")) {
        near("empty contact area", 0, sparse.contactArea); near("empty supported load", 0, sparse.supportedLoad);
        near("empty peak pressure", 0, sparse.peakPressure); near("empty slip", 0, sparse.maxSlip);
      }
      if (name.includes("side-wall") && wornFaces === 0) throw new Error("Side-wall sliding produced no wear.");
      if (name.startsWith("Y-bin query:") && wornFaces === 0) throw new Error(`${name}: fixture lost its intended sliding contact.`);
      if (name.includes("tall rectangles") && !(cpu.depths[0] > 0 && cpu.depths[3] > 0))
        throw new Error(`${name}: short faces on both rows must abrade their tall opponents.`);
      if (name.includes("nearer cap")) { near("physical cap contact area", 1, dense.contactArea); near("positive cap separation", 1.01, dense.compliantApproach); }
      if ((name.includes("beyond 256 groups") || name.startsWith("compact solve boundary:"))
        && (wornFaces !== input.surface.faces.length || gpu.lastRun?.activeFaces !== input.surface.faces.length))
        throw new Error(`${name}: expected all ${input.surface.faces.length} faces, including the partial tail, to remain active and wear.`);
      results.push({ name, faces: dense.depths.length, sparseFaces: sparse.depths.length, wornFaces, supportedLoad: sparse.supportedLoad,
        largestDepthError, denseGpuMs: denseMs, sparseGpuMs: gpu.lastRun?.totalMs ?? 0 });
    }
    return results;
  } finally { gpu.dispose(); }
}
