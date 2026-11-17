import { beforeAll, describe, expect, it, vi } from "vitest";
import Module, { type ManifoldToplevel } from "manifold-3d";
import { BoxGeometry, Vector3 } from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { buildSetupExport, crownBaseHeight, exportCrownLayout, geometryBounds, type SetupExportRequest } from "../src/wear/setupExport";
import { createCrownRows, spaceCrownRows, indexCrownGeometry, type Vec3 } from "../src/wear/crownRows";
import { identityUpperSetupPose, createUpperSetupMatrix } from "../src/wear/upperPose";
import { encodeSetupStl } from "../src/wear/export";

let wasm: ManifoldToplevel;
beforeAll(async () => { wasm = await Module(); wasm.setup(); });
const load = async () => wasm;
function fixture(row: "upper" | "lower", count = 3, baseHeight = .3): SetupExportRequest {
  const sources = [6, 0].map(y => { const g = new BoxGeometry(2, 4, 1).translate(0, y, 0); const result = indexCrownGeometry({ positions: new Float32Array(g.getAttribute("position").array), indices: new Uint32Array(g.index!.array) }); g.dispose(); return result; });
  const layouts = spaceCrownRows(createCrownRows(sources[0], sources[1]), 4), n = row === "upper" ? 0 : 1, layout = layouts[n];
  return { source: sources[n], layout, row, count, baseHeight, rotation: [0, 0, 0], pose: identityUpperSetupPose(), pivot: layout.min.map((v, a) => (v + layout.max[a]) / 2) as Vec3, format: "stl" };
}
function assertSolid(geometry: Awaited<ReturnType<typeof buildSetupExport>>) {
  const mesh = new wasm.Mesh({ numProp: 3, vertProperties: geometry.positions, triVerts: geometry.indices! }); mesh.merge();
  const solid = new wasm.Manifold(mesh);
  try { expect(solid.status()).toBe("NoError"); expect(solid.volume()).toBeGreaterThan(0); const parts = solid.decompose(); try { expect(parts).toHaveLength(1); } finally { parts.forEach(p => p.delete()); } return solid.volume(); }
  finally { solid.delete(); }
}

describe("full-resolution configured setup export", () => {
  it("exports a complete imported row with the setup pose without loading Boolean code or a wear grid", async () => {
    const request = fixture("lower"), source = request.source.positions.slice(), unused = vi.fn(load);
    request.layout = undefined; request.pose = { translation: [2, -3, 4], rotation: [23, 17, -11] };
    const result = await buildSetupExport(request, unused), e = createUpperSetupMatrix(request.pose, request.pivot);
    expect(unused).not.toHaveBeenCalled(); expect(result.indices).toEqual(request.source.indices);
    for (let i = 0; i < source.length; i += 3) {
      const expected = new Vector3(source[i], source[i + 1], source[i + 2]).applyMatrix4(e);
      expected.toArray().forEach((v, a) => expect(result.positions[i + a]).toBeCloseTo(v, 5));
    }
    expect(request.source.positions).toEqual(source);
  });
  it("changes the exported count around the same row centre while retaining pitch and stagger", () => {
    const upper = fixture("upper"), lower = fixture("lower");
    for (const count of [1, 3, 8]) {
      const a = exportCrownLayout(upper.layout!, count), b = exportCrownLayout(lower.layout!, count);
      expect(a.centers.length).toBe(count); expect(b.centers.length).toBe(count);
      expect(a.centers[0][0] - b.centers[0][0]).toBeCloseTo(2, 10);
      expect((a.centers[0][0] + a.centers.at(-1)![0]) / 2).toBeCloseTo(2, 10);
      for (let i = 1; i < count; i++) expect(a.centers[i][0] - a.centers[i - 1][0]).toBeCloseTo(4, 10);
    }
    expect(upper.layout!.centers).toHaveLength(6);
  });
  it.each(["upper", "lower"] as const)("cuts protruding %s roots and unions a watertight shallow block without moving the attachment face", async row => {
    const r = fixture(row), original = r.source.positions.slice(), base = geometryBounds(r.layout!.base);
    const result = await buildSetupExport(r, load), bounds = geometryBounds(result), delta = (r.count - 6) * 4 / 2;
    const attach = row === "upper" ? base.min[1] : base.max[1], back = attach + (row === "upper" ? r.baseHeight : -r.baseHeight);
    expect(row === "upper" ? bounds.max[1] : bounds.min[1]).toBeCloseTo(back, 5);
    expect(row === "upper" ? bounds.min[1] : bounds.max[1]).toBeCloseTo(row === "upper" ? 4 : 2, 5);
    const baseVolume = (base.max[0] - base.min[0] + 2 * delta) * (base.max[2] - base.min[2]) * r.baseHeight;
    expect(assertSolid(result)).toBeCloseTo(baseVolume + 3 * 2 * 1 * 2.6, 4);
    expect(r.source.positions).toEqual(original); expect(crownBaseHeight(r.layout!)).toBeGreaterThan(r.baseHeight);
  });
  it("applies the whole-row pose after clipping, retaining the GUI pivot when count changes", async () => {
    const request = fixture("lower", 1, .2); request.rotation = [10, 15, 4];
    const local = await buildSetupExport(request, load);
    request.pose = { translation: [4, 7, -3], rotation: [20, 30, 40] };
    const result = await buildSetupExport(request, load), inverse = createUpperSetupMatrix(request.pose, request.pivot).invert();
    const back = geometryBounds(request.layout!.base).max[1] - request.baseHeight;
    for (let i = 0; i < result.positions.length; i += 3) {
      const p = new Vector3(...Array.from(result.positions.subarray(i, i + 3)) as Vec3).applyMatrix4(inverse);
      expect(p.y).toBeGreaterThanOrEqual(back - 1e-5);
    }
    expect(assertSolid(result)).toBeCloseTo(assertSolid(local), 4);
  });
  it("height zero retains all original crown triangles and roots without Boolean processing", async () => {
    const r = fixture("lower", 5, 0), unused = vi.fn(load);
    const result = await buildSetupExport(r, unused);
    expect(unused).not.toHaveBeenCalled(); expect(result.indices!.length).toBe(r.source.indices!.length * 5);
    expect(geometryBounds(result).min[1]).toBe(-2);
    const stl = encodeSetupStl(result), restored = new STLLoader().parse(stl);
    expect(new DataView(stl).getUint32(80, true)).toBe(result.indices!.length / 3);
    expect(restored.getAttribute("position").count).toBe(result.indices!.length); restored.dispose();
  });
  it("widens the base under tilted roots so shallow exports stay connected", async () => {
    const r = fixture("lower", 3, .2); r.rotation = [50, 0, 0];
    const result = await buildSetupExport(r, load), originalBase = geometryBounds(r.layout!.base), bounds = geometryBounds(result);
    expect(bounds.max[2] - bounds.min[2]).toBeGreaterThan(originalBase.max[2] - originalBase.min[2]);
    assertSolid(result);
  });
  it("rejects invalid options and intersecting crown arrays", async () => {
    for (const count of [0, 2.5, NaN, 33]) await expect(buildSetupExport({ ...fixture("lower"), count }, load)).rejects.toThrow("1–32");
    await expect(buildSetupExport({ ...fixture("lower"), baseHeight: NaN }, load)).rejects.toThrow("Base height");
    const r = fixture("lower");r.layout!.centers = r.layout!.centers.map((p, i) => [i, p[1], p[2]]);
    await expect(buildSetupExport(r, load)).rejects.toThrow("clearance");
  });
});
