import { describe, expect, it } from "vitest";
import { BoxGeometry } from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { advanceWear, extractWearSurface, prepareWear, type WearGeometry } from "../src/wear/engine";
import { encodeWearPly } from "../src/wear/export";

function boxAtY(minY: number): WearGeometry {
  const geometry = new BoxGeometry(2, 2, 2).translate(1, minY + 1, 1);
  const positions = new Float32Array(geometry.getAttribute("position").array);
  const indices = new Uint32Array(geometry.index!.array);
  geometry.dispose();
  return { positions, indices };
}

function plyHeader(buffer: ArrayBuffer) {
  const text = new TextDecoder().decode(buffer);
  return text.slice(0, text.indexOf("end_header\n") + "end_header\n".length);
}

describe("tooth wear binary PLY export", () => {
  it("round-trips indexed topology and a signed Y translation without mutating the source", () => {
    const mesh: WearGeometry = {
      positions: new Float32Array([1, 2, 3, 4, 2, 3, 4, 6, 3, 1, 6, 3]),
      indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
    const before = mesh.positions.slice();
    const buffer = encodeWearPly(mesh, -2.5);
    const header = plyHeader(buffer);
    expect(header).toContain("format binary_little_endian 1.0\n");
    expect(header).toContain("element vertex 4\n");
    expect(header).toContain("element face 2\n");
    expect(buffer.byteLength).toBe(new TextEncoder().encode(header).length + 4 * 12 + 2 * 13);

    const restored = new PLYLoader().parse(buffer);
    try {
      expect(Array.from(restored.index!.array)).toEqual(Array.from(mesh.indices!));
      expect(Array.from(restored.getAttribute("position").array)).toEqual([
        1, -0.5, 3, 4, -0.5, 3, 4, 3.5, 3, 1, 3.5, 3,
      ]);
      expect(mesh.positions).toEqual(before);
    } finally { restored.dispose(); }
  });

  it("writes sequential triangle indices for an unindexed surface", () => {
    const positions = new Float32Array([
      -4, 2, -1, 3, 2, -1, 3, 2, 5,
      -4, 2, -1, 3, 2, 5, -4, 2, 5,
    ]);
    const buffer = encodeWearPly({ positions });
    expect(plyHeader(buffer)).toContain("element vertex 6\n");
    expect(plyHeader(buffer)).toContain("element face 2\n");
    const restored = new PLYLoader().parse(buffer);
    try {
      expect(Array.from(restored.index!.array)).toEqual([0, 1, 2, 3, 4, 5]);
      expect(Array.from(restored.getAttribute("position").array)).toEqual(Array.from(positions));
    } finally { restored.dispose(); }
  });

  it("exports both worn rows in the contact pose, applying automatic opening exactly once", () => {
    const state = prepareWear(boxAtY(1), boxAtY(0), { spacing: 0.5, autoOpen: true });
    expect(state.initialOpening).toBeCloseTo(1.05, 10);
    const snapshot = advanceWear(state, state.firstContactClosure + 0.8, 0.5);
    const upper = new PLYLoader().parse(encodeWearPly(extractWearSurface(state, "upper"), -snapshot.closure));
    const lower = new PLYLoader().parse(encodeWearPly(extractWearSurface(state, "lower")));
    try {
      upper.computeBoundingBox();
      lower.computeBoundingBox();
      expect(upper.boundingBox!.min.y).toBeCloseTo(1.6, 5);
      expect(lower.boundingBox!.max.y).toBeCloseTo(1.6, 5);
      expect(upper.boundingBox!.max.y).toBeCloseTo(3.2, 5);
      expect(lower.boundingBox!.min.y).toBeCloseTo(0, 5);
      expect(snapshot.removedUpperVolume).toBeCloseTo(1.6, 8);
      expect(snapshot.removedLowerVolume).toBeCloseTo(1.6, 8);
    } finally {
      upper.dispose();
      lower.dispose();
    }
  });

  it("encodes an exhausted row as an empty PLY instead of inventing faces", () => {
    const buffer = encodeWearPly({ positions: new Float32Array() });
    const header = plyHeader(buffer);
    expect(header).toContain("element vertex 0\n");
    expect(header).toContain("element face 0\n");
    expect(buffer.byteLength).toBe(new TextEncoder().encode(header).length);
  });
});
