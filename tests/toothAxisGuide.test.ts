import { describe, expect, it } from "vitest";
import { ArrowHelper, Euler, Group, Matrix4, Quaternion, Sprite, Vector3 } from "three";
import { updateToothAxisGuide, type ToothAxisGuide } from "../src/wear/toothAxisGuide";
import { measureCrownInclination, type CrownReference } from "../src/wear/crownOrientation";

// No renderer or canvas is needed to verify attachment and reference frames.
function guide(): ToothAxisGuide {
  const group = new Group(), local = new Group(), plane = new Group(); local.matrixAutoUpdate = false;
  const arrows = Array.from({ length: 5 }, () => new ArrowHelper());
  const labels = Array.from({ length: 5 }, () => new Sprite());
  local.add(...arrows, ...labels); group.add(local, plane);
  return { group, local, plane, arrows, labels, dispose() {
    arrows.forEach(arrow => arrow.dispose()); labels.forEach(sprite => sprite.material.dispose());
  } };
}
const reference: CrownReference = { base: [2, 8, 1], tip: [3, -8, -2], edge: [6, 0, 2] };

describe("tooth-local axis marker", () => {
  it("keeps each direction attached to the crown through rotation and translation", () => {
    const marker = guide(), pivot = new Vector3(2, 0, 1);
    for (const rotation of [[20, 3, 25], [84, 12, -70]]) {
      const matrix = new Matrix4().makeRotationFromEuler(new Euler(...rotation.map(v => v * Math.PI / 180) as [number, number, number], "ZYX"));
      matrix.setPosition(80, 34, -12);
      const worldPivot = pivot.clone().applyMatrix4(matrix);
      const frozen = JSON.stringify(reference);
      updateToothAxisGuide(marker, reference, matrix, worldPivot);
      const sourceDirections = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1),
        new Vector3(...reference.tip).sub(new Vector3(...reference.base)).normalize(), new Vector3(...reference.edge).normalize()];
      marker.arrows.forEach((arrow, i) => {
        expect(arrow.getWorldPosition(new Vector3()).distanceTo(worldPivot)).toBeLessThan(1e-10);
        const direction = new Vector3(0, 1, 0).applyQuaternion(arrow.getWorldQuaternion(new Quaternion()));
        expect(direction.distanceTo(sourceDirections[i].transformDirection(matrix))).toBeLessThan(1e-10);
      });
      expect(JSON.stringify(reference)).toBe(frozen);
    }
    marker.dispose();
  });

  it("uses the same frozen long axis and edge as the displayed angle measurements", () => {
    const marker = guide();
    const crown: [number, number, number] = [73, 15, -22], row: [number, number, number] = [-12, 35, 7];
    const rotate = (v: [number, number, number]) => new Matrix4().makeRotationFromEuler(new Euler(...v.map(n => n * Math.PI / 180) as [number, number, number], "ZYX"));
    const rowMatrix = rotate(row), combined = rowMatrix.clone().multiply(rotate(crown));
    updateToothAxisGuide(marker, reference, combined, new Vector3());
    const long = new Vector3(0, 1, 0).applyQuaternion(marker.arrows[3].getWorldQuaternion(new Quaternion()));
    const edge = new Vector3(0, 1, 0).applyQuaternion(marker.arrows[4].getWorldQuaternion(new Quaternion()));
    const along = new Vector3(1, 0, 0).transformDirection(rowMatrix); along.y = 0; along.normalize();
    const across = new Vector3().crossVectors(along, new Vector3(0, 1, 0));
    const angles = measureCrownInclination(reference, crown, row, "upper")!;
    expect(Math.atan2(Math.abs(long.y), Math.abs(long.dot(along))) * 180 / Math.PI).toBeCloseTo(angles.pitch!, 9);
    expect(Math.atan2(Math.abs(long.y), Math.abs(long.dot(across))) * 180 / Math.PI).toBeCloseTo(angles.roll!, 9);
    const yaw = Math.atan2(edge.dot(across), edge.dot(along)) * 180 / Math.PI;
    expect(((yaw + 270) % 180) - 90).toBeCloseTo(angles.edgeYaw!, 9);
    marker.dispose();
  });

  it("keeps the occlusal patch normal at world Y even when the tooth and row are tilted", () => {
    const marker = guide();
    const matrix = new Matrix4().makeRotationFromEuler(new Euler(.8, -.3, 1.2)); matrix.setPosition(-80, 45, 12);
    const pivot = new Vector3(30, 25, 3);
    updateToothAxisGuide(marker, reference, matrix, pivot);
    const normal = new Vector3(0, 1, 0).applyQuaternion(marker.plane.getWorldQuaternion(new Quaternion()));
    expect(normal.distanceTo(new Vector3(0, 1, 0))).toBeLessThan(1e-12);
    expect(marker.plane.getWorldPosition(new Vector3()).distanceTo(pivot)).toBeLessThan(1e-12);
    marker.dispose();
  });
});
