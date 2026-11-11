import {
  ArrowHelper, BufferGeometry, CanvasTexture, DoubleSide, Group, LineBasicMaterial,
  LineLoop, Matrix4, Mesh, MeshBasicMaterial, PlaneGeometry, Sprite, SpriteMaterial, Vector3,
} from "three";
import type { CrownReference } from "./crownOrientation";

const colors = ["#ff7979", "#89e6a5", "#83b8ff", "#ffd45c", "#f59bee"];
const names = ["X", "Y", "Z", "L", "E"];

function label(text: string, color: string) {
  const canvas = document.createElement("canvas"); canvas.width = 64; canvas.height = 64;
  const context = canvas.getContext("2d")!;
  context.font = "bold 40px sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
  context.lineWidth = 8; context.strokeStyle = "#111a23"; context.strokeText(text, 32, 32);
  context.fillStyle = color; context.fillText(text, 32, 32);
  const map = new CanvasTexture(canvas);
  return new Sprite(new SpriteMaterial({ map, sizeAttenuation: false, depthTest: false, depthWrite: false, toneMapped: false }));
}

/** Markers are directions at the crown pivot, not traced anatomical edges.
 * All five arrows live in immutable source coordinates and receive exactly
 * the source-to-world transform used by the selected crown instance. */
export function createToothAxisGuide() {
  const group = new Group(), local = new Group(), plane = new Group();
  group.name = "selected-tooth-local-axes";
  group.visible = false; local.matrixAutoUpdate = false;
  const arrows = colors.map(color => new ArrowHelper(new Vector3(1, 0, 0), new Vector3(), 1, color));
  const labels = names.map((name, i) => label(name, colors[i]));
  local.add(...arrows, ...labels);
  const patch = new Mesh(new PlaneGeometry(2, 2), new MeshBasicMaterial({
    color: "#c4d1dd", side: DoubleSide, transparent: true, opacity: .055,
    depthTest: false, depthWrite: false, toneMapped: false,
  }));
  patch.rotation.x = -Math.PI / 2;
  const border = new LineLoop(new BufferGeometry().setFromPoints([
    new Vector3(-1, 0, -1), new Vector3(1, 0, -1), new Vector3(1, 0, 1), new Vector3(-1, 0, 1),
  ]), new LineBasicMaterial({ color: "#c4d1dd", transparent: true, opacity: .65, depthTest: false, depthWrite: false, toneMapped: false }));
  plane.add(patch, border); group.add(plane, local);
  group.traverse(object => { object.renderOrder = 110; object.userData.excludeFromWearAO = true; });
  patch.renderOrder = 101; border.renderOrder = 102;
  for (const arrow of arrows) for (const material of [arrow.line.material, arrow.cone.material].flat()) {
    material.depthTest = false; material.depthWrite = false; material.toneMapped = false;
  }
  return {
    group, local, plane, arrows, labels,
    dispose() {
      for (const arrow of arrows) arrow.dispose();
      for (const sprite of labels) { sprite.material.map?.dispose(); sprite.material.dispose(); }
      patch.geometry.dispose(); patch.material.dispose(); border.geometry.dispose(); border.material.dispose();
      group.clear();
    },
  };
}

export type ToothAxisGuide = ReturnType<typeof createToothAxisGuide>;

export function updateToothAxisGuide(guide: ToothAxisGuide, reference: CrownReference, sourceToWorld: Matrix4, pivotWorld: Vector3) {
  const center = pivotWorld.clone().applyMatrix4(sourceToWorld.clone().invert());
  const longAxis = new Vector3(...reference.tip).sub(new Vector3(...reference.base));
  const length = longAxis.length();
  if (!(length > 1e-8)) { guide.group.visible = false; return; }
  const directions = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1), longAxis.normalize(), new Vector3(...reference.edge).normalize()];
  guide.local.matrix.copy(sourceToWorld);
  directions.forEach((direction, i) => {
    const size = length * (i < 3 ? .46 : i === 3 ? .85 : .95);
    const arrow = guide.arrows[i]; arrow.position.copy(center); arrow.setDirection(direction);
    arrow.setLength(size, length * .065, length * .035);
    guide.labels[i].position.copy(center).addScaledVector(direction, size + length * .08);
    // Keep letters readable when the camera shows the full six-tooth row.
    guide.labels[i].scale.setScalar(.025);
  });
  // This patch indicates orientation only: it is centered on the marker for
  // comparison, but never inherits the crown or row's rotation.
  guide.plane.position.copy(pivotWorld); guide.plane.quaternion.identity(); guide.plane.scale.setScalar(length * .6);
  guide.group.visible = true; guide.group.updateMatrixWorld(true);
}
