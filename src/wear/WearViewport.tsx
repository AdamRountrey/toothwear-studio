import { useEffect, useRef, useState } from "react";
import {
  ACESFilmicToneMapping,
  ArrowHelper,
  AxesHelper,
  BackSide,
  Box3,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  CylinderGeometry,
  DirectionalLight,
  DoubleSide,
  EdgesGeometry,
  Euler,
  Group,
  GridHelper,
  HemisphereLight,
  InstancedMesh,
  LineBasicMaterial,
  LineDashedMaterial,
  LineSegments,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PCFSoftShadowMap,
  PerspectiveCamera,
  Quaternion,
  Scene,
  SphereGeometry,
  Vector3,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import type { VegetationDirection } from "./engine";
import type { VegetationTrace } from "./slidingVegetation";
import { createUpperSetupMatrix, type UpperSetupPose } from "./upperPose";

export type WearMesh = {
  positions: Float32Array;
  indices?: Uint32Array;
  colors?: Float32Array;
  normals?: Float32Array;
};

export type WearView = "perspective" | "front" | "side" | "top";
export type WearShading = "relief" | "studio";

export type WearViewportProps = {
  upper: WearMesh | null;
  lower: WearMesh | null;
  upperOffsetY: number;
  /** Setup pose acts on source meshes only; prepared geometry already contains it. */
  upperPose?: UpperSetupPose;
  upperPivot?: [number, number, number];
  upperPoseBaked?: boolean;
  upperPoseMode?: "translate" | "rotate" | null;
  upperPoseEditable?: boolean;
  onUpperPoseChange?: (pose: UpperSetupPose) => void;
  hinge?: { origin: [number, number, number]; axis: [number, number, number]; visible: boolean };
  hingeMode?: "translate" | "rotate" | null;
  hingeEditable?: boolean;
  onHingeChange?: (origin: [number, number, number], axis: [number, number, number]) => void;
  /** Display-only opening about the closed-pose hinge, in signed degrees. */
  hingePreviewAngle?: number;
  /** Source geometry does not contain the automatic opening baked into prepared wear surfaces. */
  originalUpperOffsetY?: number;
  lowerOffsetY?: number;
  showUpper: boolean;
  showLower: boolean;
  showOriginal: boolean;
  originalUpper?: WearMesh | null;
  originalLower?: WearMesh | null;
  view: WearView;
  fitKey: number;
  shading: WearShading;
  lightAngle: number;
  cavityStrength: number;
  vegetationBand?: { direction: VegetationDirection; centerY: number; thickness: number; bendReach: number; visible: boolean };
  vegetationTrace?: VegetationTrace | null;
  /** Plant-bundle diameter in model units; zero retains the thin-line preview. */
  vegetationStreamThickness?: number;
  showVegetationPaths?: boolean;
  showSlidingContacts?: boolean;
  vegetationXray?: boolean;
};

type SurfaceName = "upper" | "lower" | "originalUpper" | "originalLower";
type Surface = {
  source: WearMesh;
  mesh: Mesh<BufferGeometry, MeshStandardMaterial | MeshBasicMaterial>;
};
type ViewportState = {
  scene: Scene;
  camera: PerspectiveCamera;
  controls: OrbitControls;
  upperGizmo: { controls: TransformControls; pivot: Group; scene: Scene; pointerId: number | null; target: "upper" | "hinge" };
  hingeGuide: { group: Group; arrows: ArrowHelper[]; center: Mesh<SphereGeometry, MeshBasicMaterial> };
  renderer: WebGLRenderer;
  grid: GridHelper;
  axes: AxesHelper;
  surfaces: Partial<Record<SurfaceName, Surface>>;
  geometryCache: Map<WearMesh, { geometry: BufferGeometry; uses: number }>;
  fitted: boolean;
  dirty: boolean;
  composer: EffectComposer;
  ao: GTAOPass;
  hemisphere: HemisphereLight;
  key: DirectionalLight;
  fill: DirectionalLight;
  rim: DirectionalLight;
  modelRadius: number;
  lightDirection: Vector3;
  vegetationGuide: { group: Group; box: Mesh<BoxGeometry, MeshBasicMaterial>; outline: LineSegments<EdgesGeometry, LineBasicMaterial>;
    bendOutline: LineSegments<EdgesGeometry, LineDashedMaterial>; arrows: ArrowHelper[] };
  vegetationTrace: { source?: VegetationTrace | null; diameter?: number; paths: LineSegments<BufferGeometry, LineBasicMaterial>;
    bundles?: InstancedMesh<CylinderGeometry, MeshBasicMaterial>; contacts: Mesh<BufferGeometry, MeshBasicMaterial> };
};

const UPPER_COLOR = "#eddbb8";
const LOWER_COLOR = "#65bfb9";
const SURFACES: SurfaceName[] = ["upper", "lower", "originalUpper", "originalLower"];

/** Ghosts should not cast artificial cavity shading over the remaining solid. */
class WearAmbientOcclusion extends GTAOPass {
  override overrideVisibility() {
    super.overrideVisibility();
    this.scene.traverse(object => { if (object.userData.excludeFromWearAO) object.visible = false; });
  }
}

/** A Y-up scene. Closing the upper row uses world -Y after its setup pose. */
export function WearViewport(props: WearViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<ViewportState>();
  const propsRef = useRef(props);
  const [graphicsError, setGraphicsError] = useState<string>();
  propsRef.current = props;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let renderer: WebGLRenderer;
    try {
      renderer = new WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    } catch {
      setGraphicsError("The 3D view could not start. Enable hardware acceleration in your browser and reload.");
      return;
    }
    const scene = new Scene();
    scene.background = new Color("#111a23");
    const camera = new PerspectiveCamera(38, 1, 0.001, 10000);
    camera.up.set(0, 1, 0);
    camera.position.set(2.4, 1.8, 3);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.shadowMap.type = PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;
    renderer.domElement.style.display = "block";
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.setAttribute("aria-label", "Interactive tooth row view: drag to orbit, scroll to zoom, right drag to pan");
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 0.85;
    controls.rotateSpeed = 0.7;

    // A pivot proxy keeps the raw source buffers untouched while the handles
    // manipulate its rigid pose. Render handles after shading, as a UI overlay.
    const upperPivot = new Group();
    upperPivot.name = "upper-setup-pivot";
    upperPivot.userData.excludeFromWearAO = true;
    scene.add(upperPivot);
    const upperTransform = new TransformControls(camera, renderer.domElement);
    upperTransform.setSpace("world");
    upperTransform.setSize(0.85);
    upperTransform.enabled = false;
    const gizmoScene = new Scene();
    const gizmoHelper = upperTransform.getHelper();
    gizmoHelper.traverse(object => {
      object.userData.excludeFromWearAO = true;
      object.castShadow = false;
      object.receiveShadow = false;
      object.renderOrder = 100;
    });
    gizmoScene.add(gizmoHelper);

    const hingeGroup = new Group();
    hingeGroup.name = "jaw-hinge-axis";
    hingeGroup.visible = false;
    const hingeArrows = [1, -1].map(sign => new ArrowHelper(new Vector3(sign, 0, 0), new Vector3(), 1, "#f7ba68"));
    const hingeCenter = new Mesh(new SphereGeometry(1, 12, 8), new MeshBasicMaterial({ color: "#ffe2aa", depthTest: false, depthWrite: false, toneMapped: false }));
    hingeGroup.add(...hingeArrows, hingeCenter);
    hingeGroup.traverse(object => {
      object.userData.excludeFromWearAO = true;
      object.renderOrder = 5;
    });
    for (const arrow of hingeArrows) for (const object of [arrow.line, arrow.cone]) {
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) { material.depthTest = false; material.depthWrite = false; material.toneMapped = false; }
    }
    scene.add(hingeGroup);

    const hemisphere = new HemisphereLight("#dceafb", "#34404a", 2.3);
    scene.add(hemisphere);
    const key = new DirectionalLight("#fff0db", 2.8);
    key.position.set(4, 7, 6);
    scene.add(key);
    scene.add(key.target);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.bias = -0.00004;
    const fill = new DirectionalLight("#a0d3e5", 1.5);
    fill.position.set(-5, 1, -3);
    scene.add(fill);
    const rim = new DirectionalLight("#ffffff", 1.4);
    rim.position.set(1, -3, -5);
    scene.add(rim);

    const grid = new GridHelper(1, 20, "#3b4f60", "#243440");
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of gridMaterials) {
      material.transparent = true;
      material.opacity = 0.5;
      material.depthWrite = false;
    }
    grid.visible = false;
    scene.add(grid);
    const axes = new AxesHelper(1);
    axes.setColors("#bd7470", "#77ad96", "#7c9fc2");
    axes.visible = false;
    scene.add(axes);

    const vegetationGroup = new Group();
    vegetationGroup.name = "vegetation-band-guide";
    vegetationGroup.userData.excludeFromWearAO = true;
    vegetationGroup.visible = false;
    const vegetationBox = new Mesh(new BoxGeometry(1, 1, 1), new MeshBasicMaterial({ color: "#8dbb44", opacity: 0.045, transparent: true, depthWrite: false, side: DoubleSide, toneMapped: false }));
    const vegetationOutline = new LineSegments(new EdgesGeometry(vegetationBox.geometry), new LineBasicMaterial({ color: "#9bcc66", opacity: 0.55, transparent: true, depthWrite: false, toneMapped: false }));
    const vegetationBendOutline = new LineSegments(new EdgesGeometry(vegetationBox.geometry), new LineDashedMaterial({ color: "#a9c48c", opacity: 0.32, transparent: true, depthWrite: false, toneMapped: false }));
    const vegetationArrows = [0, 1, 2].map(() => new ArrowHelper(new Vector3(1, 0, 0), new Vector3(), 1, "#b8df82"));
    vegetationGroup.add(vegetationBox, vegetationOutline, vegetationBendOutline, ...vegetationArrows);
    vegetationGroup.traverse(object => {
      object.userData.excludeFromWearAO = true;
      object.castShadow = false;
      object.receiveShadow = false;
      object.renderOrder = 3;
    });
    for (const arrow of vegetationArrows) {
      for (const object of [arrow.line, arrow.cone]) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) { material.transparent = true; material.opacity = 0.8; material.depthWrite = false; material.toneMapped = false; }
      }
    }
    scene.add(vegetationGroup);

    const vegetationPaths = new LineSegments(new BufferGeometry(), new LineBasicMaterial({ color: "#ffd86d", transparent: true, opacity: 0.9, depthWrite: false, toneMapped: false }));
    vegetationPaths.name = "computed-vegetation-paths";
    const slidingContacts = new Mesh(new BufferGeometry(), new MeshBasicMaterial({ color: "#ee79dc", transparent: true, opacity: 0.72, depthWrite: false, side: DoubleSide, toneMapped: false,
      polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    slidingContacts.name = "current-sliding-contact-surfaces";
    for (const overlay of [vegetationPaths, slidingContacts]) {
      overlay.userData.excludeFromWearAO = true;
      overlay.castShadow = false;
      overlay.receiveShadow = false;
      overlay.visible = false;
      overlay.renderOrder = overlay === vegetationPaths ? 5 : 4;
      scene.add(overlay);
    }

    const composer = new EffectComposer(renderer);
    composer.renderTarget1.samples = 4;
    composer.renderTarget2.samples = 4;
    const beauty = new RenderPass(scene, camera);
    const ao = new WearAmbientOcclusion(scene, camera, 1, 1);
    ao.normalMaterial.flatShading = true;
    ao.normalMaterial.side = DoubleSide;
    ao.updateGtaoMaterial({ samples: 12, screenSpaceRadius: false, distanceFallOff: 1, scale: 1 });
    ao.updatePdMaterial({ samples: 8, radius: 3, depthPhi: 1, normalPhi: 4 });
    const output = new OutputPass();
    composer.addPass(beauty);
    composer.addPass(ao);
    composer.addPass(output);

    const state: ViewportState = {
      scene, camera, controls, renderer, grid, axes,
      upperGizmo: { controls: upperTransform, pivot: upperPivot, scene: gizmoScene, pointerId: null, target: "upper" },
      hingeGuide: { group: hingeGroup, arrows: hingeArrows, center: hingeCenter },
      surfaces: {}, geometryCache: new Map(), fitted: false, dirty: true,
      composer, ao, hemisphere, key, fill, rim, modelRadius: 1, lightDirection: new Vector3(),
      vegetationGuide: { group: vegetationGroup, box: vegetationBox, outline: vegetationOutline, bendOutline: vegetationBendOutline, arrows: vegetationArrows },
      vegetationTrace: { paths: vegetationPaths, contacts: slidingContacts },
    };
    stateRef.current = state;
    const invalidate = () => { state.dirty = true; };
    controls.addEventListener("change", invalidate);
    const onGizmoChange = () => {
      const current = propsRef.current;
      if (state.upperGizmo.target === "hinge") {
        if (!current.hingeEditable || !current.hingeMode || !current.onHingeChange) return;
        current.onHingeChange(upperPivot.position.toArray(), new Vector3(1, 0, 0).applyQuaternion(upperPivot.quaternion).normalize().toArray());
        invalidate();
        return;
      }
      if (!current.upperPoseEditable || !current.upperPoseMode || !current.onUpperPoseChange) return;
      const pivot = getUpperSourcePivot(state, current);
      const rotation = new Euler().setFromQuaternion(upperPivot.quaternion, "ZYX");
      current.onUpperPoseChange({
        translation: [upperPivot.position.x - pivot[0], upperPivot.position.y - pivot[1], upperPivot.position.z - pivot[2]],
        rotation: [rotation.x * 180 / Math.PI, rotation.y * 180 / Math.PI, rotation.z * 180 / Math.PI],
      });
      invalidate();
    };
    const onGizmoDragging = (event: { value: unknown }) => {
      controls.enabled = !event.value;
      invalidate();
    };
    const rememberGizmoPointer = (event: PointerEvent) => {
      if (upperTransform.dragging) state.upperGizmo.pointerId = event.pointerId;
    };
    const releaseGizmoPointer = () => { state.upperGizmo.pointerId = null; };
    const cancelGizmoDrag = () => { finishUpperPoseDrag(state); invalidate(); };
    upperTransform.addEventListener("change", invalidate);
    upperTransform.addEventListener("objectChange", onGizmoChange);
    upperTransform.addEventListener("dragging-changed", onGizmoDragging);
    renderer.domElement.addEventListener("pointerdown", rememberGizmoPointer);
    renderer.domElement.addEventListener("pointerup", releaseGizmoPointer);
    renderer.domElement.addEventListener("pointercancel", cancelGizmoDrag);
    renderer.domElement.addEventListener("lostpointercapture", cancelGizmoDrag);
    window.addEventListener("blur", cancelGizmoDrag);
    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      invalidate();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    const onContextLost = (event: Event) => {
      event.preventDefault();
      setGraphicsError("The browser interrupted the 3D view. Waiting for graphics to recover…");
    };
    const onContextRestored = () => { setGraphicsError(undefined); invalidate(); };
    renderer.domElement.addEventListener("webglcontextlost", onContextLost);
    renderer.domElement.addEventListener("webglcontextrestored", onContextRestored);
    let frame = 0;
    const animate = () => {
      frame = window.requestAnimationFrame(animate);
      controls.update();
      // Keep the GPU idle when the scene is unchanged, including with million-triangle sources.
      if (state.dirty) {
        updateLighting(state, propsRef.current);
        if (propsRef.current.shading === "relief") composer.render();
        else renderer.render(scene, camera);
        if (gizmoHelper.visible) {
          const autoClear = renderer.autoClear;
          renderer.autoClear = false;
          renderer.clearDepth();
          renderer.render(gizmoScene, camera);
          renderer.autoClear = autoClear;
        }
        state.dirty = false;
      }
    };
    frame = window.requestAnimationFrame(animate);

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      finishUpperPoseDrag(state);
      upperTransform.removeEventListener("change", invalidate);
      upperTransform.removeEventListener("objectChange", onGizmoChange);
      upperTransform.removeEventListener("dragging-changed", onGizmoDragging);
      renderer.domElement.removeEventListener("pointerdown", rememberGizmoPointer);
      renderer.domElement.removeEventListener("pointerup", releaseGizmoPointer);
      renderer.domElement.removeEventListener("pointercancel", cancelGizmoDrag);
      renderer.domElement.removeEventListener("lostpointercapture", cancelGizmoDrag);
      window.removeEventListener("blur", cancelGizmoDrag);
      upperTransform.dispose();
      gizmoScene.clear();
      controls.removeEventListener("change", invalidate);
      controls.dispose();
      renderer.domElement.removeEventListener("webglcontextlost", onContextLost);
      renderer.domElement.removeEventListener("webglcontextrestored", onContextRestored);
      for (const name of SURFACES) releaseSurface(state, name);
      grid.geometry.dispose();
      for (const material of gridMaterials) material.dispose();
      axes.geometry.dispose();
      const axisMaterials = Array.isArray(axes.material) ? axes.material : [axes.material];
      for (const material of axisMaterials) material.dispose();
      vegetationBox.geometry.dispose();
      vegetationBox.material.dispose();
      vegetationOutline.geometry.dispose();
      vegetationOutline.material.dispose();
      vegetationBendOutline.geometry.dispose();
      vegetationBendOutline.material.dispose();
      for (const arrow of vegetationArrows) arrow.dispose();
      for (const arrow of hingeArrows) arrow.dispose();
      hingeCenter.geometry.dispose(); hingeCenter.material.dispose();
      for (const overlay of [vegetationPaths, slidingContacts]) { overlay.geometry.dispose(); overlay.material.dispose(); }
      disposeBundleSleeves(state);
      scene.clear();
      key.shadow.dispose();
      ao.dispose();
      // r171's pass disposal omits these two shader materials.
      ao.gtaoMaterial.dispose();
      ao.blendMaterial.dispose();
      output.dispose();
      beauty.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      if (stateRef.current === state) stateRef.current = undefined;
    };
  }, []);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    replaceSurface(state, "upper", props.upper);
    replaceSurface(state, "lower", props.lower);
    replaceSurface(state, "originalUpper", props.originalUpper ?? null);
    replaceSurface(state, "originalLower", props.originalLower ?? null);
    syncVisibilityAndPose(state, propsRef.current);
    syncUpperPoseGizmo(state, propsRef.current);
    syncHingeGuide(state, propsRef.current);
    syncVegetationGuide(state, propsRef.current);
    if (!state.fitted) frameCamera(state, propsRef.current.view, propsRef.current.hinge);
    state.dirty = true;
  }, [props.upper, props.lower, props.originalUpper, props.originalLower]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    syncVisibilityAndPose(state, props);
    syncUpperPoseGizmo(state, props);
    syncHingeGuide(state, props);
    syncVegetationGuide(state, props);
    state.dirty = true;
  }, [props.upperOffsetY, props.originalUpperOffsetY, props.lowerOffsetY, props.upperPose, props.upperPivot, props.upperPoseBaked, props.showUpper, props.showLower, props.showOriginal, props.hinge, props.hingePreviewAngle]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    syncUpperPoseGizmo(state, props);
    state.dirty = true;
  }, [props.upperPoseMode, props.upperPoseEditable, props.hingeMode, props.hingeEditable]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    syncVegetationGuide(state, props);
    state.dirty = true;
  }, [props.vegetationBand?.direction, props.vegetationBand?.centerY, props.vegetationBand?.thickness, props.vegetationBand?.bendReach, props.vegetationBand?.visible]);

  useEffect(() => {
    const state = stateRef.current;
    if (!state) return;
    const overlay = state.vegetationTrace;
    const changedTrace = overlay.source !== props.vegetationTrace;
    if (changedTrace) {
      overlay.source = props.vegetationTrace;
      for (const [object, positions] of [[overlay.paths, props.vegetationTrace?.pathPositions], [overlay.contacts, props.vegetationTrace?.contactPositions]] as const) {
        object.geometry.dispose();
        object.geometry = new BufferGeometry();
        if (positions?.length) {
          object.geometry.setAttribute("position", new BufferAttribute(positions, 3));
          object.geometry.computeBoundingSphere();
        }
      }
    }
    const diameter = props.vegetationStreamThickness ?? 0;
    if (changedTrace || overlay.diameter !== diameter) {
      overlay.diameter = diameter;
      disposeBundleSleeves(state);
      if (diameter > 0 && Number.isFinite(diameter) && props.vegetationTrace?.pathPositions.length) {
        overlay.bundles = makeBundleSleeves(props.vegetationTrace.pathPositions, diameter);
        state.scene.add(overlay.bundles);
      }
    }
    overlay.paths.visible = !!props.showVegetationPaths && !!props.vegetationTrace?.pathPositions.length;
    if (overlay.bundles) {
      overlay.bundles.visible = overlay.paths.visible;
      overlay.bundles.material.depthTest = !props.vegetationXray;
      overlay.bundles.material.opacity = props.vegetationXray ? 0.08 : 0.16;
    }
    overlay.contacts.visible = !!props.showSlidingContacts && !!props.vegetationTrace?.contactPositions.length;
    overlay.paths.material.depthTest = !props.vegetationXray;
    overlay.contacts.material.depthTest = !props.vegetationXray;
    overlay.contacts.material.opacity = props.vegetationXray ? 0.4 : 0.72;
    state.dirty = true;
  }, [props.vegetationTrace, props.vegetationStreamThickness, props.showVegetationPaths, props.showSlidingContacts, props.vegetationXray]);

  useEffect(() => {
    const state = stateRef.current;
    if (state) { frameCamera(state, props.view, props.hinge); syncHingeGuide(state, props); }
  }, [props.fitKey, props.view]);

  useEffect(() => {
    const state = stateRef.current;
    if (state) state.dirty = true;
  }, [props.shading, props.lightAngle, props.cavityStrength]);

  return (
    <div className="wear-viewport" style={{ position: "absolute", inset: 0, overflow: "hidden", background: "#111a23" }}>
      <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />
      {graphicsError && (
        <div role="alert" style={{ position: "absolute", inset: 24, display: "grid", placeItems: "center", color: "#eddbb8", textAlign: "center", pointerEvents: "none" }}>
          {graphicsError}
        </div>
      )}
    </div>
  );
}

/** Preserve the routed corners while collapsing collinear grid steps to one instance. */
function makeBundleSleeves(positions: Float32Array, diameter: number) {
  const runs: number[] = [];
  let lastDirection = 0;
  for (let i = 0; i + 5 < positions.length; i += 6) {
    const dx = positions[i + 3] - positions[i], dy = positions[i + 4] - positions[i + 1], dz = positions[i + 5] - positions[i + 2];
    if (dx === 0 && dy === 0 && dz === 0) continue;
    const direction = dx !== 0 ? Math.sign(dx) : dy !== 0 ? Math.sign(dy) * 2 : Math.sign(dz) * 3;
    const end = runs.length;
    if (end && direction === lastDirection && runs[end - 3] === positions[i] && runs[end - 2] === positions[i + 1] && runs[end - 1] === positions[i + 2]) {
      runs[end - 3] = positions[i + 3]; runs[end - 2] = positions[i + 4]; runs[end - 1] = positions[i + 5];
    } else {
      runs.push(positions[i], positions[i + 1], positions[i + 2], positions[i + 3], positions[i + 4], positions[i + 5]);
    }
    lastDirection = direction;
  }
  // One instanced draw for the bounded representative routes, not one scene
  // object per routing edge. Open cylinders keep contact overlays legible.
  const sleeves = new InstancedMesh(new CylinderGeometry(1, 1, 1, 8, 1, true),
    new MeshBasicMaterial({ color: "#ffd86d", transparent: true, opacity: 0.16, depthWrite: false, toneMapped: false }), runs.length / 6);
  sleeves.name = "vegetation-bundle-diameter";
  sleeves.userData.excludeFromWearAO = true;
  sleeves.renderOrder = 3;
  const start = new Vector3(), end = new Vector3(), direction = new Vector3(), midpoint = new Vector3();
  const up = new Vector3(0, 1, 0), scale = new Vector3(), rotation = new Quaternion(), matrix = new Matrix4();
  for (let i = 0; i < runs.length; i += 6) {
    start.fromArray(runs, i); end.fromArray(runs, i + 3);
    direction.subVectors(end, start);
    const length = direction.length();
    rotation.setFromUnitVectors(up, direction.divideScalar(length));
    midpoint.addVectors(start, end).multiplyScalar(0.5);
    scale.set(diameter / 2, length, diameter / 2);
    sleeves.setMatrixAt(i / 6, matrix.compose(midpoint, rotation, scale));
  }
  sleeves.instanceMatrix.needsUpdate = true;
  sleeves.computeBoundingSphere();
  return sleeves;
}

function disposeBundleSleeves(state: ViewportState) {
  const sleeves = state.vegetationTrace.bundles;
  if (!sleeves) return;
  sleeves.removeFromParent();
  sleeves.geometry.dispose(); sleeves.material.dispose(); sleeves.dispose();
  state.vegetationTrace.bundles = undefined;
}

function releaseSurface(state: ViewportState, name: SurfaceName) {
  const surface = state.surfaces[name];
  if (!surface) return;
  surface.mesh.removeFromParent();
  surface.mesh.material.dispose();
  const cached = state.geometryCache.get(surface.source);
  if (cached && --cached.uses === 0) {
    cached.geometry.dispose();
    state.geometryCache.delete(surface.source);
  }
  delete state.surfaces[name];
}

function replaceSurface(state: ViewportState, name: SurfaceName, source: WearMesh | null) {
  if (state.surfaces[name]?.source === source) return;
  releaseSurface(state, name);
  if (!source || source.positions.length < 9) return;
  let cached = state.geometryCache.get(source);
  if (!cached) {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(source.positions, 3));
    if (source.indices) geometry.setIndex(new BufferAttribute(source.indices, 1));
    if (source.colors?.length === source.positions.length) {
      geometry.setAttribute("color", new BufferAttribute(source.colors, 3));
    }
    // The worker already supplies face normals for sampled surfaces. Shadow
    // bias needs vertex normals even when visible shading uses flat derivatives.
    if (source.normals?.length === source.positions.length) {
      geometry.setAttribute("normal", new BufferAttribute(source.normals, 3));
    } else {
      geometry.computeVertexNormals();
    }
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    cached = { geometry, uses: 0 };
    state.geometryCache.set(source, cached);
  }
  cached.uses++;
  const original = name.startsWith("original");
  const color = name.toLowerCase().includes("upper") ? UPPER_COLOR : LOWER_COLOR;
  const material = original
    ? new MeshBasicMaterial({ color, transparent: true, opacity: 0.12, depthWrite: false, side: DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })
    : new MeshStandardMaterial({ color: cached.geometry.hasAttribute("color") ? "#ffffff" : color, vertexColors: cached.geometry.hasAttribute("color"), roughness: 0.66, metalness: 0.025, flatShading: true, side: DoubleSide });
  const mesh = new Mesh(cached.geometry, material);
  // Closed solids cast from their back faces, avoiding self-shadow speckling.
  if (!original) material.shadowSide = BackSide;
  mesh.name = `wear-${name}`;
  mesh.userData.excludeFromWearAO = original;
  mesh.castShadow = !original;
  mesh.receiveShadow = !original;
  mesh.renderOrder = original ? 1 : 0;
  state.scene.add(mesh);
  state.surfaces[name] = { source, mesh };
}

function getUpperSourcePivot(state: ViewportState, props: WearViewportProps): [number, number, number] {
  const sourceUpper = state.surfaces.originalUpper ?? state.surfaces.upper;
  const sourceCenter = sourceUpper?.mesh.geometry.boundingBox?.getCenter(new Vector3());
  return props.upperPivot ?? (sourceCenter?.toArray() as [number, number, number] | undefined) ?? [0, 0, 0];
}

function finishUpperPoseDrag(state: ViewportState) {
  const { controls, pointerId } = state.upperGizmo;
  state.upperGizmo.pointerId = null;
  controls.pointerUp(null);
  const canvas = state.renderer.domElement;
  if (pointerId !== null && canvas.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
  state.controls.enabled = true;
}

function syncUpperPoseGizmo(state: ViewportState, props: WearViewportProps) {
  const gizmo = state.upperGizmo;
  const hinge = !!props.hinge?.visible && !!props.hingeEditable && !!props.hingeMode && !!props.onHingeChange;
  const active = !!state.surfaces.upper && (hinge || !!props.upperPoseEditable && !!props.upperPoseMode && !!props.onUpperPoseChange && props.showUpper);
  const target = hinge ? "hinge" : "upper";
  const mode = hinge ? props.hingeMode : props.upperPoseMode;
  if (!active) {
    finishUpperPoseDrag(state);
    gizmo.controls.enabled = false;
    gizmo.controls.detach();
    return;
  }
  if (gizmo.controls.mode !== mode || gizmo.target !== target) {
    finishUpperPoseDrag(state);
    gizmo.controls.setMode(mode!);
    gizmo.target = target;
  }
  // React receives every objectChange. Never feed that pose back into the
  // active drag: doing so changes the drag origin and can jump Euler angles.
  if (!gizmo.controls.dragging) {
    if (hinge) {
      gizmo.pivot.position.fromArray(props.hinge!.origin);
      gizmo.pivot.quaternion.setFromUnitVectors(new Vector3(1, 0, 0), new Vector3(...props.hinge!.axis).normalize());
    } else {
      const pivot = getUpperSourcePivot(state, props);
      const pose = props.upperPose;
      gizmo.pivot.position.set(pivot[0] + (pose?.translation[0] ?? 0), pivot[1] + (pose?.translation[1] ?? 0), pivot[2] + (pose?.translation[2] ?? 0));
      const [rx, ry, rz] = pose?.rotation.map(angle => angle % 360 * Math.PI / 180) ?? [0, 0, 0];
      gizmo.pivot.quaternion.setFromEuler(new Euler(rx, ry, rz, "ZYX"));
    }
    gizmo.pivot.updateMatrixWorld(true);
  }
  if (gizmo.controls.object !== gizmo.pivot) gizmo.controls.attach(gizmo.pivot);
  gizmo.controls.enabled = true;
}

function syncVisibilityAndPose(state: ViewportState, props: WearViewportProps) {
  const pivot = getUpperSourcePivot(state, props);
  const setupMatrix = props.upperPose ? createUpperSetupMatrix(props.upperPose, pivot) : new Matrix4();
  const previewMatrix = new Matrix4();
  if (props.hinge && props.hingePreviewAngle && Number.isFinite(props.hingePreviewAngle)) {
    const origin = new Vector3(...props.hinge.origin);
    previewMatrix.makeTranslation(origin.x, origin.y, origin.z)
      .multiply(new Matrix4().makeRotationAxis(new Vector3(...props.hinge.axis).normalize(), props.hingePreviewAngle * Math.PI / 180))
      .multiply(new Matrix4().makeTranslation(-origin.x, -origin.y, -origin.z));
  }
  for (const name of SURFACES) {
    const surface = state.surfaces[name];
    if (!surface) continue;
    const upper = name.toLowerCase().includes("upper");
    const original = name.startsWith("original");
    const offsetY = upper
      ? (original ? props.originalUpperOffsetY ?? props.upperOffsetY : props.upperOffsetY)
      : props.lowerOffsetY ?? 0;
    // Reuse the original vertex buffers while editing the setup pose. Closure
    // and automatic opening always move along world Y, after setup rotation.
    const mesh = surface.mesh;
    mesh.matrixAutoUpdate = false;
    if (upper && (original || !props.upperPoseBaked)) mesh.matrix.copy(setupMatrix);
    else mesh.matrix.identity();
    mesh.matrix.elements[13] += offsetY;
    if (upper) mesh.matrix.premultiply(previewMatrix);
    mesh.updateMatrixWorld(true);
    surface.mesh.visible = (upper ? props.showUpper : props.showLower) && (!original || props.showOriginal);
  }
}

function syncHingeGuide(state: ViewportState, props: WearViewportProps) {
  const guide = state.hingeGuide;
  guide.group.visible = !!props.hinge?.visible && !!state.surfaces.upper;
  if (!guide.group.visible || !props.hinge) return;
  guide.group.position.fromArray(props.hinge.origin);
  const axis = new Vector3(...props.hinge.axis).normalize();
  const length = Math.max(state.modelRadius * 0.65, 0.001);
  guide.arrows.forEach((arrow, index) => {
    arrow.setDirection(axis.clone().multiplyScalar(index === 0 ? 1 : -1));
    arrow.setLength(length, length * 0.07, length * 0.035);
  });
  guide.center.scale.setScalar(length * 0.018);
}

function updateLighting(state: ViewportState, props: WearViewportProps) {
  const relief = props.shading === "relief";
  state.renderer.toneMappingExposure = relief ? 0.95 : 1.12;
  state.renderer.shadowMap.enabled = relief;
  state.renderer.shadowMap.needsUpdate = relief;
  state.hemisphere.intensity = relief ? 0.38 : 2.3;
  state.key.intensity = relief ? 3.1 : 2.8;
  state.fill.intensity = relief ? 0.18 : 1.5;
  state.rim.intensity = relief ? 0.35 : 1.4;
  state.ao.enabled = relief && props.cavityStrength > 0;
  // GTAO's blend is a linear mix: values above one can produce negative RGB.
  // Use its AO exponent for the stronger half of the contrast control instead.
  state.ao.blendIntensity = Math.min(1, props.cavityStrength);
  state.ao.updateGtaoMaterial({ scale: Math.max(1, props.cavityStrength) });
  if (relief) {
    // Rotate the key around the view direction, keeping relief readable as the camera orbits.
    const angle = props.lightAngle * Math.PI / 180;
    state.lightDirection.set(Math.cos(angle), Math.sin(angle), 0.75)
      .normalize().applyQuaternion(state.camera.quaternion);
    state.key.position.copy(state.controls.target).addScaledVector(state.lightDirection, state.modelRadius * 3);
    state.key.target.position.copy(state.controls.target);
  } else {
    state.key.position.set(4, 7, 6);
    state.key.target.position.set(0, 0, 0);
  }
}

function syncVegetationGuide(state: ViewportState, props: WearViewportProps) {
  const guide = state.vegetationGuide;
  const band = props.vegetationBand;
  guide.group.visible = !!band?.visible;
  if (!band?.visible) return;
  const bounds = new Box3();
  // Sources keep the guide stable as material is removed; helpers never enter camera fitting.
  for (const name of ["upper", "lower"] as const) {
    const originalName = name === "upper" ? "originalUpper" : "originalLower";
    const surface = state.surfaces[originalName] ?? state.surfaces[name];
    if (surface?.mesh.geometry.boundingBox) bounds.union(surface.mesh.geometry.boundingBox.clone().applyMatrix4(surface.mesh.matrixWorld));
  }
  if (bounds.isEmpty()) { guide.group.visible = false; return; }
  const size = bounds.getSize(new Vector3());
  const center = bounds.getCenter(new Vector3());
  const margin = Math.max(size.x, size.z, 1e-6) * 0.035;
  const width = Math.max(size.x, margin) + margin * 2;
  const depth = Math.max(size.z, margin) + margin * 2;
  guide.group.position.set(center.x, band.centerY, center.z);
  guide.box.scale.set(width, band.thickness, depth);
  guide.outline.scale.copy(guide.box.scale);
  const alongX = band.direction.endsWith("x");
  guide.bendOutline.visible = band.bendReach > 0;
  if (guide.bendOutline.visible) {
    // This is only an upper bound on the reachable region, not a fabricated path trace.
    // Store world-sized vertices so dashes have the same length on all box edges.
    const positions = guide.bendOutline.geometry.getAttribute("position");
    const bendWidth = width + (alongX ? 0 : band.bendReach * 2);
    const bendHeight = band.thickness + band.bendReach * 2;
    const bendDepth = depth + (alongX ? band.bendReach * 2 : 0);
    for (let index = 0; index < positions.count; index++) {
      positions.setXYZ(index, Math.sign(positions.getX(index)) * bendWidth / 2,
        Math.sign(positions.getY(index)) * bendHeight / 2, Math.sign(positions.getZ(index)) * bendDepth / 2);
    }
    positions.needsUpdate = true;
    guide.bendOutline.geometry.computeBoundingSphere();
    guide.bendOutline.computeLineDistances();
    guide.bendOutline.material.dashSize = margin * 0.5;
    guide.bendOutline.material.gapSize = margin * 0.4;
  }
  const sign = band.direction.startsWith("+") ? 1 : -1;
  const direction = new Vector3(alongX ? sign : 0, 0, alongX ? 0 : sign);
  const length = (alongX ? width : depth) * 0.72;
  const span = (alongX ? depth : width) * 0.3;
  guide.arrows.forEach((arrow, index) => {
    arrow.setDirection(direction);
    arrow.position.set(alongX ? -sign * length / 2 : (index - 1) * span, 0, alongX ? (index - 1) * span : -sign * length / 2);
    const headLength = Math.min(length * 0.12, Math.max(width, depth) * 0.035);
    arrow.setLength(length, headLength, headLength * 0.45);
  });
}

function frameCamera(state: ViewportState, view: WearView, hinge?: WearViewportProps["hinge"]) {
  const bounds = new Box3();
  for (const name of SURFACES) {
    const surface = state.surfaces[name];
    if (!surface || (name.startsWith("original") && !surface.mesh.visible)) continue;
    const localBounds = surface.mesh.geometry.boundingBox;
    if (localBounds) bounds.union(localBounds.clone().applyMatrix4(surface.mesh.matrixWorld));
  }
  if (bounds.isEmpty()) return;
  // The jaw joint is posterior to the isolated toothrow, so include its origin
  // when fitting; otherwise the corrected transverse hinge can be off screen.
  if (hinge?.visible) bounds.expandByPoint(new Vector3(...hinge.origin));
  const size = bounds.getSize(new Vector3());
  const center = bounds.getCenter(new Vector3());
  const radius = Math.max(size.length() / 2, 1e-6);
  state.modelRadius = radius;
  state.key.shadow.camera.left = state.key.shadow.camera.bottom = -radius * 1.25;
  state.key.shadow.camera.right = state.key.shadow.camera.top = radius * 1.25;
  state.key.shadow.camera.near = radius * 0.05;
  state.key.shadow.camera.far = radius * 7;
  state.key.shadow.normalBias = radius * 0.001;
  state.key.shadow.camera.updateProjectionMatrix();
  state.ao.updateGtaoMaterial({ radius: radius * 0.045, thickness: radius * 0.025 });
  const verticalFov = state.camera.fov * Math.PI / 180;
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * state.camera.aspect);
  const distance = radius * 1.12 / Math.sin(Math.min(verticalFov, horizontalFov) / 2);
  const direction = view === "front" ? new Vector3(0, 0, 1)
    : view === "side" ? new Vector3(1, 0, 0)
      : view === "top" ? new Vector3(0, 1, 0.00001).normalize()
        : new Vector3(1, 0.65, 1.2).normalize();
  // Keep OrbitControls' Y-up convention stable; a tiny top-view tilt avoids its polar singularity.
  state.camera.up.set(0, 1, 0);
  state.camera.position.copy(center).addScaledVector(direction, distance);
  state.camera.near = Math.max(radius / 10000, 1e-8);
  state.camera.far = Math.max(distance + radius * 100, 1);
  state.camera.updateProjectionMatrix();
  state.controls.target.copy(center);
  state.controls.minDistance = radius * 0.025;
  state.controls.maxDistance = distance * 25;
  state.controls.update();
  state.controls.saveState();
  const gridSize = Math.max(size.x, size.z, radius) * 1.8;
  state.grid.scale.setScalar(gridSize);
  state.grid.position.set(center.x, bounds.min.y - radius * 0.14, center.z);
  state.grid.visible = true;
  state.axes.scale.setScalar(gridSize * 0.12);
  state.axes.position.set(center.x - gridSize * 0.45, state.grid.position.y, center.z + gridSize * 0.45);
  state.axes.visible = true;
  state.fitted = true;
  state.dirty = true;
}
