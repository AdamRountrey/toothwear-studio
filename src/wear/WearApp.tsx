import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowDownToLine, ArrowLeftRight, Check, ChevronRight, Eye, Focus, Layers3, Leaf, LoaderCircle, Move3d, Pause, Play, Rotate3d, RotateCcw, SkipForward, Upload } from "lucide-react";
import { WearViewport, type WearMesh, type WearView, type WearShading } from "./WearViewport";
import type { VegetationDirection, VegetationSettings, WearSnapshot } from "./engine";
import type { ComputePreference, SourceInfo, WearComputeStatus, WearRequest, WearResponse, WearTimings } from "./protocol";
import type { VegetationTrace } from "./slidingVegetation";
import { identityUpperSetupPose, isValidUpperSetupPose, type UpperSetupPose } from "./upperPose";
import { createDefaultBiteCycleSettings, validateBiteCycleSettings, type BiteCycleMetrics, type BiteCycleSettings } from "./biteCycles";
import { makeCrownMatrices, crownRowPitch, spaceCrownRows, type CrownRowLayout } from "./crownRows";
import type { CrownSpacingCheck } from "./crownSpacing";
import nonImbricatedPreset from "./presets/nonImbricated.json";
import s128Preset from "./presets/s128.json";
import y13S083Preset from "./presets/y13S083.json";
import { crownRotationForEdit, estimateCrownReference, measureCrownInclination, type CrownAngleEdit } from "./crownOrientation";
import { CrownAngles } from "./CrownAngles";
import { geometricRunComplete, nextGeometricWearRequest, type GeometricRunSettings, type JawMotion } from "./runControl";

type Row = "upper" | "lower";
type Rotation = [number, number, number];
type SetupTarget = Row | "upper-crown" | "lower-crown" | "hinge";
const identityCrownRotations = (): [Rotation, Rotation] => [[0, 0, 0], [0, 0, 0]];
const usageNotice = "Experimental — not for research use. Simulated wear has not been validated.";

type HistoryPoint = { closure: number; upper: number; lower: number; contact: number; contactUpper: number; contactLower: number; chippedUpper: number; chippedLower: number; vegetationPull: number; vegetationUpper: number; vegetationLower: number; vegetationSettings: VegetationSettings | null };
const format = (value: number | undefined, digits = 2) => value === undefined || !Number.isFinite(value) ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
const duration = (milliseconds: number) => milliseconds < 1000 ? `${format(milliseconds, 0)} ms` : `${format(milliseconds / 1000, 2)} s`;
const download = (data: BlobPart, name: string, type: string) => {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 30000);
};

export function WearApp() {
  const worker = useRef<Worker>();
  const run = useRef(false);
  const timer = useRef<number>();
  const manualPullPending = useRef(false);
  const initializeVegetationBand = useRef(true);
  const traceRequestId = useRef(0);
  const settings = useRef<GeometricRunSettings & { mode: "geometric" | "cycles"; cycleTarget: number; cycleSettings?: BiteCycleSettings }>({ target: 0, step: 0.25, share: 0.5, mode: "geometric", cycleTarget: 10, jawMotion: "closing", pullTarget: 250, pullStep: 25 });
  const [original, setOriginal] = useState<[WearMesh, WearMesh] | null>(null);
  const [surfaces, setSurfaces] = useState<[WearMesh, WearMesh] | null>(null);
  const [info, setInfo] = useState<[SourceInfo, SourceInfo] | null>(null);
  const [upperPose, setUpperPose] = useState<UpperSetupPose>(identityUpperSetupPose);
  const [upperPoseMode, setUpperPoseMode] = useState<"translate" | "rotate" | null>("translate");
  const [lowerPose, setLowerPose] = useState<UpperSetupPose>(identityUpperSetupPose);
  const [lowerPoseMode, setLowerPoseMode] = useState<"translate" | "rotate" | null>(null);
  const [sourceCrownRows, setCrownRows] = useState<[CrownRowLayout, CrownRowLayout] | null>(null);
  const [crownPitch, setCrownPitch] = useState(27);
  const validCrownPitch = Number.isFinite(crownPitch) && crownPitch > 0 && crownPitch <= 1e6;
  const crownRows = useMemo(() => sourceCrownRows ? spaceCrownRows(sourceCrownRows, validCrownPitch ? crownPitch : crownRowPitch(sourceCrownRows[0])) : null, [sourceCrownRows, crownPitch, validCrownPitch]);
  const [crownRotations, setCrownRotations] = useState<[Rotation, Rotation]>(identityCrownRotations);
  const crownCheckKey = JSON.stringify([crownPitch, crownRotations]);
  const crownCheckRequest = useRef({ id: 0, key: "" });
  const [crownCheck, setCrownCheck] = useState<{ key: string; result: CrownSpacingCheck } | null>(null);
  const [crownCheckError, setCrownCheckError] = useState("");
  const currentCrownCheck = crownCheck?.key === crownCheckKey ? crownCheck.result : null;
  const validCrownSpacing = !sourceCrownRows || validCrownPitch && currentCrownCheck?.clear === true;
  const [selectedCrowns, setSelectedCrowns] = useState<[number, number]>([2, 2]);
  const [crownEditing, setCrownEditing] = useState(false);
  const [importKind, setImportKind] = useState<"rows" | "crowns">("rows");
  const [mode, setMode] = useState<"geometric" | "cycles">("geometric");
  const [jawMotion, setJawMotion] = useState<JawMotion>("closing");
  const [panel, setPanel] = useState<"run" | "setup" | "display">("run");
  const rail = useRef<HTMLElement>(null);
  const [cycleSettings, setCycleSettings] = useState<BiteCycleSettings>(() => createDefaultBiteCycleSettings({ min: [-1, -1, -1], max: [1, 1, 1] }, 0.5));
  const [cycle, setCycle] = useState<BiteCycleMetrics | null>(null);
  const [cycleTarget, setCycleTarget] = useState(10);
  const [setupTarget, setSetupTarget] = useState<SetupTarget>("upper");
  const [hingeMode, setHingeMode] = useState<"translate" | "rotate" | null>(null);
  const [hingePreview, setHingePreview] = useState(0);
  const [showHinge, setShowHinge] = useState(true);
  const [hingePreset, setHingePreset] = useState<"negative" | "positive" | "custom">("negative");
  const rowAxis = info && info[0].max[2] - info[0].min[2] > info[0].max[0] - info[0].min[0] ? "Z" : "X";
  const upperPreviewPose = useRef<UpperSetupPose>(identityUpperSetupPose());
  const upperPivot = useMemo<[number, number, number]>(() => info
    ? [0, 1, 2].map(axis => ((crownRows ?? info)[0].min[axis] + (crownRows ?? info)[0].max[axis]) / 2) as [number, number, number]
    : [0, 0, 0], [info, crownRows]);
  const lowerPivot = useMemo<Rotation>(() => info
    ? [0, 1, 2].map(axis => ((crownRows ?? info)[1].min[axis] + (crownRows ?? info)[1].max[axis]) / 2) as Rotation
    : [0, 0, 0], [info, crownRows]);
  const crownArrays = useMemo(() => crownRows && original ? {
    upper: { crown: original[0], transforms: makeCrownMatrices(crownRows[0], crownRotations[0]), pivots: crownRows[0].centers, base: crownRows[0].base },
    lower: { crown: original[1], transforms: makeCrownMatrices(crownRows[1], crownRotations[1]), pivots: crownRows[1].centers, base: crownRows[1].base },
  } : undefined, [crownRows, original, crownRotations]);
  const selectedRow: Row = setupTarget.startsWith("lower") ? "lower" : "upper";
  const selectedRowIndex = selectedRow === "upper" ? 0 : 1;
  const selectedPose = selectedRow === "upper" ? upperPose : lowerPose;
  const eachCrownTarget = setupTarget.endsWith("-crown");
  const crownReferences = useMemo(() => sourceCrownRows && original
    ? original.map((mesh, row) => estimateCrownReference(mesh, row === 0 ? "upper" : "lower")) : null, [sourceCrownRows, original]);
  const crownInclinations = useMemo(() => crownReferences?.map((reference, row) => reference
    ? measureCrownInclination(reference, crownRotations[row], row === 0 ? upperPose.rotation : lowerPose.rotation, row === 0 ? "upper" : "lower") : null),
    [crownReferences, crownRotations, upperPose.rotation, lowerPose.rotation]);
  const activeSetupMode = setupTarget === "hinge" ? hingeMode : eachCrownTarget ? crownEditing ? "rotate" : null : selectedRow === "upper" ? upperPoseMode : lowerPoseMode;
  const [files, setFiles] = useState<[File | null, File | null]>([null, null]);
  const [snapshot, setSnapshot] = useState<WearSnapshot | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [grid, setGrid] = useState<{ nx: number; nz: number; spacing: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Load two aligned toothrows to begin.");
  const [error, setError] = useState("");
  const [workerFailed, setWorkerFailed] = useState(false);
  const [compute, setCompute] = useState<WearComputeStatus>({ preference: "auto", backend: "initializing" });
  const [computeSwitching, setComputeSwitching] = useState(false);
  const [timings, setTimings] = useState<WearTimings | null>(null);
  const [spacing, setSpacing] = useState(0.5);
  const [contactSpread, setContactSpread] = useState(0);
  const [chipThickness, setChipThickness] = useState(0);
  const [step, setStep] = useState(0.25);
  const [depth, setDepth] = useState(5);
  const [stopMode, setStopMode] = useState<"contact" | "total">("contact");
  const [share, setShare] = useState(50);
  const [autoOpen, setAutoOpen] = useState(true);
  const [dirty, setDirty] = useState(false);
  const [unit, setUnit] = useState("u");
  const [view, setView] = useState<WearView>("front");
  const [fitKey, setFitKey] = useState(0);
  const [showUpper, setShowUpper] = useState(true);
  const [showLower, setShowLower] = useState(true);
  const [showOriginal, setShowOriginal] = useState(false);
  const [shading, setShading] = useState<WearShading>("relief");
  const [lightAngle, setLightAngle] = useState(135);
  const [cavityStrength, setCavityStrength] = useState(1.5);
  const [vegetationEnabled, setVegetationEnabled] = useState(false);
  const [vegetationDirection, setVegetationDirection] = useState<VegetationDirection>("+x");
  const [vegetationPathMode, setVegetationPathMode] = useState<"straight" | "flexible">("flexible");
  const [vegetationBendReach, setVegetationBendReach] = useState(2.5);
  const [vegetationCenterY, setVegetationCenterY] = useState(0);
  const [vegetationThickness, setVegetationThickness] = useState(5);
  const [vegetationStreamThickness, setVegetationStreamThickness] = useState(1);
  const [abrasionRate, setAbrasionRate] = useState(0.02);
  const [pullPerClosure, setPullPerClosure] = useState(10);
  const [pullDistance, setPullDistance] = useState(25);
  const [pullTarget, setPullTarget] = useState(250);
  const [showVegetation, setShowVegetation] = useState(true);
  const [showVegetationPaths, setShowVegetationPaths] = useState(true);
  const [showSlidingContacts, setShowSlidingContacts] = useState(true);
  const [vegetationXray, setVegetationXray] = useState(false);
  const [vegetationTrace, setVegetationTrace] = useState<VegetationTrace | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState("");
  const vegetationSettings: VegetationSettings = { direction: vegetationDirection, centerY: vegetationCenterY, thickness: vegetationThickness, abrasionRate, pullPerClosure,
    pathMode: vegetationPathMode, bendReach: vegetationPathMode === "flexible" ? vegetationBendReach : 0, streamThickness: vegetationStreamThickness };
  const validVegetation = Number.isFinite(vegetationCenterY) && Number.isFinite(vegetationThickness) && vegetationThickness > 0
    && Number.isFinite(vegetationStreamThickness) && vegetationStreamThickness >= 0
    && Number.isFinite(abrasionRate) && abrasionRate > 0 && Number.isFinite(pullPerClosure) && pullPerClosure >= 0
    && (vegetationPathMode === "straight" || Number.isFinite(vegetationBendReach) && vegetationBendReach >= 0);
  const first = snapshot?.firstContactClosure ?? 0;
  const target = depth + (stopMode === "contact" ? first : 0);
  const closure = snapshot?.closure ?? 0;
  const isCycles = mode === "cycles";
  const fixedJaw = !isCycles && jawMotion === "fixed";
  const cumulativePull = snapshot?.vegetation.pullDistance ?? 0;
  const noMaterial = !!snapshot && snapshot.remainingUpperVolume <= 0 && snapshot.remainingLowerVolume <= 0;
  const maxContactSpread = Number.isFinite(spacing) && spacing > 0 ? spacing * 16 / 3 : 0;
  const validContactSpread = isCycles || Number.isFinite(contactSpread) && contactSpread >= 0 && contactSpread <= maxContactSpread;
  const validChipping = isCycles || Number.isFinite(chipThickness) && chipThickness >= 0 && chipThickness <= spacing * 8 && !(chipThickness > 0 && vegetationEnabled);
  const gaussianPrepared = !isCycles && (snapshot?.contactSpread ?? 0) > 0;
  let cycleSettingsError = "";
  try { validateBiteCycleSettings(cycleSettings); } catch (cause) { cycleSettingsError = cause instanceof Error ? cause.message : "Check bite-cycle settings."; }
  const completedCycles = cycle?.completedCycles ?? 0;
  const runSettings: GeometricRunSettings = { jawMotion, target, step, share: share / 100,
    vegetation: vegetationEnabled && !isCycles ? vegetationSettings : undefined, pullTarget, pullStep: pullDistance };
  const complete = !!snapshot && !dirty && (isCycles ? completedCycles >= cycleTarget : geometricRunComplete(snapshot, runSettings));
  const validRun = isCycles ? !cycleSettingsError && Number.isInteger(cycleTarget) && cycleTarget > 0
    : fixedJaw ? vegetationEnabled && validVegetation && Number.isFinite(pullTarget) && pullTarget > 0
      && Number.isFinite(pullDistance) && pullDistance > 0 && chipThickness === 0
    : Number.isFinite(target) && depth > 0 && Number.isFinite(step) && step > 0 && (!vegetationEnabled || validVegetation);
  const locked = busy || running || workerFailed || computeSwitching;
  const toothAxisReference = crownReferences?.[selectedRowIndex];
  const showToothAxes = !!crownRows && !!toothAxisReference && !locked && !surfaces && setupTarget !== "hinge" && activeSetupMode === "rotate";
  const vegetationStarted = (snapshot?.vegetation.pullDistance ?? 0) > 0;
  const vegetationLocked = locked || vegetationStarted || isCycles || chipThickness > 0;
  const setupStarted = isCycles ? completedCycles > 0 : closure > 0 || vegetationStarted;
  const poseLocked = locked || !original || setupStarted;
  const cycleSettingsLocked = locked || completedCycles > 0;
  const validUpperPose = isValidUpperSetupPose(upperPose);
  const validSetup = validUpperPose && isValidUpperSetupPose(lowerPose);
  if (validUpperPose) upperPreviewPose.current = upperPose;
  settings.current = { ...runSettings, mode, cycleTarget, cycleSettings };

  const post = (message: WearRequest) => worker.current?.postMessage(message);
  const clearTrace = () => { traceRequestId.current++; setVegetationTrace(null); setTraceLoading(false); setTraceError(""); post({ type: "trace-off" }); };
  const pause = () => { run.current = false; setRunning(false); window.clearTimeout(timer.current); setStatus(busy ? "Pausing after the current increment…" : "Paused. Continue the run or inspect the current pose."); };

  useEffect(() => {
    // A development hot refresh restarts workers: never keep a stale visible
    // result whose geometry is no longer owned by the active worker.
    setOriginal(null); setSurfaces(null); setInfo(null); setSnapshot(null);
    setUpperPose(identityUpperSetupPose());
    setUpperPoseMode("translate");
    setLowerPose(identityUpperSetupPose()); setLowerPoseMode(null);
    setCrownRows(null); setCrownRotations(identityCrownRotations()); setCrownEditing(false); setSelectedCrowns([2, 2]);
    setCrownCheck(null); setCrownCheckError(""); crownCheckRequest.current.id++;
    setCycle(null); setHingeMode(null); setHingePreview(0); setSetupTarget("upper");
    setHistory([]); setWarnings([]); setGrid(null); setDirty(false);
    setBusy(false); setRunning(false); setError(""); setProgress(0); setWorkerFailed(false);
    setCompute({ preference: "auto", backend: "initializing" }); setComputeSwitching(false); setTimings(null);
    setVegetationTrace(null); setTraceLoading(false); setTraceError(""); traceRequestId.current++;
    setStatus("Load two aligned toothrows to begin.");
    initializeVegetationBand.current = true;
    const instance = new Worker(new URL("./wear.worker.ts", import.meta.url), { type: "module" });
    worker.current = instance;
    // An explicit launch link starts a fresh, unworn setup in this tab only.
    const presetId = new URLSearchParams(window.location.search).get("preset")?.toLowerCase();
    const launchPreset = presetId === "non-imbricated" ? nonImbricatedPreset
      : presetId === "s128" ? s128Preset : presetId === "y13-s083" ? y13S083Preset : null;
    let presetPending = !!launchPreset;
    const fail = (message: string) => {
      run.current = false; manualPullPending.current = false; setRunning(false); setBusy(false); setComputeSwitching(false); setError(message); setStatus("Check the setup and try again.");
    };
    instance.onerror = event => {
      setWorkerFailed(true);
      fail(event.message || "The wear worker stopped unexpectedly. Restart the app to continue.");
    };
    instance.onmessage = (event: MessageEvent<WearResponse>) => {
      const message = event.data;
      if (message.type === "crown-check" || message.type === "crown-check-error") {
        if (message.requestId === crownCheckRequest.current.id) {
          setCrownCheck(message.type === "crown-check" ? { key: crownCheckRequest.current.key, result: message.result } : null);
          setCrownCheckError(message.type === "crown-check-error" ? message.message : "");
        }
        return;
      }
      if (message.type === "compute-status") { setCompute(message.status); setComputeSwitching(message.status.backend === "initializing"); return; }
      if (message.type === "trace") {
        if (message.requestId === traceRequestId.current) { setVegetationTrace(message.trace); setTraceLoading(false); setTraceError(""); }
        return;
      }
      if (message.type === "trace-error") {
        if (message.requestId === traceRequestId.current) { setVegetationTrace(null); setTraceLoading(false); setTraceError(message.message); }
        return;
      }
      if (message.type === "progress") { setProgress(message.fraction); setStatus(message.message); return; }
      if (message.type === "error") { fail(message.message); return; }
      if (message.type === "exported") {
        download(message.buffer, `${message.row}-worn.ply`, "application/octet-stream");
        setBusy(false); setStatus(`Exported ${message.row} row in its current assembly pose.`); return;
      }
      if (message.type === "loaded") {
        setOriginal([message.upper, message.lower]); setSurfaces(null); setInfo(message.info);
        setUpperPose(identityUpperSetupPose());
        setUpperPoseMode("translate");
        setLowerPose(identityUpperSetupPose()); setLowerPoseMode(null);
        setCrownRows(message.crownRows ?? null); setCrownRotations(identityCrownRotations()); setCrownEditing(false); setSelectedCrowns([2, 2]);
        setCrownPitch(message.crownRows ? crownRowPitch(message.crownRows[0]) : 27); setCrownCheck(null); setCrownCheckError("");
        setCycle(null); setHingeMode(null); setHingePreview(0); setSetupTarget("upper");
        setCycleSettings(createDefaultBiteCycleSettings(message.info[0], message.suggestedSpacing));
        setHingePreset("negative");
        setSnapshot(null); setHistory([]); setWarnings([]); setGrid(null); setDirty(false); setTimings(null);
        setSpacing(message.suggestedSpacing); setContactSpread(0); setStep(message.suggestedSpacing / 2);
        setVegetationThickness(message.suggestedSpacing * 10); setPullDistance(message.suggestedSpacing * 50); setPullTarget(message.suggestedSpacing * 500);
        setVegetationStreamThickness(message.suggestedSpacing * 2);
        setVegetationBendReach(message.suggestedSpacing * 5);
        initializeVegetationBand.current = true;
        setBusy(false); setProgress(0); setFitKey(key => key + 1);
        setStatus(message.crownRows ? "Six upper and six lower crowns ready. Select a whole row or rotate each crown together, then prepare the wear grid." : "Surfaces loaded. Adjust the row setup poses, then prepare the wear grid.");
        if (presetPending && launchPreset && message.crownRows) {
          presetPending = false;
          const { setup, settings: p } = launchPreset;
          const upper = { translation: setup.upper.translation, rotation: setup.upper.rotationDegrees };
          const lower = { translation: setup.lower.translation, rotation: setup.lower.rotationDegrees };
          if (!isValidUpperSetupPose(upper) || !isValidUpperSetupPose(lower)) { fail("Invalid saved crown setup."); return; }
          const rotations = setup.crowns.rotationsDegrees as [Rotation, Rotation];
          setUpperPose(upper); setLowerPose(lower); setUpperPoseMode(null); setLowerPoseMode(null);
          setCrownRotations(rotations); setCrownPitch(setup.crowns.pitch);
          setMode("geometric"); setJawMotion("closing"); setSpacing(p.spacing); setContactSpread(p.contactSpread); setChipThickness(p.chipThickness);
          setAutoOpen(p.autoOpen); setShare(p.upperWearShare * 100); setStep(p.increment);
          setStopMode(p.stopMode === "total" ? "total" : "contact"); setDepth(p.stopDistance); setVegetationEnabled(false);
          setBusy(true); setStatus(`Preparing ${launchPreset.name}: restoring crown poses, spacing, and wear settings.`);
          // The worker independently checks crown clearance before sampling.
          instance.postMessage({ type: "prepare", spacing: p.spacing, autoOpen: p.autoOpen,
            contactSpread: p.contactSpread, chipThickness: p.chipThickness, upperPose: upper, lowerPose: lower,
            crownRotations: rotations, crownPitch: setup.crowns.pitch, mode: "geometric" } satisfies WearRequest);
        }
        return;
      }
      if (message.type === "frame") {
        setCycle(message.cycle ?? null);
        setTimings(message.timings ?? null);
        setSurfaces([message.upper, message.lower]); setSnapshot(message.snapshot);
        setVegetationTrace(message.traceRequestId === traceRequestId.current ? message.trace ?? null : null);
        if (message.traceRequestId === traceRequestId.current && message.trace) { setTraceLoading(false); setTraceError(""); }
        setWarnings(message.warnings); setGrid(message.grid); setDirty(false); setProgress(1); setBusy(false);
        const s = message.snapshot;
        setSpacing(message.grid.spacing);
        if (message.mode) setMode(message.mode);
        if (message.preparation) {
          setAutoOpen(message.preparation.autoOpen); setUpperPose(message.preparation.upperPose);
          setLowerPose(message.preparation.lowerPose ?? identityUpperSetupPose());
          setCrownRotations(message.preparation.crownRotations ?? identityCrownRotations());
          if (message.preparation.crownPitch) setCrownPitch(message.preparation.crownPitch);
        }
        if (message.mode !== "cycles") {
          setContactSpread(s.contactSpread ?? 0); setChipThickness(s.chipThickness ?? 0);
          if ((s.chipThickness ?? 0) > 0) { setVegetationEnabled(false); setJawMotion("closing"); }
        }
        if (initializeVegetationBand.current) { setVegetationCenterY(Number(s.recommendedVegetationCenterY.toPrecision(7))); initializeVegetationBand.current = false; }
        setHistory(previous => {
          const point: HistoryPoint = { closure: message.mode === "cycles" ? message.cycle?.completedCycles ?? 0 : s.closure, upper: s.removedUpperVolume, lower: s.removedLowerVolume, contact: message.cycle?.contactArea ?? s.contactArea,
            contactUpper: s.contactRemovedUpperVolume, contactLower: s.contactRemovedLowerVolume,
            chippedUpper: s.chippedUpperVolume ?? 0, chippedLower: s.chippedLowerVolume ?? 0,
            vegetationPull: s.vegetation.pullDistance, vegetationUpper: s.vegetation.removedUpperVolume, vegetationLower: s.vegetation.removedLowerVolume, vegetationSettings: s.vegetation.settings };
          if (message.mode === "cycles" ? !message.cycle?.completedCycles : s.closure === 0 && s.vegetation.pullDistance === 0) return [point];
          const last = previous.at(-1);
          return [...(last?.closure === point.closure && last.vegetationPull === point.vegetationPull ? previous.slice(0, -1) : previous), point];
        });
        const config = settings.current;
        if (message.mode === "cycles") {
          const bites = message.cycle?.completedCycles ?? 0;
          const finished = bites >= config.cycleTarget;
          setStatus(finished ? "Preset bite count reached. Worn surfaces are ready to export."
            : bites === 0 ? "Closed reference pose ready. Position the hinge, preview opening, then run repeated bites."
            : `Completed ${format(bites, 0)} bite${bites === 1 ? "" : "s"}. Contact pressure and sliding distance determine wear.`);
          if (run.current && !finished) {
            timer.current = window.setTimeout(() => {
              if (!run.current || !config.cycleSettings) return;
              setBusy(true);
              instance.postMessage({ type: "cycles", count: 1, settings: config.cycleSettings } satisfies WearRequest);
            }, 45);
          } else { run.current = false; setRunning(false); }
          return;
        }
        const finished = geometricRunComplete(s, config);
        setStatus(config.jawMotion === "fixed" ? s.remainingUpperVolume <= 0 && s.remainingLowerVolume <= 0
          ? "Stopped: no tooth material remains."
          : finished ? "Vegetation pull target reached. Jaw position stayed fixed; worn surfaces are ready to export."
          : run.current ? "Vegetation wear is advancing. Both toothrows stay at the current pose."
          : "Fixed-jaw vegetation ready. Run or step vegetation without closing the jaw."
          : manualPullPending.current ? `Vegetation pulled at fixed closure. ${format(s.vegetation.effectiveRecession, 3)} model units of cumulative nominal recession applied continuously at sliding contacts.`
          : s.materialExhausted ? `Stopped: all opposing material paths have been exhausted.${config.vegetation ? " Vegetation can still be pulled at this pose." : ""}`
          : finished ? "Preset stopping point reached. Worn surfaces are ready to export."
          : s.closure === 0 ? "Wear grid ready. Start the run or advance one increment."
          : s.contactColumns === 0 ? (s.contactSpread ?? 0) > 0
            ? "No direct same-cell contact at this pose. Gaussian neighboring-cell wear remains active during closure."
            : "No tooth-to-tooth contact at the current pose. Continue closure or inspect vegetation wear."
          : run.current ? "Mutual wear is advancing at the contacting surfaces." : "Current pose is in contact. Continue the run or export the worn surfaces.");
        manualPullPending.current = false;
        if (run.current && !finished) {
          timer.current = window.setTimeout(() => {
            if (!run.current) return;
            const request = nextGeometricWearRequest(s, settings.current);
            if (!request) { run.current = false; setRunning(false); return; }
            setBusy(true);
            instance.postMessage(request);
          }, 45);
        } else { run.current = false; setRunning(false); }
      }
    };
    if (launchPreset) {
      setBusy(true); setStatus(`Loading ${launchPreset.name} · six teeth per row…`);
      instance.postMessage({ type: "load", kind: "crowns", sources: [
        { name: launchPreset.sources[0], url: new URL(`${import.meta.env.BASE_URL}models/${launchPreset.sources[0]}`, document.baseURI).href },
        { name: launchPreset.sources[1], url: new URL(`${import.meta.env.BASE_URL}models/${launchPreset.sources[1]}`, document.baseURI).href },
      ] } satisfies WearRequest);
    }
    return () => { run.current = false; window.clearTimeout(timer.current); instance.terminate(); worker.current = undefined; };
  }, []);

  useEffect(() => {
    const requestId = ++crownCheckRequest.current.id;
    crownCheckRequest.current.key = crownCheckKey;
    setCrownCheck(null); setCrownCheckError("");
    if (!sourceCrownRows || !validCrownPitch || workerFailed) return;
    const debounce = window.setTimeout(() => post({ type: "crown-check", requestId, pitch: crownPitch, rotations: crownRotations }), 250);
    return () => { window.clearTimeout(debounce); if (crownCheckRequest.current.id === requestId) crownCheckRequest.current.id++; };
  }, [sourceCrownRows, crownCheckKey, validCrownPitch, workerFailed]);

  const preparedForTrace = !!snapshot && !dirty;
  const wantsTrace = !isCycles && vegetationEnabled && (showVegetationPaths || showSlidingContacts);
  useEffect(() => {
    const requestId = ++traceRequestId.current;
    // Drop remembered previews immediately: an older setting must never appear
    // over newer geometry while the debounced request is being prepared.
    post({ type: "trace-off" });
    setVegetationTrace(null); setTraceError("");
    if (!preparedForTrace || !wantsTrace || !validVegetation || workerFailed) { setTraceLoading(false); return; }
    setTraceLoading(true);
    const debounce = window.setTimeout(() => post({ type: "trace", requestId, settings: {
      direction: vegetationDirection, pathMode: vegetationPathMode, bendReach: vegetationPathMode === "flexible" ? vegetationBendReach : 0,
      centerY: vegetationCenterY, thickness: vegetationThickness, streamThickness: vegetationStreamThickness, abrasionRate, pullPerClosure,
    } }), 140);
    return () => { window.clearTimeout(debounce); if (traceRequestId.current === requestId) traceRequestId.current++; };
  }, [preparedForTrace, wantsTrace, validVegetation, workerFailed, vegetationDirection, vegetationPathMode, vegetationBendReach, vegetationCenterY, vegetationThickness, vegetationStreamThickness, abrasionRate, pullPerClosure]);

  const load = (local: boolean) => {
    clearTrace();
    pause(); setBusy(true); setError(""); setStatus("Reading toothrow files…"); setProgress(0);
    post({ type: "load", kind: !local && importKind === "crowns" ? "crowns" : undefined, sources: local ? [
      { name: "1aligned.ply", url: new URL(`${import.meta.env.BASE_URL}models/1aligned.ply`, document.baseURI).href }, { name: "2aligned.ply", url: new URL(`${import.meta.env.BASE_URL}models/2aligned.ply`, document.baseURI).href }
    ] : [{ name: files[0]!.name, file: files[0]! }, { name: files[1]!.name, file: files[1]! }] });
  };
  const loadCrowns = () => {
    clearTrace(); pause(); setBusy(true); setError(""); setStatus("Loading single crowns and building six teeth per row…"); setProgress(0);
    post({ type: "load", kind: "crowns", sources: [
      { name: "Camarasaurus_crown_hole_upper.stl", url: new URL(`${import.meta.env.BASE_URL}models/Camarasaurus_crown_hole_upper.stl`, document.baseURI).href },
      { name: "Camarasaurus_crown_hole_lower.stl", url: new URL(`${import.meta.env.BASE_URL}models/Camarasaurus_crown_hole_lower.stl`, document.baseURI).href },
    ] });
  };
  const stopSetupTools = () => { setUpperPoseMode(null); setLowerPoseMode(null); setCrownEditing(false); setHingeMode(null); };
  const prepare = (reset = false, forFixedJaw = false) => {
    if (!original || locked || !reset && (!validSetup || !validCrownSpacing || !validContactSpread || !forFixedJaw && !validChipping || !Number.isFinite(spacing) || spacing <= 0)) return;
    if (forFixedJaw) { setChipThickness(0); setJawMotion("fixed"); setVegetationEnabled(true); }
    showPanel("run");
    clearTrace();
    pause(); setBusy(true); setError(""); setStatus("Sampling solid tooth geometry…"); setProgress(0);
    setSnapshot(null); setSurfaces(null); setHistory([]); setTimings(null);
    setCycle(null); setHingePreview(0); stopSetupTools();
    post(reset ? { type: "reset" } : { type: "prepare", spacing, autoOpen, contactSpread: isCycles ? 0 : contactSpread, chipThickness: isCycles || forFixedJaw ? 0 : chipThickness, upperPose, lowerPose, crownRotations, crownPitch, mode });
  };
  const invalidateSetup = (message: string) => {
    clearTrace();
    setDirty(true); setError("");
    // A setup edit invalidates sampling. Preview raw sources in the new pose;
    // simulation and exports stay unavailable until that pose is prepared.
    setSnapshot(null); setSurfaces(null); setHistory([]); setGrid(null); setWarnings([]); setTimings(null); setCycle(null); setHingePreview(0);
    setStatus(message);
  };
  const changeRowPose = (row: Row, next: UpperSetupPose) => {
    if (poseLocked) return;
    if (!isValidUpperSetupPose(next)) { setError("Setup pose must contain finite translation and rotation values."); return; }
    (row === "upper" ? setUpperPose : setLowerPose)(next);
    invalidateSetup("Whole-row setup preview. Prepare the wear grid to apply this pose.");
  };
  const changeUpperPose = (next: UpperSetupPose) => changeRowPose("upper", next);
  const changeLowerPose = (next: UpperSetupPose) => changeRowPose("lower", next);
  const changeCrownRotation = (row: Row, rotation: Rotation) => {
    if (poseLocked || !crownRows) return;
    if (!isValidUpperSetupPose({ translation: [0, 0, 0], rotation })) return;
    setCrownRotations(current => row === "upper" ? [[...rotation], current[1]] : [current[0], [...rotation]]);
    invalidateSetup(`All six ${row} crowns rotate about their own centers. Prepare the wear grid to apply this tilt.`);
  };
  const changeCrownAngles = (row: Row, edit: CrownAngleEdit): string | null => {
    if (poseLocked) return "Reset to unworn geometry before editing the setup.";
    const index = row === "upper" ? 0 : 1, reference = crownReferences?.[index];
    if (!reference) return "A long axis could not be measured for this crown.";
    try {
      const rotation = crownRotationForEdit(reference, crownRotations[index], row === "upper" ? upperPose.rotation : lowerPose.rotation, row, edit);
      stopSetupTools(); setSetupTarget(`${row}-crown`); setCrownEditing(true);
      changeCrownRotation(row, rotation);
      return null;
    } catch (cause) { return cause instanceof Error ? cause.message : String(cause); }
  };
  const changeCrownPitch = (value: number) => {
    if (poseLocked || !sourceCrownRows) return;
    setCrownPitch(value);
    invalidateSetup("Tooth spacing changed. Checking crown clearance before preparation.");
  };
  const restoreSetup = async (file: File) => {
    if (poseLocked) return;
    setBusy(true); setError("");
    try {
      const report = JSON.parse(await file.text()) as { setup?: { upper?: { translation?: unknown; rotationDegrees?: unknown }; lower?: { translation?: unknown; rotationDegrees?: unknown }; crowns?: { rotationsDegrees?: unknown; pitch?: unknown } } } | null;
      const pose = { translation: report?.setup?.upper?.translation, rotation: report?.setup?.upper?.rotationDegrees };
      if (!isValidUpperSetupPose(pose)) throw new Error("This JSON does not contain a valid exported upper-row setup pose.");
      const lower = report?.setup?.lower ? { translation: report.setup.lower.translation, rotation: report.setup.lower.rotationDegrees } : identityUpperSetupPose();
      if (!isValidUpperSetupPose(lower)) throw new Error("This JSON does not contain a valid lower-row setup pose.");
      const rotations = report?.setup?.crowns?.rotationsDegrees ?? identityCrownRotations();
      if (report?.setup?.crowns && !crownRows) throw new Error("Load the single-crown arrays before restoring this crown setup.");
      if (!Array.isArray(rotations) || rotations.length !== 2 || !rotations.every(rotation => isValidUpperSetupPose({ translation: [0, 0, 0], rotation }))) throw new Error("This JSON does not contain valid crown rotations.");
      const restoredPitch = report?.setup?.crowns?.pitch ?? (sourceCrownRows ? crownRowPitch(sourceCrownRows[0]) : 27);
      if (typeof restoredPitch !== "number" || !Number.isFinite(restoredPitch) || restoredPitch <= 0 || restoredPitch > 1e6) throw new Error("This JSON does not contain valid tooth spacing.");
      setUpperPose({ translation: [...pose.translation], rotation: [...pose.rotation] });
      setLowerPose({ translation: [...lower.translation], rotation: [...lower.rotation] });
      setCrownRotations(rotations as [Rotation, Rotation]);
      setCrownPitch(restoredPitch);
      invalidateSetup("Restored row poses and crown tilts. Prepare the grid to apply them; worn material and run settings were not restored.");
    } catch (cause) {
      setError(cause instanceof SyntaxError ? "Choose a valid run-settings JSON file." : cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(false); }
  };
  const editSetup = (mode: "translate" | "rotate", target: SetupTarget = setupTarget) => {
    if (poseLocked) return;
    showPanel(target === "hinge" ? "run" : "setup");
    stopSetupTools(); setSetupTarget(target); setHingePreview(0);
    if (target === "hinge") { setHingeMode(mode); setShowHinge(true); return; }
    // Return to the raw setup pose before showing the handles. An automatic
    // opening belongs to preparation, never to the user's rigid transform.
    if (surfaces) invalidateSetup("Setup preview. Prepare the wear grid to apply the edited geometry.");
    if (target.endsWith("-crown")) { if (crownRows) setCrownEditing(true); }
    else (target === "upper" ? setUpperPoseMode : setLowerPoseMode)(mode);
  };
  const selectCrown = (row: Row, index: number) => {
    if (poseLocked || !crownRows) return;
    setSelectedCrowns(current => row === "upper" ? [index, current[1]] : [current[0], index]);
    editSetup("rotate", `${row}-crown`);
  };
  const advance = (continuous: boolean) => {
    if (!snapshot || !validRun || dirty || complete || locked) return;
    const request = isCycles ? { type: "cycles" as const, count: 1, settings: cycleSettings }
      : nextGeometricWearRequest(snapshot, settings.current);
    if (!request) return;
    setError(""); run.current = continuous; setRunning(continuous); setBusy(true);
    setHingePreview(0); stopSetupTools();
    post(request);
  };
  const contactJump = () => {
    setBusy(true); setError("");
    post({ type: "advance", target: Math.min(target, first), share: share / 100, vegetation: settings.current.vegetation });
  };
  const pullVegetation = () => {
    if (!snapshot || dirty || locked || isCycles || !vegetationEnabled || !validVegetation || !Number.isFinite(pullDistance) || pullDistance <= 0) return;
    manualPullPending.current = true;
    setBusy(true); setError(""); setStatus("Pulling abrasive vegetation through the current pose…");
    post({ type: "vegetation", distance: pullDistance, settings: vegetationSettings });
  };
  const changeCompute = (preference: ComputePreference) => {
    if (locked || traceLoading || compute.backend === "initializing") return;
    setComputeSwitching(true);
    setCompute(current => ({ ...current, preference }));
    post({ type: "compute", preference });
  };
  const changeMode = (next: "geometric" | "cycles") => {
    if (locked || next === mode) return;
    clearTrace(); setMode(next); setSnapshot(null); setSurfaces(null); setCycle(null); setHistory([]); setGrid(null);
    setTimings(null); setWarnings([]); setError(""); setDirty(true); setHingePreview(0); stopSetupTools();
    setSetupTarget(next === "cycles" ? "hinge" : "upper"); setHingeMode(null);
    setFitKey(key => key + 1);
    setStatus("Prepare the wear grid for the selected simulation mode. Switching modes starts from unworn geometry.");
  };
  const changeJawMotion = (next: JawMotion) => {
    if (locked || isCycles || next === jawMotion || next === "fixed" && (chipThickness > 0 || (snapshot?.chipThickness ?? 0) > 0)) return;
    setJawMotion(next); setError("");
    if (next === "fixed") {
      setVegetationEnabled(true);
      // A completed closure run may already have pulled more than the default
      // target. Offer another block of travel when switching to held wear.
      if (!Number.isFinite(pullTarget) || pullTarget <= cumulativePull) {
        setPullTarget(cumulativePull + (Number.isFinite(pullDistance) && pullDistance > 0 ? pullDistance * 10 : 250));
      }
    }
    setStatus(next === "fixed" ? "Jaw held at the current pose. Set the total vegetation pull distance, then run."
      : "Closure wear selected. Continue closing from the current pose.");
  };
  function showPanel(next: "run" | "setup" | "display") {
    setPanel(next);
    rail.current?.scrollTo({ top: 0 });
  }
  const changeCycleSetting = <K extends keyof BiteCycleSettings>(key: K, value: BiteCycleSettings[K]) => {
    if (cycleSettingsLocked) return;
    setCycleSettings(current => ({ ...current, [key]: value })); setError("");
  };
  const editHinge = (next: "translate" | "rotate") => {
    editSetup(next, "hinge");
  };
  const placeHinge = (preset: "negative" | "positive" | "custom") => {
    if (!info || cycleSettingsLocked) return;
    setHingePreset(preset);
    if (preset === "custom") return;
    const suggested = createDefaultBiteCycleSettings(info[0], spacing, preset === "negative" ? -1 : 1);
    setCycleSettings(current => ({ ...current, hingeOrigin: suggested.hingeOrigin, hingeAxis: suggested.hingeAxis,
      openingAngle: Math.abs(current.openingAngle || 8) * Math.sign(suggested.openingAngle) }));
    setHingePreview(0); setShowHinge(true); setSetupTarget("hinge"); stopSetupTools(); setHingeMode("rotate");
    setFitKey(key => key + 1);
  };
  const exportReport = () => {
    setHingePreview(0);
    download(JSON.stringify({ application: "Toothwear Studio", version: 12, usageNotice, mode, chippingModel: "contact-seeded-axis-width-cap-trimming-v1", vegetationModel: "continuous-finite-bundle-sliding-v3", model: isCycles
      ? "Repeated hinge opening and closing about a common closed reference pose, with compliant distributed contact pressure and Archard wear depth = coefficient × pressure × sliding distance. Uncalibrated mechanical approximation; no plant coupling or biological timescale."
      : `Geometric mutual contact wear. Contact spread sigma ${(snapshot?.contactSpread ?? 0) > 0 ? "enables normalized Gaussian-weighted neighboring grid-cell contact with a vertical-contact check; direct local overlap is still resolved" : "is zero, preserving exact same-cell contact"}. Contact spread is a geometric approximation, not contact pressure. Continuous fractional vegetation recession follows complete straight or bounded flexible sliding routes. Stream thickness is plant-bundle diameter; bundles can slide against multiple teeth and both rows at once. Total volume loss is geometric; the contact/vegetation split is estimated within each integration interval. No calibrated branch mechanics, forces, or biological wear rates.`,
      exportedAt: new Date().toISOString(), units: unit === "u" ? "native model units (unspecified)" : unit,
      sources: info, grid, movement: isCycles ? "Both row setup poses and any local crown tilts are applied before sampling. A single common closed pose is established for the toothrow, then the upper rotates about the configured hinge. The lower stays fixed during simulation. Exported PLYs use the closed reference assembly pose; the opening preview is display-only." : fixedJaw ? "Vegetation-only run: both toothrows remain at the current assembly pose. Vegetation slides and abrades exposed surfaces without advancing closure. Any earlier closure/contact wear remains in the history. Exported PLYs include setup poses, initial opening, and the held closure." : "Both row setup poses and any local crown tilts are applied before sampling. During closure the upper translates along world -Y; lower stays fixed. Exported PLYs include setup poses, initial opening, and final closure.",
      setup: { upper: { translation: upperPose.translation, rotationDegrees: upperPose.rotation, pivot: upperPivot,
        rotationOrder: "world X, then Y, then Z about the source bounding-box center; translation applied afterward" },
        lower: { translation: lowerPose.translation, rotationDegrees: lowerPose.rotation, pivot: lowerPivot, rotationOrder: "world X, then Y, then Z about the source bounding-box center; translation applied afterward" },
        crowns: crownRows ? { countPerRow: 6, pitch: crownPitch, spacingCheck: currentCrownCheck, rotationsDegrees: crownRotations, centers: crownRows.map(row => row.centers), sourcePivots: crownRows.map(row => row.sourcePivot),
          rotationOrder: "row-local X, then Y, then Z about each crown center; the same rotation applies to each of the six crowns before the whole-row pose; bases follow only the whole-row pose",
          orientationReadout: { method: "Unworn source section-area centers at 20% and 80% crown height. Pitch = along-row projection, roll = across-row projection, 90 degrees perpendicular to fixed world XZ occlusal plane. Row directions follow the projected row X axis. Edge yaw is a middle-section width-chord estimate, distinct from multi-section matrix scoring. Includes whole-row rotation; independent of closure and wear.", sourceReferences: crownReferences, anglesDegrees: crownInclinations } } : null },
      settings: isCycles ? { autoOpen, targetCycles: cycleTarget, ...cycleSettings, loadSchedule: "Opening is unloaded; force smoothly rises during closure to the configured peak at the shared closed pose.", pressureUnits: `N/${unit}²`, stiffnessUnits: `N/${unit}³`, coefficientUnits: `${unit}²/N` }
        : { autoOpen, jawMotion, targetPullDistance: fixedJaw ? pullTarget : null, pullIncrement: fixedJaw ? pullDistance : null, heldClosure: fixedJaw ? closure : null, contactSpread: snapshot?.contactSpread ?? 0, chipThickness: snapshot?.chipThickness ?? 0, upperWearShare: share / 100, increment: step, stopMode, stopDistance: depth, targetClosure: target, vegetation: snapshot?.vegetation.settings ?? (vegetationEnabled ? vegetationSettings : null) },
      compute: { ...compute, latestFrameTimings: timings }, snapshot, cycle, warnings, history }, null, 2), "toothwear-run.json", "application/json");
  };
  const progressTarget = isCycles ? cycleTarget : fixedJaw ? pullTarget : target;
  const progressValue = isCycles ? completedCycles : fixedJaw ? cumulativePull : closure;
  const percent = snapshot && Number.isFinite(progressTarget) && progressTarget > 0 ? Math.min(100, progressValue / progressTarget * 100) : 0;
  const canAdvance = !!snapshot && !dirty && !complete && validRun && !locked;
  const canPull = !!snapshot && !dirty && !locked && !isCycles && vegetationEnabled && validVegetation && Number.isFinite(pullDistance) && pullDistance > 0
    && !!surfaces?.some(surface => surface.positions.length > 0);
  const gpuStages = isCycles ? [...(compute.cycleBackend === "gpu" ? ["contact pressure & sliding"] : []), ...(compute.materialBackend === "gpu" ? ["material removal"] : []), ...(compute.meshBackend === "gpu" ? ["surface generation"] : [])]
    : ["routing", ...(compute.contactBackend === "gpu" ? ["sliding contacts"] : []), ...(compute.materialBackend === "gpu" ? ["vegetation removal"] : []), ...(compute.meshBackend === "gpu" ? ["surface generation"] : [])];
  const cpuStages = isCycles ? ["cycle orchestration, surface preparation & bookkeeping", ...(compute.cycleBackend === "gpu" ? [] : ["contact pressure & sliding"]), ...(compute.materialBackend === "gpu" ? [] : ["material removal"]), ...(compute.meshBackend === "gpu" ? [] : ["surface generation"])]
    : [fixedJaw ? "run bookkeeping" : (snapshot?.contactSpread ?? 0) > 0 ? "geometric tooth contact" : "exact tooth contact", ...(compute.contactBackend === "gpu" ? [] : ["sliding contacts"]), ...(compute.materialBackend === "gpu" ? [] : ["vegetation removal"]), ...(compute.meshBackend === "gpu" ? [] : ["surface generation"])];

  return (
    <div className="wear-app">
      <header className="wear-header">
        <div className="wear-brand"><span className="wear-brand-mark"><Layers3 size={23} /></span><div><h1>Toothwear <span>Studio</span></h1><p>Progressive occlusion & mutual wear</p></div></div>
        <div className="wear-header-right"><span className="wear-method">{isCycles ? "CYCLIC CONTACT WEAR" : fixedJaw ? "FIXED-JAW VEGETATION" : "GEOMETRIC WEAR"}</span></div>
      </header>
      <aside className="wear-experimental-notice" role="note" aria-label="Experimental use warning"><strong>Experimental — not for research use.</strong><span>Simulated wear has not been validated.</span></aside>
      <main className="wear-workspace">
        <section className="wear-stage" aria-label="Toothrow visualization">
          <WearViewport upper={(surfaces ?? original)?.[0] ?? null} lower={(surfaces ?? original)?.[1] ?? null}
            originalUpper={original?.[0]} originalLower={original?.[1]}
            upperPose={upperPreviewPose.current} upperPivot={upperPivot} upperPoseBaked={!!surfaces}
            upperPoseMode={upperPoseMode} upperPoseEditable={!poseLocked} onUpperPoseChange={changeUpperPose}
            lowerPose={lowerPose} lowerPivot={lowerPivot} lowerPoseBaked={!!surfaces}
            lowerPoseMode={lowerPoseMode} lowerPoseEditable={!poseLocked} onLowerPoseChange={changeLowerPose}
            crownArrays={crownArrays}
            crownEdit={crownEditing && crownRows && eachCrownTarget ? { row: selectedRow, index: selectedCrowns[selectedRowIndex], rotation: crownRotations[selectedRowIndex] } : undefined}
            toothAxes={showToothAxes && toothAxisReference ? { row: selectedRow, index: selectedCrowns[selectedRowIndex], reference: toothAxisReference } : undefined}
            onCrownSelect={selectCrown} onCrownRotationChange={changeCrownRotation}
            hinge={isCycles ? { origin: cycleSettings.hingeOrigin, axis: cycleSettings.hingeAxis, visible: showHinge } : undefined}
            hingeMode={hingeMode} hingeEditable={isCycles && !poseLocked}
            onHingeChange={(origin, axis) => { if (!poseLocked) { setCycleSettings(current => ({ ...current, hingeOrigin: origin, hingeAxis: axis })); setHingePreset("custom"); } }}
            hingePreviewAngle={isCycles ? hingePreview * (Number.isFinite(cycleSettings.openingAngle) ? cycleSettings.openingAngle : 0) : 0}
            upperOffsetY={-closure} originalUpperOffsetY={(snapshot?.initialOpening ?? 0) - closure}
            showUpper={showUpper} showLower={showLower} showOriginal={showOriginal && !!surfaces} view={view} fitKey={fitKey}
            shading={shading} lightAngle={lightAngle} cavityStrength={cavityStrength}
            vegetationTrace={!isCycles && vegetationEnabled && !dirty ? vegetationTrace : null} showVegetationPaths={showVegetationPaths} showSlidingContacts={showSlidingContacts} vegetationXray={vegetationXray}
            vegetationStreamThickness={vegetationStreamThickness}
            vegetationBand={!isCycles && vegetationEnabled && validVegetation ? { direction: vegetationDirection, centerY: vegetationCenterY, thickness: vegetationThickness,
              bendReach: vegetationPathMode === "flexible" ? vegetationBendReach : 0, visible: showVegetation } : undefined} />
          <div className="wear-stage-top"><div className="wear-view-tabs">{(["front", "perspective", "side", "top"] as WearView[]).map(v => <button key={v} className={view === v ? "selected" : ""} onClick={() => setView(v)}>{v === "perspective" ? "3D" : v}</button>)}</div><button className="wear-stage-button" onClick={() => setFitKey(k => k + 1)} title="Fit toothrows to view"><Focus size={16} /> Fit view</button></div>
          {original && !setupStarted && <div className="wear-transform-toolbar" role="group" aria-label="Toothrow and hinge setup tools">
            <select aria-label="Setup gizmo target" value={setupTarget} disabled={poseLocked} onChange={event => { const next = event.target.value as SetupTarget; editSetup(next.endsWith("-crown") ? "rotate" : "translate", next); }}>
              <option value="upper">Whole upper row</option><option value="lower">Whole lower row</option>
              {crownRows && <><option value="upper-crown">Each upper crown</option><option value="lower-crown">Each lower crown</option></>}
              {isCycles && <option value="hinge">Jaw hinge</option>}
            </select>
            {eachCrownTarget && crownRows && <select aria-label="Crown carrying rotation gizmo" disabled={poseLocked} value={selectedCrowns[selectedRowIndex]} onChange={event => selectCrown(selectedRow, Number(event.target.value))}>{crownRows[selectedRowIndex].centers.map((_, i) => <option key={i} value={i}>Tooth {i + 1} of 6</option>)}</select>}
            {!eachCrownTarget && <button aria-label={setupTarget === "hinge" ? "Move jaw hinge" : `Move ${selectedRow} teeth`} aria-pressed={activeSetupMode === "translate"} className={activeSetupMode === "translate" ? "selected" : ""} disabled={poseLocked} onClick={() => editSetup("translate")}><Move3d size={14} /> Move</button>}
            <button aria-label={setupTarget === "hinge" ? "Rotate jaw hinge" : eachCrownTarget ? `Rotate each ${selectedRow} crown` : `Rotate ${selectedRow} teeth`} aria-pressed={activeSetupMode === "rotate"} className={activeSetupMode === "rotate" ? "selected" : ""} disabled={poseLocked} onClick={() => editSetup("rotate")}><Rotate3d size={14} />{eachCrownTarget ? "Rotate each" : "Rotate"}</button>
            <button aria-label="Finish positioning" disabled={poseLocked || activeSetupMode === null} onClick={stopSetupTools}><Check size={14} /> Done</button>
          </div>}
          {crownRows && !setupStarted && <div className="wear-crown-hint">{eachCrownTarget ? "Drag a ring: all six crowns tilt at their own centers. Click a crown to move the handle." : "Whole row moves the crowns and base together. Choose Each crown to change tooth tilt."}</div>}
          {showToothAxes && <div className="wear-tooth-axis-key" role="note" aria-label="Tooth axis marker legend">
            <strong>{selectedRow === "upper" ? "Upper" : "Lower"} tooth {selectedCrowns[selectedRowIndex] + 1} · tooth-local directions</strong>
            <span><b className="axis-x">X</b> <b className="axis-y">Y</b> <b className="axis-z">Z</b> source axes · <b className="axis-long">L</b> long axis · <b className="axis-edge">E</b> edge estimate</span>
            <span>Arrows rotate with the tooth. Gray plane keeps world XZ orientation.</span>
            <span>Rings use local X/Y/Z; sliders use occlusal-frame axes.</span>
            <span>Plane position is a visual guide, not a contact surface.</span>
          </div>}
          {isCycles && original && <div className="wear-cycle-preview"><label><span>Opening preview <strong>{format(hingePreview * cycleSettings.openingAngle, 1)}°</strong></span><input aria-label="Jaw opening preview" type="range" min={0} max={1} step={0.01} value={hingePreview} disabled={locked || !Number.isFinite(cycleSettings.openingAngle)} onChange={event => { setHingePreview(Number(event.target.value)); stopSetupTools(); }} /></label><span>{hingePreview > 0 ? "Preview only · run and export return to closed pose" : "Closed reference pose · all teeth share this endpoint"}</span></div>}
          {!original && <div className="wear-empty"><div className="wear-empty-icon"><Layers3 size={42} strokeWidth={1} /></div><span className="wear-eyebrow">FROM UNWORN TO OCCLUDED</span><h2>Let the contact<br />shape the teeth.</h2><p>Bring your aligned toothrows together along Y.<br />Wear develops wherever the two surfaces meet.</p><button className="wear-primary" disabled={locked} onClick={loadCrowns}><Layers3 size={16} /> Build 6 × 6 crown rows <ChevronRight size={16} /></button><button className="wear-text-button" disabled={locked} onClick={() => load(true)}><Upload size={14} /> Load example toothrows</button><small>Two example PLY files · 96 MB download</small></div>}
          {original && <div className="wear-legend"><span><i className="upper" />Upper · {isCycles ? "hinged" : fixedJaw ? "fixed" : "moving −Y"}</span><span><i className="lower" />Lower · fixed</span><span><i className="worn" />{(snapshot?.chipThickness ?? 0) > 0 ? "Contact wear + chips" : "Contact wear"}</span>{isCycles && showHinge && <span><i className="hinge" />Hinge axis</span>}{!isCycles && vegetationEnabled && <><span><i className="vegetation" />Vegetation wear</span>{showVegetationPaths && <span><i className="vegetation-path" />Computed paths</span>}{showSlidingContacts && <span><i className="sliding-contact" />Sliding now</span>}</>}</div>}
          <div className="wear-stage-bottom"><span>Y ↑ <b>·</b> Drag to orbit <b>·</b> Scroll to zoom <b>·</b> Right drag to pan</span><span>{grid ? `${grid.nx} × ${grid.nz} columns · ${format(grid.spacing)} ${unit} pitch` : original ? crownRows ? "6 upper + 6 lower · linked crown rotations" : "Row setup preview" : "Source coordinates preserved"}</span></div>
        </section>

        <aside className="wear-rail" ref={rail}>
          <nav className="wear-panel-nav" aria-label="Wear controls">
            <button aria-pressed={panel === "run"} onClick={() => showPanel("run")}><Play size={13} /> Run</button>
            <button aria-pressed={panel === "setup"} onClick={() => showPanel("setup")}><Layers3 size={13} /> Setup{dirty && original && <i aria-label="Unprepared changes" />}</button>
            <button aria-pressed={panel === "display"} onClick={() => showPanel("display")}><Eye size={13} /> Display & export</button>
          </nav>
          <section className="wear-section" hidden={panel !== "setup"}>
            <h2><span className="wear-section-number">01</span> Toothrow surfaces <Upload size={15} /></h2>
            <button className="wear-secondary full wear-build-crowns" disabled={locked} onClick={loadCrowns}>Build 6 × 6 crown rows <span>STL × 2</span></button>
            <button className="wear-secondary full" disabled={locked} onClick={() => load(true)}>Load example toothrows <span>96 MB</span></button>
            {crownRows && <p className="wear-note wear-crown-description">Six copies of each single crown on a base. Choose <strong>Each upper crown</strong> or <strong>Each lower crown</strong> to apply the same tilt to all six teeth in that row. Assignment follows Y position, regardless of filename.</p>}
            {info && <div className="wear-file-list">{info.map((row, i) => <div className="wear-file" key={i}><i className={i === 0 ? "upper" : "lower"} /><div><strong>{i === 0 ? "Upper" : "Lower"} <span>{row.name}</span></strong><small>{format(row.triangles, 0)} triangles</small></div><Check size={14} /></div>)}<button className="wear-text-button" disabled={locked} onClick={() => { clearTrace(); setBusy(true); setError(""); post({ type: "swap" }); }}><ArrowLeftRight size={13} /> Swap upper / lower assignment</button></div>}
            <details className="wear-import" open><summary>Import your own aligned pair</summary><div>
              <label className="wear-field"><span>Mesh contents</span><select aria-label="Imported mesh contents" disabled={locked} value={importKind} onChange={event => setImportKind(event.target.value as "rows" | "crowns")}><option value="rows">Complete toothrows</option><option value="crowns">Single crowns · six copies per row</option></select></label>
              {[0, 1].map(i => <label key={i}>Surface {i + 1}<input aria-label={`Surface ${i + 1} file`} type="file" accept=".ply,.stl" disabled={locked} onChange={event => setFiles(current => { const next: [File | null, File | null] = [...current]; next[i] = event.target.files?.[0] ?? null; return next; })} /></label>)}<button className="wear-secondary full" onClick={() => load(false)} disabled={locked || !files[0] || !files[1]}>Load selected pair</button><p className="wear-note">Closed triangle PLY or STL. Files are processed in your browser and are not uploaded. Higher Y centre is assigned upper; use Swap if needed.</p></div></details>
          </section>

          <section className="wear-section" hidden={panel !== "setup"}>
            <h2><span className="wear-section-number">02</span> Prepare geometry <Layers3 size={15} /></h2>
            <fieldset className="wear-pose" disabled={poseLocked}>
              <legend>{selectedRow === "upper" ? "Upper" : "Lower"} {eachCrownTarget && crownRows ? "crown tilt · linked across all six" : "whole-row setup pose"}</legend>
              <p className="wear-note">Choose <strong>Move</strong> or <strong>Rotate</strong> above the model, then drag its colored arrows or rings. Drag away from the handles to orbit the view.</p>
              {!eachCrownTarget && <dl className="wear-pose-readout"><div><dt>Position · {unit}</dt><dd>{selectedPose.translation.map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div></dl>}
              {crownRows && <CrownAngles key={selectedRow} value={crownInclinations?.[selectedRowIndex] ?? null} row={selectedRow === "upper" ? "Upper" : "Lower"} disabled={poseLocked} onChange={target => changeCrownAngles(selectedRow, target)} />}
              <details className="wear-angle-details" open={crownRows ? undefined : true}><summary>Source rotations · X / Y / Z</summary>
                <dl className="wear-pose-readout"><div><dt>{eachCrownTarget ? "Each crown" : "Whole row"} · °</dt><dd>{(eachCrownTarget ? crownRotations[selectedRowIndex] : selectedPose.rotation).map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div></dl>
                <p className="wear-note">Euler rotations from the original mesh orientation, applied X then Y then Z. These are rotation commands, not angles to the occlusal plane.</p>
              </details>
              {eachCrownTarget && crownRows
                ? <button className="wear-text-button" disabled={!crownRotations[selectedRowIndex].some(v => v !== 0)} onClick={() => changeCrownRotation(selectedRow, [0, 0, 0])}><RotateCcw size={12} /> Restore crown tilt</button>
                : <button className="wear-text-button" disabled={!selectedPose.translation.some(v => v !== 0) && !selectedPose.rotation.some(v => v !== 0)} onClick={() => changeRowPose(selectedRow, identityUpperSetupPose())}><RotateCcw size={12} /> Restore source pose</button>}
              <details className="wear-import"><summary>Restore setup pose from run JSON</summary><div><label>Run settings JSON<input aria-label="Restore setup pose from run JSON" type="file" accept=".json,application/json" disabled={poseLocked} onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void restoreSetup(file); }} /></label><p className="wear-note">Restores both row poses and linked crown tilts. Load the same original files and row type first. Worn material and other run settings are not restored.</p></div></details>
            </fieldset>
            <p className="wear-note">Preview changes live, then prepare the grid. Whole-row controls move the base and all teeth together about the row center. Crown controls rotate each tooth about its own center using row-local axes.</p>
            {crownRows && <div className="wear-crown-spacing">
              <NumberField label="Tooth spacing" value={crownPitch} min={0.001} max={1e6} step="any" suffix={unit} disabled={poseLocked} onChange={changeCrownPitch} />
              <p className="wear-note">Center-to-center distance along each row. Both rows use the same spacing and retain their half-tooth stagger; bases expand with the rows.</p>
              <div role="status" aria-live="polite" className={currentCrownCheck?.clear ? "wear-prepared" : "wear-warning"}>
                {!validCrownPitch ? "Enter positive tooth spacing within the supported range." : crownCheckError || (!currentCrownCheck ? "Checking individual crowns…" : currentCrownCheck.clear ? "Crown clearance verified in both rows." : currentCrownCheck.rows.some(row => row.status === "intersecting") ? "Neighboring crowns intersect. Increase tooth spacing before preparing." : "No sampled intersection, but clearance is not yet verified. Increase tooth spacing before preparing.")}
              </div>
              {currentCrownCheck && !currentCrownCheck.clear && <>
                <p className="wear-note">{currentCrownCheck.rows.filter(row => row.intersectingPairs > 0).map(row => `${row.row === "upper" ? "Upper" : "Lower"}: ${row.intersectingPairs} intersecting pairs`).join(" · ")}</p>
                <button className="wear-secondary full" disabled={poseLocked} onClick={() => changeCrownPitch(Math.ceil(currentCrownCheck.suggestedPitch * 1000) / 1000)}>Space teeth to clear · {format(Math.ceil(currentCrownCheck.suggestedPitch * 1000) / 1000, 3)} {unit}</button>
              </>}
              <p className="wear-note">Checks include the complete crowns, excluding the shared support base. Automatic spacing uses a conservative clearance margin; it may leave a little extra space. Rechecked after each tooth tilt or spacing change.</p>
            </div>}
            {setupStarted && <p className="wear-warning wear-pose-warning">Reset to unworn geometry to adjust the setup pose{isCycles ? " or hinge" : ""}.</p>}
            {!validSetup && <p className="wear-warning wear-pose-warning">Use finite pose values and translations within the supported coordinate range.</p>}
            <div className="wear-field-pair"><NumberField label="Column pitch" value={spacing} min={0.001} step="any" suffix={unit} disabled={locked} onChange={v => { setSpacing(v); setDirty(true); }} /><label className="wear-field"><span>Unit label</span><select value={unit} disabled={locked} onChange={e => setUnit(e.target.value)}><option value="u">Native units</option><option value="mm">Millimetres</option><option value="cm">Centimetres</option></select></label></div>
            <p className="wear-note">Smaller pitch retains finer detail. Y surfaces stay continuous. Unit labels do not rescale your files.</p>
            {!isCycles && <>
              <NumberField label="Contact spread σ" value={contactSpread} min={0} max={maxContactSpread} step="any" suffix={unit} disabled={locked} onChange={value => { setContactSpread(value); setDirty(true); setStatus("Prepare the wear grid to apply the contact spread. This starts from unworn geometry."); }} />
              <p className="wear-note">0 keeps exact cell contact. σ sets the footprint in model units, reaching 3σ sideways. Try {format(spacing, 3)}–{format(2 * spacing, 3)} {unit} for broader contact. Direct overlap is still resolved. Wider footprints cost more; radius is limited to 16 grid cells. Prepare again after changing it.</p>
              {!validContactSpread && <p className="wear-warning">Enter a finite spread from 0 to {format(maxContactSpread, 4)} {unit}.</p>}
              <NumberField label="Thin-wall chipping" value={chipThickness} min={0} max={spacing * 8} step="any" suffix={unit} disabled={locked || vegetationEnabled} onChange={value => { setChipThickness(value); setDirty(true); }} />
              <p className="wear-note">0 disables chipping. Otherwise, narrow caps below this width can break back to nearby contact-wear depth. Use more than one grid pitch to resolve it. A geometric approximation, not a fracture model; vegetation must be off.</p>
              {!validChipping && <p className="wear-warning">Use a thickness from 0 to eight pitches with vegetation off.</p>}
              {(snapshot?.chipThickness ?? 0) > 0 && <p className="wear-note">Chipped volume: upper {format(snapshot?.chippedUpperVolume ?? 0)} · lower {format(snapshot?.chippedLowerVolume ?? 0)} {unit}³. Included in total removal; separate from contact wear.</p>}
            </>}
            <label className="wear-check"><input type="checkbox" checked={autoOpen} disabled={locked} onChange={e => { setAutoOpen(e.target.checked); setDirty(true); }} /> Automatically clear initial overlap along Y</label>
            <button className="wear-secondary full" disabled={!original || locked || !validSetup || !validCrownSpacing || !validContactSpread || !validChipping || !Number.isFinite(spacing) || spacing <= 0} onClick={() => prepare()}><Layers3 size={15} />{surfaces && !dirty ? "Rebuild wear grid" : "Prepare wear grid"}</button>
            {snapshot && <div className="wear-prepared"><Check size={13} /><span>{isCycles ? "Closed reference offset" : gaussianPrepared ? "Initial direct contact at" : "Initial first contact at"} <strong>{format(isCycles ? closure : first, 3)} {unit}</strong>{isCycles ? " along −Y" : " closure"}{snapshot.initialOpening > 0 ? ` · opened ${format(snapshot.initialOpening, 3)} ${unit} to clear overlap` : ""}</span></div>}

          </section>

          <section className="wear-section wear-jaw-section" hidden={panel !== "run"} aria-labelledby="jaw-motion-heading">
            <h2 id="jaw-motion-heading">Jaw motion <ArrowDown size={16} /></h2>
            {!isCycles && <>
              <div className="wear-jaw-options" role="group" aria-label="Jaw motion">
                <button aria-pressed={!fixedJaw} disabled={locked} onClick={() => changeJawMotion("closing")}><ArrowDown size={17} /><strong>Close jaw</strong><span>Tooth contact + optional vegetation</span></button>
                <button aria-pressed={fixedJaw} disabled={locked || chipThickness > 0 || (snapshot?.chipThickness ?? 0) > 0} onClick={() => changeJawMotion("fixed")}><Leaf size={17} /><strong>Hold jaw fixed</strong><span>Vegetation wear only</span></button>
              </div>
              {(chipThickness > 0 || (snapshot?.chipThickness ?? 0) > 0) && <div className="wear-fixed-preparation">
                <p>This setup uses thin-wall chipping, which cannot run with vegetation. Rebuilding turns chipping off and resets wear to the unworn setup.</p>
                <button className="wear-secondary full" disabled={!original || locked || !validSetup || !validCrownSpacing || !validContactSpread || !Number.isFinite(spacing) || spacing <= 0} onClick={() => prepare(false, true)}>Rebuild for fixed-jaw vegetation</button>
              </div>}
            </>}
            {(!original || !snapshot || dirty) && <button className="wear-secondary full wear-setup-link" onClick={() => showPanel("setup")}><Layers3 size={14} />{!original ? "Load teeth in Setup" : "Prepare geometry in Setup"}<ChevronRight size={13} /></button>}
            <div className="wear-run-buttons"><button className="wear-primary" disabled={running ? false : !canAdvance} onClick={() => running ? pause() : advance(true)}>{running ? <Pause size={16} /> : <Play size={16} />}{running ? "Pause" : complete ? "Complete" : isCycles ? "Run bites" : fixedJaw ? "Run vegetation" : "Run wear"}</button><button className="wear-secondary" disabled={!canAdvance} onClick={() => advance(false)} title={isCycles ? "Run one complete bite cycle" : fixedJaw ? "Pull one vegetation increment without moving the jaw" : "Advance one closure increment"}><SkipForward size={16} />{isCycles ? "One bite" : "Step"}</button><button className="wear-secondary wear-icon" disabled={!snapshot || locked} onClick={() => prepare(true)} title="Reset to unworn geometry using the last prepared settings"><RotateCcw size={16} /></button></div>
            <details className="wear-run-details"><summary>Wear model · {isCycles ? "repeated bites" : "fast geometric"}</summary>
            <label className="wear-field"><span>Simulation mode</span><select value={mode} disabled={locked} onChange={event => changeMode(event.target.value as "geometric" | "cycles")}><option value="geometric">Fast geometric wear · default</option><option value="cycles">Repeated bite cycles · experimental</option></select></label>
            {isCycles ? <p className="wear-note">Experimental pressure-and-sliding model; improved facet realism has not been validated. The whole upper row opens about the hinge and returns to the shared closed pose on every cycle.</p>
              : <p className="wear-note">Fast geometric wear supports jaw closure and fixed-jaw vegetation. Changing the wear model starts from unworn geometry.</p>}
            </details>
            {isCycles ? <>
              <fieldset className="wear-pose wear-hinge-panel" disabled={cycleSettingsLocked || !original}>
                <legend>Jaw hinge</legend>
                <label className="wear-field"><span>Joint-end preset</span><select value={hingePreset} onChange={event => placeHinge(event.target.value as typeof hingePreset)}><option value="negative">{rowAxis === "X" ? "Left / posterior · −X" : "Behind the −Z end"}</option><option value="positive">{rowAxis === "X" ? "Right · +X" : "Behind the +Z end"}</option><option value="custom">Custom gizmo placement</option></select></label>
                <div className="wear-hinge-buttons"><button className="wear-secondary" aria-pressed={setupTarget === "hinge" && hingeMode === "translate"} onClick={() => editHinge("translate")}><Move3d size={13} /> Move hinge</button><button className="wear-secondary" aria-pressed={setupTarget === "hinge" && hingeMode === "rotate"} onClick={() => editHinge("rotate")}><Rotate3d size={13} /> Rotate hinge</button></div>
                <p className="wear-note">The hinge crosses the toothrow ({rowAxis === "X" ? "Z axis" : "X axis"}) beyond its posterior end. Use the arrows to adjust the estimated joint position and the rings to adjust its axis. The toothrows alone do not locate the joint.</p>
                <dl className="wear-pose-readout"><div><dt>Origin · {unit}</dt><dd>{cycleSettings.hingeOrigin.map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div><div><dt>Axis</dt><dd>{cycleSettings.hingeAxis.map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div></dl>
                <NumberField label="Opening angle" value={cycleSettings.openingAngle} min={-90} step="1" suffix="°" onChange={value => changeCycleSetting("openingAngle", value)} />
                <p className="wear-note">Use the opening preview above the model to check direction. A negative angle reverses the opening; the closed endpoint stays fixed.</p>
              </fieldset>
              <label className="wear-check"><input type="checkbox" checked={showHinge} onChange={event => { setShowHinge(event.target.checked); if (!event.target.checked) setHingeMode(null); }} /> Show hinge axis</label>
              <div className="wear-field-pair"><NumberField label="Stop after bites" value={cycleTarget} min={1} step="1" suffix="cycles" disabled={locked} onChange={setCycleTarget} /><NumberField label="Steps per cycle" value={cycleSettings.stepsPerCycle} min={2} step="1" suffix="steps" disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("stepsPerCycle", value)} /></div>
              <div className="wear-field-pair"><NumberField label="Peak bite force" value={cycleSettings.biteForce} min={0} step="any" suffix="N" disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("biteForce", value)} /><NumberField label="Contact stiffness" value={cycleSettings.contactStiffness} min={0.000001} step="any" suffix={`N/${unit}³`} disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("contactStiffness", value)} /></div>
              <p className="wear-note">Opening releases the bite load. During closure, force rises smoothly to the configured peak at the shared closed pose.</p>
              <NumberField label="Compliant contact reach" value={cycleSettings.contactReach} min={0.000001} step="any" suffix={unit} disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("contactReach", value)} />
              <p className="wear-note">One bite load is distributed across opposing surfaces near the shared closed pose. Contact reach bounds the shared seating adjustment and contact search; it does not force separated teeth to touch.</p>
              <div className="wear-field-pair"><NumberField label="Upper wear coefficient" value={cycleSettings.wearCoefficientUpper} min={0} step="any" suffix={`${unit}²/N`} disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("wearCoefficientUpper", value)} /><NumberField label="Lower wear coefficient" value={cycleSettings.wearCoefficientLower} min={0} step="any" suffix={`${unit}²/N`} disabled={cycleSettingsLocked} onChange={value => changeCycleSetting("wearCoefficientLower", value)} /></div>
              <p className="wear-note">Wear depth = coefficient × contact pressure × sliding distance. Pressure without sliding produces no wear. These are uncalibrated model parameters; use units consistently. Steps may be refined to resolve the hinge motion.</p>
              {cycleSettingsError && <p className="wear-warning">{cycleSettingsError}</p>}
              {cycle && <dl className="wear-cycle-metrics"><div><dt>Completed bites</dt><dd>{format(completedCycles, 0)}</dd></div><div><dt>Contact area at peak load</dt><dd>{format(cycle.contactArea, 3)} {unit}²</dd></div><div><dt>Peak supported load</dt><dd>{format(cycle.supportedLoad, 2)} N</dd></div><div><dt>Peak pressure</dt><dd>{format(cycle.peakPressure, 3)} N/{unit}²</dd></div><div><dt>Maximum sliding increment</dt><dd>{format(cycle.maxSlip, 4)} {unit}</dd></div><div><dt>Compliant seating at peak load</dt><dd>{format(cycle.compliantApproach, 4)} {unit}</dd></div><div><dt>Upper removed · last bite</dt><dd>{format(cycle.removedUpperVolume, 4)} {unit}³</dd></div><div><dt>Lower removed · last bite</dt><dd>{format(cycle.removedLowerVolume, 4)} {unit}³</dd></div><div><dt>Resolved substeps · last bite</dt><dd>{format(cycle.substeps, 0)}</dd></div></dl>}
            </> : fixedJaw ? <>
              <div className="wear-field-pair"><NumberField label="Stop at total pull distance" value={pullTarget} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setPullTarget} /><NumberField label="Pull increment" value={pullDistance} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setPullDistance} /></div>
              <p className="wear-note">Both toothrows stay at their current pose{snapshot ? ` (${format(closure, 3)} ${unit} closure)` : " after grid preparation"}. Only vegetation advances. The stopping distance includes any vegetation already pulled in this run; increase it to continue.</p>
              <p className="wear-note">Set the band and bundle size below so vegetation slides against the teeth. Existing tooth-to-tooth wear is preserved; fixed-jaw pulls add vegetation wear only.</p>
              {!validRun && <p className="wear-warning">Use positive, finite pull distances and valid vegetation settings below.</p>}
            </> : <>
            <label className="wear-field"><span>Preset stopping point</span><select value={stopMode} disabled={locked} onChange={e => setStopMode(e.target.value as "contact" | "total")}><option value="contact">Distance beyond initial {gaussianPrepared ? "direct" : "first"} contact</option><option value="total">Total closure from prepared pose</option></select></label>
            <div className="wear-field-pair"><NumberField label="Stop distance" value={depth} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setDepth} /><NumberField label="Closure increment" value={step} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setStep} /></div>
            <div className="wear-share"><div><span>Wear allocation</span><strong>{share}% upper / {100 - share}% lower</strong></div><input aria-label="Upper wear percentage" type="range" min={0} max={100} step={5} value={share} disabled={locked || closure > 0} onChange={e => setShare(Number(e.target.value))} /><div className="wear-range-labels"><span>Lower only</span><span>Equal</span><span>Upper only</span></div></div>
            <p className="wear-note">Equal wear removes the same local depth from both rows. The upper translates −Y; the lower stays fixed. Tooth-to-tooth wear stays active during closure, including when vegetation is enabled.</p>
            {gaussianPrepared && <p className="wear-note">The stopping reference and projected contact area measure direct same-cell contact. Gaussian neighboring-cell wear can begin earlier.</p>}
            </>}

            {!isCycles && !fixedJaw && snapshot && closure < first && <button className="wear-text-button" disabled={!canAdvance} onClick={contactJump}>Move to initial {gaussianPrepared ? "direct-contact" : "first-contact"} closure <ChevronRight size={12} /></button>}
            {dirty && snapshot && <p className="wear-warning">Prepare the grid to apply edited settings, or reset to restore the last prepared settings.</p>}
            {snapshot && (isCycles ? cycleTarget < completedCycles : fixedJaw ? pullTarget < cumulativePull - 1e-8 : target < closure - 1e-8) && <p className="wear-warning">Reset the run to use a stopping point before the current {isCycles ? "bite count" : fixedJaw ? "cumulative pull distance" : "closure"}.</p>}
          </section>

          <section className="wear-section wear-vegetation-section" hidden={panel !== "run"}>
            <h2>Vegetation abrasion <Leaf size={15} /></h2>
            {isCycles && <p className="wear-note">Vegetation mode is available with geometric closure. Repeated bite cycles currently model tooth-to-tooth contact only.</p>}
            <label className="wear-check"><input type="checkbox" checked={vegetationEnabled} disabled={vegetationLocked || fixedJaw} onChange={e => setVegetationEnabled(e.target.checked)} /> Enable sideways branch stripping</label>
            {!isCycles && <p className="wear-note">Vegetation wears surfaces as it slides alongside them, including areas without tooth-to-tooth contact. A head-on obstruction alone does not cause abrasion.</p>}
            {!isCycles && vegetationEnabled && <>
              <label className="wear-field"><span>Vegetation path</span><select value={vegetationPathMode} disabled={vegetationLocked} onChange={e => setVegetationPathMode(e.target.value as "straight" | "flexible")}><option value="flexible">Flexible around teeth</option><option value="straight">Straight sliding paths</option></select></label>
              <label className="wear-field"><span>Pull direction</span><select value={vegetationDirection} disabled={vegetationLocked} onChange={e => setVegetationDirection(e.target.value as VegetationDirection)}><option value="+x">+X · left to right in front view</option><option value="-x">−X · right to left in front view</option><option value="+z">+Z · toward the front view</option><option value="-z">−Z · away from the front view</option></select></label>
              <div className="wear-field-pair"><NumberField label="Incoming band centre Y" value={vegetationCenterY} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationCenterY} /><NumberField label="Incoming thickness" value={vegetationThickness} min={0.001} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationThickness} /></div>
              <NumberField label="Stream thickness · bundle diameter" value={vegetationStreamThickness} min={0} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationStreamThickness} />
              <p className="wear-note">The band distributes bundle centres; each bundle extends half its diameter beyond its path. A bundle can slide against several teeth, including both rows at once. Wider bundles need wider gaps; diameter stays fixed while bending. Use 0 for the original thin paths.</p>
              {vegetationPathMode === "flexible" && <><NumberField label="Bend reach" value={vegetationBendReach} min={0} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationBendReach} /><p className="wear-note">Total vertical and sideways detour allowed, including return to the band at the exit. Paths need an open way through. Bends use column-pitch steps; less than one pitch gives straight paths.</p></>}
              <div className="wear-field-pair"><NumberField label="Abrasion per pull distance" value={abrasionRate} min={0.000001} step="any" suffix="u/u" disabled={vegetationLocked} onChange={setAbrasionRate} />{!fixedJaw && <NumberField label="Pull per closure distance" value={pullPerClosure} min={0} step="any" suffix="u/u" disabled={vegetationLocked} onChange={setPullPerClosure} />}</div>
              <p className="wear-note">The entrance and exit band stay at this world Y. {fixedJaw ? "The jaw stays fixed while vegetation is pulled to the total distance above." : "Closure combines tooth contact wear with vegetation pulls at the ratio above; use 0 for manual pulls only."}</p>
              <label className="wear-check"><input type="checkbox" checked={showVegetation} onChange={e => setShowVegetation(e.target.checked)} /> Show incoming band & {vegetationPathMode === "flexible" ? "bend reach" : "pull direction"}</label>
              {showVegetation && vegetationPathMode === "flexible" && <p className="wear-note">Solid outline: incoming band. Dashed outline: possible bend range. Arrows show pull direction, not individual paths.</p>}
              <div className="wear-path-display">
                <label className="wear-check"><input type="checkbox" checked={showVegetationPaths} onChange={e => setShowVegetationPaths(e.target.checked)} /><i className="vegetation-path" /> Show vegetation paths</label>
                <label className="wear-check"><input type="checkbox" checked={showSlidingContacts} onChange={e => setShowSlidingContacts(e.target.checked)} /><i className="sliding-contact" /> Show sliding contacts</label>
                <label className="wear-check"><input type="checkbox" checked={vegetationXray} disabled={!showVegetationPaths && !showSlidingContacts} onChange={e => setVegetationXray(e.target.checked)} /> See paths & contacts through teeth</label>
                {wantsTrace && <div className="wear-path-readout" role="status" aria-live="polite">
                  {traceLoading ? <span><LoaderCircle size={11} className="spin" /> Computing current paths…</span>
                    : traceError ? <p className="wear-warning">Path preview: {traceError}</p>
                      : vegetationTrace ? <><strong>{format(vegetationTrace.pathCount, 0)} paths shown · {format(vegetationTrace.totalContactFaces, 0)} sliding faces</strong><span>{format(vegetationTrace.blockedInletCount, 0)} of {format(vegetationTrace.inletCount, 0)} entrance positions blocked</span>{vegetationTrace.truncated && <span>Representative path sample; contact shading covers the computed sliding region{vegetationTrace.contactPositions.length / 18 < vegetationTrace.totalContactFaces ? " with a display limit" : ""}.</span>}{vegetationTrace.totalContactFaces === 0 && <span>No sliding contact at this pose. Adjust the band or close the jaws.</span>}</>
                        : <span>Prepare the wear grid to preview paths.</span>}
                </div>}
                <p className="wear-note">Yellow paths follow computed routes; translucent sleeves show bundle diameter. Magenta marks all sliding contacts, including simultaneous contact with both rows. Contacts use conservative grid cells and may appear beyond a sleeve; finer column pitch improves precision. Green records vegetation wear already removed. Previewing does not wear the teeth.</p>
              </div>
              {!fixedJaw && <div className="wear-manual-pull"><NumberField label="Manual pull distance" value={pullDistance} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setPullDistance} /><button className="wear-secondary full" disabled={!canPull} onClick={pullVegetation}><Leaf size={14} /> Pull vegetation at current pose</button></div>}
              <p className="wear-note">Every pull adds proportional recession at sliding contacts, including pulls smaller than the column pitch. {fixedJaw ? "Run or step vegetation using the fixed-jaw controls above." : "A manual pull holds the jaws fixed; choose Hold jaw fixed above for a continuous vegetation-only run."}</p>
              {!validVegetation && <p className="wear-warning">Enter a finite band centre, positive band thickness and abrasion rate, and nonnegative bundle diameter, bend reach, and pull-to-closure ratio.</p>}
              {snapshot && <dl className="wear-vegetation-metrics"><div className="wear-contact-measurement"><dt>Tooth contact · upper removed</dt><dd>{format(snapshot.contactRemovedUpperVolume)} {unit}³</dd></div><div className="wear-contact-measurement"><dt>Tooth contact · lower removed</dt><dd>{format(snapshot.contactRemovedLowerVolume)} {unit}³</dd></div><div className="wear-vegetation-measurement"><dt>Vegetation · upper removed</dt><dd>{format(snapshot.vegetation.removedUpperVolume)} {unit}³</dd></div><div><dt>Vegetation · lower removed</dt><dd>{format(snapshot.vegetation.removedLowerVolume)} {unit}³</dd></div><div><dt>Cumulative pull</dt><dd>{format(snapshot.vegetation.pullDistance, 3)} {unit}</dd></div><div><dt>Nominal recession</dt><dd>{format(snapshot.vegetation.effectiveRecession, 3)} {unit}</dd></div></dl>}
              <p className="wear-note">Total volume loss is geometric; the contact/vegetation split is estimated within each integration interval.</p>
              <p className="wear-note wear-vegetation-model">Recession is continuous; route detail still follows the column pitch. Paths that cannot reach the exit add no abrasion. No branch forces or tension are solved. Surface colors show the latest wear type; measurements separate both sources. Settings lock after the first vegetation pull; reset to change them.</p>
            </>}
          </section>

          <section className="wear-section wear-display-section" hidden={panel !== "display"}>
            <h2><Eye size={15} /> Display & export</h2>
            <label className="wear-field"><span>Surface shading</span><select value={shading} onChange={e => setShading(e.target.value as WearShading)}><option value="relief">Relief · directional light & cavities</option><option value="studio">Soft studio · original lighting</option></select></label>
            {shading === "relief" && <div className="wear-shading-controls">
              <label><span>Light direction <strong>{lightAngle}°</strong></span><input aria-label="Light direction" type="range" min={0} max={360} step={5} value={lightAngle} onChange={e => setLightAngle(Number(e.target.value))} /></label>
              <label><span>Cavity contrast <strong>{cavityStrength.toFixed(1)}</strong></span><input aria-label="Cavity contrast" type="range" min={0} max={2} step={0.1} value={cavityStrength} onChange={e => setCavityStrength(Number(e.target.value))} /></label>
              <p className="wear-note">Rotate the light to reveal ridges and depressions. These controls change the display only.</p>
            </div>}
            <div className="wear-visibility"><label><input type="checkbox" checked={showUpper} onChange={e => setShowUpper(e.target.checked)} /> Upper</label><label><input type="checkbox" checked={showLower} onChange={e => setShowLower(e.target.checked)} /> Lower</label><label><input type="checkbox" checked={showOriginal} onChange={e => setShowOriginal(e.target.checked)} /> Original ghost</label></div>
            <div className="wear-export-buttons">{(["upper", "lower"] as const).map(row => <button key={row} className="wear-secondary" disabled={!snapshot || locked || dirty} onClick={() => { setHingePreview(0); setBusy(true); setError(""); post({ type: "export", row }); }}><ArrowDownToLine size={14} />{row === "upper" ? "Upper" : "Lower"} PLY</button>)}</div>
            <button className="wear-text-button" disabled={!snapshot || locked || dirty} onClick={exportReport}><ArrowDownToLine size={13} /> Download run settings & measurements</button>
            <p className="wear-note">Exports contain the sampled, worn solids in {isCycles ? "the closed reference assembly pose" : "their current assembly pose"}. Source files stay untouched.</p>
          </section>
          <section className="wear-section" hidden={panel !== "run"}>
            <details className="wear-compute wear-run-details"><summary>Computation & performance</summary>
              <label className="wear-field"><span>Computation</span><select value={compute.preference} disabled={locked || traceLoading || compute.backend === "initializing"} onChange={e => changeCompute(e.target.value as ComputePreference)}><option value="auto">Automatic · GPU when available</option><option value="cpu">CPU reference</option></select></label>
              <div className={`wear-compute-status ${compute.backend}`} role="status" aria-live="polite">
                <strong>{computeSwitching || compute.backend === "initializing" ? <><LoaderCircle size={11} className="spin" /> Checking compute backend…</> : compute.backend === "webgpu" ? "GPU + CPU" : compute.preference === "cpu" ? "CPU reference" : "CPU fallback"}</strong>
                {!computeSwitching && compute.backend !== "initializing" && <>
                  <span>{compute.backend === "webgpu" ? `GPU: ${gpuStages.join(", ") || "no active stages"}. CPU: ${cpuStages.join(", ")}.` : isCycles ? "Contact pressure, sliding, wear, and surface generation on CPU." : "Wear, routing, and surface generation on CPU."}</span>
                  {compute.label && <span className="wear-compute-device">{compute.label}</span>}
                  {compute.reason && <span>{compute.reason}</span>}
                </>}
              </div>
              {timings && <div className="wear-compute-timings"><span>Last geometry update</span><dl><div><dt>Solve</dt><dd>{duration(timings.solveMs)}</dd></div><div><dt>Surfaces</dt><dd>{duration(timings.meshMs)}</dd></div><div><dt>Paths</dt><dd>{duration(timings.previewMs)}</dd></div></dl></div>}
              <p className="wear-note">Switch while idle to compare backends. The current worn geometry is preserved.</p>
            </details>
          </section>
          <div className="wear-model-note" hidden={panel !== "run"}>{isCycles ? "Hinge kinematics, compliant contact, and pressure × sliding wear. Force and material parameters are user assumptions, not a calibrated prediction of Camarasaurus feeding or biological time." : "Geometric contact and material removal. This model does not predict bite forces, wear rates, or elapsed biological time."}</div>
        </aside>

        <section className="wear-results" aria-label="Wear measurements">
          <div className="wear-progress-block"><div><span className="wear-eyebrow">{running ? "RUNNING" : !isCycles && (fixedJaw ? noMaterial : snapshot?.materialExhausted) ? "MATERIAL EXHAUSTED" : complete ? "STOPPING POINT REACHED" : snapshot ? "READY TO ADVANCE" : "SESSION PROGRESS"}</span><strong>{format(percent, 0)}<small>%</small></strong></div><div className="wear-progress-track"><i style={{ width: `${percent}%` }} /></div><p>{isCycles ? `${format(completedCycles, 0)} / ${format(cycleTarget, 0)} bites` : fixedJaw ? `${format(cumulativePull, 3)} / ${format(pullTarget, 3)} ${unit} vegetation pull · jaw held at ${format(closure, 3)} ${unit}` : `${format(closure, 3)} / ${snapshot ? format(target, 3) : "—"} ${unit} closure`}</p></div>
          <Metric label="Upper removed" value={format(snapshot?.removedUpperVolume)} unit={`${unit}³`} color="upper" />
          <Metric label="Lower removed" value={format(snapshot?.removedLowerVolume)} unit={`${unit}³`} color="lower" />
          <Metric label={isCycles ? "Contact at peak load" : gaussianPrepared ? "Projected direct contact" : "Projected contact"} value={format(isCycles ? cycle?.contactArea : snapshot?.contactArea)} unit={`${unit}²`} />
          <div className="wear-chart"><span>Removed volume / {isCycles ? "bite count" : fixedJaw ? "vegetation pull" : "closure"}</span><WearChart history={history} cycles={isCycles} vegetation={fixedJaw} /></div>
        </section>
      </main>
      <footer className={`wear-status ${error ? "has-error" : ""}`} role="status" aria-live="polite"><span>{busy ? <LoaderCircle size={14} className="spin" /> : <i className={error ? "error-dot" : "status-dot"} />}{error || status}</span>{workerFailed && <button className="wear-secondary" onClick={() => window.location.reload()}>Restart app</button>}{busy && progress > 0 && progress < 1 && <strong>{Math.round(progress * 100)}%</strong>}{warnings.length > 0 && <details><summary>{warnings.length} geometry note{warnings.length > 1 ? "s" : ""}</summary><div>{warnings.map((warning, i) => <p key={i}>{warning}</p>)}</div></details>}</footer>
    </div>
  );
}

function NumberField({ label, value, suffix, onChange, ...props }: { label: string; value: number; suffix: string; onChange: (value: number) => void; disabled?: boolean; min?: number; max?: number; step: string }) {
  return <label className="wear-field"><span>{label}</span><div className="wear-number"><input type="number" value={Number.isFinite(value) ? value : ""} onChange={e => onChange(e.target.value === "" ? NaN : Number(e.target.value))} {...props} /><small>{suffix}</small></div></label>;
}
function Metric({ label, value, unit, color }: { label: string; value: string; unit: string; color?: string }) {
  return <div className="wear-metric"><span>{color && <i className={color} />}{label}</span><strong>{value} <small>{unit}</small></strong></div>;
}
function WearChart({ history, cycles = false, vegetation = false }: { history: HistoryPoint[]; cycles?: boolean; vegetation?: boolean }) {
  const x = (p: HistoryPoint) => vegetation ? p.vegetationPull : p.closure;
  const maxX = Math.max(1e-9, ...history.map(x));
  const maxY = Math.max(1e-9, ...history.flatMap(p => [p.upper, p.lower]));
  const path = (key: "upper" | "lower") => history.map((p, i) => `${i ? "L" : "M"}${4 + x(p) / maxX * 192},${47 - p[key] / maxY * 40}`).join(" ");
  return <svg viewBox="0 0 200 52" role="img" aria-label={`Removed volume versus ${cycles ? "bite count" : vegetation ? "vegetation pull" : "closure"}, cream for upper and teal for lower`}><path d="M4 7H196 M4 27H196 M4 47H196" stroke="#dce4e8" strokeWidth="1" fill="none" />{history.length > 1 && <><path d={path("upper")} stroke="#b59865" strokeWidth="3" fill="none" /><path d={path("lower")} stroke="#178d87" strokeWidth="1.5" strokeDasharray="4 2" fill="none" /></>}</svg>;
}
