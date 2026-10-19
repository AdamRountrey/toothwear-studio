import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowDownToLine, ArrowLeftRight, Check, ChevronRight, Eye, Focus, Layers3, Leaf, LoaderCircle, Move3d, Pause, Play, Rotate3d, RotateCcw, SkipForward, Upload } from "lucide-react";
import { WearViewport, type WearMesh, type WearView, type WearShading } from "./WearViewport";
import type { VegetationDirection, VegetationSettings, WearSnapshot } from "./engine";
import type { ComputePreference, SourceInfo, WearComputeStatus, WearRequest, WearResponse, WearTimings } from "./protocol";
import type { VegetationTrace } from "./slidingVegetation";
import { identityUpperSetupPose, isValidUpperSetupPose, type UpperSetupPose } from "./upperPose";
import { createDefaultBiteCycleSettings, validateBiteCycleSettings, type BiteCycleMetrics, type BiteCycleSettings } from "./biteCycles";

const usageNotice = "Experimental — not for research use. Simulated wear has not been validated.";

type HistoryPoint = { closure: number; upper: number; lower: number; contact: number; contactUpper: number; contactLower: number; vegetationPull: number; vegetationUpper: number; vegetationLower: number; vegetationSettings: VegetationSettings | null };
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
  const settings = useRef<{ target: number; step: number; share: number; vegetation?: VegetationSettings; mode: "geometric" | "cycles"; cycleTarget: number; cycleSettings?: BiteCycleSettings }>({ target: 0, step: 0.25, share: 0.5, mode: "geometric", cycleTarget: 10 });
  const [original, setOriginal] = useState<[WearMesh, WearMesh] | null>(null);
  const [surfaces, setSurfaces] = useState<[WearMesh, WearMesh] | null>(null);
  const [info, setInfo] = useState<[SourceInfo, SourceInfo] | null>(null);
  const [upperPose, setUpperPose] = useState<UpperSetupPose>(identityUpperSetupPose);
  const [upperPoseMode, setUpperPoseMode] = useState<"translate" | "rotate" | null>("translate");
  const [mode, setMode] = useState<"geometric" | "cycles">("geometric");
  const [cycleSettings, setCycleSettings] = useState<BiteCycleSettings>(() => createDefaultBiteCycleSettings({ min: [-1, -1, -1], max: [1, 1, 1] }, 0.5));
  const [cycle, setCycle] = useState<BiteCycleMetrics | null>(null);
  const [cycleTarget, setCycleTarget] = useState(10);
  const [setupTarget, setSetupTarget] = useState<"upper" | "hinge">("upper");
  const [hingeMode, setHingeMode] = useState<"translate" | "rotate" | null>(null);
  const [hingePreview, setHingePreview] = useState(0);
  const [showHinge, setShowHinge] = useState(true);
  const [hingePreset, setHingePreset] = useState<"negative" | "positive" | "custom">("negative");
  const rowAxis = info && info[0].max[2] - info[0].min[2] > info[0].max[0] - info[0].min[0] ? "Z" : "X";
  const upperPreviewPose = useRef<UpperSetupPose>(identityUpperSetupPose());
  const upperPivot = useMemo<[number, number, number]>(() => info
    ? [0, 1, 2].map(axis => (info[0].min[axis] + info[0].max[axis]) / 2) as [number, number, number]
    : [0, 0, 0], [info]);
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
  const maxContactSpread = Number.isFinite(spacing) && spacing > 0 ? spacing * 2 / 3 : 0;
  const validContactSpread = isCycles || Number.isFinite(contactSpread) && contactSpread >= 0 && contactSpread <= maxContactSpread;
  const gaussianPrepared = !isCycles && (snapshot?.contactSpread ?? 0) > 0;
  let cycleSettingsError = "";
  try { validateBiteCycleSettings(cycleSettings); } catch (cause) { cycleSettingsError = cause instanceof Error ? cause.message : "Check bite-cycle settings."; }
  const completedCycles = cycle?.completedCycles ?? 0;
  const complete = !!snapshot && !dirty && (isCycles ? completedCycles >= cycleTarget : closure >= target - 1e-8 || snapshot.materialExhausted);
  const validRun = isCycles ? !cycleSettingsError && Number.isInteger(cycleTarget) && cycleTarget > 0
    : Number.isFinite(target) && depth > 0 && Number.isFinite(step) && step > 0 && (!vegetationEnabled || validVegetation);
  const locked = busy || running || workerFailed || computeSwitching;
  const vegetationStarted = (snapshot?.vegetation.pullDistance ?? 0) > 0;
  const vegetationLocked = locked || vegetationStarted || isCycles;
  const setupStarted = isCycles ? completedCycles > 0 : closure > 0 || vegetationStarted;
  const poseLocked = locked || !original || setupStarted;
  const cycleSettingsLocked = locked || completedCycles > 0;
  const validUpperPose = isValidUpperSetupPose(upperPose);
  if (validUpperPose) upperPreviewPose.current = upperPose;
  settings.current = { target, step, share: share / 100, vegetation: vegetationEnabled && !isCycles ? vegetationSettings : undefined, mode, cycleTarget, cycleSettings };

  const post = (message: WearRequest) => worker.current?.postMessage(message);
  const clearTrace = () => { traceRequestId.current++; setVegetationTrace(null); setTraceLoading(false); setTraceError(""); post({ type: "trace-off" }); };
  const pause = () => { run.current = false; setRunning(false); window.clearTimeout(timer.current); setStatus(busy ? "Pausing after the current increment…" : "Paused. Continue the run or inspect the current pose."); };

  useEffect(() => {
    // A development hot refresh restarts workers: never keep a stale visible
    // result whose geometry is no longer owned by the active worker.
    setOriginal(null); setSurfaces(null); setInfo(null); setSnapshot(null);
    setUpperPose(identityUpperSetupPose());
    setUpperPoseMode("translate");
    setCycle(null); setHingeMode(null); setHingePreview(0); setSetupTarget("upper");
    setHistory([]); setWarnings([]); setGrid(null); setDirty(false);
    setBusy(false); setRunning(false); setError(""); setProgress(0); setWorkerFailed(false);
    setCompute({ preference: "auto", backend: "initializing" }); setComputeSwitching(false); setTimings(null);
    setVegetationTrace(null); setTraceLoading(false); setTraceError(""); traceRequestId.current++;
    setStatus("Load two aligned toothrows to begin.");
    initializeVegetationBand.current = true;
    const instance = new Worker(new URL("./wear.worker.ts", import.meta.url), { type: "module" });
    worker.current = instance;
    const fail = (message: string) => {
      run.current = false; manualPullPending.current = false; setRunning(false); setBusy(false); setComputeSwitching(false); setError(message); setStatus("Check the setup and try again.");
    };
    instance.onerror = event => {
      setWorkerFailed(true);
      fail(event.message || "The wear worker stopped unexpectedly. Restart the app to continue.");
    };
    instance.onmessage = (event: MessageEvent<WearResponse>) => {
      const message = event.data;
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
        setCycle(null); setHingeMode(null); setHingePreview(0); setSetupTarget("upper");
        setCycleSettings(createDefaultBiteCycleSettings(message.info[0], message.suggestedSpacing));
        setHingePreset("negative");
        setSnapshot(null); setHistory([]); setWarnings([]); setGrid(null); setDirty(false); setTimings(null);
        setSpacing(message.suggestedSpacing); setContactSpread(0); setStep(message.suggestedSpacing / 2);
        setVegetationThickness(message.suggestedSpacing * 10); setPullDistance(message.suggestedSpacing * 50);
        setVegetationStreamThickness(message.suggestedSpacing * 2);
        setVegetationBendReach(message.suggestedSpacing * 5);
        initializeVegetationBand.current = true;
        setBusy(false); setProgress(0); setFitKey(key => key + 1);
        setStatus("Surfaces loaded. Adjust the upper-row setup pose, then prepare the wear grid."); return;
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
        if (message.preparation) { setAutoOpen(message.preparation.autoOpen); setUpperPose(message.preparation.upperPose); }
        if (message.mode !== "cycles") setContactSpread(s.contactSpread ?? 0);
        if (initializeVegetationBand.current) { setVegetationCenterY(Number(s.recommendedVegetationCenterY.toPrecision(7))); initializeVegetationBand.current = false; }
        setHistory(previous => {
          const point: HistoryPoint = { closure: message.mode === "cycles" ? message.cycle?.completedCycles ?? 0 : s.closure, upper: s.removedUpperVolume, lower: s.removedLowerVolume, contact: message.cycle?.contactArea ?? s.contactArea,
            contactUpper: s.contactRemovedUpperVolume, contactLower: s.contactRemovedLowerVolume,
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
        const finished = s.closure >= config.target - 1e-8 || s.materialExhausted;
        setStatus(manualPullPending.current ? `Vegetation pulled at fixed closure. ${format(s.vegetation.effectiveRecession, 3)} model units of cumulative nominal recession applied continuously at sliding contacts.`
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
            setBusy(true);
            instance.postMessage({ type: "advance", target: Math.min(config.target, s.closure + config.step), share: config.share, vegetation: config.vegetation } satisfies WearRequest);
          }, 45);
        } else { run.current = false; setRunning(false); }
      }
    };
    return () => { run.current = false; window.clearTimeout(timer.current); instance.terminate(); worker.current = undefined; };
  }, []);

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
    post({ type: "load", sources: local ? [
      { name: "1aligned.ply", url: new URL(`${import.meta.env.BASE_URL}models/1aligned.ply`, document.baseURI).href }, { name: "2aligned.ply", url: new URL(`${import.meta.env.BASE_URL}models/2aligned.ply`, document.baseURI).href }
    ] : [{ name: files[0]!.name, file: files[0]! }, { name: files[1]!.name, file: files[1]! }] });
  };
  const prepare = (reset = false) => {
    if (!original || locked || !reset && (!validUpperPose || !validContactSpread || !Number.isFinite(spacing) || spacing <= 0)) return;
    clearTrace();
    pause(); setBusy(true); setError(""); setStatus("Sampling solid tooth geometry…"); setProgress(0);
    setSnapshot(null); setSurfaces(null); setHistory([]); setTimings(null);
    setCycle(null); setHingePreview(0); setHingeMode(null); setUpperPoseMode(null);
    post(reset ? { type: "reset" } : { type: "prepare", spacing, autoOpen, contactSpread: isCycles ? 0 : contactSpread, upperPose, mode });
  };
  const changeUpperPose = (next: UpperSetupPose) => {
    if (poseLocked) return;
    if (!isValidUpperSetupPose(next)) { setError("Upper setup pose must contain finite translation and rotation values."); return; }
    clearTrace();
    setUpperPose(next); setDirty(true); setError("");
    // A setup edit invalidates sampling. Preview raw sources in the new pose;
    // simulation and exports stay unavailable until that pose is prepared.
    setSnapshot(null); setSurfaces(null); setHistory([]); setGrid(null); setWarnings([]); setTimings(null); setCycle(null); setHingePreview(0);
    setStatus("Upper-row setup preview. Prepare the wear grid to apply this pose.");
  };
  const restoreUpperPose = async (file: File) => {
    if (poseLocked) return;
    setBusy(true); setError("");
    try {
      const report = JSON.parse(await file.text()) as { setup?: { upper?: { translation?: unknown; rotationDegrees?: unknown } } } | null;
      const pose = { translation: report?.setup?.upper?.translation, rotation: report?.setup?.upper?.rotationDegrees };
      if (!isValidUpperSetupPose(pose)) throw new Error("This JSON does not contain a valid exported upper-row setup pose.");
      changeUpperPose({ translation: [...pose.translation], rotation: [...pose.rotation] });
      setStatus("Restored the upper-row setup pose. Prepare the grid to apply it; worn material and run settings were not restored.");
    } catch (cause) {
      setError(cause instanceof SyntaxError ? "Choose a valid run-settings JSON file." : cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(false); }
  };
  const editUpperPose = (mode: "translate" | "rotate") => {
    if (poseLocked) return;
    setSetupTarget("upper"); setHingeMode(null); setHingePreview(0);
    // Return to the raw setup pose before showing the handles. An automatic
    // opening belongs to preparation, never to the user's rigid transform.
    if (surfaces) changeUpperPose(upperPose);
    setUpperPoseMode(mode);
  };
  const advance = (continuous: boolean) => {
    if (!snapshot || !validRun || dirty || complete || locked) return;
    setError(""); run.current = continuous; setRunning(continuous); setBusy(true);
    setHingePreview(0); setHingeMode(null); setUpperPoseMode(null);
    if (isCycles) post({ type: "cycles", count: 1, settings: cycleSettings });
    else post({ type: "advance", target: Math.min(target, closure + step), share: share / 100, vegetation: settings.current.vegetation });
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
    setTimings(null); setWarnings([]); setError(""); setDirty(true); setHingePreview(0); setUpperPoseMode(null);
    setSetupTarget(next === "cycles" ? "hinge" : "upper"); setHingeMode(null);
    setFitKey(key => key + 1);
    setStatus("Prepare the wear grid for the selected simulation mode. Switching modes starts from unworn geometry.");
  };
  const changeCycleSetting = <K extends keyof BiteCycleSettings>(key: K, value: BiteCycleSettings[K]) => {
    if (cycleSettingsLocked) return;
    setCycleSettings(current => ({ ...current, [key]: value })); setError("");
  };
  const editHinge = (next: "translate" | "rotate") => {
    if (poseLocked) return;
    setSetupTarget("hinge"); setUpperPoseMode(null); setHingeMode(next); setShowHinge(true); setHingePreview(0);
  };
  const placeHinge = (preset: "negative" | "positive" | "custom") => {
    if (!info || cycleSettingsLocked) return;
    setHingePreset(preset);
    if (preset === "custom") return;
    const suggested = createDefaultBiteCycleSettings(info[0], spacing, preset === "negative" ? -1 : 1);
    setCycleSettings(current => ({ ...current, hingeOrigin: suggested.hingeOrigin, hingeAxis: suggested.hingeAxis,
      openingAngle: Math.abs(current.openingAngle || 8) * Math.sign(suggested.openingAngle) }));
    setHingePreview(0); setShowHinge(true); setSetupTarget("hinge"); setUpperPoseMode(null); setHingeMode("rotate");
    setFitKey(key => key + 1);
  };
  const exportReport = () => {
    setHingePreview(0);
    download(JSON.stringify({ application: "Toothwear Studio", version: 9, usageNotice, mode, vegetationModel: "continuous-finite-bundle-sliding-v3", model: isCycles
      ? "Repeated hinge opening and closing about a common closed reference pose, with compliant distributed contact pressure and Archard wear depth = coefficient × pressure × sliding distance. Uncalibrated mechanical approximation; no plant coupling or biological timescale."
      : `Geometric mutual contact wear. Contact spread sigma ${(snapshot?.contactSpread ?? 0) > 0 ? "enables normalized Gaussian-weighted neighboring grid-cell contact with a vertical-contact check; direct local overlap is still resolved" : "is zero, preserving exact same-cell contact"}. Contact spread is a geometric approximation, not contact pressure. Continuous fractional vegetation recession follows complete straight or bounded flexible sliding routes. Stream thickness is plant-bundle diameter; bundles can slide against multiple teeth and both rows at once. Total volume loss is geometric; the contact/vegetation split is estimated within each integration interval. No calibrated branch mechanics, forces, or biological wear rates.`,
      exportedAt: new Date().toISOString(), units: unit === "u" ? "native model units (unspecified)" : unit,
      sources: info, grid, movement: isCycles ? "Upper setup pose is applied before sampling. A single common closed pose is established for the toothrow, then the upper rotates about the configured hinge. The lower stays fixed. Exported PLYs use the closed reference assembly pose; the opening preview is display-only." : "Upper setup pose is applied before sampling. During closure the upper translates along world -Y; lower stays fixed. Exported PLYs include setup pose, initial opening, and final closure.",
      setup: { upper: { translation: upperPose.translation, rotationDegrees: upperPose.rotation, pivot: upperPivot,
        rotationOrder: "world X, then Y, then Z about the source bounding-box center; translation applied afterward" } },
      settings: isCycles ? { autoOpen, targetCycles: cycleTarget, ...cycleSettings, loadSchedule: "Opening is unloaded; force smoothly rises during closure to the configured peak at the shared closed pose.", pressureUnits: `N/${unit}²`, stiffnessUnits: `N/${unit}³`, coefficientUnits: `${unit}²/N` }
        : { autoOpen, contactSpread: snapshot?.contactSpread ?? 0, upperWearShare: share / 100, increment: step, stopMode, stopDistance: depth, targetClosure: target, vegetation: snapshot?.vegetation.settings ?? (vegetationEnabled ? vegetationSettings : null) },
      compute: { ...compute, latestFrameTimings: timings }, snapshot, cycle, warnings, history }, null, 2), "toothwear-run.json", "application/json");
  };
  const percent = snapshot && (isCycles ? cycleTarget : target) > 0 ? Math.min(100, isCycles ? completedCycles / cycleTarget * 100 : closure / target * 100) : 0;
  const canAdvance = !!snapshot && !dirty && !complete && validRun && !locked;
  const canPull = !!snapshot && !dirty && !locked && !isCycles && vegetationEnabled && validVegetation && Number.isFinite(pullDistance) && pullDistance > 0
    && !!surfaces?.some(surface => surface.positions.length > 0);
  const gpuStages = isCycles ? [...(compute.cycleBackend === "gpu" ? ["contact pressure & sliding"] : []), ...(compute.materialBackend === "gpu" ? ["material removal"] : []), ...(compute.meshBackend === "gpu" ? ["surface generation"] : [])]
    : ["routing", ...(compute.contactBackend === "gpu" ? ["sliding contacts"] : []), ...(compute.materialBackend === "gpu" ? ["vegetation removal"] : []), ...(compute.meshBackend === "gpu" ? ["surface generation"] : [])];
  const cpuStages = isCycles ? ["cycle orchestration, surface preparation & bookkeeping", ...(compute.cycleBackend === "gpu" ? [] : ["contact pressure & sliding"]), ...(compute.materialBackend === "gpu" ? [] : ["material removal"]), ...(compute.meshBackend === "gpu" ? [] : ["surface generation"])]
    : [(snapshot?.contactSpread ?? 0) > 0 ? "geometric tooth contact" : "exact tooth contact", ...(compute.contactBackend === "gpu" ? [] : ["sliding contacts"]), ...(compute.materialBackend === "gpu" ? [] : ["vegetation removal"]), ...(compute.meshBackend === "gpu" ? [] : ["surface generation"])];

  return (
    <div className="wear-app">
      <header className="wear-header">
        <div className="wear-brand"><span className="wear-brand-mark"><Layers3 size={23} /></span><div><h1>Toothwear <span>Studio</span></h1><p>Progressive occlusion & mutual wear</p></div></div>
        <div className="wear-header-right"><span className="wear-method">{isCycles ? "CYCLIC CONTACT WEAR" : "GEOMETRIC WEAR"}</span></div>
      </header>
      <aside className="wear-experimental-notice" role="note" aria-label="Experimental use warning"><strong>Experimental — not for research use.</strong><span>Simulated wear has not been validated.</span></aside>
      <main className="wear-workspace">
        <section className="wear-stage" aria-label="Toothrow visualization">
          <WearViewport upper={(surfaces ?? original)?.[0] ?? null} lower={(surfaces ?? original)?.[1] ?? null}
            originalUpper={original?.[0]} originalLower={original?.[1]}
            upperPose={upperPreviewPose.current} upperPivot={upperPivot} upperPoseBaked={!!surfaces}
            upperPoseMode={upperPoseMode} upperPoseEditable={!poseLocked} onUpperPoseChange={changeUpperPose}
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
            {isCycles ? <select aria-label="Setup gizmo target" value={setupTarget} disabled={poseLocked} onChange={event => { const next = event.target.value as "upper" | "hinge"; setSetupTarget(next); setUpperPoseMode(null); setHingeMode(null); setHingePreview(0); }}><option value="upper">Upper row</option><option value="hinge">Jaw hinge</option></select> : <span>Upper row</span>}
            <button aria-label={setupTarget === "hinge" && isCycles ? "Move jaw hinge" : "Move upper teeth"} aria-pressed={(setupTarget === "hinge" && isCycles ? hingeMode : upperPoseMode) === "translate"} className={(setupTarget === "hinge" && isCycles ? hingeMode : upperPoseMode) === "translate" ? "selected" : ""} disabled={poseLocked} onClick={() => setupTarget === "hinge" && isCycles ? editHinge("translate") : editUpperPose("translate")}><Move3d size={14} /> Move</button>
            <button aria-label={setupTarget === "hinge" && isCycles ? "Rotate jaw hinge" : "Rotate upper teeth"} aria-pressed={(setupTarget === "hinge" && isCycles ? hingeMode : upperPoseMode) === "rotate"} className={(setupTarget === "hinge" && isCycles ? hingeMode : upperPoseMode) === "rotate" ? "selected" : ""} disabled={poseLocked} onClick={() => setupTarget === "hinge" && isCycles ? editHinge("rotate") : editUpperPose("rotate")}><Rotate3d size={14} /> Rotate</button>
            <button aria-label="Finish positioning" disabled={poseLocked || upperPoseMode === null && hingeMode === null} onClick={() => { setUpperPoseMode(null); setHingeMode(null); }}><Check size={14} /> Done</button>
          </div>}
          {isCycles && original && <div className="wear-cycle-preview"><label><span>Opening preview <strong>{format(hingePreview * cycleSettings.openingAngle, 1)}°</strong></span><input aria-label="Jaw opening preview" type="range" min={0} max={1} step={0.01} value={hingePreview} disabled={locked || !Number.isFinite(cycleSettings.openingAngle)} onChange={event => { setHingePreview(Number(event.target.value)); setUpperPoseMode(null); setHingeMode(null); }} /></label><span>{hingePreview > 0 ? "Preview only · run and export return to closed pose" : "Closed reference pose · all teeth share this endpoint"}</span></div>}
          {!original && <div className="wear-empty"><div className="wear-empty-icon"><Layers3 size={42} strokeWidth={1} /></div><span className="wear-eyebrow">FROM UNWORN TO OCCLUDED</span><h2>Let the contact<br />shape the teeth.</h2><p>Bring your aligned toothrows together along Y.<br />Wear develops wherever the two surfaces meet.</p><button className="wear-primary" disabled={locked} onClick={() => load(true)}><Upload size={16} /> Load example toothrows <ChevronRight size={16} /></button><small>Two example PLY files · 96 MB download</small></div>}
          {original && <div className="wear-legend"><span><i className="upper" />Upper · {isCycles ? "hinged" : "moving −Y"}</span><span><i className="lower" />Lower · fixed</span><span><i className="worn" />Contact wear</span>{isCycles && showHinge && <span><i className="hinge" />Hinge axis</span>}{!isCycles && vegetationEnabled && <><span><i className="vegetation" />Vegetation wear</span>{showVegetationPaths && <span><i className="vegetation-path" />Computed paths</span>}{showSlidingContacts && <span><i className="sliding-contact" />Sliding now</span>}</>}</div>}
          <div className="wear-stage-bottom"><span>Y ↑ <b>·</b> Drag to orbit <b>·</b> Scroll to zoom <b>·</b> Right drag to pan</span><span>{grid ? `${grid.nx} × ${grid.nz} columns · ${format(grid.spacing)} ${unit} pitch` : original ? "Upper-row setup preview" : "Source coordinates preserved"}</span></div>
        </section>

        <aside className="wear-rail">
          <section className="wear-section">
            <h2><span className="wear-section-number">01</span> Toothrow surfaces <Upload size={15} /></h2>
            <button className="wear-secondary full" disabled={locked} onClick={() => load(true)}>Load example toothrows <span>96 MB</span></button>
            {info && <div className="wear-file-list">{info.map((row, i) => <div className="wear-file" key={i}><i className={i === 0 ? "upper" : "lower"} /><div><strong>{i === 0 ? "Upper" : "Lower"} <span>{row.name}</span></strong><small>{format(row.triangles, 0)} triangles</small></div><Check size={14} /></div>)}<button className="wear-text-button" disabled={locked} onClick={() => { clearTrace(); setBusy(true); setError(""); post({ type: "swap" }); }}><ArrowLeftRight size={13} /> Swap upper / lower assignment</button></div>}
            <details className="wear-import" open><summary>Import your own aligned pair</summary><div>{[0, 1].map(i => <label key={i}>Surface {i + 1}<input aria-label={`Surface ${i + 1} file`} type="file" accept=".ply,.stl" disabled={locked} onChange={event => setFiles(current => { const next: [File | null, File | null] = [...current]; next[i] = event.target.files?.[0] ?? null; return next; })} /></label>)}<button className="wear-secondary full" onClick={() => load(false)} disabled={locked || !files[0] || !files[1]}>Load selected pair</button><p className="wear-note">Closed triangle PLY or STL. Files are processed in your browser and are not uploaded. Higher Y centre is assigned upper; use Swap if needed.</p></div></details>
          </section>

          <section className="wear-section">
            <h2><span className="wear-section-number">02</span> Prepare geometry <Layers3 size={15} /></h2>
            <label className="wear-field"><span>Simulation mode</span><select value={mode} disabled={locked} onChange={event => changeMode(event.target.value as "geometric" | "cycles")}><option value="geometric">Fast geometric wear · default</option><option value="cycles">Repeated bite cycles · experimental</option></select></label>
            {isCycles ? <p className="wear-note">Experimental pressure-and-sliding model; improved facet realism has not been validated. The whole upper row opens about the hinge and returns to the shared closed pose on every cycle.</p>
              : <p className="wear-note">Default workflow: close the rows along Y and wear both teeth where they meet. Optional vegetation abrasion can run alongside contact wear.</p>}
            <fieldset className="wear-pose" disabled={poseLocked}>
              <legend>Upper-row setup pose</legend>
              <p className="wear-note">Choose <strong>Move</strong> or <strong>Rotate</strong> above the model, then drag its colored arrows or rings. Drag away from the handles to orbit the view.</p>
              <dl className="wear-pose-readout"><div><dt>Position · {unit}</dt><dd>{upperPose.translation.map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div><div><dt>Rotation · °</dt><dd>{upperPose.rotation.map((value, axis) => <span key={axis}>{["X", "Y", "Z"][axis]} <b>{format(value, 2)}</b></span>)}</dd></div></dl>
              <button className="wear-text-button" disabled={!upperPose.translation.some(v => v !== 0) && !upperPose.rotation.some(v => v !== 0)} onClick={() => changeUpperPose(identityUpperSetupPose())}><RotateCcw size={12} /> Restore source pose</button>
              <details className="wear-import"><summary>Restore setup pose from run JSON</summary><div><label>Run settings JSON<input aria-label="Restore setup pose from run JSON" type="file" accept=".json,application/json" disabled={poseLocked} onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void restoreUpperPose(file); }} /></label><p className="wear-note">Restores the upper-row position and rotation only. Load the original toothrows first. Worn material and other run settings are not restored.</p></div></details>
            </fieldset>
            <p className="wear-note">Preview changes live, then prepare the grid. The gizmo uses world axes and rotates about the upper row’s center.</p>
            {setupStarted && <p className="wear-warning wear-pose-warning">Reset to unworn geometry to adjust the setup pose{isCycles ? " or hinge" : ""}.</p>}
            {!validUpperPose && <p className="wear-warning wear-pose-warning">Use finite pose values and translations within the supported coordinate range.</p>}
            <div className="wear-field-pair"><NumberField label="Column pitch" value={spacing} min={0.001} step="any" suffix={unit} disabled={locked} onChange={v => { setSpacing(v); setDirty(true); }} /><label className="wear-field"><span>Unit label</span><select value={unit} disabled={locked} onChange={e => setUnit(e.target.value)}><option value="u">Native units</option><option value="mm">Millimetres</option><option value="cm">Centimetres</option></select></label></div>
            <p className="wear-note">Smaller pitch retains finer detail. Y surfaces stay continuous. Unit labels do not rescale your files.</p>
            {!isCycles && <>
              <NumberField label="Contact spread σ" value={contactSpread} min={0} max={maxContactSpread} step="any" suffix={unit} disabled={locked} onChange={value => { setContactSpread(value); setDirty(true); setStatus("Prepare the wear grid to apply the contact spread. This starts from unworn geometry."); }} />
              <p className="wear-note">0 keeps exact cell contact. Try {format(spacing / 2, 4)} {unit} (half a pitch) for a finite Gaussian contact footprint. Direct overlap is still resolved. This controls geometry, not pressure. Maximum {format(maxContactSpread, 4)} {unit} keeps it fast. Prepare again after changing it.</p>
              {!validContactSpread && <p className="wear-warning">Enter a finite contact spread from 0 to two-thirds of the column pitch ({format(maxContactSpread, 4)} {unit}).</p>}
            </>}
            <label className="wear-check"><input type="checkbox" checked={autoOpen} disabled={locked} onChange={e => { setAutoOpen(e.target.checked); setDirty(true); }} /> Automatically clear initial overlap along Y</label>
            <button className="wear-secondary full" disabled={!original || locked || !validUpperPose || !validContactSpread || !Number.isFinite(spacing) || spacing <= 0} onClick={() => prepare()}><Layers3 size={15} />{surfaces && !dirty ? "Rebuild wear grid" : "Prepare wear grid"}</button>
            {snapshot && <div className="wear-prepared"><Check size={13} /><span>{isCycles ? "Closed reference offset" : gaussianPrepared ? "Initial direct contact at" : "Initial first contact at"} <strong>{format(isCycles ? closure : first, 3)} {unit}</strong>{isCycles ? " along −Y" : " closure"}{snapshot.initialOpening > 0 ? ` · opened ${format(snapshot.initialOpening, 3)} ${unit} to clear overlap` : ""}</span></div>}
            <div className="wear-compute">
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
            </div>
          </section>

          <section className="wear-section">
            <h2><span className="wear-section-number">03</span> {isCycles ? "Bite cycles & wear" : "Closure & wear"} <ArrowDown size={16} /></h2>
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
            </> : <>
            <label className="wear-field"><span>Preset stopping point</span><select value={stopMode} disabled={locked} onChange={e => setStopMode(e.target.value as "contact" | "total")}><option value="contact">Distance beyond initial {gaussianPrepared ? "direct" : "first"} contact</option><option value="total">Total closure from prepared pose</option></select></label>
            <div className="wear-field-pair"><NumberField label="Stop distance" value={depth} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setDepth} /><NumberField label="Closure increment" value={step} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setStep} /></div>
            <div className="wear-share"><div><span>Wear allocation</span><strong>{share}% upper / {100 - share}% lower</strong></div><input aria-label="Upper wear percentage" type="range" min={0} max={100} step={5} value={share} disabled={locked || closure > 0} onChange={e => setShare(Number(e.target.value))} /><div className="wear-range-labels"><span>Lower only</span><span>Equal</span><span>Upper only</span></div></div>
            <p className="wear-note">Equal wear removes the same local depth from both rows. The upper translates −Y; the lower stays fixed. Tooth-to-tooth wear stays active during closure, including when vegetation is enabled.</p>
            {gaussianPrepared && <p className="wear-note">The stopping reference and projected contact area measure direct same-cell contact. Gaussian neighboring-cell wear can begin earlier.</p>}
            </>}
            <div className="wear-run-buttons"><button className="wear-primary" disabled={running ? false : !canAdvance} onClick={() => running ? pause() : advance(true)}>{running ? <Pause size={16} /> : <Play size={16} />}{running ? "Pause" : complete ? "Complete" : isCycles ? "Run bites" : "Run wear"}</button><button className="wear-secondary" disabled={!canAdvance} onClick={() => advance(false)} title={isCycles ? "Run one complete bite cycle" : "Advance one closure increment"}><SkipForward size={16} />{isCycles ? "One bite" : "Step"}</button><button className="wear-secondary wear-icon" disabled={!snapshot || locked} onClick={() => prepare(true)} title="Reset to unworn geometry using the last prepared settings"><RotateCcw size={16} /></button></div>
            {!isCycles && snapshot && closure < first && <button className="wear-text-button" disabled={!canAdvance} onClick={contactJump}>Move to initial {gaussianPrepared ? "direct-contact" : "first-contact"} closure <ChevronRight size={12} /></button>}
            {dirty && snapshot && <p className="wear-warning">Prepare the grid to apply edited settings, or reset to restore the last prepared settings.</p>}
            {snapshot && (isCycles ? cycleTarget < completedCycles : target < closure - 1e-8) && <p className="wear-warning">Reset the run to use a stopping point before the current {isCycles ? "bite count" : "closure"}.</p>}
          </section>

          <section className="wear-section wear-vegetation-section">
            <h2><span className="wear-section-number">04</span> Vegetation abrasion <Leaf size={15} /></h2>
            {isCycles && <p className="wear-note">Vegetation mode is available with geometric closure. Repeated bite cycles currently model tooth-to-tooth contact only.</p>}
            <label className="wear-check"><input type="checkbox" checked={vegetationEnabled} disabled={vegetationLocked} onChange={e => setVegetationEnabled(e.target.checked)} /> Enable sideways branch stripping</label>
            {!isCycles && <p className="wear-note">Vegetation wears surfaces as it slides alongside them, including areas without tooth-to-tooth contact. A head-on obstruction alone does not cause abrasion.</p>}
            {!isCycles && vegetationEnabled && <>
              <label className="wear-field"><span>Vegetation path</span><select value={vegetationPathMode} disabled={vegetationLocked} onChange={e => setVegetationPathMode(e.target.value as "straight" | "flexible")}><option value="flexible">Flexible around teeth</option><option value="straight">Straight sliding paths</option></select></label>
              <label className="wear-field"><span>Pull direction</span><select value={vegetationDirection} disabled={vegetationLocked} onChange={e => setVegetationDirection(e.target.value as VegetationDirection)}><option value="+x">+X · left to right in front view</option><option value="-x">−X · right to left in front view</option><option value="+z">+Z · toward the front view</option><option value="-z">−Z · away from the front view</option></select></label>
              <div className="wear-field-pair"><NumberField label="Incoming band centre Y" value={vegetationCenterY} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationCenterY} /><NumberField label="Incoming thickness" value={vegetationThickness} min={0.001} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationThickness} /></div>
              <NumberField label="Stream thickness · bundle diameter" value={vegetationStreamThickness} min={0} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationStreamThickness} />
              <p className="wear-note">The band distributes bundle centres; each bundle extends half its diameter beyond its path. A bundle can slide against several teeth, including both rows at once. Wider bundles need wider gaps; diameter stays fixed while bending. Use 0 for the original thin paths.</p>
              {vegetationPathMode === "flexible" && <><NumberField label="Bend reach" value={vegetationBendReach} min={0} step="any" suffix={unit} disabled={vegetationLocked} onChange={setVegetationBendReach} /><p className="wear-note">Total vertical and sideways detour allowed, including return to the band at the exit. Paths need an open way through. Bends use column-pitch steps; less than one pitch gives straight paths.</p></>}
              <div className="wear-field-pair"><NumberField label="Abrasion per pull distance" value={abrasionRate} min={0.000001} step="any" suffix="u/u" disabled={vegetationLocked} onChange={setAbrasionRate} /><NumberField label="Pull per closure distance" value={pullPerClosure} min={0} step="any" suffix="u/u" disabled={vegetationLocked} onChange={setPullPerClosure} /></div>
              <p className="wear-note">The entrance and exit band stay at this world Y. Closure combines tooth contact wear with vegetation pulls at the ratio above; use 0 for manual pulls only.</p>
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
              <div className="wear-manual-pull"><NumberField label="Manual pull distance" value={pullDistance} min={0.001} step="any" suffix={unit} disabled={locked} onChange={setPullDistance} /><button className="wear-secondary full" disabled={!canPull} onClick={pullVegetation}><Leaf size={14} /> Pull vegetation at current pose</button></div>
              <p className="wear-note">Every pull adds proportional recession at sliding contacts, including pulls smaller than the column pitch. A manual pull holds the jaws fixed; continue closure to produce more tooth-to-tooth wear.</p>
              {!validVegetation && <p className="wear-warning">Enter a finite band centre, positive band thickness and abrasion rate, and nonnegative bundle diameter, bend reach, and pull-to-closure ratio.</p>}
              {snapshot && <dl className="wear-vegetation-metrics"><div className="wear-contact-measurement"><dt>Tooth contact · upper removed</dt><dd>{format(snapshot.contactRemovedUpperVolume)} {unit}³</dd></div><div className="wear-contact-measurement"><dt>Tooth contact · lower removed</dt><dd>{format(snapshot.contactRemovedLowerVolume)} {unit}³</dd></div><div className="wear-vegetation-measurement"><dt>Vegetation · upper removed</dt><dd>{format(snapshot.vegetation.removedUpperVolume)} {unit}³</dd></div><div><dt>Vegetation · lower removed</dt><dd>{format(snapshot.vegetation.removedLowerVolume)} {unit}³</dd></div><div><dt>Cumulative pull</dt><dd>{format(snapshot.vegetation.pullDistance, 3)} {unit}</dd></div><div><dt>Nominal recession</dt><dd>{format(snapshot.vegetation.effectiveRecession, 3)} {unit}</dd></div></dl>}
              <p className="wear-note">Total volume loss is geometric; the contact/vegetation split is estimated within each integration interval.</p>
              <p className="wear-note wear-vegetation-model">Recession is continuous; route detail still follows the column pitch. Paths that cannot reach the exit add no abrasion. No branch forces or tension are solved. Surface colors show the latest wear type; measurements separate both sources. Settings lock after the first vegetation pull; reset to change them.</p>
            </>}
          </section>

          <section className="wear-section wear-display-section">
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
          <div className="wear-model-note">{isCycles ? "Hinge kinematics, compliant contact, and pressure × sliding wear. Force and material parameters are user assumptions, not a calibrated prediction of Camarasaurus feeding or biological time." : "Geometric contact and material removal. This model does not predict bite forces, wear rates, or elapsed biological time."}</div>
        </aside>

        <section className="wear-results" aria-label="Wear measurements">
          <div className="wear-progress-block"><div><span className="wear-eyebrow">{running ? "RUNNING" : !isCycles && snapshot?.materialExhausted ? "MATERIAL EXHAUSTED" : complete ? "STOPPING POINT REACHED" : snapshot ? "READY TO ADVANCE" : "SESSION PROGRESS"}</span><strong>{format(percent, 0)}<small>%</small></strong></div><div className="wear-progress-track"><i style={{ width: `${percent}%` }} /></div><p>{isCycles ? `${format(completedCycles, 0)} / ${format(cycleTarget, 0)} bites` : `${format(closure, 3)} / ${snapshot ? format(target, 3) : "—"} ${unit} closure`}</p></div>
          <Metric label="Upper removed" value={format(snapshot?.removedUpperVolume)} unit={`${unit}³`} color="upper" />
          <Metric label="Lower removed" value={format(snapshot?.removedLowerVolume)} unit={`${unit}³`} color="lower" />
          <Metric label={isCycles ? "Contact at peak load" : gaussianPrepared ? "Projected direct contact" : "Projected contact"} value={format(isCycles ? cycle?.contactArea : snapshot?.contactArea)} unit={`${unit}²`} />
          <div className="wear-chart"><span>Removed volume / {isCycles ? "bite count" : "closure"}</span><WearChart history={history} cycles={isCycles} /></div>
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
function WearChart({ history, cycles = false }: { history: HistoryPoint[]; cycles?: boolean }) {
  const maxX = Math.max(1e-9, ...history.map(p => p.closure));
  const maxY = Math.max(1e-9, ...history.flatMap(p => [p.upper, p.lower]));
  const path = (key: "upper" | "lower") => history.map((p, i) => `${i ? "L" : "M"}${4 + p.closure / maxX * 192},${47 - p[key] / maxY * 40}`).join(" ");
  return <svg viewBox="0 0 200 52" role="img" aria-label={`Removed volume versus ${cycles ? "bite count" : "closure"}, cream for upper and teal for lower`}><path d="M4 7H196 M4 27H196 M4 47H196" stroke="#dce4e8" strokeWidth="1" fill="none" />{history.length > 1 && <><path d={path("upper")} stroke="#b59865" strokeWidth="3" fill="none" /><path d={path("lower")} stroke="#178d87" strokeWidth="1.5" strokeDasharray="4 2" fill="none" /></>}</svg>;
}
