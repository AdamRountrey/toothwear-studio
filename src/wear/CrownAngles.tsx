import { useEffect, useId, useRef, useState } from "react";
import type { CrownAngleEdit, CrownInclination } from "./crownOrientation";

const angle = (n: number | null) => n === null ? "—" : `${n.toFixed(1)}°`;
const editableAngle = (n: number | null) => n === null ? "" : String(Number(n.toFixed(4)));

function AngleControl({ title, name, value, min, max, disabled, onChange }: { title: string; name: string; value: number | null; min: number; max: number; disabled: boolean; onChange: (value: number) => string | null }) {
  const id = useId();
  const [draft, setDraft] = useState(editableAngle(value));
  const [error, setError] = useState<string | null>(null);
  const dirty = useRef(false);
  const unavailable = disabled || value === null;
  useEffect(() => {
    setDraft(editableAngle(value));
    setError(null);
    dirty.current = false;
  }, [value, disabled]);
  const commit = () => {
    if (unavailable || !dirty.current) return;
    const next = Number(draft);
    if (!draft.trim() || !Number.isFinite(next) || next < min || next > max) {
      setError(`Enter an angle from ${min}° to ${max}°.`);
      return;
    }
    // Use the same edit callback as the slider, including row linking, lean
    // direction, gizmo selection and setup invalidation. Incomplete typing
    // never moves the crown; apply only on Enter or leaving the input.
    if (onChange(next)) return;
    dirty.current = false;
    setDraft(editableAngle(next));
    setError(null);
  };
  return <div className="wear-angle-setting">
    <label htmlFor={id}><span>{title} <strong>{angle(value)}</strong></span></label>
    <div className="wear-angle-inputs">
      <input id={id} aria-label={`${name} angle`} type="range" min={min} max={max} step="0.1" value={value ?? min} disabled={unavailable} onChange={event => {
        if (!onChange(Number(event.target.value))) { dirty.current = false; setDraft(editableAngle(Number(event.target.value))); setError(null); }
      }} />
      <div className="wear-angle-number">
        <input aria-label={`${name} degrees`} aria-invalid={error ? true : undefined} aria-describedby={error ? `${id}-error` : undefined} type="number" min={min} max={max} step="any" value={draft} disabled={unavailable}
          onChange={event => { setDraft(event.target.value); dirty.current = true; setError(null); }} onBlur={commit}
          onKeyDown={event => {
            if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); }
            if (event.key === "Escape") { event.preventDefault(); dirty.current = false; setDraft(editableAngle(value)); setError(null); }
          }} />
        <span aria-hidden="true">°</span>
      </div>
    </div>
    {error && <p id={`${id}-error`} className="wear-warning" role="alert">{error}</p>}
  </div>;
}

function Projection({ name, subtitle, value, horizontal, vertical, axis }: { name: string; subtitle: string; value: number | null; horizontal: number; vertical: number; axis: string }) {
  const length = Math.hypot(horizontal, vertical);
  const x = length > 1e-8 ? 56 + 40 * horizontal / length : 56;
  const y = length > 1e-8 ? 51 - 40 * Math.abs(vertical) / length : 51;
  return <div className="wear-inclination-card">
    <span>{name} <strong>{angle(value)}</strong></span><small>{subtitle}</small>
    <svg viewBox="0 0 112 65" role="img" aria-label={`${name}: ${angle(value)} to the occlusal plane`}>
      <path d="M8 51H104" stroke="#9caeb6" strokeWidth="2" />
      <path d="M56 51V8" stroke="#c4d2d8" strokeDasharray="3 3" />
      {value !== null && <><path d={`M56 51L${x} ${y}`} stroke="#27877d" strokeWidth="3" /><circle cx={x} cy={y} r="3" fill="#27877d" /></>}
      <text x="7" y="63">−{axis}</text><text x="105" y="63" textAnchor="end">+{axis}</text>
    </svg>
    <small>{value === null ? "No projection in this view" : Math.abs(horizontal) < 1e-5 ? "Upright in this view" : `Tip leans toward ${horizontal > 0 ? "+" : "−"}${axis}`}</small>
  </div>;
}

export function CrownAngles({ value, axialYaw, row, disabled, onChange }: { value: CrownInclination | null; axialYaw: number | null; row: string; disabled: boolean; onChange: (edit: CrownAngleEdit) => string | null }) {
  const [alongSign, setAlongSign] = useState(value && value.along < 0 ? -1 : 1);
  const [acrossSign, setAcrossSign] = useState(value && value.across < 0 ? -1 : 1);
  const [controlError, setControlError] = useState<string | null>(null);
  useEffect(() => {
    if (value && Math.abs(value.along) > 1e-8) setAlongSign(value.along < 0 ? -1 : 1);
    if (value && Math.abs(value.across) > 1e-8) setAcrossSign(value.across < 0 ? -1 : 1);
    setControlError(null);
  }, [value]);
  const change = (edit: CrownAngleEdit) => {
    if (disabled) return "Controls unavailable";
    const error = onChange(edit);
    setControlError(error);
    return error;
  };
  return <div className="wear-inclinations" aria-label={`${row} crown angles to occlusal plane`}>
    <p className="wear-inclination-title">{row} crown · angles to occlusal plane</p>
    {value ? <>
      <div className="wear-inclination-pair">
        <Projection name="Pitch" subtitle="Along the row" value={value.pitch} horizontal={value.along} vertical={value.vertical} axis="row" />
        <Projection name="Roll" subtitle="Across the row" value={value.roll} horizontal={value.across} vertical={value.vertical} axis="across" />
      </div>
      <p className="wear-note">90° = perpendicular · 0° = parallel. Green line: projected base-to-tip axis. Gray line: occlusal plane. Both rows are drawn tip-up for comparison.</p>
      <div className="wear-angle-controls" role="group" aria-label={`${row} linked crown rotation controls`}>
        <p>Rotate each {row.toLowerCase()} crown</p>
        <AngleControl title="Pitch · along row" name={`${row} crown pitch`} value={value.pitch} min={0} max={90} disabled={disabled} onChange={next => change({ angle: "pitch", value: next, leanSign: alongSign })} />
        <select aria-label={`${row} pitch lean direction`} value={alongSign} disabled={disabled || value.pitch === null} onChange={event => { const sign = Number(event.target.value); if (!change({ angle: "pitch", value: value.pitch!, leanSign: sign })) setAlongSign(sign); }}>
          <option value={1}>Tip toward +row</option><option value={-1}>Tip toward −row</option>
        </select>
        <AngleControl title="Roll · across row" name={`${row} crown roll`} value={value.roll} min={0} max={90} disabled={disabled} onChange={next => change({ angle: "roll", value: next, leanSign: acrossSign })} />
        <select aria-label={`${row} roll lean direction`} value={acrossSign} disabled={disabled || value.roll === null} onChange={event => { const sign = Number(event.target.value); if (!change({ angle: "roll", value: value.roll!, leanSign: sign })) setAcrossSign(sign); }}>
          <option value={1}>Tip toward +across</option><option value={-1}>Tip toward −across</option>
        </select>
        <p className="wear-note">All six crowns rotate about their own centers. Roll tilts the long axis across the row; pitch tilts it along the row. Both preserve axial yaw. These two projected inclinations can affect each other.</p>
        <p className="wear-note">Pitch and roll: 0–90°. Type a value and press Enter or leave the field to apply; Escape cancels typing. Sliders and typed values select the matching ring above the model. A missing projection disables only that control. Prepare the wear grid after editing.</p>
        {controlError && <p className="wear-warning" role="alert">{controlError}</p>}
      </div>
      {!value.towardOcclusion && <p className="wear-warning">This crown points away from the opposing row.</p>}
      <details className="wear-angle-details"><summary>Axis definition & other angles</summary>
        <p className="wear-note">During rotation, X/Y/Z mark the crown’s original local axes; L marks its estimated long axis (the yaw axis) and E its estimated cutting-edge direction. All are fixed to the original crown. E is a direction estimate, not a fitted cutting-surface plane.</p>
        <p className="wear-note">The gray reference patch stays parallel to world XZ, perpendicular to closure along Y. Only its display position follows the selected tooth. Rotating a whole row changes its projected jawline direction, but does not tilt this plane.</p>
        <p className="wear-note">Long axis estimated between cross-section centers at 20% and 80% of the unworn crown height. Whole-row rotation is included. The reference plane stays perpendicular to Y closure; along-row follows the row’s projected X axis. These are setup measurements, including during wear.</p>
        <p className="wear-note">True 3D inclination: <strong>{angle(value.inclination)}</strong><br />Cutting-edge heading along jaw (top view): <strong>{angle(value.edgeYaw)}</strong></p>
        <p className="wear-note">Cutting-edge heading is a separate measurement, not axial yaw. It uses the widest direction of the middle crown section viewed from above, and may differ slightly from the multi-section measurement used in the parameter matrix.</p>
      </details>
    </> : <p className="wear-note">Pitch/roll are unavailable because the crown axis or projected jawline could not be measured. Axial yaw remains available when its reference is defined.</p>}
    <div className="wear-angle-controls" role="group" aria-label={`${row} axial yaw control`}>
      <AngleControl title="Yaw · spin about tooth long axis" name={`${row} crown yaw`} value={axialYaw} min={-180} max={180} disabled={disabled} onChange={next => change({ angle: "yaw", value: next })} />
      <p className="wear-note">Yaw: −180–180°. Type a value and press Enter or leave the field to apply; Escape cancels typing.</p>
      <p className="wear-note">Yaw 0° aligns the cutting-edge heading with the row when pitch and roll are both 90°. Positive yaw turns uppers and lowers in the same direction when upright. The slider and ring spin about the tooth’s long axis (gold L); pitch, roll and true inclination stay unchanged. Tilting preserves yaw; the top-view edge heading can change.</p>
      {value && <p className="wear-note">Separate edge heading along jaw: <strong>{angle(value.edgeYaw)}</strong></p>}
      {controlError && !value && <p className="wear-warning" role="alert">{controlError}</p>}
    </div>
  </div>;
}
