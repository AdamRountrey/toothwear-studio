import { useEffect, useState } from "react";
import type { CrownAngleEdit, CrownInclination } from "./crownOrientation";

const angle = (n: number | null) => n === null ? "—" : `${n.toFixed(1)}°`;

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

export function CrownAngles({ value, row, disabled, onChange }: { value: CrownInclination | null; row: string; disabled: boolean; onChange: (edit: CrownAngleEdit) => string | null }) {
  const [alongSign, setAlongSign] = useState(value && value.along < 0 ? -1 : 1);
  const [acrossSign, setAcrossSign] = useState(value && value.across < 0 ? -1 : 1);
  const [controlError, setControlError] = useState<string | null>(null);
  useEffect(() => {
    if (value && Math.abs(value.along) > 1e-8) setAlongSign(value.along < 0 ? -1 : 1);
    if (value && Math.abs(value.across) > 1e-8) setAcrossSign(value.across < 0 ? -1 : 1);
    setControlError(null);
  }, [value]);
  const change = (edit: CrownAngleEdit) => {
    if (!value || disabled) return "Controls unavailable";
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
        <label><span>Pitch · along row <strong>{angle(value.pitch)}</strong></span>
          <input aria-label={`${row} crown pitch angle`} type="range" min="0" max="90" step="0.1" value={value.pitch ?? 90} disabled={disabled || value.pitch === null} onChange={event => change({ angle: "pitch", value: Number(event.target.value), leanSign: alongSign })} />
        </label>
        <select aria-label={`${row} pitch lean direction`} value={alongSign} disabled={disabled || value.pitch === null} onChange={event => { const sign = Number(event.target.value); if (!change({ angle: "pitch", value: value.pitch!, leanSign: sign })) setAlongSign(sign); }}>
          <option value={1}>Tip toward +row</option><option value={-1}>Tip toward −row</option>
        </select>
        <label><span>Roll · across row <strong>{angle(value.roll)}</strong></span>
          <input aria-label={`${row} crown roll angle`} type="range" min="0" max="90" step="0.1" value={value.roll ?? 90} disabled={disabled || value.roll === null} onChange={event => change({ angle: "roll", value: Number(event.target.value), leanSign: acrossSign })} />
        </label>
        <select aria-label={`${row} roll lean direction`} value={acrossSign} disabled={disabled || value.roll === null} onChange={event => { const sign = Number(event.target.value); if (!change({ angle: "roll", value: value.roll!, leanSign: sign })) setAcrossSign(sign); }}>
          <option value={1}>Tip toward +across</option><option value={-1}>Tip toward −across</option>
        </select>
        <label><span>Yaw · estimated edge <strong>{angle(value.edgeYaw)}</strong></span>
          <input aria-label={`${row} crown yaw angle`} type="range" min="-90" max="90" step="0.1" value={value.edgeYaw ?? 0} disabled={disabled || value.edgeYaw === null} onChange={event => change({ angle: "edgeYaw", value: Number(event.target.value) })} />
        </label>
        <p className="wear-note">All six crowns rotate about their own centers. Roll turns about the jawline; pitch about the across-row axis; yaw about the occlusal-plane normal. The other measured angles update naturally.</p>
        <p className="wear-note">Pitch and roll: 0–90°. Yaw 0° aligns the estimated cutting edge with the jawline. A missing projection disables only that control. Prepare the wear grid after editing.</p>
        {controlError && <p className="wear-warning" role="alert">{controlError}</p>}
      </div>
      {!value.towardOcclusion && <p className="wear-warning">This crown points away from the opposing row.</p>}
      <details className="wear-angle-details"><summary>Axis definition & other angles</summary>
        <p className="wear-note">During rotation, X/Y/Z mark the crown’s original local axes; L marks its estimated long axis and E its estimated cutting-edge direction. All are fixed to the original crown. E is a direction estimate, not a fitted cutting-surface plane.</p>
        <p className="wear-note">The gray reference patch stays parallel to world XZ, perpendicular to closure along Y. Only its display position follows the selected tooth. Rotating a whole row changes its projected jawline direction, but does not tilt this plane.</p>
        <p className="wear-note">Long axis estimated between cross-section centers at 20% and 80% of the unworn crown height. Whole-row rotation is included. The reference plane stays perpendicular to Y closure; along-row follows the row’s projected X axis. These are setup measurements, including during wear.</p>
        <p className="wear-note">True 3D inclination: <strong>{angle(value.inclination)}</strong><br />Estimated cutting-edge yaw: <strong>{angle(value.edgeYaw)}</strong></p>
        <p className="wear-note">Edge yaw uses the widest direction of the middle crown section, viewed from above. It may differ slightly from the multi-section edge measurement used in the parameter matrix.</p>
      </details>
    </> : <p className="wear-note">A crown axis could not be measured in this pose. Source rotations remain available below.</p>}
  </div>;
}
