import { useEffect, useState } from "react";
import type { CrownAngleTarget, CrownInclination } from "./crownOrientation";

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

export function CrownAngles({ value, row, disabled, onChange }: { value: CrownInclination | null; row: string; disabled: boolean; onChange: (target: CrownAngleTarget) => string | null }) {
  const [alongSign, setAlongSign] = useState(value && value.along < 0 ? -1 : 1);
  const [acrossSign, setAcrossSign] = useState(value && value.across < 0 ? -1 : 1);
  const [controlError, setControlError] = useState<string | null>(null);
  useEffect(() => {
    if (value && Math.abs(value.along) > 1e-8) setAlongSign(value.along < 0 ? -1 : 1);
    if (value && Math.abs(value.across) > 1e-8) setAcrossSign(value.across < 0 ? -1 : 1);
    setControlError(null);
  }, [value]);
  const controlsReady = value?.pitch !== null && value?.roll !== null && value?.edgeYaw !== null && value !== null;
  const change = (patch: Partial<CrownAngleTarget>) => {
    if (!controlsReady || !value || disabled) return "Controls unavailable";
    const error = onChange({ pitch: value.pitch!, roll: value.roll!, edgeYaw: value.edgeYaw!, alongSign, acrossSign, ...patch });
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
          <input aria-label={`${row} crown pitch angle`} type="range" min="1" max="90" step="0.1" value={value.pitch ?? 90} disabled={disabled || !controlsReady} onChange={event => change({ pitch: Number(event.target.value) })} />
        </label>
        <select aria-label={`${row} pitch lean direction`} value={alongSign} disabled={disabled || !controlsReady} onChange={event => { const sign = Number(event.target.value); if (!change({ alongSign: sign })) setAlongSign(sign); }}>
          <option value={1}>Tip toward +row</option><option value={-1}>Tip toward −row</option>
        </select>
        <label><span>Roll · across row <strong>{angle(value.roll)}</strong></span>
          <input aria-label={`${row} crown roll angle`} type="range" min="1" max="90" step="0.1" value={value.roll ?? 90} disabled={disabled || !controlsReady} onChange={event => change({ roll: Number(event.target.value) })} />
        </label>
        <select aria-label={`${row} roll lean direction`} value={acrossSign} disabled={disabled || !controlsReady} onChange={event => { const sign = Number(event.target.value); if (!change({ acrossSign: sign })) setAcrossSign(sign); }}>
          <option value={1}>Tip toward +across</option><option value={-1}>Tip toward −across</option>
        </select>
        <label><span>Yaw · estimated edge <strong>{angle(value.edgeYaw)}</strong></span>
          <input aria-label={`${row} crown yaw angle`} type="range" min="-90" max="90" step="0.1" value={value.edgeYaw ?? 0} disabled={disabled || !controlsReady} onChange={event => change({ edgeYaw: Number(event.target.value) })} />
        </label>
        <p className="wear-note">All six crowns rotate about their own centers. Each slider holds the other two angles fixed. Yaw 0° aligns the estimated cutting edge with the jawline. Whole-row positioning uses the gizmo.</p>
        <p className="wear-note">Pitch and roll: 1–90°. Use the gizmo for horizontal or inverted poses. Prepare the wear grid after editing.</p>
        {controlError && <p className="wear-warning" role="alert">{controlError}</p>}
      </div>
      {!value.towardOcclusion && <p className="wear-warning">This crown points away from the opposing row.</p>}
      <details className="wear-angle-details"><summary>Axis definition & other angles</summary>
        <p className="wear-note">Long axis estimated between cross-section centers at 20% and 80% of the unworn crown height. Whole-row rotation is included. The reference plane stays perpendicular to Y closure; along-row follows the row’s projected X axis. These are setup measurements, including during wear.</p>
        <p className="wear-note">True 3D inclination: <strong>{angle(value.inclination)}</strong><br />Estimated cutting-edge yaw: <strong>{angle(value.edgeYaw)}</strong></p>
        <p className="wear-note">Edge yaw uses the widest direction of the middle crown section, viewed from above. It may differ slightly from the multi-section edge measurement used in the parameter matrix.</p>
      </details>
    </> : <p className="wear-note">A crown axis could not be measured in this pose. Source rotations remain available below.</p>}
  </div>;
}
