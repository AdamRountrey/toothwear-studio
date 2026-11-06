import type { Interval, WearState } from "./engine";

export function validateChipThickness(spacing: number, thickness: number): void {
  if (!Number.isFinite(thickness) || thickness < 0 || thickness > spacing * 8)
    throw new Error("Thin-wall thickness must be between zero and eight grid pitches.");
}

/**
 * A deliberately simple geometric breakage proxy, not a fracture law.
 * Trim only a narrow cap connected to a contact-worn neighborhood, no deeper
 * than that neighborhood's actual contact recession. Width is measured along
 * grid X and Z; voids and undercuts remain voids. One simultaneous pass prevents
 * scan-order cascades. No new columns, patches or mesh subdivisions are made.
 */
export function chipThinWalls(state: WearState): void {
  const thickness = state.chipThickness ?? 0;
  const { spacing: h, nx, nz } = state.grid;
  if (thickness <= h) return; // Cannot resolve a narrower cross-section.
  const radius = Math.ceil(thickness / h);
  const area = h * h;
  for (const row of ["upper", "lower"] as const) {
    const upper = row === "upper";
    let furthestContact = upper ? -Infinity : Infinity;
    for (const column of state.columns) for (const v of column[row]) {
      if (upper && v.contactMin !== undefined) furthestContact = Math.max(furthestContact, v.contactMin);
      if (!upper && v.contactMax !== undefined) furthestContact = Math.min(furthestContact, v.contactMax);
    }
    const changes: Array<{ cell: number; index: number; cap: number }> = [];
    const solid = (x: number, z: number, y: number): boolean =>
      x >= 0 && x < nx && z >= 0 && z < nz &&
      state.columns[z * nx + x][row].some(v => y > v.min && y < v.max);
    for (let cell = 0; cell < state.columns.length; cell++) {
      const column = state.columns[cell];
      const index = upper ? column.upperStart : column.lowerEnd;
      const cap = column[row][index];
      if (!cap || cap.max <= cap.min) continue;
      if (upper ? cap.min >= furthestContact : cap.max <= furthestContact) continue;
      const x = cell % nx, z = Math.floor(cell / nx);
      let limit = upper ? cap.min : cap.max;
      // Only contact-produced markers seed chipping; chipped surfaces do not.
      for (let dz = -radius; dz <= radius; dz++) for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dz * dz > radius * radius) continue;
        const xx = x + dx, zz = z + dz;
        if (xx < 0 || xx >= nx || zz < 0 || zz >= nz) continue;
        for (const v of state.columns[zz * nx + xx][row]) {
          if (upper && v.contactMin !== undefined && v.contactMin > v.initialMin)
            limit = Math.max(limit, v.contactMin);
          if (!upper && v.contactMax !== undefined && v.contactMax < v.initialMax)
            limit = Math.min(limit, v.contactMax);
        }
      }
      limit = Math.max(cap.min, Math.min(cap.max, limit));
      const start = upper ? cap.min : cap.max;
      if (Math.abs(limit - start) <= state.epsilon) continue;
      // Cross-section changes only at a neighboring interval boundary.
      const levels = new Set<number>([start, limit]);
      const add = (v: Interval) => {
        for (const y of [v.min, v.max])
          if (y > Math.min(start, limit) && y < Math.max(start, limit)) levels.add(y);
      };
      for (const [dx, dz] of [[1, 0], [0, 1]]) for (let k = -radius; k <= radius; k++) {
        const xx = x + dx * k, zz = z + dz * k;
        if (xx >= 0 && xx < nx && zz >= 0 && zz < nz)
          state.columns[zz * nx + xx][row].forEach(add);
      }
      const sorted = [...levels].sort((a, b) => upper ? a - b : b - a);
      let trimmed = start;
      for (let j = 1; j < sorted.length; j++) {
        const y = (sorted[j - 1] + sorted[j]) / 2;
        const narrow = [[1, 0], [0, 1]].some(([dx, dz]) => {
          let cells = 1;
          for (const sign of [-1, 1]) {
            for (let k = 1; k <= radius; k++) {
              if (!solid(x + sign * dx * k, z + sign * dz * k, y)) break;
              cells++;
            }
          }
          return cells * h < thickness - state.epsilon;
        });
        if (!narrow) break;
        trimmed = sorted[j];
      }
      if (trimmed !== start) changes.push({ cell, index, cap: trimmed });
    }
    for (const change of changes) {
      const column = state.columns[change.cell], interval = column[row][change.index];
      const volume = Math.abs(change.cap - (upper ? interval.min : interval.max)) * area;
      column[row][change.index] = upper ? { ...interval, min: change.cap } : { ...interval, max: change.cap };
      if (upper) {
        state.removedUpperVolume += volume;
        state.chippedUpperVolume = (state.chippedUpperVolume ?? 0) + volume;
        if (change.cap === interval.max) column.upperStart++;
      } else {
        state.removedLowerVolume += volume;
        state.chippedLowerVolume = (state.chippedLowerVolume ?? 0) + volume;
        if (change.cap === interval.min) column.lowerEnd--;
      }
    }
  }
}
