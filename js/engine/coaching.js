// So sánh động tác người học với video mẫu (DTW theo bộ phận) và sinh nhận xét tiếng Việt.
// Chép từ backend/coaching/ (features.py, dtw.py, compare.py, feedback.py).
import { NECK, SHO_L, SHO_R, V, smoothSequence } from './fusion.js';

// Python round(): làm tròn nửa về số chẵn
export function pyRound(x) {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const clip01 = (x) => Math.min(1, Math.max(0, x));
const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

// ------------------------------------------------------------------
// features.py
// ------------------------------------------------------------------
export const HANDS = {
  right: { label: 'phải', base: 0, elbow: 27 },
  left: { label: 'trái', base: 11, elbow: 26 },
};
const SIDES = ['right', 'left'];
export const FINGERS = [
  ['ngón cái', [1, 2]],
  ['ngón trỏ', [3, 4]],
  ['ngón giữa', [5, 6]],
  ['ngón áp út', [7, 8]],
  ['ngón út', [9, 10]],
];
const MIDDLE_BASE = 5;
const DEFAULT_HAND_SIZE = 0.3;

export const PARTS = [
  { key: 'right_shape', side: 'right', kind: 'shape', name: 'Hình dạng bàn tay phải' },
  { key: 'right_motion', side: 'right', kind: 'motion', name: 'Vị trí, chuyển động tay phải' },
  { key: 'left_shape', side: 'left', kind: 'shape', name: 'Hình dạng bàn tay trái' },
  { key: 'left_motion', side: 'left', kind: 'motion', name: 'Vị trí, chuyển động tay trái' },
];

const MIRROR_INDEX = [
  ...Array.from({ length: 11 }, (_, i) => 11 + i),
  ...Array.from({ length: 11 }, (_, i) => i),
  23, 22, 24, 25, 27, 26,
];

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

// Sequence: { T, xy: Float64Array(T*28*2), visible: {side: bool[]}, parts: {key: {P, data}}, handSize }
const XY = (t, v) => (t * V + v) * 2;

function buildParts(xy, T, visible) {
  const parts = {};
  const sizes = {};
  for (const side of SIDES) {
    const { base: b, elbow } = HANDS[side];
    const span = new Float64Array(T);
    const seen = [];
    for (let t = 0; t < T; t++) {
      const dx = xy[XY(t, b + MIDDLE_BASE)] - xy[XY(t, b)];
      const dy = xy[XY(t, b + MIDDLE_BASE) + 1] - xy[XY(t, b) + 1];
      span[t] = Math.sqrt(dx * dx + dy * dy);
      if (visible[side][t] && span[t] > 1e-4) seen.push(span[t]);
    }
    let size = seen.length ? median(seen) : DEFAULT_HAND_SIZE;
    size = Math.max(size, 0.05);
    sizes[side] = size;

    const shape = new Float64Array(T * 10 * 2);
    const motion = new Float64Array(T * 2 * 2);
    for (let t = 0; t < T; t++) {
      const wx = xy[XY(t, b)];
      const wy = xy[XY(t, b) + 1];
      for (let p = 0; p < 10; p++) {
        shape[(t * 10 + p) * 2] = (xy[XY(t, b + 1 + p)] - wx) / size;
        shape[(t * 10 + p) * 2 + 1] = (xy[XY(t, b + 1 + p) + 1] - wy) / size;
      }
      motion[t * 4] = wx;
      motion[t * 4 + 1] = wy;
      motion[t * 4 + 2] = xy[XY(t, elbow)];
      motion[t * 4 + 3] = xy[XY(t, elbow) + 1];
    }
    parts[`${side}_shape`] = { P: 10, data: shape };
    parts[`${side}_motion`] = { P: 2, data: motion };
  }
  return { parts, sizes };
}

/** Keypoint thô (Float64Array T*28*3) -> Sequence. */
export function prepare(raw, T) {
  const sm = smoothSequence(raw, T);
  const xy = new Float64Array(T * V * 2);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) {
      xy[XY(t, v)] = sm[(t * V + v) * 3];
      xy[XY(t, v) + 1] = sm[(t * V + v) * 3 + 1];
    }
    const nx = (xy[XY(t, SHO_L)] + xy[XY(t, SHO_R)]) / 2;
    const ny = (xy[XY(t, SHO_L) + 1] + xy[XY(t, SHO_R) + 1]) / 2;
    xy[XY(t, NECK)] = nx;
    xy[XY(t, NECK) + 1] = ny;
    for (let v = 0; v < V; v++) {
      xy[XY(t, v)] -= nx;
      xy[XY(t, v) + 1] -= ny;
    }
  }
  const visible = {};
  for (const side of SIDES) {
    const w = HANDS[side].base;
    visible[side] = Array.from({ length: T }, (_, t) => raw[(t * V + w) * 3] !== 0 || raw[(t * V + w) * 3 + 1] !== 0);
  }
  const { parts, sizes } = buildParts(xy, T, visible);
  return { T, xy, visible, parts, handSize: sizes };
}

/** Soi gương: đổi tay trái <-> phải và lật trục x. */
export function mirror(seq) {
  const { T } = seq;
  const xy = new Float64Array(T * V * 2);
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) {
      const s = MIRROR_INDEX[v];
      xy[XY(t, v)] = -seq.xy[XY(t, s)];
      xy[XY(t, v) + 1] = seq.xy[XY(t, s) + 1];
    }
  }
  const visible = { right: seq.visible.left, left: seq.visible.right };
  const { parts, sizes } = buildParts(xy, T, visible);
  return { T, xy, visible, parts, handSize: sizes };
}

function variance(values) {
  const m = mean(values);
  return mean(values.map((x) => (x - m) ** 2));
}

export function handActivity(seq, side) {
  const { base, elbow } = HANDS[side];
  let lift = -Infinity;
  const wx = [];
  const wy = [];
  for (let t = 0; t < seq.T; t++) {
    lift = Math.max(lift, seq.xy[XY(t, elbow) + 1] - seq.xy[XY(t, base) + 1]);
    wx.push(seq.xy[XY(t, base)]);
    wy.push(seq.xy[XY(t, base) + 1]);
  }
  const travel = Math.sqrt(variance(wx) + variance(wy));
  const visible = seq.visible[side].filter(Boolean).length / seq.T;
  return { lift, travel, visible };
}

/** a, b: part {P, data} của 2 chuỗi -> ma trận (Ta, Tb) khoảng cách Euclid trung bình trên P điểm. */
function pairwiseDistance(a, Ta, b, Tb) {
  const P = a.P;
  const out = new Float64Array(Ta * Tb);
  for (let i = 0; i < Ta; i++) {
    for (let j = 0; j < Tb; j++) {
      let s = 0;
      for (let p = 0; p < P; p++) {
        const dx = a.data[(i * P + p) * 2] - b.data[(j * P + p) * 2];
        const dy = a.data[(i * P + p) * 2 + 1] - b.data[(j * P + p) * 2 + 1];
        s += Math.sqrt(dx * dx + dy * dy);
      }
      out[i * Tb + j] = s / P;
    }
  }
  return out;
}

// ------------------------------------------------------------------
// dtw.py: DTW với hai đầu nới lỏng. Bước đi (1,1), (1,0), (0,1).
// ------------------------------------------------------------------
export function dtw(cost, n, m, margin = 0.15) {
  const startA = Math.max(1, pyRound(n * margin));
  const startB = Math.max(1, pyRound(m * margin));
  const acc = new Float64Array(n * m).fill(Infinity);
  const len = new Int32Array(n * m);
  const move = new Int8Array(n * m).fill(-1);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const k = i * m + j;
      const c = cost[k];
      if ((i === 0 && j < startB) || (j === 0 && i < startA)) {
        acc[k] = c; len[k] = 1; move[k] = -1;
        continue;
      }
      let best = Infinity;
      let bestLen = 0;
      let bestMove = -1;
      if (i > 0 && j > 0 && acc[k - m - 1] < best) { best = acc[k - m - 1]; bestLen = len[k - m - 1]; bestMove = 0; }
      if (i > 0 && acc[k - m] < best) { best = acc[k - m]; bestLen = len[k - m]; bestMove = 1; }
      if (j > 0 && acc[k - 1] < best) { best = acc[k - 1]; bestLen = len[k - 1]; bestMove = 2; }
      acc[k] = best + c;
      len[k] = bestLen + 1;
      move[k] = bestMove;
    }
  }

  const endA = n - Math.max(1, pyRound(n * margin));
  const endB = m - Math.max(1, pyRound(m * margin));
  const candidates = [];
  for (let j = Math.max(endB, 0); j < m; j++) candidates.push([n - 1, j]);
  for (let i = Math.max(endA, 0); i < n; i++) candidates.push([i, m - 1]);
  let end = candidates[0];
  let endVal = Infinity;
  for (const [i, j] of candidates) {
    const val = acc[i * m + j] / Math.max(len[i * m + j], 1);
    if (val < endVal) { endVal = val; end = [i, j]; }
  }

  const path = [];
  let [i, j] = end;
  for (;;) {
    path.push([i, j]);
    const mv = move[i * m + j];
    if (mv === -1) break;
    if (mv === 0) { i -= 1; j -= 1; } else if (mv === 1) i -= 1; else j -= 1;
  }
  path.reverse();
  const e = end[0] * m + end[1];
  return { path, meanCost: acc[e] / Math.max(len[e], 1) };
}

// ------------------------------------------------------------------
// compare.py
// ------------------------------------------------------------------
export const PHASES = ['đầu', 'giữa', 'cuối'];
const DTW_MARGIN = 0.15;
const INACTIVE_WEIGHT = 0.15;

export const DEFAULT_CALIBRATION = {
  right_shape: { scale: 0.43, good: 0.59, good_strict: 0.43, bad: 0.76 },
  right_motion: { scale: 0.18, good: 0.24, good_strict: 0.18, bad: 0.31 },
  left_shape: { scale: 0.49, good: 0.73, good_strict: 0.49, bad: 1.00 },
  left_motion: { scale: 0.18, good: 0.24, good_strict: 0.18, bad: 0.38 },
};

export function mergeCalibration(data) {
  const calib = {};
  for (const [k, v] of Object.entries(DEFAULT_CALIBRATION)) calib[k] = { ...v };
  for (const [key, values] of Object.entries(data?.parts || {})) {
    if (!calib[key]) continue;
    for (const f of ['scale', 'good', 'good_strict', 'bad']) if (f in values) calib[key][f] = Number(values[f]);
  }
  return calib;
}

const toScore = (d, good, bad) => clip01((bad - d) / Math.max(bad - good, 1e-6)) * 100;

export function handUsage(seq, side) {
  const a = handActivity(seq, side);
  const byLift = clip01((a.lift + 0.6) / 0.4);
  const byTravel = clip01((a.travel - 0.05) / 0.20);
  return [Math.max(byLift, byTravel), a];
}

function partWeights(ref) {
  const usage = {};
  for (const side of SIDES) usage[side] = handUsage(ref, side)[0];
  const weights = {};
  for (const part of PARTS) weights[part.key] = INACTIVE_WEIGHT + (1 - INACTIVE_WEIGHT) * usage[part.side];
  return { weights, usage };
}

function align(user, ref, weights, calib) {
  const n = user.T;
  const m = ref.T;
  const dist = {};
  for (const p of PARTS) dist[p.key] = pairwiseDistance(user.parts[p.key], n, ref.parts[p.key], m);
  let totalW = 0;
  for (const p of PARTS) totalW += weights[p.key];
  const cost = new Float64Array(n * m);
  for (let k = 0; k < n * m; k++) {
    let s = 0;
    for (const p of PARTS) s += (weights[p.key] * dist[p.key][k]) / calib[p.key].scale;
    cost[k] = s / totalW;
  }
  const { path, meanCost } = dtw(cost, n, m, DTW_MARGIN);
  const errors = {};
  for (const p of PARTS) errors[p.key] = path.map(([i, j]) => dist[p.key][i * m + j]);
  return { path, errors, cost: meanCost };
}

function phasesOf(path) {
  const js = path.map((p) => p[1]);
  const lo = Math.min(...js);
  const hi = Math.max(...js);
  return js.map((j) => Math.min(Math.floor(((j - lo) / Math.max(hi - lo, 1)) * 3), 2));
}

/** strict=true khi model không nhận ra ký hiệu: ngưỡng 100 điểm chặt hơn để chỉ ra chỗ sai. */
export function compareSequences(userRaw, userT, refRaw, refT, calib, strict = false) {
  const goodKey = strict ? 'good_strict' : 'good';
  let user = prepare(userRaw, userT);
  const ref = prepare(refRaw, refT);

  const { weights, usage: usageRef } = partWeights(ref);
  let { path, errors, cost } = align(user, ref, weights, calib);

  const userMirror = mirror(user);
  const m = align(userMirror, ref, weights, calib);
  const mirroredBetter = m.cost < 0.7 * cost;
  if (mirroredBetter) {
    user = userMirror;
    ({ path, errors } = m);
  }

  const phase = phasesOf(path);
  const partScores = {};
  const phaseScores = {};
  for (const part of PARTS) {
    const k = part.key;
    const { bad } = calib[k];
    const good = calib[k][goodKey];
    partScores[k] = toScore(mean(errors[k]), good, bad);
    phaseScores[k] = [0, 1, 2].map((p) => {
      const e = errors[k].filter((_, s) => phase[s] === p);
      return e.length ? toScore(mean(e), good, bad) : null;
    });
  }

  const fingerErrors = {};
  const wristOffset = {};
  const amplitudeRatio = {};
  const usageUser = {};
  const visibleUser = {};
  const sh = (seq, side, t, p) => {
    const d = seq.parts[`${side}_shape`].data;
    return [d[(t * 10 + p) * 2], d[(t * 10 + p) * 2 + 1]];
  };
  for (const side of SIDES) {
    const b = HANDS[side].base;
    const perPoint = path.map(([ui, rj]) => Array.from({ length: 10 }, (_, p) => {
      const [ux, uy] = sh(user, side, ui, p);
      const [rx, ry] = sh(ref, side, rj, p);
      return Math.hypot(ux - rx, uy - ry);
    }));
    fingerErrors[side] = {};
    wristOffset[side] = {};
    for (let p = 0; p < 3; p++) {
      const steps = path.map((_, s) => s).filter((s) => phase[s] === p);
      if (!steps.length) continue;
      fingerErrors[side][p] = FINGERS.map(([, [a, t]]) => {
        let s = 0;
        for (const st of steps) s += perPoint[st][a - 1] + perPoint[st][t - 1];
        return s / (steps.length * 2);
      });
      let dx = 0;
      let dy = 0;
      for (const st of steps) {
        const [ui, rj] = path[st];
        dx += user.xy[XY(ui, b)] - ref.xy[XY(rj, b)];
        dy += user.xy[XY(ui, b) + 1] - ref.xy[XY(rj, b) + 1];
      }
      wristOffset[side][p] = [dx / steps.length, dy / steps.length];
    }
    const travel = (seq, idx) => Math.sqrt(
      variance(idx.map((t) => seq.xy[XY(t, b)])) + variance(idx.map((t) => seq.xy[XY(t, b) + 1])));
    const tu = travel(user, path.map((p) => p[0]));
    const tr = travel(ref, path.map((p) => p[1]));
    amplitudeRatio[side] = tu / Math.max(tr, 1e-3);
    usageUser[side] = handUsage(user, side)[0];
    visibleUser[side] = user.visible[side].filter(Boolean).length / user.T;
  }

  let totalW = 0;
  let sim = 0;
  for (const part of PARTS) {
    totalW += weights[part.key];
    sim += weights[part.key] * partScores[part.key];
  }

  return {
    path, phase, errors, partScores, phaseScores, weights,
    usageRef, usageUser, visibleUser, fingerErrors, wristOffset, amplitudeRatio,
    similarity: sim / totalW,
    mirroredBetter,
  };
}

// ------------------------------------------------------------------
// feedback.py
// ------------------------------------------------------------------
const GOOD_SCORE = 75;
const WEAK_SCORE = 50;
const ACTIVE_WEIGHT = 0.5;
const OFFSET_THRESHOLD = 0.2;
const MODEL_WEIGHT = 0.7;

export function verdictOf(rank) {
  if (rank === 1) return 'pass';
  if (rank <= 5) return 'close';
  return 'fail';
}

export function lessonScore(prob, similarity) {
  return pyRound(100 * (MODEL_WEIGHT * prob + (1 - MODEL_WEIGHT) * similarity / 100));
}

export function headline(verdict, target, rank, top1) {
  if (verdict === 'pass') return `Đúng ký hiệu. Mô hình nhận ra “${target}”.`;
  if (verdict === 'close') return `Gần đúng. “${target}” đứng thứ ${rank}, mô hình đang nghiêng về “${top1}”.`;
  return `Chưa đúng. Mô hình nhận thành “${top1}”.`;
}

function join(items) {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} và ${items[items.length - 1]}`;
}

export function physicalSide(comp, side) {
  if (comp.mirroredBetter) return side === 'right' ? 'left' : 'right';
  return side;
}

const handLabel = (comp, side) => HANDS[physicalSide(comp, side)].label;

export function partName(comp, part) {
  const key = `${physicalSide(comp, part.side)}_${part.kind}`;
  return PARTS.find((p) => p.key === key).name;
}

function worstPhase(scores) {
  let best = null;
  scores.forEach((s, p) => {
    if (s === null) return;
    if (best === null || s < best[0]) best = [s, p];
  });
  return best ? best[1] : 1;
}

function shapeMessage(comp, side, phase) {
  const label = handLabel(comp, side);
  const errs = comp.fingerErrors[side][phase];
  let text = `Hình dạng bàn tay ${label} chưa giống mẫu ở đoạn ${PHASES[phase]} động tác`;
  if (errs) {
    const asc = errs.map((e, i) => [e, i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map((x) => x[1]);
    const order = asc.reverse();
    const top = [FINGERS[order[0]][0]];
    if (errs[order[1]] >= 0.75 * errs[order[0]]) top.push(FINGERS[order[1]][0]);
    text += `, lệch nhiều nhất ở ${join(top)}`;
  }
  return `${text}. Xem chậm video mẫu và để ý cách co, duỗi các ngón này.`;
}

function motionMessage(comp, side, phase) {
  const label = handLabel(comp, side);
  let [dx, dy] = comp.wristOffset[side][phase] || [0, 0];
  if (comp.mirroredBetter) dx = -dx;
  const hints = [];
  if (dy > OFFSET_THRESHOLD) hints.push('đưa tay lên cao hơn');
  else if (dy < -OFFSET_THRESHOLD) hints.push('hạ tay thấp xuống');
  if (dx > OFFSET_THRESHOLD) hints.push('đưa tay về phía bên phải của bạn');
  else if (dx < -OFFSET_THRESHOLD) hints.push('đưa tay về phía bên trái của bạn');
  if (hints.length) return `Vị trí tay ${label} lệch ở đoạn ${PHASES[phase]} động tác: ${join(hints)}.`;

  const ratio = comp.amplitudeRatio[side];
  if (ratio < 0.65) return `Chuyển động tay ${label} còn nhỏ so với mẫu. Hãy làm rộng và dứt khoát hơn.`;
  if (ratio > 1.6) return `Chuyển động tay ${label} rộng hơn mẫu. Hãy thu gọn lại.`;
  return `Đường đi của tay ${label} chưa giống mẫu ở đoạn ${PHASES[phase]} động tác. `
    + 'Xem lại hướng di chuyển trong video mẫu.';
}

export function buildFeedback(comp, verdict) {
  const items = [];
  const lenient = verdict === 'pass';

  for (const side of SIDES) {
    if (comp.usageRef[side] >= ACTIVE_WEIGHT && comp.visibleUser[side] < 0.6) {
      const pct = pyRound(comp.visibleUser[side] * 100);
      items.push({
        level: 'warn', part: `${side}_shape`,
        text: `Camera chỉ thấy tay ${handLabel(comp, side)} trong ${pct}% thời gian. `
          + 'Giữ tay trong khung hình, đủ sáng và đừng di chuyển quá nhanh.',
      });
    }
  }

  if (comp.mirroredBetter) {
    const main = comp.usageRef.right >= comp.usageRef.left ? 'right' : 'left';
    const other = main === 'right' ? 'left' : 'right';
    items.push({
      level: 'tip', part: null,
      text: `Bạn đang dùng tay ${HANDS[other].label} làm tay chính, còn người mẫu dùng tay `
        + `${HANDS[main].label}. Người thuận tay ${HANDS[other].label} có thể làm như vậy; `
        + 'điểm bên dưới đã đổi tay để so sánh. Lưu ý video mẫu không lật gương: '
        + 'tay phải của người mẫu nằm ở bên trái màn hình.',
    });
  }

  const active = PARTS.filter((p) => comp.weights[p.key] >= ACTIVE_WEIGHT);
  const sorted = active.map((p, i) => [p, i])
    .sort((a, b) => comp.partScores[a[0].key] - comp.partScores[b[0].key] || a[1] - b[1])
    .map((x) => x[0]);
  for (const part of sorted) {
    const score = comp.partScores[part.key];
    if (score >= GOOD_SCORE) continue;
    const phase = worstPhase(comp.phaseScores[part.key]);
    const text = part.kind === 'shape' ? shapeMessage(comp, part.side, phase) : motionMessage(comp, part.side, phase);
    if (lenient) {
      items.push({ level: 'tip', part: part.key, text: `Để giống mẫu hơn: ${text[0].toLowerCase()}${text.slice(1)}` });
    } else {
      items.push({ level: score < WEAK_SCORE ? 'warn' : 'tip', part: part.key, text });
    }
  }

  for (const side of SIDES) {
    const other = side === 'right' ? 'left' : 'right';
    if (comp.usageRef[side] < 0.3 && comp.usageUser[side] > 0.7 && comp.partScores[`${side}_motion`] < 60) {
      items.push({
        level: 'tip', part: `${side}_motion`,
        text: `Ký hiệu này chỉ dùng tay ${handLabel(comp, other)}. Tay ${handLabel(comp, side)} nên để yên, buông tự nhiên.`,
      });
    }
  }

  const good = active.filter((p) => comp.partScores[p.key] >= GOOD_SCORE).map((p) => partName(comp, p).toLowerCase());
  if (lenient && !items.length) {
    items.push({ level: 'good', part: null, text: 'Ký hiệu rõ ràng và gần giống mẫu. Giữ nhịp như vậy.' });
  } else if (lenient) {
    items.unshift({ level: 'good', part: null,
      text: 'Mô hình đã nhận đúng ký hiệu. Các gợi ý dưới đây chỉ giúp động tác giống người mẫu hơn.' });
  } else if (good.length && !items.length) {
    items.push({ level: 'good', part: null, text: 'Các bộ phận chính đều gần giống mẫu. Giữ nhịp như vậy.' });
  } else if (good.length) {
    items.push({ level: 'good', part: null, text: `Làm tốt: ${join(good)}.` });
  }
  return items;
}
