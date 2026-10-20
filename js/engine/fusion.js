// Tiền xử lý keypoint, chép từ backend/pipeline/fusion.py và feeder.py (giống hệt lúc train).
//   raw (T, 28, 3) -> interpolate_missing -> normalize_pose -> Kalman RTS -> fit_length -> (3, 64, 28)
//   feeder: điểm 24 = cổ -> trừ cổ -> valid_crop_resize -> (1, 3, 64, 28, 1)
// Chuỗi keypoint lưu phẳng: Float64Array độ dài T*28*3, chỉ số (t*28 + v)*3 + c.

export const V = 28;
export const C = 3;
export const SHO_L = 22;
export const SHO_R = 23;
export const NECK = 24;
export const HIP_C = 25;

/** np.interp: ngoài khoảng thì giữ giá trị ở biên. xp tăng dần. */
export function interp(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + t * (fp[hi] - fp[lo]);
}

/** Điểm thiếu (x = y = 0) -> nội suy tuyến tính theo thời gian; thiếu ở biên giữ giá trị gần nhất. */
export function interpolateMissing(raw, T) {
  const k = Float64Array.from(raw);
  for (let v = 0; v < V; v++) {
    const okT = [];
    const missT = [];
    for (let t = 0; t < T; t++) {
      const i = (t * V + v) * C;
      if (k[i] !== 0 || k[i + 1] !== 0) okT.push(t);
      else missT.push(t);
    }
    if (missT.length === 0 || okT.length === 0) continue;
    for (let c = 0; c < C; c++) {
      const fp = okT.map((t) => k[(t * V + v) * C + c]);
      for (const t of missT) k[(t * V + v) * C + c] = interp(t, okT, fp);
    }
  }
  return k;
}

/** Dời tâm về (2 vai + 2 x tâm hông) / 4 từng frame, chia cho chiều rộng vai trung bình. */
export function normalizePose(pose, T) {
  const d = Float64Array.from(pose);
  const empty = new Array(V).fill(true);
  for (let v = 0; v < V; v++) {
    for (let t = 0; t < T && empty[v]; t++) {
      const i = (t * V + v) * C;
      if (d[i] !== 0 || d[i + 1] !== 0) empty[v] = false;
    }
  }
  let scaleSum = 0;
  const cx = new Float64Array(T);
  const cy = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const l = (t * V + SHO_L) * C;
    const r = (t * V + SHO_R) * C;
    const h = (t * V + HIP_C) * C;
    cx[t] = (d[l] + d[r] + 2 * d[h]) / 4;
    cy[t] = (d[l + 1] + d[r + 1] + 2 * d[h + 1]) / 4;
    const dx = d[l] - d[r];
    const dy = d[l + 1] - d[r + 1];
    scaleSum += Math.sqrt(dx * dx + dy * dy);
  }
  let scale = scaleSum / T;
  if (scale < 1e-6) scale = 1;
  for (let t = 0; t < T; t++) {
    for (let v = 0; v < V; v++) {
      const i = (t * V + v) * C;
      if (empty[v]) {
        d[i] = 0; d[i + 1] = 0; d[i + 2] = 0;
        continue;
      }
      d[i] = (d[i] - cx[t]) / scale;
      d[i + 1] = (d[i + 1] - cy[t]) / scale;
      d[i + 2] = d[i + 2] / scale;
    }
  }
  return d;
}

/** Kalman 1D random-walk cho từng khớp x kênh, lọc xuôi + làm mượt ngược (RTS). */
export function applyKalmanFilter(x, T, Q = 1e-3, R = 1e-3) {
  const D = V * C;
  const filt = new Float64Array(T * D);
  const Pf = new Float64Array(T);
  const Pp = new Float64Array(T);
  let P = R;
  for (let j = 0; j < D; j++) filt[j] = x[j];
  Pf[0] = P;
  Pp[0] = P;
  for (let t = 1; t < T; t++) {
    const prior = P + Q;
    const K = prior / (prior + R);
    for (let j = 0; j < D; j++) {
      const s = filt[(t - 1) * D + j];
      filt[t * D + j] = s + K * (x[t * D + j] - s);
    }
    P = (1 - K) * prior;
    Pf[t] = P;
    Pp[t] = prior;
  }
  const sm = Float64Array.from(filt);
  for (let t = T - 2; t >= 0; t--) {
    const G = Pf[t] / Pp[t + 1];
    for (let j = 0; j < D; j++) {
      sm[t * D + j] = filt[t * D + j] + G * (sm[(t + 1) * D + j] - filt[t * D + j]);
    }
  }
  return sm;
}

/** Dài hơn maxFrames: lấy mẫu đều (nội suy tuyến tính). Ngắn hơn: pad 0 ở cuối. */
export function fitLength(k, T, maxFrames = 64) {
  const D = V * C;
  const out = new Float64Array(maxFrames * D);
  if (T > maxFrames) {
    const src = Array.from({ length: T }, (_, i) => i);
    const col = new Float64Array(T);
    for (let j = 0; j < D; j++) {
      for (let t = 0; t < T; t++) col[t] = k[t * D + j];
      for (let i = 0; i < maxFrames; i++) {
        const dst = maxFrames === 1 ? 0 : (i * (T - 1)) / (maxFrames - 1);
        out[i * D + j] = interp(dst, src, col);
      }
    }
  } else {
    out.set(k.subarray(0, T * D));
  }
  return out;
}

/** (T, 28, 3) thô -> đã nội suy, chuẩn hóa, làm mượt; chưa đổi độ dài. */
export function smoothSequence(raw, T) {
  return applyKalmanFilter(normalizePose(interpolateMissing(raw, T), T), T);
}

/** (T, 28, 3) thô -> (3, 64, 28) phẳng, chỉ số (c*64 + t)*28 + v. */
export function fuse(raw, T, maxFrames = 64) {
  const k = fitLength(smoothSequence(raw, T), T, maxFrames);
  const out = new Float64Array(C * maxFrames * V);
  for (let t = 0; t < maxFrames; t++) {
    for (let v = 0; v < V; v++) {
      for (let c = 0; c < C; c++) out[(c * maxFrames + t) * V + v] = k[(t * V + v) * C + c];
    }
  }
  return out;
}

/** (3, 64, 28) từ fuse() -> Float32Array (1, 3, 64, 28, 1), đúng như Feeder ở pha test. */
export function toModelInput(fused, window = 64) {
  const T = fused.length / (C * V);
  const d = Float64Array.from(fused);
  const at = (c, t, v) => (c * T + t) * V + v;
  for (let c = 0; c < C; c++) {
    for (let t = 0; t < T; t++) {
      d[at(c, t, NECK)] = (d[at(c, t, SHO_L)] + d[at(c, t, SHO_R)]) / 2;
      const neck = d[at(c, t, NECK)];
      for (let v = 0; v < V; v++) d[at(c, t, v)] -= neck;
    }
  }
  let valid = 0;
  for (let t = 0; t < T; t++) {
    let s = 0;
    for (let c = 0; c < C; c++) {
      let sv = 0;
      for (let v = 0; v < V; v++) sv += d[at(c, t, v)];
      s += sv;
    }
    if (s !== 0) valid++;
  }
  if (valid === 0) valid = T;

  // F.interpolate(bilinear, align_corners=False) theo trục thời gian: valid frame đầu -> window frame
  const out = new Float32Array(C * window * V);
  const scale = valid / window;
  for (let i = 0; i < window; i++) {
    let src = scale * (i + 0.5) - 0.5;
    if (src < 0) src = 0;
    const i0 = Math.floor(src);
    const i1 = i0 < valid - 1 ? i0 + 1 : i0;
    const l1 = src - i0;
    const l0 = 1 - l1;
    for (let c = 0; c < C; c++) {
      for (let v = 0; v < V; v++) {
        out[(c * window + i) * V + v] = l0 * d[at(c, i0, v)] + l1 * d[at(c, i1, v)];
      }
    }
  }
  return out;
}
