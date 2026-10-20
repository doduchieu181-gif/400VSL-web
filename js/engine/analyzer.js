// Bộ phân tích chạy hoàn toàn trong trình duyệt (thay cho backend FastAPI):
//   ảnh / video -> MediaPipe Tasks (Pose FULL + Hand, chế độ VIDEO) -> keypoint (T, 28, 3)
//   -> fusion + feeder -> SDA-GCN (ONNX Runtime Web) -> so với video mẫu -> kết quả
// Kết quả cùng dạng với /api/analyze cũ nên giao diện giữ nguyên.
import { FilesetResolver, HandLandmarker, PoseLandmarker } from '../../vendor/mediapipe/vision_bundle.mjs';
import * as ort from '../../vendor/ort/ort.wasm.min.mjs';
import {
  PARTS, buildFeedback, compareSequences, handUsage, headline, lessonScore, mergeCalibration,
  partName, physicalSide, prepare, pyRound, verdictOf,
} from './coaching.js';
import { V, fuse, toModelInput } from './fusion.js';

const asset = (path) => new URL(`../../${path}`, import.meta.url).href;

export const DEFAULT_FPS = 25;
export const MAX_FRAMES = 500;
const WINDOW = 64;
const TOP_K = 5;
const PROGRESS_EVERY = 4;
const ACTIVE_WEIGHT = 0.5;
const PREVIEW_SIZE = 384;
const PREVIEW_QUALITY = 0.8;
const MAX_DECODE_SIDE = 1080;  // khung lớn hơn thu nhỏ lại trước MediaPipe (video train 1080 x 1080)
const TARGET_HAND_INDICES = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20];
const CONF = 0.5;

// ------------------------------------------------------------------
// Nạp tài nguyên (một lần, trình duyệt tự cache)
// ------------------------------------------------------------------
let enginePromise = null;

async function fetchBytes(path) {
  const res = await fetch(asset(path));
  if (!res.ok) throw new Error(`Không tải được ${path} (${res.status}).`);
  return new Uint8Array(await res.arrayBuffer());
}

async function fetchJSON(path) {
  const res = await fetch(asset(path));
  if (!res.ok) throw new Error(`Không tải được ${path} (${res.status}).`);
  return res.json();
}

async function loadEngine() {
  ort.env.wasm.wasmPaths = asset('vendor/ort/');
  ort.env.wasm.numThreads = 1; // web tĩnh không bật được cross-origin isolation cho đa luồng
  const [fileset, poseModel, handModel, session, calibration] = await Promise.all([
    FilesetResolver.forVisionTasks(asset('vendor/mediapipe/wasm')),
    fetchBytes('model/pose_landmarker_full.task'),
    fetchBytes('model/hand_landmarker.task'),
    fetchBytes('model/sdagcn.onnx').then((bytes) => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })),
    fetchJSON('data/calibration.json').then(mergeCalibration).catch(() => mergeCalibration(null)),
  ]);
  return { fileset, poseModel, handModel, session, calibration };
}

/** Nạp mô hình (gọi sớm để lượt phân tích đầu tiên không phải chờ). */
export function loadAnalyzer() {
  if (!enginePromise) {
    enginePromise = loadEngine();
    enginePromise.catch(() => { enginePromise = null; });
  }
  return enginePromise;
}

// ------------------------------------------------------------------
// MediaPipe: tạo landmarker MỚI cho mỗi lượt (chế độ VIDEO cần timestamp tăng dần), giống lúc train
// ------------------------------------------------------------------
async function createLandmarkers(engine) {
  const base = (buffer) => ({ modelAssetBuffer: buffer.slice(), delegate: 'CPU' });
  const hands = await HandLandmarker.createFromOptions(engine.fileset, {
    baseOptions: base(engine.handModel),
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: CONF,
    minHandPresenceConfidence: CONF,
    minTrackingConfidence: CONF,
  });
  const pose = await PoseLandmarker.createFromOptions(engine.fileset, {
    baseOptions: base(engine.poseModel),
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: CONF,
    minPosePresenceConfidence: CONF,
    minTrackingConfidence: CONF,
  });
  return { hands, pose };
}

function frameTimestamp(i, fps) {
  return pyRound((i * 1000) / (fps > 0 ? fps : DEFAULT_FPS));
}

/** Một ảnh -> 28 x 3 keypoint (ghi vào out tại vị trí frame t). Điểm không phát hiện = 0. */
function extractFrame(image, lm, ts, out, t) {
  const put = (v, x, y, z) => {
    const i = (t * V + v) * 3;
    out[i] = x; out[i + 1] = y; out[i + 2] = z;
  };
  const pose = lm.pose.detectForVideo(image, ts);
  if (pose.landmarks && pose.landmarks.length) {
    const p = pose.landmarks[0];
    put(22, p[11].x, p[11].y, p[11].z);
    put(23, p[12].x, p[12].y, p[12].z);
    put(24, p[0].x, p[0].y, p[0].z);
    put(25, (p[23].x + p[24].x) / 2, (p[23].y + p[24].y) / 2, (p[23].z + p[24].z) / 2);
    put(26, p[13].x, p[13].y, p[13].z);
    put(27, p[14].x, p[14].y, p[14].z);
  }
  const hands = lm.hands.detectForVideo(image, ts);
  const handed = hands.handednesses || hands.handedness || [];
  (hands.landmarks || []).forEach((h, k) => {
    const base = handed[k] && handed[k][0] && handed[k][0].categoryName === 'Right' ? 0 : 11;
    TARGET_HAND_INDICES.forEach((idx, i) => {
      if (idx < h.length) put(base + i, h[idx].x, h[idx].y, h[idx].z);
    });
  });
}

// ------------------------------------------------------------------
// Đọc đầu vào: chuỗi ảnh JPEG (webcam) hoặc file video (lấy mẫu 25 fps)
// ------------------------------------------------------------------
function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

function fitSize(w, h, maxSide) {
  const s = Math.min(1, maxSide / Math.max(w, h));
  return [Math.max(1, Math.round(w * s)), Math.max(1, Math.round(h * s))];
}

function previewOf(source, w, h) {
  const [pw, ph] = fitSize(w, h, PREVIEW_SIZE);
  const c = makeCanvas(pw, ph);
  c.getContext('2d').drawImage(source, 0, 0, pw, ph);
  return new Promise((resolve) => c.toBlob((b) => resolve(b ? URL.createObjectURL(b) : null), 'image/jpeg', PREVIEW_QUALITY));
}

function once(target, event) {
  return new Promise((resolve, reject) => {
    const ok = () => { cleanup(); resolve(); };
    const fail = () => { cleanup(); reject(new Error('Không đọc được file video. Hãy thử file .mp4 hoặc .webm khác.')); };
    const cleanup = () => {
      target.removeEventListener(event, ok);
      target.removeEventListener('error', fail);
    };
    target.addEventListener(event, ok);
    target.addEventListener('error', fail);
  });
}

/** Sinh lần lượt các frame (dạng ảnh vẽ được) để không giữ cả video trong bộ nhớ. */
async function openFrames(input) {
  if (input.kind === 'frames') {
    if (!input.blobs.length) throw new Error('Không nhận được frame nào từ webcam.');
    if (input.blobs.length > MAX_FRAMES) throw new Error(`Quá nhiều frame (${input.blobs.length} > ${MAX_FRAMES}).`);
    const first = await createImageBitmap(input.blobs[0]);
    const size = [first.width, first.height];
    first.close();
    return {
      count: input.blobs.length,
      fps: input.fps || DEFAULT_FPS,
      size,
      async get(i) {
        const bmp = await createImageBitmap(input.blobs[i]);
        if (bmp.width !== size[0] || bmp.height !== size[1]) {
          bmp.close();
          throw new Error('Các frame webcam có kích thước khác nhau.');
        }
        return { image: bmp, release: () => bmp.close() };
      },
      close() {},
    };
  }

  const url = URL.createObjectURL(input.file);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = url;
  try {
    await once(video, 'loadedmetadata');
    if (video.readyState < 2) await once(video, 'loadeddata');
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0 || !video.videoWidth) {
    URL.revokeObjectURL(url);
    throw new Error('Video không có frame nào đọc được.');
  }
  const count = Math.max(1, Math.floor(duration * DEFAULT_FPS));
  if (count > MAX_FRAMES) {
    URL.revokeObjectURL(url);
    throw new Error(`Video dài quá ${MAX_FRAMES} frame (khoảng ${Math.round(MAX_FRAMES / DEFAULT_FPS)} giây). `
      + 'Mỗi video chỉ nên chứa một ký hiệu, hãy cắt ngắn lại.');
  }
  const [w, h] = fitSize(video.videoWidth, video.videoHeight, MAX_DECODE_SIDE);
  const canvas = makeCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  return {
    count,
    fps: DEFAULT_FPS,
    size: [w, h],
    async get(i) {
      const t = Math.min((i + 0.5) / DEFAULT_FPS, Math.max(0, duration - 1e-3));
      const seeked = once(video, 'seeked');
      video.currentTime = t;
      await seeked;
      ctx.drawImage(video, 0, 0, w, h);
      return { image: canvas, release: () => {} };
    },
    close() {
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
    },
  };
}

// ------------------------------------------------------------------
// Chấm điểm
// ------------------------------------------------------------------
function softmax(logits) {
  let max = -Infinity;
  for (const z of logits) max = Math.max(max, z);
  const e = Array.from(logits, (z) => Math.exp(z - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

export async function predict(engine, raw, T) {
  const input = toModelInput(fuse(raw, T, WINDOW), WINDOW);
  const tensor = new ort.Tensor('float32', input, [1, 3, WINDOW, V, 1]);
  const out = await engine.session.run({ [engine.session.inputNames[0]]: tensor });
  return softmax(out[engine.session.outputNames[0]].data);
}

function assessQuality(raw, T) {
  const seen = (v) => {
    let n = 0;
    for (let t = 0; t < T; t++) {
      const i = (t * V + v) * 3;
      if (raw[i] !== 0 || raw[i + 1] !== 0) n++;
    }
    return n / T;
  };
  const pose = seen(22);
  const right = seen(0);
  const left = seen(11);
  const warnings = [];
  if (T < 20) warnings.push(`Đoạn video quá ngắn (${T} frame). Một ký hiệu thường dài khoảng 2 đến 3 giây.`);
  if (pose < 0.7) {
    warnings.push(`Chỉ thấy vai và thân người trong ${pyRound(pose * 100)}% số frame. `
      + 'Hãy lùi ra để camera thấy từ đầu tới hông.');
  }
  if (Math.max(right, left) < 0.3) {
    warnings.push('Gần như không thấy bàn tay nào. Kiểm tra ánh sáng và giữ tay trong khung hình.');
  } else if (T >= 10 && pose >= 0.7) {
    const seq = prepare(raw, T);
    if (Math.max(handUsage(seq, 'right')[0], handUsage(seq, 'left')[0]) < 0.3) {
      warnings.push('Gần như không thấy tay giơ lên hay di chuyển, kết quả dưới đây có thể không có nghĩa. '
        + 'Hãy bắt đầu làm ký hiệu ngay khi đếm ngược kết thúc.');
    }
  }
  return { frames: T, pose, hands: { right, left }, warnings };
}

function rankOrder(probs) {
  return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]);
}

function practiceResult(engine, catalog, raw, T, probs, target, ref) {
  const gloss = catalog.gloss[target];
  const order = rankOrder(probs);
  const rank = order.indexOf(target - 1) + 1;
  const verdict = verdictOf(rank);
  const top1 = catalog.gloss[order[0] + 1];
  const comp = compareSequences(raw, T, ref.data, ref.T, engine.calibration, verdict !== 'pass');
  const parts = PARTS.map((part) => {
    const weight = comp.weights[part.key];
    return {
      key: part.key,
      name: partName(comp, part),
      side: physicalSide(comp, part.side),
      kind: part.kind,
      score: pyRound(comp.partScores[part.key]),
      phases: comp.phaseScores[part.key].map((s) => (s === null ? null : pyRound(s))),
      weight: Math.round(weight * 100) / 100,
      active: weight >= ACTIVE_WEIGHT,
    };
  });
  return {
    target: { label: target, gloss },
    verdict,
    rank,
    prob: probs[target - 1],
    headline: headline(verdict, gloss, rank, top1),
    score: lessonScore(probs[target - 1], comp.similarity),
    similarity: pyRound(comp.similarity),
    parts,
    feedback: buildFeedback(comp, verdict),
    mirrored: comp.mirroredBetter,
  };
}

/**
 * input: { kind: 'frames', blobs, fps } | { kind: 'video', file }
 * catalog: { gloss: {label: tên} }, getReference(label) -> Promise<{data, T}>
 * onProgress nhận các bước giống backend cũ: load, decode, extract, model, compare, preview.
 */
export async function analyzeLocally(input, { target = null, catalog, getReference, onProgress = () => {}, signal } = {}) {
  const check = () => { if (signal?.aborted) throw new DOMException('Đã huỷ', 'AbortError'); };
  const timing = {};
  let t0 = performance.now();
  onProgress({ type: 'progress', stage: 'load', current: 0, total: 1 });
  const engine = await loadAnalyzer();
  check();

  onProgress({ type: 'progress', stage: 'decode', current: 0, total: 1 });
  const src = await openFrames(input);
  const lm = await createLandmarkers(engine);
  const T = src.count;
  const raw = new Float64Array(T * V * 3);
  const preview = [];
  const wantPreview = input.kind !== 'frames';
  try {
    t0 = performance.now();
    onProgress({ type: 'progress', stage: 'extract', current: 0, total: T });
    for (let i = 0; i < T; i++) {
      check();
      const frame = await src.get(i);
      try {
        extractFrame(frame.image, lm, frameTimestamp(i, src.fps), raw, i);
        if (wantPreview) preview.push(await previewOf(frame.image, src.size[0], src.size[1]));
      } finally {
        frame.release();
      }
      if ((i + 1) % PROGRESS_EVERY === 0 || i + 1 === T) {
        onProgress({ type: 'progress', stage: 'extract', current: i + 1, total: T });
        await new Promise((r) => setTimeout(r, 0)); // nhường cho giao diện vẽ lại thanh tiến độ
      }
    }
    timing.extract_ms = Math.round(performance.now() - t0);
  } finally {
    lm.pose.close();
    lm.hands.close();
    src.close();
  }

  check();
  onProgress({ type: 'progress', stage: 'model', current: 0, total: 1 });
  t0 = performance.now();
  const probs = await predict(engine, raw, T);
  timing.model_ms = Math.round(performance.now() - t0);

  let practice = null;
  if (target !== null && target !== undefined) {
    onProgress({ type: 'progress', stage: 'compare', current: 0, total: 1 });
    t0 = performance.now();
    const ref = await getReference(target);
    practice = practiceResult(engine, catalog, raw, T, probs, target, ref);
    timing.compare_ms = Math.round(performance.now() - t0);
  }

  const top5 = rankOrder(probs).slice(0, TOP_K).map((i) => ({ label: i + 1, gloss: catalog.gloss[i + 1], prob: probs[i] }));
  return {
    frames: T,
    fps: src.fps,
    size: src.size,
    top5,
    quality: assessQuality(raw, T),
    preview: wantPreview ? preview.filter(Boolean) : input.blobs.map((b) => URL.createObjectURL(b)),
    timing,
    practice,
    ...(globalThis.__VSL_DEBUG ? { keypoints: Array.from(raw) } : {}),
  };
}
