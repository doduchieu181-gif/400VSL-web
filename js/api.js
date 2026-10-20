// Bản web tĩnh: không có máy chủ. Các hàm giữ đúng tên và dạng dữ liệu như bản gọi FastAPI cũ,
// nhưng đọc danh mục từ data/catalog.json, chạy nhận diện ngay trong trình duyệt (engine/analyzer.js)
// và lưu tiến độ học trong localStorage của trình duyệt này.
import { analyzeLocally, loadAnalyzer } from './engine/analyzer.js';

const asset = (path) => new URL(`../${path}`, import.meta.url).href;

// ------------------------------------------------------------------
// Danh mục
// ------------------------------------------------------------------
let dataPromise = null;

function loadData() {
  if (!dataPromise) {
    dataPromise = fetch(asset('data/catalog.json')).then((res) => {
      if (!res.ok) throw new Error(`Không tải được danh mục bài học (${res.status}).`);
      return res.json();
    }).then((raw) => {
      const gloss = Object.fromEntries(Object.entries(raw.gloss).map(([k, v]) => [Number(k), v]));
      const references = Object.fromEntries(Object.entries(raw.references).map(([k, v]) => [Number(k), v]));
      const topics = {};
      const topicOf = {};
      for (const group of raw.groups) {
        for (const topic of group.topics) {
          topics[topic.id] = { ...topic, groupName: group.name };
          for (const label of topic.labels) topicOf[label] = topic.id;
        }
      }
      return { ...raw, gloss, references, topics, topicOf };
    });
    dataPromise.catch(() => { dataPromise = null; });
  }
  return dataPromise;
}

const videoUrl = (id) => asset(`media/reference/videos/${id}.mp4`);

export async function getCatalog() {
  const d = await loadData();
  const lessons = Object.keys(d.gloss).map(Number).sort((a, b) => a - b).map((label) => ({
    label, gloss: d.gloss[label], topic: d.topicOf[label], reference: Boolean(d.references[label]),
  }));
  return { groups: d.groups, lessons, signer: d.signer };
}

export async function getLesson(label) {
  const d = await loadData();
  label = Number(label);
  if (!d.gloss[label]) throw new Error(`Không có bài học số ${label}.`);
  const topic = d.topics[d.topicOf[label]];
  const pos = topic.labels.indexOf(label);
  const brief = (lab) => (lab === undefined ? null : { label: lab, gloss: d.gloss[lab] });
  const ref = d.references[label];
  return {
    label,
    gloss: d.gloss[label],
    topic: { id: topic.id, name: topic.name, group: topic.groupName },
    position: pos + 1,
    count: topic.labels.length,
    prev: brief(pos > 0 ? topic.labels[pos - 1] : undefined),
    next: brief(pos + 1 < topic.labels.length ? topic.labels[pos + 1] : undefined),
    reference: ref ? {
      video_url: videoUrl(ref.video_id), video_id: ref.video_id, signer: d.signer, fps: d.fps, frames: ref.frames,
    } : null,
  };
}

const refCache = new Map();

async function getReference(label) {
  const d = await loadData();
  const ref = d.references[label];
  if (!ref) throw new Error(`Bài ${label} chưa có video mẫu.`);
  if (!refCache.has(label)) {
    const p = fetch(asset(`media/reference/keypoints/${ref.video_id}.bin`))
      .then((res) => {
        if (!res.ok) throw new Error(`Không tải được khung xương mẫu (${res.status}).`);
        return res.arrayBuffer();
      })
      .then((buf) => {
        const data = Float64Array.from(new Float32Array(buf));
        return { data, T: data.length / (28 * 3) };
      });
    p.catch(() => refCache.delete(label));
    refCache.set(label, p);
  }
  return refCache.get(label);
}

// ------------------------------------------------------------------
// Trạng thái (không còn máy chủ, hàng đợi)
// ------------------------------------------------------------------
export async function getStatus() {
  const d = await loadData();
  return {
    ready: true,
    error: null,
    classes: Object.keys(d.gloss).length,
    reference: { signer: d.signer, count: Object.keys(d.references).length },
    capture: { countdown: 3, frames: 64, fps: 25, max_side: 540, jpeg_quality: 0.85 },
    queue: { waiting: 0, running: 0 },
  };
}

// ------------------------------------------------------------------
// "Tài khoản": web tĩnh không có đăng nhập, mỗi trình duyệt là một người học
// ------------------------------------------------------------------
const LOCAL_USER = { id: 'local', username: 'Bạn' };
export const getMe = async () => ({ user: LOCAL_USER });
export const login = async () => ({ user: LOCAL_USER });
export const register = async () => ({ user: LOCAL_USER });
export const logout = async () => ({ user: LOCAL_USER });

// ------------------------------------------------------------------
// Tiến độ: { label: { attempts, best, passed, last } } trong localStorage
// ------------------------------------------------------------------
const KEYS = { lesson: 'vsl400.progress.v1', drill: 'vsl400.drill.v1' };

function readStore(kind) {
  try {
    return JSON.parse(localStorage.getItem(KEYS[kind]) || '{}') || {};
  } catch {
    return {};
  }
}

function writeStore(kind, data) {
  try {
    localStorage.setItem(KEYS[kind], JSON.stringify(data));
  } catch {
    // trình duyệt chặn lưu trữ: tiến độ chỉ còn trong phiên này
  }
}

const memory = { lesson: null, drill: null };
const load = (kind) => {
  if (!memory[kind]) memory[kind] = readStore(kind);
  return memory[kind];
};

export const fetchProgress = async () => ({ ...load('lesson') });
export const fetchDrill = async () => ({ ...load('drill') });

export async function resetDrill(labels) {
  const data = load('drill');
  for (const label of labels) if (data[label]) data[label] = { ...data[label], passed: false };
  writeStore('drill', data);
  return { ...data };
}

function record(practice, drill) {
  const kind = drill ? 'drill' : 'lesson';
  const data = load(kind);
  const label = practice.target.label;
  const row = data[label] || { attempts: 0, best: 0, passed: false, last: 0 };
  const entry = {
    attempts: row.attempts + 1,
    best: Math.max(row.best, Math.round(practice.score)),
    passed: row.passed || practice.verdict === 'pass',
    last: Date.now(),
  };
  data[label] = entry;
  writeStore(kind, data);
  return entry;
}

// ------------------------------------------------------------------
// Phân tích
// ------------------------------------------------------------------

/** Nạp sẵn mô hình trong nền để lượt đầu tiên nhanh hơn. */
export function preloadModels() {
  return loadAnalyzer();
}

/**
 * input: { kind: 'frames', blobs: Blob[], fps } | { kind: 'video', file: File }
 * target: label bài đang luyện (tuỳ chọn) -> thêm phần so sánh với video mẫu
 * drill: true khi luyện tập không xem mẫu -> ghi vào tiến độ luyện tập thay cho bài học
 */
export async function analyze(input, { target = null, drill = false, onProgress = () => {}, signal } = {}) {
  const d = await loadData();
  if (target !== null && target !== undefined && !d.gloss[target]) throw new Error(`Không có bài học số ${target}.`);
  if (drill && (target === null || target === undefined)) throw new Error('Luyện tập cần ký hiệu mục tiêu.');
  const result = await analyzeLocally(input, { target, catalog: d, getReference, onProgress, signal });
  if (result.practice) result.practice.progress = record(result.practice, drill);
  return result;
}
