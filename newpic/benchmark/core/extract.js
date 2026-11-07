import { tf } from './tf.js';
import { TRANSFER_MODEL_OPTIONS, DEFAULT_CONFIG } from '../../src/ml/config/defaults.js';
import { ModelBuilder } from '../../src/ml/ModelBuilder.js';
import { DataProcessor } from '../../src/ml/DataProcessor.js';
import { ReusableCanvasAdapter } from './adapter.js';
import { cacheGet, cachePut } from './cache.js';

// Bump when preprocessing (crop/resize/normalization code path) changes, to invalidate the cache.
export const PREPROCESSING_VERSION = 1;

const STALL_MS = 60000; // one image taking longer than this means the pipeline is stuck
const CHECKPOINT_EVERY = 500;

const withTimeout = (promise, ms, what) => {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`stalled for ${ms / 1000}s: ${what}`)), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
};

export function modelConfig(modelKey) {
  const transferModel = TRANSFER_MODEL_OPTIONS[modelKey];
  if (!transferModel) throw new Error(`unknown model "${modelKey}"`);
  return { ...DEFAULT_CONFIG, transferModel };
}

// Same feature path as ImageClassifier._extractFeatures: graph models execute() up to the
// truncation node, layers models are already truncated.
export const extractor = (model, transferModel) => (x) =>
  transferModel.modelFormat === 'graph' ? model.execute(x, transferModel.truncationLayer) : model.predict(x);

export async function loadBackbone(modelKey) {
  const config = modelConfig(modelKey);
  const t0 = performance.now();
  const model = await new ModelBuilder(config).getTransferModel();
  return {
    model, config,
    loadMs: performance.now() - t0,
    extract: extractor(model, config.transferModel),
    processor: new DataProcessor(config, new ReusableCanvasAdapter())
  };
}

export const featureCacheKey = (modelKey, fileListHash) => {
  const { normalization = 'mobilenet' } = TRANSFER_MODEL_OPTIONS[modelKey];
  return `features|${modelKey}|${normalization}|p${PREPROCESSING_VERSION}|${fileListHash}`;
};

// Streams the dataset through the frozen backbone one image at a time and keeps the feature maps
// on the CPU (N * H*W*C floats), so nothing accumulates on the GPU. Each image runs inside its own
// tf scope because DataProcessor.convertImage leaves its intermediate tensors undisposed.
export async function extractFeatures({ modelKey, dataset, onProgress, useCache = true }) {
  const key = featureCacheKey(modelKey, dataset.fileListHash);
  if (useCache) {
    const hit = await cacheGet(key);
    if (hit && hit.count === dataset.samples.length) return { ...hit, fromCache: true, features: new Float32Array(hit.buffer) };
  }
  const backbone = await loadBackbone(modelKey);
  const { outputShape } = backbone.config.transferModel;
  const featLen = outputShape.reduce((a, b) => a * b, 1);
  const n = dataset.samples.length;
  const features = new Float32Array(n * featLen);
  // resume from a partial checkpoint left by an interrupted extraction of the same files
  let start = 0, priorMs = 0;
  const partial = useCache ? await cacheGet(`${key}|partial`) : undefined;
  if (partial && partial.count === n && partial.done > 0) {
    features.set(new Float32Array(partial.buffer));
    start = partial.done; priorMs = partial.elapsedMs;
  }
  const t0 = performance.now();
  for (let i = start; i < n; i++) {
    tf.engine().startScope();
    try {
      features.set(await withTimeout((async () => {
        const img = await backbone.processor.convertImage(dataset.samples[i].url);
        return (await backbone.extract(img.expandDims(0)).data());
      })(), STALL_MS, `image ${i} ${dataset.samples[i].id}`), i * featLen);
    } catch (error) {
      if (useCache && i > start) await cachePut(`${key}|partial`, { count: n, done: i, buffer: features.buffer, elapsedMs: priorMs + performance.now() - t0 });
      throw error;
    } finally { tf.engine().endScope(); }
    if (i % 10 === 9 || i === n - 1) { onProgress?.({ stage: 'extract', model: modelKey, done: i + 1, total: n }); await new Promise(r => setTimeout(r)); }
    if (useCache && (i + 1) % CHECKPOINT_EVERY === 0 && i + 1 < n) await cachePut(`${key}|partial`, { count: n, done: i + 1, buffer: features.buffer, elapsedMs: priorMs + performance.now() - t0 });
  }
  const result = {
    count: n, shape: outputShape, loadMs: backbone.loadMs, extractMs: priorMs + performance.now() - t0,
    buffer: features.buffer, key
  };
  backbone.model.dispose();
  if (useCache) await cachePut(key, result);
  return { ...result, fromCache: false, features };
}
