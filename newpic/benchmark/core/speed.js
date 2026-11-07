import { tf } from './tf.js';
import { loadBackbone, modelConfig } from './extract.js';
import { quantile, mean } from './metrics.js';

const timed = async (fn) => { const t = performance.now(); const r = await fn(); return [performance.now() - t, r]; };

// Size of the model files the browser downloads, and the parameter count from the weight shapes in model.json.
// Both are read from the files, so v1 (layers) and v3 (graph) are counted the same way. The count includes any
// layers after the truncation point, because those weights are downloaded too.
async function modelFiles(modelKey) {
  const { url } = modelConfig(modelKey).transferModel;
  const manifest = await (await fetch(url)).json();
  const base = url.slice(0, url.lastIndexOf('/') + 1);
  let bytes = JSON.stringify(manifest).length, parameters = 0;
  for (const group of manifest.weightsManifest) {
    for (const p of group.paths) bytes += (await (await fetch(base + p)).arrayBuffer()).byteLength;
    for (const w of group.weights) parameters += w.shape.reduce((a, b) => a * b, 1);
  }
  return { bytes, parameters };
}

// Per-model latency/throughput. Warm-up iterations are discarded (they compile WebGL shaders), and every
// timed region ends in `await tensor.data()` so GPU work is included.
export async function benchmarkSpeed({ modelKey, samples, warmup = 5, runs = 50, onProgress }) {
  const files = await modelFiles(modelKey);
  const memBefore = tf.memory().numBytes;
  const backbone = await loadBackbone(modelKey);
  const memAfterLoad = tf.memory().numBytes;
  const urls = samples.slice(0, warmup + runs).map(s => s.url);
  const feat = (x) => backbone.extract(x);

  // backbone only (batch 1, already-preprocessed tensor)
  tf.engine().startScope();
  const pre = await backbone.processor.convertImage(urls[0]);
  const x = pre.expandDims(0);
  const [inH, inW, inC] = [x.shape[1], x.shape[2], x.shape[3]];
  const backboneMs = [];
  for (let i = 0; i < warmup + runs; i++) {
    const [ms] = await timed(async () => { const f = feat(x); await f.data(); f.dispose(); });
    if (i >= warmup) backboneMs.push(ms);
  }
  tf.engine().endScope();

  // end to end from an image URL: decode + crop/resize/normalize + backbone (the part training repeats per image)
  const e2eMs = [];
  for (let i = 0; i < urls.length; i++) {
    tf.engine().startScope();
    const [ms] = await timed(async () => {
      const img = await backbone.processor.convertImage(urls[i]);
      const f = feat(img.expandDims(0));
      await f.data();
    });
    tf.engine().endScope();
    if (i >= warmup) e2eMs.push(ms);
    onProgress?.({ stage: 'speed', model: modelKey, done: i + 1, total: urls.length });
  }
  const out = {
    model: modelKey,
    inputShape: [inH, inW, inC],
    loadMs: backbone.loadMs, // includes the HTTP fetch; the browser may serve repeat loads from its cache
    downloadBytes: files.bytes,
    parameters: files.parameters, // in the downloaded files, incl. layers after the truncation point
    gpuBytesAfterLoad: memAfterLoad - memBefore,
    jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
    backboneMs: { p50: quantile(backboneMs, 0.5), p95: quantile(backboneMs, 0.95), mean: mean(backboneMs), n: backboneMs.length },
    endToEndMs: { p50: quantile(e2eMs, 0.5), p95: quantile(e2eMs, 0.95), mean: mean(e2eMs), n: e2eMs.length },
    imagesPerSecond: 1000 / mean(e2eMs)
  };
  backbone.model.dispose();
  return out;
}
