import { tf } from './tf.js';
import { loadDataset } from './dataset.js';
import { extractFeatures } from './extract.js';
import { trainAndPredict, APP_DEFAULT } from './head.js';
import { stratifiedSplit, subsamplePerClass } from './splits.js';
import { allMetrics, tprOnGrid } from './metrics.js';
import { benchmarkSpeed } from './speed.js';
import { collectEnvironment } from './env.js';
import { cacheGet, cachePut } from './cache.js';

export const BENCHMARK_CONFIG = {
  name: 'mobilenet v1 vs v3 on benign-vs-malignant moles',
  models: ['mobilenet_local', 'mobilenet_v3_small_local'],
  // Train / validation / test split, repeated on several random splits so the result does not hinge on one lucky split.
  split: { train: 0.7, validation: 0.15, test: 0.15, splits: 5, seed: 1 },
  headSeeds: [0, 1, 2],
  conditions: { 'app-default': APP_DEFAULT }, // the one training setup: the app's own settings
  learningCurve: { sizes: [10, 25, 50, 100, 250, 500], splits: [0], seeds: 2, conditions: ['app-default'] },
  speed: { warmup: 5, runs: 50 }
};

// Bump when the code that produces a run record changes, so checkpoints from older code are not reused.
export const RUN_VERSION = 7;

const round4 = (xs) => Array.from(xs, v => Math.round(v * 1e4) / 1e4);
const round5 = (v) => Math.round(v * 1e5) / 1e5;
const nextTick = () => new Promise(r => setTimeout(r));

// Runs the whole benchmark and returns one JSON-serialisable results object. Finished runs are checkpointed in
// IndexedDB under a key that includes the config hash and file-list hash, so an interrupted run resumes.
export async function runBenchmark({ config = BENCHMARK_CONFIG, onProgress, resume = true } = {}) {
  await tf.setBackend('webgl'); await tf.ready();
  const dataset = await loadDataset();
  const env = await collectEnvironment(config);
  const { labels } = dataset;
  const parts = Array.from({ length: config.split.splits }, (_, s) => stratifiedSplit(labels, config.split, config.split.seed * 7919 + s));
  const results = {
    schema: 2, config, env,
    dataset: { counts: dataset.counts, fileListHash: dataset.fileListHash, labels, ids: dataset.samples.map(s => s.id) },
    splits: parts.map(p => ({ validation: p.validation, test: p.test })), extraction: {}, runs: [], speed: [],
    notes: ['positive class = malignant', 'splits are shared by both models, so comparisons are paired']
  };

  const plan = [];
  const lc = config.learningCurve;
  for (const model of config.models) {
    for (const cond of Object.keys(config.conditions))
      for (let split = 0; split < config.split.splits; split++)
        for (const seed of config.headSeeds) plan.push({ model, cond, split, seed, n: 'all' });
    for (const cond of lc.conditions)
      for (const split of lc.splits)
        for (let seed = 0; seed < lc.seeds; seed++)
          for (const n of lc.sizes) plan.push({ model, cond, split, seed: config.headSeeds[seed], n });
  }

  let done = 0;
  for (const model of config.models) {
    const ex = await extractFeatures({ modelKey: model, dataset, onProgress });
    results.extraction[model] = { count: ex.count, shape: ex.shape, loadMs: ex.loadMs, extractMs: ex.extractMs, fromCache: ex.fromCache };
    const features = tf.tensor4d(ex.features, [ex.count, ...ex.shape]);
    for (const step of plan.filter(p => p.model === model)) {
      const key = `run|v${RUN_VERSION}|${results.env.configHash}|${dataset.fileListHash}|${model}|${step.cond}|${step.split}|${step.seed}|${step.n}`;
      let record = resume ? await cacheGet(key) : undefined;
      if (!record) {
        const part = parts[step.split];
        const full = step.n === 'all';
        const trainIdx = full ? part.train : subsamplePerClass(part.train, labels, step.n, step.split + 1);
        // full-size runs also record training and validation loss per epoch; the validation images are only monitored, never trained on
        const out = await trainAndPredict({ features, y: labels, trainIdx, testIdx: part.test, valIdx: full ? part.validation : null, cond: config.conditions[step.cond], seed: step.seed });
        const yTest = part.test.map(i => labels[i]);
        record = {
          ...step, nTrain: trainIdx.length, nTrainPerClass: Math.min(...[0, 1].map(c => trainIdx.filter(i => labels[i] === c).length)),
          nTest: part.test.length, metrics: allMetrics(out.scores, yTest), tpr: round4(tprOnGrid(out.scores, yTest)),
          trainMs: out.trainMs, leakedTensors: out.leaked,
          history: full ? out.history.map(h => ({ loss: h.loss, acc: h.acc, trainLoss: round5(h.trainLoss), trainAcc: round5(h.trainAcc), valLoss: round5(h.valLoss), valAcc: round5(h.valAcc) })) : out.history,
          ...(full ? { nVal: part.validation.length, scores: round4(out.scores) } : {})
        };
        await cachePut(key, record);
      }
      results.runs.push(record);
      onProgress?.({ stage: 'runs', model, done: ++done, total: plan.length, last: record });
      await nextTick();
    }
    features.dispose();
  }

  for (const model of config.models) results.speed.push(await benchmarkSpeed({ modelKey: model, samples: dataset.samples, ...config.speed, onProgress }));
  results.env.finished = new Date().toISOString();
  return results;
}
