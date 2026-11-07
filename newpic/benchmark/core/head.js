import { tf } from './tf.js';
import { DEFAULT_CONFIG } from '../../src/ml/config/defaults.js';
import { ModelBuilder } from '../../src/ml/ModelBuilder.js';
import { shuffled } from './rng.js';

// The one training setup: exactly what the app trains (DEFAULT_CONFIG).
export const APP_DEFAULT = {
  epochs: DEFAULT_CONFIG.training.epochs,
  learningRate: DEFAULT_CONFIG.customModel.optimizer.learningRate,
  batchSizeFraction: DEFAULT_CONFIG.training.batchSizeFraction
};

// 'varianceScaling' string initializers are unseeded; swap in seeded instances with the same settings.
const seededLayers = (layers, seed) => layers.map((layer, i) => layer.kernelInitializer === 'varianceScaling'
  ? { ...layer, kernelInitializer: tf.initializers.varianceScaling({ scale: 1, mode: 'fanIn', distribution: 'normal', seed: seed * 101 + i }) }
  : layer);

const gatherRows = (x, idx) => tf.tidy(() => tf.gather(x, tf.tensor1d(idx, 'int32')));

function buildConvHead(featShape, cond, seed) {
  const customModel = { ...DEFAULT_CONFIG.customModel, layers: seededLayers(DEFAULT_CONFIG.customModel.layers, seed), optimizer: { type: 'adam', learningRate: cond.learningRate } };
  return new ModelBuilder({ transferModel: { outputShape: featShape }, customModel }).buildCustomModel(2);
}

// features: tf.Tensor4D [N,H,W,C] (kept by the caller), y: 0/1 array, trainIdx/testIdx: row indices.
// Returns P(malignant) for each test row plus training stats. All tensors created here are disposed.
// With valIdx (rows the model never trains on), each epoch's history entry also holds trainLoss/trainAcc and valLoss/valAcc,
// measured after the epoch with the same weights. They are only monitored: nothing is chosen from them. (The `loss`/`acc`
// that fit reports are running averages while the weights are still changing.)
export async function trainAndPredict({ features, y, trainIdx, testIdx, valIdx = null, cond, seed }) {
  const before = tf.memory().numTensors;
  const t0 = performance.now();
  const [, H, W, C] = features.shape;
  let model, optimizer;
  const history = [];

  model = buildConvHead([H, W, C], cond, seed);
  optimizer = model.optimizer;
  const batchSize = Math.max(1, Math.floor(trainIdx.length * cond.batchSizeFraction));
  for (let e = 0; e < cond.epochs; e++) {
    // our own seeded order each epoch (fit's shuffle uses Math.random)
    const order = shuffled(trainIdx, seed * 100003 + e);
    const xs = gatherRows(features, order), ys = tf.tidy(() => tf.oneHot(tf.tensor1d(order.map(i => y[i]), 'int32'), 2));
    const h = await model.fit(xs, ys, { epochs: 1, batchSize, shuffle: false, verbose: 0 });
    const entry = { loss: h.history.loss[0], acc: h.history.acc?.[0] ?? h.history.accuracy?.[0] };
    if (valIdx) {
      const train = await evaluateRows(model, features, y, trainIdx), val = await evaluateRows(model, features, y, valIdx);
      Object.assign(entry, { trainLoss: train.loss, trainAcc: train.acc, valLoss: val.loss, valAcc: val.acc });
    }
    history.push(entry);
    xs.dispose(); ys.dispose();
  }
  const trainMs = performance.now() - t0;
  const scores = await predictInBatches(model, features, testIdx);
  model.dispose(); optimizer.dispose();
  return { scores, trainMs, history, leaked: tf.memory().numTensors - before };
}

// Mean log loss and accuracy of the model on the given rows, from tfjs's own loss and metric, in batches to bound memory.
async function evaluateRows(model, x, y, idx, batch = 256) {
  let loss = 0, correct = 0;
  for (let s = 0; s < idx.length; s += batch) {
    const part = idx.slice(s, s + batch);
    const [batchLoss, batchAcc] = tf.tidy(() => {
      const probs = model.predict(gatherRows(x, part)).slice([0, 1], [-1, 1]); // P(malignant), one column
      const labels = tf.tensor2d(part.map(i => [y[i]]));
      return [tf.losses.logLoss(labels, probs), tf.metrics.binaryAccuracy(labels, probs).mean()];
    });
    const [lossValue, accValue] = await Promise.all([batchLoss.data(), batchAcc.data()]);
    loss += lossValue[0] * part.length; correct += accValue[0] * part.length;
    batchLoss.dispose(); batchAcc.dispose();
  }
  return { loss: loss / idx.length, acc: correct / idx.length };
}

async function predictInBatches(model, x, idx, batch = 256) {
  const out = new Float32Array(idx.length);
  for (let s = 0; s < idx.length; s += batch) {
    const part = idx.slice(s, s + batch);
    const probs = tf.tidy(() => model.predict(gatherRows(x, part)).slice([0, 1], [-1, 1]).flatten());
    out.set(await probs.data(), s);
    probs.dispose();
  }
  return out;
}
