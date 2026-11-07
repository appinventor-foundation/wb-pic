// Pure post-processing of a results object (see bench.js) into report tables. No tf, no DOM.
import { mean, std, confusion, perClassMetrics } from './metrics.js';

export const METRIC_KEYS = ['accuracy', 'balancedAccuracy', 'recall', 'specificity', 'precision', 'f1', 'rocAuc', 'prAuc', 'sensAtSpec90', 'sensAtSpec95', 'brier', 'ece'];

const full = (results) => results.runs.filter(r => r.n === 'all');
const groupBy = (items, keyFn) => {
  const out = new Map();
  for (const it of items) { const k = keyFn(it); if (!out.has(k)) out.set(k, []); out.get(k).push(it); }
  return out;
};

// Seed-average within each random split first, then mean +- std across the splits (the splits are the unit of replication).
function acrossSplits(runs, pick) {
  const perSplit = [...groupBy(runs, r => r.split).values()].map(rs => mean(rs.map(pick)));
  return { mean: mean(perSplit), std: std(perSplit), n: perSplit.length };
}

export function summaryRows(results) {
  const rows = [];
  for (const [key, runs] of groupBy(full(results), r => `${r.model}|${r.cond}`)) {
    const [model, cond] = key.split('|');
    const row = { model, cond, splits: new Set(runs.map(r => r.split)).size, seeds: new Set(runs.map(r => r.seed)).size };
    for (const m of METRIC_KEYS) row[m] = acrossSplits(runs, r => r.metrics[m]);
    row.trainMs = acrossSplits(runs, r => r.trainMs);
    row.finalLoss = acrossSplits(runs, r => r.history.at(-1).loss);
    row.firstLoss = acrossSplits(runs, r => r.history[0].loss);
    rows.push(row);
  }
  return rows;
}

export function rocRows(results, cond) {
  const rows = [];
  for (const [model, runs] of groupBy(full(results).filter(r => r.cond === cond), r => r.model)) {
    const grid = runs[0].tpr.map((_, g) => mean(runs.map(r => r.tpr[g])));
    grid.forEach((tpr, g) => rows.push({ model, fpr: g / (grid.length - 1), tpr }));
  }
  return rows;
}

export function learningCurveRows(results) {
  const { learningCurve } = results.config;
  const sizeOf = (r) => (r.n === 'all' ? r.nTrainPerClass : r.n);
  const used = results.runs.filter(r => r.n !== 'all'
    || (learningCurve.splits.includes(r.split) && r.seed < learningCurve.seeds && learningCurve.conditions.includes(r.cond)));
  const rows = [];
  for (const [key, runs] of groupBy(used, r => `${r.model}|${r.cond}|${sizeOf(r)}`)) {
    const [model, cond, n] = key.split('|');
    rows.push({ model, cond, nPerClass: +n, runs: runs.length,
      rocAuc: mean(runs.map(r => r.metrics.rocAuc)), recall: mean(runs.map(r => r.metrics.recall)), accuracy: mean(runs.map(r => r.metrics.accuracy)) });
  }
  return rows.sort((a, b) => a.model.localeCompare(b.model) || a.cond.localeCompare(b.cond) || a.nPerClass - b.nPerClass);
}

// Confusion matrix at threshold 0.5 on the test images, averaged over the random splits (one head seed). Rows are the actual
// class and columns the predicted class (benign, malignant). Counts are the mean per test set, so they are not whole numbers.
export function meanConfusion(results, model, cond, seed = results.config.headSeeds[0]) {
  const runs = full(results).filter(r => r.model === model && r.cond === cond && r.seed === seed && r.scores);
  if (!runs.length) return null;
  const labels = results.dataset.labels;
  const parts = runs.map(r => confusion(r.scores, results.splits[r.split].test.map(i => labels[i])));
  const avg = (key) => mean(parts.map(p => p[key]));
  const tn = avg('tn'), fp = avg('fp'), fn = avg('fn'), tp = avg('tp');
  return { tn, fp, fn, tp, matrix: [[tn, fp], [fn, tp]], classes: ['benign', 'malignant'], testSets: runs.length };
}

// Mean training and validation loss/accuracy per epoch for each model, across the random splits and seeds (full-size runs).
// Every value is measured after the epoch with the same weights: training on the train images, validation on the held-out
// validation images. The test images are not involved.
export function curveRows(results) {
  const rows = [];
  for (const [model, runs] of groupBy(full(results).filter(r => r.history[0]?.valLoss != null), r => r.model)) {
    const epochs = Math.min(...runs.map(r => r.history.length));
    for (let e = 0; e < epochs; e++) {
      const at = (key) => mean(runs.map(r => r.history[e][key]));
      rows.push({ model, epoch: e + 1, trainLoss: at('trainLoss'), valLoss: at('valLoss'), trainAcc: at('trainAcc'), valAcc: at('valAcc'), runs: runs.length });
    }
  }
  return rows;
}

// Per-class precision/recall/F1 (each class as the positive class) and macro averages, computed from each full-size run's
// stored test scores. Mean +- std across the random splits, seeds averaged within a split. `support` is the number of
// benign and malignant images in one test set.
export function perClassRows(results) {
  const labels = results.dataset.labels;
  const testLabels = (r) => results.splits[r.split].test.map(i => labels[i]);
  const cache = new Map();
  const metricsOf = (r) => {
    if (!cache.has(r)) cache.set(r, perClassMetrics(r.scores, testLabels(r)));
    return cache.get(r);
  };
  const rows = [];
  for (const [key, runs] of groupBy(full(results).filter(r => r.scores), r => `${r.model}|${r.cond}`)) {
    const [model, cond] = key.split('|');
    const row = { model, cond };
    for (const cls of ['benign', 'malignant', 'macro'])
      for (const m of ['precision', 'recall', 'f1']) row[`${cls}.${m}`] = acrossSplits(runs, r => metricsOf(r)[cls][m]);
    const y = testLabels(runs[0]);
    row.support = { benign: y.filter(v => v === 0).length, malignant: y.filter(v => v === 1).length };
    rows.push(row);
  }
  return rows;
}
