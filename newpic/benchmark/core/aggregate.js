// Pure post-processing of a results object (see bench.js) into report tables. No tf, no DOM.
import { shuffled } from './rng.js';
import { mean, std, mcnemarExact, pairedBootstrap, rocAuc, allMetrics, confusion, perClassMetrics } from './metrics.js';

export const METRIC_KEYS = ['accuracy', 'balancedAccuracy', 'recall', 'specificity', 'precision', 'f1', 'rocAuc', 'prAuc', 'sensAtSpec90', 'sensAtSpec95', 'brier', 'ece'];

const full = (results) => results.runs.filter(r => r.n === 'all');
const groupBy = (items, keyFn) => {
  const out = new Map();
  for (const it of items) { const k = keyFn(it); if (!out.has(k)) out.set(k, []); out.get(k).push(it); }
  return out;
};

// Seed-average within each fold first, then mean +- std across folds (folds are the unit of replication).
function acrossFolds(runs, pick) {
  const perFold = [...groupBy(runs, r => r.fold).values()].map(rs => mean(rs.map(pick)));
  return { mean: mean(perFold), std: std(perFold), n: perFold.length };
}

export function summaryRows(results) {
  const rows = [];
  for (const [key, runs] of groupBy(full(results), r => `${r.model}|${r.cond}`)) {
    const [model, cond] = key.split('|');
    const row = { model, cond, folds: new Set(runs.map(r => r.fold)).size, seeds: new Set(runs.map(r => r.seed)).size };
    for (const m of METRIC_KEYS) row[m] = acrossFolds(runs, r => r.metrics[m]);
    row.trainMs = acrossFolds(runs, r => r.trainMs);
    row.finalLoss = acrossFolds(runs, r => r.history.at(-1).loss);
    row.firstLoss = acrossFolds(runs, r => r.history[0].loss);
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
    || (learningCurve.folds.includes(r.fold) && r.seed < learningCurve.seeds && learningCurve.conditions.includes(r.cond)));
  const rows = [];
  for (const [key, runs] of groupBy(used, r => `${r.model}|${r.cond}|${sizeOf(r)}`)) {
    const [model, cond, n] = key.split('|');
    rows.push({ model, cond, nPerClass: +n, runs: runs.length,
      rocAuc: mean(runs.map(r => r.metrics.rocAuc)), recall: mean(runs.map(r => r.metrics.recall)), accuracy: mean(runs.map(r => r.metrics.accuracy)) });
  }
  return rows.sort((a, b) => a.model.localeCompare(b.model) || a.cond.localeCompare(b.cond) || a.nPerClass - b.nPerClass);
}

// Out-of-fold P(malignant) for every sample, from one head seed: each sample is scored by the fold that held it out.
export function outOfFoldScores(results, model, cond, seed) {
  const N = results.dataset.labels.length;
  const scores = new Array(N).fill(null);
  for (const r of full(results).filter(r => r.model === model && r.cond === cond && r.seed === seed)) {
    results.folds[r.fold].test.forEach((idx, i) => { scores[idx] = r.scores[i]; });
  }
  return scores.every(s => s !== null) ? scores : null;
}

// Paired comparison of two models on identical out-of-fold predictions.
export function pairedComparison(results, modelA, modelB, cond, { resamples = 1000, seed = 1 } = {}) {
  const s = results.config.headSeeds[0];
  const a = outOfFoldScores(results, modelA, cond, s), b = outOfFoldScores(results, modelB, cond, s);
  if (!a || !b) return null;
  const y = results.dataset.labels;
  const correct = (sc) => sc.map((v, i) => (v >= 0.5 ? 1 : 0) === y[i]);
  return {
    cond, modelA, modelB, n: y.length,
    accuracyA: mean(correct(a)), accuracyB: mean(correct(b)),
    mcnemar: mcnemarExact(correct(a), correct(b)),
    aucA: rocAuc(a, y), aucB: rocAuc(b, y),
    aucDiff: pairedBootstrap(a, b, y, rocAuc, { resamples, seed }),
    pooled: { [modelA]: allMetrics(a, y), [modelB]: allMetrics(b, y) }
  };
}

// Pooled out-of-fold confusion matrix at threshold 0.5 for one head seed: every image is counted once, in the fold
// that held it out. Rows are the actual class, columns the predicted class (benign, malignant).
export function pooledConfusion(results, model, cond, seed = results.config.headSeeds[0]) {
  const scores = outOfFoldScores(results, model, cond, seed);
  if (!scores) return null;
  const { tp, fp, tn, fn } = confusion(scores, results.dataset.labels);
  return { tn, fp, fn, tp, matrix: [[tn, fp], [fn, tp]], classes: ['benign', 'malignant'] };
}

// Mean training loss/accuracy per epoch, across the folds and seeds of each model and condition (full-size runs only).
// These are training-set values: the app-default head is not trained with a validation set, so there is no val loss.
export function lossCurveRows(results) {
  const rows = [];
  for (const [key, runs] of groupBy(full(results), r => `${r.model}|${r.cond}`)) {
    const [model, cond] = key.split('|');
    const epochs = Math.min(...runs.map(r => r.history.length));
    for (let e = 0; e < epochs; e++) {
      const loss = runs.map(r => r.history[e].loss), acc = runs.map(r => r.history[e].acc);
      rows.push({ model, cond, epoch: e + 1, loss: mean(loss), lossStd: std(loss), acc: mean(acc), runs: runs.length });
    }
  }
  return rows;
}

// Per-class precision/recall/F1 (each class as the positive class) and macro averages. Computed from the stored
// out-of-fold scores of each full-size run, so runs saved before this metric existed still work.
// Mean +- std across folds, seeds averaged within a fold.
export function perClassRows(results) {
  const labels = results.dataset.labels;
  const cache = new Map();
  const metricsOf = (r) => {
    if (!cache.has(r)) {
      const test = results.folds[r.fold].test;
      cache.set(r, perClassMetrics(r.scores, test.map(i => labels[i])));
    }
    return cache.get(r);
  };
  const rows = [];
  for (const [key, runs] of groupBy(full(results).filter(r => r.scores), r => `${r.model}|${r.cond}`)) {
    const [model, cond] = key.split('|');
    const row = { model, cond };
    for (const cls of ['benign', 'malignant', 'macro'])
      for (const m of ['precision', 'recall', 'f1']) row[`${cls}.${m}`] = acrossFolds(runs, r => metricsOf(r)[cls][m]);
    row.support = metricsOf(runs[0]) && { benign: labels.filter(v => v === 0).length, malignant: labels.filter(v => v === 1).length };
    rows.push(row);
  }
  return rows;
}

// ---- galleries (use dataset.ids, so they need a results file saved with this schema) ----

const CLASS_NAMES = ['benign', 'malignant'];

// A seeded random sample of `perClass` images from each class: [{ idx, id, y, label }].
export function sampleImages(results, perClass, seed = 1) {
  const { labels, ids } = results.dataset;
  return [0, 1].flatMap(y => shuffled([...labels.keys()].filter(i => labels[i] === y), seed * 131 + y)
    .slice(0, perClass).map(idx => ({ idx, id: ids[idx], y, label: CLASS_NAMES[y] })));
}

// Random images with each model's out-of-fold P(malignant): [{ idx, id, y, label, scores: { [model]: p } }].
// Out-of-fold means the model that scored an image never trained on it, so these are genuine test predictions.
export function testPredictionCards(results, cond, perClass, seed = 1) {
  const { models } = results.config, s = results.config.headSeeds[0];
  const oof = Object.fromEntries(models.map(m => [m, outOfFoldScores(results, m, cond, s)]));
  if (models.some(m => !oof[m])) return [];
  return sampleImages(results, perClass, seed).map(c => ({ ...c, scores: Object.fromEntries(models.map(m => [m, oof[m][c.idx]])) }));
}

// The model's most confident mistakes: malignant images scored lowest (missed) and benign scored highest (false alarms).
export function confidentMistakes(results, model, cond, k) {
  const scores = outOfFoldScores(results, model, cond, results.config.headSeeds[0]);
  if (!scores) return { fn: [], fp: [] };
  const { labels, ids } = results.dataset;
  const pick = (y, cmp, wrong) => [...labels.keys()].filter(i => labels[i] === y && wrong(scores[i]))
    .sort((a, b) => cmp(scores[a], scores[b])).slice(0, k).map(idx => ({ idx, id: ids[idx], p: scores[idx], y, label: CLASS_NAMES[y] }));
  return {
    fn: pick(1, (a, b) => a - b, p => p < 0.5),
    fp: pick(0, (a, b) => b - a, p => p >= 0.5)
  };
}
