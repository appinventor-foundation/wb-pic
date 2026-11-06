import test from 'node:test';
import assert from 'node:assert/strict';
import { summaryRows, rocRows, learningCurveRows, outOfFoldScores, pooledConfusion, lossCurveRows, perClassRows } from '../core/aggregate.js';
import { stratifiedKFold } from '../core/splits.js';
import { allMetrics, tprOnGrid } from '../core/metrics.js';

// Synthetic results: model "good" separates the classes, model "bad" is near chance.
const labels = Array.from({ length: 60 }, (_, i) => (i < 30 ? 0 : 1));
const folds = stratifiedKFold(labels, 3, 1);
const scoreFor = (model, y, i) => (model === 'good' ? (y ? 0.85 : 0.15) : (i % 3 === 0 ? 1 - (y ? 0.6 : 0.4) : (y ? 0.6 : 0.4)));
const config = { models: ['good', 'bad'], headSeeds: [0, 1], learningCurve: { folds: [0], seeds: 1, conditions: ['c'], sizes: [5] } };
const runs = [];
for (const model of ['good', 'bad'])
  for (let fold = 0; fold < 3; fold++)
    for (const seed of [0, 1]) {
      const test = folds[fold].test, y = test.map(i => labels[i]);
      const scores = test.map((idx, i) => scoreFor(model, labels[idx], idx));
      runs.push({ model, cond: 'c', fold, seed, n: 'all', nTrainPerClass: 20, metrics: allMetrics(scores, y), tpr: tprOnGrid(scores, y), trainMs: 10, history: [{ loss: 1 }, { loss: 0.5 }], scores });
      if (fold === 0 && seed === 0) runs.push({ model, cond: 'c', fold, seed, n: 5, nTrainPerClass: 5, metrics: allMetrics(scores, y), tpr: tprOnGrid(scores, y), trainMs: 1, history: [{ loss: 1 }] });
    }
const results = { config, dataset: { labels, ids: labels.map((y, i) => `${y ? 'malignant' : 'benign'}/img${i}.jpg`) }, folds: folds.map(f => ({ test: f.test })), runs };

test('summary has one row per model and condition with mean/std across folds', () => {
  const rows = summaryRows(results);
  assert.equal(rows.length, 2);
  const good = rows.find(r => r.model === 'good');
  assert.equal(good.folds, 3); assert.equal(good.seeds, 2);
  assert.equal(good.rocAuc.mean, 1); assert.equal(good.rocAuc.n, 3);
  assert.ok(rows.find(r => r.model === 'bad').rocAuc.mean < 1);
});

test('roc rows cover the grid for each model', () => {
  const rows = rocRows(results, 'c');
  assert.equal(rows.length, 202);
});

test('learning curve includes the small run and the full-size point', () => {
  const rows = learningCurveRows(results).filter(r => r.model === 'good');
  assert.deepEqual(rows.map(r => r.nPerClass), [5, 20]);
});

test('out-of-fold scores cover every sample exactly once', () => {
  const s = outOfFoldScores(results, 'good', 'c', 0);
  assert.equal(s.length, 60);
  assert.ok(s.every((v, i) => v === (labels[i] ? 0.85 : 0.15)));
});

test('pooled confusion counts every sample once and matches the scores', () => {
  const c = pooledConfusion(results, 'good', 'c', 0);
  assert.deepEqual(c.matrix, [[30, 0], [0, 30]]); // the good model is perfect at 0.5
  const bad = pooledConfusion(results, 'bad', 'c', 0);
  assert.equal(bad.tn + bad.fp + bad.fn + bad.tp, 60);
  assert.equal(bad.tn + bad.fp, 30); assert.equal(bad.fn + bad.tp, 30);
});

test('loss curve has one row per epoch for each model and condition', () => {
  const rows = lossCurveRows(results);
  assert.equal(rows.length, 4); // 2 models x 2 epochs in the synthetic history
  const first = rows.find(r => r.model === 'good' && r.epoch === 1);
  assert.equal(first.loss, 1); assert.equal(first.runs, 6); // 3 folds x 2 seeds
  assert.equal(rows.find(r => r.model === 'good' && r.epoch === 2).loss, 0.5);
});

test('per-class rows: perfect model scores 1 everywhere, near-chance model does not', () => {
  const rows = perClassRows(results);
  const good = rows.find(r => r.model === 'good'), bad = rows.find(r => r.model === 'bad');
  for (const k of ['benign.precision', 'benign.recall', 'benign.f1', 'malignant.precision', 'malignant.recall', 'malignant.f1', 'macro.f1'])
    assert.equal(good[k].mean, 1);
  assert.ok(bad['macro.f1'].mean < 1);
  assert.equal(good.support.benign, 30); assert.equal(good.support.malignant, 30);
});

