import test from 'node:test';
import assert from 'node:assert/strict';
import { summaryRows, rocRows, learningCurveRows, meanConfusion, curveRows, perClassRows } from '../core/aggregate.js';
import { stratifiedSplit } from '../core/splits.js';
import { allMetrics, tprOnGrid } from '../core/metrics.js';

// Synthetic results: model "good" separates the classes, model "bad" is near chance. Three random splits of 60 images.
const labels = Array.from({ length: 60 }, (_, i) => (i < 30 ? 0 : 1));
const splits = Array.from({ length: 3 }, (_, s) => stratifiedSplit(labels, { validation: 0.2, test: 0.2 }, s + 1)); // 36 train / 12 validation / 12 test
const scoreFor = (model, y, i) => (model === 'good' ? (y ? 0.85 : 0.15) : (i % 3 === 0 ? 1 - (y ? 0.6 : 0.4) : (y ? 0.6 : 0.4)));
const config = { models: ['good', 'bad'], headSeeds: [0, 1], learningCurve: { splits: [0], seeds: 1, conditions: ['c'], sizes: [5] } };
const history = [
  { loss: 1, acc: 0.5, trainLoss: 0.9, trainAcc: 0.5, valLoss: 1.0, valAcc: 0.5 },
  { loss: 0.5, acc: 0.6, trainLoss: 0.6, trainAcc: 0.7, valLoss: 0.8, valAcc: 0.6 }
];
const runs = [];
for (const model of ['good', 'bad'])
  for (let split = 0; split < 3; split++)
    for (const seed of [0, 1]) {
      const test = splits[split].test, y = test.map(i => labels[i]);
      const scores = test.map(idx => scoreFor(model, labels[idx], idx));
      runs.push({ model, cond: 'c', split, seed, n: 'all', nTrainPerClass: 18, metrics: allMetrics(scores, y), tpr: tprOnGrid(scores, y), trainMs: 10, history, scores });
      if (split === 0 && seed === 0) runs.push({ model, cond: 'c', split, seed, n: 5, nTrainPerClass: 5, metrics: allMetrics(scores, y), tpr: tprOnGrid(scores, y), trainMs: 1, history: [{ loss: 1 }] });
    }
const results = { config, dataset: { labels, ids: labels.map((y, i) => `${y ? 'malignant' : 'benign'}/img${i}.jpg`) }, splits: splits.map(p => ({ validation: p.validation, test: p.test })), runs };

test('summary has one row per model and condition with mean/std across the random splits', () => {
  const rows = summaryRows(results);
  assert.equal(rows.length, 2);
  const good = rows.find(r => r.model === 'good');
  assert.equal(good.splits, 3); assert.equal(good.seeds, 2);
  assert.equal(good.rocAuc.mean, 1); assert.equal(good.rocAuc.n, 3);
  assert.ok(rows.find(r => r.model === 'bad').rocAuc.mean < 1);
});

test('roc rows cover the grid for each model', () => {
  const rows = rocRows(results, 'c');
  assert.equal(rows.length, 202);
});

test('learning curve includes the small run and the full-size point', () => {
  const rows = learningCurveRows(results).filter(r => r.model === 'good');
  assert.deepEqual(rows.map(r => r.nPerClass), [5, 18]);
});

test('mean confusion averages the counts over the test sets and matches the scores', () => {
  const c = meanConfusion(results, 'good', 'c', 0);
  assert.deepEqual(c.matrix, [[6, 0], [0, 6]]); // each test set holds 6 benign and 6 malignant; the good model is perfect at 0.5
  assert.equal(c.testSets, 3);
  const bad = meanConfusion(results, 'bad', 'c', 0);
  assert.ok(Math.abs(bad.tn + bad.fp - 6) < 1e-9 && Math.abs(bad.fn + bad.tp - 6) < 1e-9);
  assert.equal(meanConfusion(results, 'good', 'c', 99), null); // no runs for that seed
});

test('curves average training and validation loss per epoch across splits and seeds, per model', () => {
  const rows = curveRows(results);
  assert.equal(rows.length, 4); // 2 models x 2 epochs
  const first = rows.find(r => r.model === 'good' && r.epoch === 1);
  assert.equal(first.runs, 6); // 3 splits x 2 seeds
  assert.ok(Math.abs(first.trainLoss - 0.9) < 1e-9 && Math.abs(first.valLoss - 1.0) < 1e-9);
  assert.ok(Math.abs(rows.find(r => r.model === 'good' && r.epoch === 2).valAcc - 0.6) < 1e-9);
  // runs without per-epoch validation values (such as learning-curve runs) are skipped
  assert.equal(curveRows({ runs: [{ model: 'm', n: 5, split: 0, seed: 0, history: [{ loss: 1 }] }] }).length, 0);
});

test('per-class rows: perfect model scores 1 everywhere, near-chance model does not', () => {
  const rows = perClassRows(results);
  const good = rows.find(r => r.model === 'good'), bad = rows.find(r => r.model === 'bad');
  for (const k of ['benign.precision', 'benign.recall', 'benign.f1', 'malignant.precision', 'malignant.recall', 'malignant.f1', 'macro.f1'])
    assert.equal(good[k].mean, 1);
  assert.ok(bad['macro.f1'].mean < 1);
  assert.equal(good.support.benign, 6); assert.equal(good.support.malignant, 6); // images per test set
});
