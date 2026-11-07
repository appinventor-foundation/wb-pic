import test from 'node:test';
import assert from 'node:assert/strict';
import * as m from '../core/metrics.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('perfect and inverted rankings', () => {
  const y = [0, 0, 1, 1];
  close(m.rocAuc([0.1, 0.2, 0.8, 0.9], y), 1);
  close(m.rocAuc([0.9, 0.8, 0.2, 0.1], y), 0);
  close(m.averagePrecision([0.1, 0.2, 0.8, 0.9], y), 1);
});

test('AUC with ties equals the Mann-Whitney value', () => {
  // positives {0.5, 0.8}, negatives {0.5, 0.2}: pairs win=3, tie=1 of 4 -> 0.875
  close(m.rocAuc([0.5, 0.2, 0.5, 0.8], [0, 0, 1, 1]), 0.875);
});

test('average precision matches a hand computation', () => {
  // ranking by score: pos, neg, pos -> precision at the positives 1 and 2/3, recall steps of 1/2
  close(m.averagePrecision([0.9, 0.8, 0.7], [1, 0, 1]), 0.5 * 1 + 0.5 * (2 / 3));
});

test('classification metrics from a known confusion matrix', () => {
  // tp=2 fp=1 tn=1 fn=0 at 0.5
  const s = [0.9, 0.6, 0.7, 0.2], y = [1, 0, 1, 0];
  assert.deepEqual(m.confusion(s, y), { tp: 2, fp: 1, tn: 1, fn: 0 });
  const r = m.classificationMetrics(s, y);
  close(r.accuracy, 0.75); close(r.recall, 1); close(r.specificity, 0.5); close(r.precision, 2 / 3);
  close(r.f1, 0.8);
});

test('sensitivity at specificity', () => {
  const y = [0, 0, 0, 0, 1, 1, 1, 1];
  const s = [0.1, 0.2, 0.3, 0.6, 0.5, 0.7, 0.8, 0.9];
  // one false positive allowed (0.6) -> threshold 0.5 catches every positive
  close(m.sensitivityAtSpecificity(s, y, 0.75), 1);
  // no false positive allowed -> only scores above 0.6 count: 0.7, 0.8, 0.9
  close(m.sensitivityAtSpecificity(s, y, 1), 0.75);
});

test('tprOnGrid is monotone and ends at 1; g[0] is the TPR at zero false positives', () => {
  const g = m.tprOnGrid([0.9, 0.4, 0.6, 0.3, 0.8], [1, 0, 1, 0, 0]);
  assert.equal(g.length, 101);
  close(g[0], 0.5); close(g[100], 1); // 0.9 is the only positive above the top negative (0.8)
  for (let i = 1; i < g.length; i++) assert.ok(g[i] >= g[i - 1] - 1e-12);
});

test('brier and ece', () => {
  close(m.brier([1, 0], [1, 0]), 0);
  close(m.brier([0.5, 0.5], [1, 0]), 0.25);
  close(m.expectedCalibrationError([1, 0, 1, 0], [1, 0, 1, 0]), 0);
  close(m.expectedCalibrationError([0.9, 0.9], [1, 0]), 0.4); // conf 0.9, accuracy 0.5
});

test('mean, std, quantile', () => {
  close(m.mean([1, 2, 3]), 2); close(m.std([1, 2, 3]), 1); close(m.quantile([1, 2, 3, 4], 0.5), 2.5);
});

test('per-class metrics: each class as positive, macro average', () => {
  // classes: 0,0,0,1,1 ; predictions at 0.5: 0,0,1,1,0  -> class 0: tp=2 fp=1 fn=1 ; class 1: tp=1 fp=1 fn=1
  const y = [0, 0, 0, 1, 1], s = [0.1, 0.2, 0.7, 0.9, 0.3];
  const r = m.perClassMetrics(s, y);
  close(r.benign.precision, 2 / 3); close(r.benign.recall, 2 / 3); close(r.benign.f1, 2 / 3);
  close(r.malignant.precision, 1 / 2); close(r.malignant.recall, 1 / 2); close(r.malignant.f1, 1 / 2);
  assert.equal(r.benign.support, 3); assert.equal(r.malignant.support, 2);
  close(r.macro.f1, (2 / 3 + 1 / 2) / 2);
  // the malignant view equals the plain classification metrics
  const base = m.classificationMetrics(s, y);
  close(r.malignant.precision, base.precision); close(r.malignant.recall, base.recall); close(r.malignant.f1, base.f1);
  // benign recall is the plain specificity
  close(r.benign.recall, base.specificity);
});

test('log loss: chance is ln 2, confident and right is near 0, confident and wrong is large', () => {
  close(m.logLoss([0.5, 0.5], [1, 0]), Math.LN2);
  assert.ok(m.logLoss([0.999, 0.001], [1, 0]) < 0.01);
  assert.ok(m.logLoss([0.001, 0.999], [1, 0]) > 5);
  assert.ok(Number.isFinite(m.logLoss([0, 1], [1, 0]))); // clipped, so even a certain mistake is finite
});
