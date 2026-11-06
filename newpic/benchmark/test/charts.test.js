import test from 'node:test';
import assert from 'node:assert/strict';
import { confusionMatrixSvg, rocSvg, trainingLossSvg, trainingAccuracySvg, allCharts, lineChartSvg, escapeXml, modelLabel } from '../core/charts.js';
import { stratifiedKFold } from '../core/splits.js';
import { allMetrics, tprOnGrid } from '../core/metrics.js';

// Synthetic results: "good" separates the classes, "bad" is near chance (same shape as a saved run).
const labels = Array.from({ length: 60 }, (_, i) => (i < 30 ? 0 : 1));
const folds = stratifiedKFold(labels, 3, 1);
const scoreFor = (model, y, i) => (model === 'good' ? (y ? 0.85 : 0.15) : (i % 3 === 0 ? 1 - (y ? 0.6 : 0.4) : (y ? 0.6 : 0.4)));
const runs = [];
for (const model of ['good', 'bad'])
  for (let fold = 0; fold < 3; fold++) {
    const test = folds[fold].test, y = test.map(i => labels[i]);
    const scores = test.map(idx => scoreFor(model, labels[idx], idx));
    runs.push({ model, cond: 'c', fold, seed: 0, n: 'all', nTrainPerClass: 20, metrics: allMetrics(scores, y), tpr: tprOnGrid(scores, y), trainMs: 1, scores,
      history: Array.from({ length: 10 }, (_, e) => ({ loss: 0.8 - 0.05 * e, acc: 0.5 + 0.04 * e })) });
  }
const results = { config: { models: ['good', 'bad'], headSeeds: [0], conditions: { c: {} }, learningCurve: { folds: [], seeds: 0, conditions: [], sizes: [] } },
  dataset: { labels }, folds: folds.map(f => ({ test: f.test })), runs };

test('every chart is a standalone svg document with a white background', () => {
  for (const [name, svg] of Object.entries(allCharts(results))) {
    assert.match(svg, /^<svg xmlns="http:\/\/www\.w3\.org\/2000\/svg" width="\d+" height="\d+"/, name);
    assert.ok(svg.trimEnd().endsWith('</svg>'), name);
    assert.ok(svg.includes('fill="#ffffff"'), name);
  }
  assert.deepEqual(Object.keys(allCharts(results)), ['confusion-matrix', 'roc-curve', 'training-loss', 'training-accuracy']);
});

test('confusion matrix prints the pooled counts for each model', () => {
  const svg = confusionMatrixSvg(results);
  assert.equal((svg.match(/>30</g) || []).length, 2);              // the perfect model: 30 benign and 30 malignant correct
  assert.ok(svg.includes('100% of row'));
  assert.ok(svg.includes('predicted malignant') && svg.includes('actual benign'));
});

test('ROC chart draws one curve per model plus the chance line, with AUC in the legend', () => {
  const svg = rocSvg(results);
  assert.equal((svg.match(/<polyline/g) || []).length, 2);
  assert.ok(svg.includes('stroke-dasharray'));
  assert.match(svg, /\(AUC 1\.000\)/);
});

test('training charts have one line per model and one point per epoch', () => {
  for (const svg of [trainingLossSvg(results), trainingAccuracySvg(results)]) {
    assert.equal((svg.match(/<polyline/g) || []).length, 2);
    assert.equal((svg.match(/<circle/g) || []).length, 20);
    assert.ok(svg.includes('epoch'));
  }
});

test('labels are XML-escaped and unknown models fall back to their key', () => {
  assert.equal(escapeXml('a<b & "c"'), 'a&lt;b &amp; &quot;c&quot;');
  assert.equal(modelLabel('mobilenet_local'), 'MobileNet v1');
  assert.equal(modelLabel('something_else'), 'something_else');
  const svg = lineChartSvg({ title: 'x < y', xLabel: 'x', yLabel: 'y', series: [{ label: 'a & b', color: '#000', points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }], xDomain: [0, 1], yDomain: [0, 1], xTicks: [0, 1], yTicks: [0, 1] });
  assert.ok(svg.includes('x &lt; y') && svg.includes('a &amp; b'));
});
