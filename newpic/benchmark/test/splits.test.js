import test from 'node:test';
import assert from 'node:assert/strict';
import { stratifiedKFold, subsamplePerClass } from '../core/splits.js';
import { hashStrings } from '../core/rng.js';

const labels = [...Array(40).fill(0), ...Array(60).fill(1)];

test('folds partition the data and keep the class ratio', () => {
  const folds = stratifiedKFold(labels, 5, 1);
  const seen = folds.flatMap(f => f.test).sort((a, b) => a - b);
  assert.deepEqual(seen, [...labels.keys()]);
  for (const f of folds) {
    assert.equal(f.train.length + f.test.length, 100);
    assert.equal(new Set([...f.train, ...f.test]).size, 100);
    assert.equal(f.test.filter(i => labels[i] === 1).length, 12);
  }
});

test('same seed gives same folds, different seed differs', () => {
  assert.deepEqual(stratifiedKFold(labels, 5, 1), stratifiedKFold(labels, 5, 1));
  assert.notDeepEqual(stratifiedKFold(labels, 5, 1), stratifiedKFold(labels, 5, 2));
});

test('subsample is per class, nested and within the given indices', () => {
  const train = stratifiedKFold(labels, 5, 1)[0].train;
  const small = subsamplePerClass(train, labels, 5, 3);
  const big = subsamplePerClass(train, labels, 10, 3);
  assert.equal(small.filter(i => labels[i] === 0).length, 5);
  assert.equal(small.filter(i => labels[i] === 1).length, 5);
  assert.ok(small.every(i => big.includes(i)));
  assert.ok(big.every(i => train.includes(i)));
});

test('hashStrings is stable and order sensitive', () => {
  assert.equal(hashStrings(['a', 'b']), hashStrings(['a', 'b']));
  assert.notEqual(hashStrings(['a', 'b']), hashStrings(['b', 'a']));
});
