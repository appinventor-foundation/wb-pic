import test from 'node:test';
import assert from 'node:assert/strict';
import { subsamplePerClass, stratifiedSplit } from '../core/splits.js';
import { hashStrings } from '../core/rng.js';

const labels = [...Array(40).fill(0), ...Array(60).fill(1)];

test('subsample is per class, nested and within the given indices', () => {
  const train = stratifiedSplit(labels, { validation: 0.15, test: 0.15 }, 1).train;
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

test('train/validation/test split: disjoint, complete, in proportion, stratified and seeded', () => {
  const p = stratifiedSplit(labels, { validation: 0.15, test: 0.15 }, 1);
  assert.deepEqual([p.train.length, p.validation.length, p.test.length], [70, 15, 15]);
  const all = [...p.train, ...p.validation, ...p.test].sort((a, b) => a - b);
  assert.deepEqual(all, [...labels.keys()]);                                   // every image in exactly one part
  for (const part of [p.train, p.validation, p.test]) {
    const malignant = part.filter(i => labels[i] === 1).length;
    assert.ok(Math.abs(malignant / part.length - 0.6) < 0.05);                 // class ratio kept (60% class 1)
  }
  assert.deepEqual(stratifiedSplit(labels, { validation: 0.15, test: 0.15 }, 1), p);
  assert.notDeepEqual(stratifiedSplit(labels, { validation: 0.15, test: 0.15 }, 2).test, p.test);
});
