import { shuffled } from './rng.js';

const byClass = (labels) => {
  const groups = new Map();
  labels.forEach((y, i) => {
    if (!groups.has(y)) groups.set(y, []);
    groups.get(y).push(i);
  });
  return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, idx]) => idx);
};

// Up to nPerClass samples of each class from `indices` (seeded); used by the learning curve.
// Nested: a smaller n is always a prefix of a larger one for the same seed.
export function subsamplePerClass(indices, labels, nPerClass, seed) {
  const groups = byClass(indices.map(i => labels[i])).map(g => g.map(p => indices[p]));
  return groups
    .flatMap((idx, c) => shuffled(idx, seed * 7919 + c).slice(0, nPerClass))
    .sort((a, b) => a - b);
}

// Stratified train/validation/test split: each class is shuffled with the seed and cut in the given proportions, so every part
// keeps the class ratio. `validation` and `test` are fractions; train is the remainder. Returns sorted index arrays.
export function stratifiedSplit(labels, { validation, test }, seed) {
  const parts = { train: [], validation: [], test: [] };
  byClass(labels).forEach((idx, c) => {
    const order = shuffled(idx, seed * 1013 + c);
    const nTest = Math.round(idx.length * test), nVal = Math.round(idx.length * validation);
    parts.test.push(...order.slice(0, nTest));
    parts.validation.push(...order.slice(nTest, nTest + nVal));
    parts.train.push(...order.slice(nTest + nVal));
  });
  for (const k of Object.keys(parts)) parts[k].sort((a, b) => a - b);
  return parts;
}
