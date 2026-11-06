import { shuffled } from './rng.js';

const byClass = (labels) => {
  const groups = new Map();
  labels.forEach((y, i) => {
    if (!groups.has(y)) groups.set(y, []);
    groups.get(y).push(i);
  });
  return [...groups.entries()].sort((a, b) => a[0] - b[0]).map(([, idx]) => idx);
};

// Stratified k-fold: each class is shuffled with the seed and dealt round-robin, so every
// fold keeps the class ratio. Returns [{ train: number[], test: number[] }] with sorted indices.
export function stratifiedKFold(labels, k, seed) {
  const folds = Array.from({ length: k }, () => []);
  byClass(labels).forEach((idx, c) => {
    shuffled(idx, seed * 1009 + c).forEach((sample, n) => folds[n % k].push(sample));
  });
  return folds.map((test, f) => {
    const testSet = new Set(test);
    const train = [];
    for (let i = 0; i < labels.length; i++) if (!testSet.has(i)) train.push(i);
    return { train, test: test.sort((a, b) => a - b) };
  });
}

// Up to nPerClass samples of each class from `indices` (seeded); used by the learning curve.
// Nested: a smaller n is always a prefix of a larger one for the same seed.
export function subsamplePerClass(indices, labels, nPerClass, seed) {
  const groups = byClass(indices.map(i => labels[i])).map(g => g.map(p => indices[p]));
  return groups
    .flatMap((idx, c) => shuffled(idx, seed * 7919 + c).slice(0, nPerClass))
    .sort((a, b) => a - b);
}
