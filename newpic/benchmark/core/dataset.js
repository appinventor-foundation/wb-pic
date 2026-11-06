import { hashStrings } from './rng.js';

// Class order is fixed so label 1 is always the positive class (malignant).
export const CLASS_ORDER = ['benign', 'malignant'];
export const POSITIVE = 'malignant';

// Reads /dataset/manifest.json (served from DATASET_DIR, see vite.js) and uses every image in both class folders.
export async function loadDataset({ baseUrl = '/dataset' } = {}) {
  const manifest = await (await fetch(`${baseUrl}/manifest.json`)).json();
  const missing = CLASS_ORDER.filter(c => !manifest.classes[c]);
  if (missing.length) throw new Error(`dataset is missing class folders: ${missing.join(', ')}`);
  const samples = [];
  CLASS_ORDER.forEach((label, y) => {
    for (const file of manifest.classes[label]) samples.push({ url: `${baseUrl}/${label}/${file}`, id: `${label}/${file}`, label, y });
  });
  return {
    samples,
    labels: samples.map(s => s.y),
    counts: Object.fromEntries(CLASS_ORDER.map(c => [c, samples.filter(s => s.label === c).length])),
    fileListHash: hashStrings(samples.map(s => s.id))
  };
}
