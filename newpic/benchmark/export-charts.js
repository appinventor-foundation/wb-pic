// Writes the result charts (confusion matrix, ROC curve, training loss, training accuracy) as SVG and PNG files
// from a saved results file.
//   node benchmark/export-charts.js [results.json] [outDir]
// Defaults: the newest benchmark-*.json in benchmark/results/, written to benchmark/results/charts/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { allCharts } from './core/charts.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const resultsDir = path.join(here, 'results');

function newestResults() {
  const files = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir).filter(f => /^benchmark-.*\.json$/.test(f)).sort() : [];
  if (!files.length) throw new Error(`no benchmark-*.json in ${resultsDir}; pass a results file as the first argument`);
  return path.join(resultsDir, files.at(-1));
}

const file = process.argv[2] ? path.resolve(process.argv[2]) : newestResults();
const outDir = process.argv[3] ? path.resolve(process.argv[3]) : path.join(resultsDir, 'charts');
const results = JSON.parse(fs.readFileSync(file, 'utf8'));
fs.mkdirSync(outDir, { recursive: true });

let sharp = null;
try { sharp = (await import('sharp')).default; } catch { console.warn('sharp is not installed: writing SVG only (npm install --legacy-peer-deps adds it)'); }

for (const [name, svg] of Object.entries(allCharts(results))) {
  fs.writeFileSync(path.join(outDir, `${name}.svg`), svg);
  if (sharp) await sharp(Buffer.from(svg), { density: 144 }).png().toFile(path.join(outDir, `${name}.png`));
  console.log(`wrote ${path.relative(process.cwd(), path.join(outDir, name))}.${sharp ? '{svg,png}' : 'svg'}`);
}
console.log(`from ${path.relative(process.cwd(), file)}`);
