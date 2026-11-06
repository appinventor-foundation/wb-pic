// Shared Vite setup for the benchmark notebook (the `notebooks` CLI takes no Vite config,
// so preview/build go through serve.js and build.js).
import fs from 'node:fs';
import { execSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mergeConfig } from 'vite';
import { config, observable } from '@observablehq/notebook-kit/vite';

const here = path.dirname(fileURLToPath(import.meta.url));
export const appRoot = path.resolve(here, '..');
export const notebooksRoot = path.join(here, 'notebooks');
const IMAGE = /\.(jpe?g|png)$/i;
export const resultsDir = path.join(here, 'results');
const RESULT_NAME = /^[\w.-]+\.json$/;
const git = (cmd) => { try { return execSync(`git ${cmd}`, { cwd: appRoot }).toString().trim(); } catch { return null; } };

// Serves the benchmark inputs next to the notebook:
//   /models/...             newpic/public/models (npm run copy-models)
//   /dataset/manifest.json  { classes: { <label>: [file, ...] } } read from DATASET_DIR
//   /dataset/<label>/<file> the images
function benchmarkAssets({ datasetDir }) {
  const modelsDir = path.join(appRoot, 'public', 'models');
  const send = (res, file) => {
    res.setHeader('Content-Type', file.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
  };
  const inside = (root, rel) => {
    const full = path.resolve(root, rel);
    return full.startsWith(root + path.sep) && fs.existsSync(full) && fs.statSync(full).isFile() ? full : null;
  };
  return {
    name: 'benchmark-assets',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = decodeURIComponent((req.url || '').split('?')[0]);
        if (url === '/env.json') {
          res.setHeader('Content-Type', 'application/json');
          return res.end(JSON.stringify({ gitCommit: git('rev-parse HEAD'), gitDirty: !!git('status --porcelain -- ../benchmark ../src/ml'), serverNode: process.version }));
        }
        // /results/: GET list.json | GET <name>.json | POST <name>.json (saved into benchmark/results/)
        if (url.startsWith('/results/')) {
          const name = url.slice('/results/'.length);
          if (name === 'list.json') {
            res.setHeader('Content-Type', 'application/json');
            return res.end(JSON.stringify(fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir).filter(f => RESULT_NAME.test(f)) : []));
          }
          if (!RESULT_NAME.test(name)) return ((res.statusCode = 400), res.end('bad name'));
          if (req.method === 'POST') {
            const chunks = [];
            req.on('data', c => chunks.push(c));
            req.on('end', () => { fs.mkdirSync(resultsDir, { recursive: true }); fs.writeFileSync(path.join(resultsDir, name), Buffer.concat(chunks)); res.end('saved'); });
            return;
          }
          const file = inside(resultsDir, name);
          return file ? send(res, file) : ((res.statusCode = 404), res.end('no such results file'));
        }
        if (url.startsWith('/models/')) {
          const file = inside(modelsDir, url.slice('/models/'.length));
          return file ? send(res, file) : ((res.statusCode = 404), res.end('model not found (run npm run copy-models)'));
        }
        if (url.startsWith('/dataset/')) {
          if (!datasetDir) return ((res.statusCode = 404), res.end('set DATASET_DIR'));
          if (url === '/dataset/manifest.json') {
            const classes = {};
            for (const entry of fs.readdirSync(datasetDir, { withFileTypes: true }).filter(e => e.isDirectory())) {
              classes[entry.name] = fs.readdirSync(path.join(datasetDir, entry.name)).filter(f => IMAGE.test(f)).sort();
            }
            res.setHeader('Content-Type', 'application/json');
            return res.end(JSON.stringify({ classes }));
          }
          const file = inside(path.resolve(datasetDir), url.slice('/dataset/'.length));
          return file ? send(res, file) : ((res.statusCode = 404), res.end('not found'));
        }
        next();
      });
    }
  };
}

export function viteConfig({ datasetDir = process.env.DATASET_DIR } = {}) {
  return mergeConfig(config(), {
    root: notebooksRoot,
    plugins: [observable(), benchmarkAssets({ datasetDir })],
    // src/ml reads process.env through src/config/runtime.js (inlined by CRA in the app).
    define: {
      'process.env.PUBLIC_URL': '""',
      'process.env.REACT_APP_MODELS_URL': '"/models"',
      'process.env.REACT_APP_SERVER_URL': '""'
    },
    server: { fs: { allow: [appRoot] } }
  });
}
