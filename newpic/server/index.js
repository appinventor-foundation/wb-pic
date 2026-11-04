'use strict';

// tfjs-node 4.x still calls util helpers that Node 23 removed
// (e.g. "util.isNullOrUndefined is not a function" when training starts).
// Restore them before TensorFlow loads so the server also runs on newer Node.
const util = require('util');
const legacyUtilHelpers = {
  isArray: Array.isArray,
  isBoolean: v => typeof v === 'boolean',
  isNull: v => v === null,
  isNullOrUndefined: v => v === null || v === undefined,
  isNumber: v => typeof v === 'number',
  isString: v => typeof v === 'string',
  isSymbol: v => typeof v === 'symbol',
  isUndefined: v => v === undefined,
  isObject: v => v !== null && typeof v === 'object',
  isFunction: v => typeof v === 'function',
  isRegExp: v => v instanceof RegExp,
  isDate: v => v instanceof Date,
  isError: v => v instanceof Error,
  isPrimitive: v => v === null || (typeof v !== 'object' && typeof v !== 'function')
};
for (const [name, helper] of Object.entries(legacyUtilHelpers)) {
  if (typeof util[name] !== 'function') util[name] = helper;
}

// GPU/CPU auto-detection — must happen before any tf usage.
// @tensorflow/tfjs-node-gpu installs successfully on Mac (uses Apple Metal / CPU)
// and on Linux with CUDA. We load it first and let TF pick the best device.
// Check GET /health for the actual backend TF chose at runtime.
let tf;
let loadedPackage;
try {
  tf = require('@tensorflow/tfjs-node-gpu');
  loadedPackage = 'tfjs-node-gpu';
} catch (e) {
  try {
    tf = require('@tensorflow/tfjs-node');
    loadedPackage = 'tfjs-node';
  } catch (e2) {
    console.error('Neither tfjs-node-gpu nor tfjs-node could be loaded:', e2.message);
    process.exit(1);
  }
}

// Make tf available globally so route modules can access it without re-requiring
global.tf = tf;

const express = require('express');
const cors = require('cors');
const path = require('path');
const trainRouter = require('./routes/train');

const app = express();

app.use(cors());
app.enable('trust proxy');
app.use(express.json({ limit: '2gb' }));
app.use('/mobilenet_v3', express.static(path.resolve(__dirname, '../../mobilenetv3')));
app.use('/mobilenet_v2', express.static(path.resolve(__dirname, '../../mobilenetv2')));
app.use('/mobilenet', express.static(path.resolve(__dirname, '../../mobilenet')));
app.use('/api', trainRouter);

app.get('/health', (_req, res) => {
  res.json({ ok: true, backend: tf.getBackend() });
});

const PORT = process.env.PORT || 3100;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`  Package:  ${loadedPackage}`);
  console.log(`  Backend:  ${tf.getBackend()} (check /health for live info)`);
});
