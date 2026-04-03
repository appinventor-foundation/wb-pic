'use strict';

const express = require('express');
const { v4: uuid } = require('uuid');
const sharp = require('sharp');
const unzipper = require('unzipper');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

const router = express.Router();

// In-memory job store: { [jobId]: { status, progress, message, modelFiles, error } }
const jobs = {};

// In-memory session store for imported datasets: { [sessionId]: { dir, classes } }
// classes: { [className]: [filename, ...] }
const sessions = {};

// In-memory import job store: { [importJobId]: { status, filesProcessed, totalFiles, sessionId?, imageMap?, error? } }
const importJobs = {};

const TARGET_SIZE = 224;
const SUPPORTED_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'webp', 'gif']);

// Cache transfer models by URL to avoid re-downloading when the same model is reused
const transferModelCache = {};

// ─── POST /api/import-zip ─────────────────────────────────────────────────────
// Accepts a raw ZIP file, extracts images server-side, returns imageMap URLs.
// Browser never decompresses anything — no beach ball.

router.post('/import-zip',
  express.raw({ type: ['application/zip', 'application/octet-stream'], limit: '2gb' }),
  async (req, res) => {
    if (!req.body || !req.body.length) {
      return res.status(400).json({ error: 'Empty body' });
    }

    const importJobId = uuid();
    importJobs[importJobId] = { status: 'running', filesProcessed: 0, totalFiles: 0 };
    console.log(`[${importJobId}] Import started — ${(req.body.length / 1024 / 1024).toFixed(1)} MB received`);
    res.json({ importJobId });

    // Run extraction async — do not await
    runImport(importJobId, req.body, req.protocol, req.get('host')).catch(err => {
      console.error(`[${importJobId}] Import error:`, err);
      importJobs[importJobId] = { status: 'error', error: err.message };
    });
  }
);

// ─── GET /api/import-status/:importJobId ──────────────────────────────────────

router.get('/import-status/:importJobId', (req, res) => {
  const job = importJobs[req.params.importJobId];
  if (!job) return res.status(404).json({ status: 'not_found' });
  res.json(job);
});

// ─── Import extraction logic ──────────────────────────────────────────────────

async function runImport(importJobId, buffer, protocol, host) {
  const sessionId = uuid();
  const sessionDir = path.join(os.tmpdir(), 'pic-sessions', sessionId);

  console.log(`[${importJobId}] Opening ZIP...`);
  const directory = await unzipper.Open.buffer(buffer);
  const classes = {}; // { className: [filename, ...] }

  // Detect a single wrapper folder (e.g. zipping a folder instead of its contents).
  // If all files share one top-level name AND some are nested 3+ levels deep,
  // skip that wrapper so the next level becomes the class name.
  const topLevelNames = new Set();
  let hasDeepFiles = false;
  for (const file of directory.files) {
    if (file.type === 'Directory') continue;
    const p = file.path.replace(/\\/g, '/').split('/');
    if (p[0] === '__MACOSX') continue;
    if (p.length < 2) continue;
    topLevelNames.add(p[0]);
    if (p.length >= 3) hasDeepFiles = true;
  }
  const pathOffset = (topLevelNames.size === 1 && hasDeepFiles) ? 1 : 0;

  // Count total image files first for accurate progress reporting
  const imageFiles = directory.files.filter(file => {
    if (file.type === 'Directory') return false;
    const parts = file.path.replace(/\\/g, '/').split('/');
    if (parts[0] === '__MACOSX') return false;
    if (parts.length < 2 + pathOffset) return false;
    const ext = parts[parts.length - 1].split('.').pop().toLowerCase();
    return SUPPORTED_EXTENSIONS.has(ext);
  });

  importJobs[importJobId].totalFiles = imageFiles.length;
  console.log(`[${importJobId}] Extracting ${imageFiles.length} images to ${sessionDir}`);
  let filesProcessed = 0;
  let jobs = [];

  for (const file of imageFiles) {
    const parts = file.path.replace(/\\/g, '/').split('/');
    const className = parts[pathOffset];
    const ext = parts[parts.length - 1].split('.').pop().toLowerCase();

    if (!classes[className]) classes[className] = [];

    const destDir = path.join(sessionDir, className);
    await fs.mkdir(destDir, { recursive: true });

    // Use a safe filename: index + original extension
    const index = classes[className].length;
    const filename = `${index}.${ext}`;
    const destPath = path.join(destDir, filename);

    jobs.push(new Promise(async (res, rej) => {
      try {
	const content = await file.buffer();
	await fs.writeFile(destPath, content);
	classes[className].push(filename);
	filesProcessed++;
	importJobs[importJobId].filesProcessed = filesProcessed;
	if (filesProcessed % 100 === 0) {
	  console.log(`[${importJobId}] ${filesProcessed}/${imageFiles.length} files extracted`);
	}
	res();
      } catch (e) {
	rej(e);
      }
    }));
  }

  console.log('Jobs: ' + jobs.length);
  await Promise.all(jobs);

  if (Object.keys(classes).length === 0) {
    importJobs[importJobId] = {
      status: 'error',
      error: 'No valid class folders found in ZIP. Expected: className/image.jpg'
    };
    return;
  }

  sessions[sessionId] = { dir: sessionDir, classes };

  // Build imageMap: { className: [serverUrl, ...] }
  const baseUrl = `${protocol}://${host}`;
  const imageMap = {};
  for (const [className, files] of Object.entries(classes)) {
    imageMap[className] = files.map(
      f => `${baseUrl}/api/images/${sessionId}/${encodeURIComponent(className)}/${f}`
    );
  }

  const classSummary = Object.entries(classes).map(([k, v]) => `${k}:${v.length}`).join(', ');
  console.log(`[${importJobId}] Done — ${filesProcessed} files, session ${sessionId} [${classSummary}]`);
  importJobs[importJobId] = { status: 'done', filesProcessed, totalFiles: filesProcessed, sessionId, imageMap };
}

// ─── GET /api/images/:sessionId/:className/:filename ──────────────────────────
// Serves individual images from an imported session.

router.get('/images/:sessionId/:className/:filename', (req, res) => {
  const { sessionId, className, filename } = req.params;
  const session = sessions[sessionId];
  if (!session) return res.status(404).send('Session not found');

  const filePath = path.join(session.dir, className, filename);
  res.sendFile(filePath, (err) => {
    if (err && !res.headersSent) res.status(404).send('Image not found');
  });
});

// ─── POST /api/train ───────────────────────────────────────────────────────────
// Accepts either:
//   { imageMap: { className: [dataUrl, ...] }, config }   ← base64 images
//   { sessionId, config }                                  ← reuse import session

router.post('/train', async (req, res) => {
  const { imageMap, sessionId, config = {} } = req.body;

  // Build resolvedImageMap: start with any browser-supplied imageMap, then
  // overlay session classes (file paths) so the server reads them from disk.
  // This supports mixed datasets: some classes from ZIP import, some added manually.
  const resolvedImageMap = Object.assign({}, imageMap);
  if (sessionId) {
    const session = sessions[sessionId];
    if (!session) return res.status(400).json({ error: 'Session not found. Import your dataset again.' });
    // File-path entries override same-named entries from imageMap (session is authoritative)
    for (const [className, files] of Object.entries(session.classes)) {
      resolvedImageMap[className] = files.map(f => path.join(session.dir, className, f));
    }
  }

  if (!resolvedImageMap || Object.keys(resolvedImageMap).length < 2) {
    return res.status(400).json({ error: 'imageMap must contain at least 2 classes' });
  }

  const jobId = uuid();
  jobs[jobId] = { status: 'running', progress: 0, message: 'Starting...' };
  res.json({ jobId });

  // Run training asynchronously — do not await
  runTraining(jobId, resolvedImageMap, config).catch((err) => {
    console.error(`[${jobId}] Unhandled error:`, err);
    jobs[jobId] = { status: 'error', error: err.message };
  });
});

// ─── GET /api/status/:jobId ───────────────────────────────────────────────────

router.get('/status/:jobId', (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).json({ status: 'not_found' });
  res.json(job);
});

// ─── Training logic ───────────────────────────────────────────────────────────

async function runTraining(jobId, imageMap, config) {
  const tf = global.tf;

  const notify = (progress, message) => {
    jobs[jobId] = { ...jobs[jobId], progress, message };
    console.log(`[${jobId}] ${progress}% — ${message}`);
  };

  try {
    // 1. Load transfer model (cached by URL to avoid re-downloading)
    const { name, truncationLayer, modelFormat = 'layers' } = config.transferModel || {};
    let { url } = config.transferModel || {};
    if (!url || !truncationLayer) {
      throw new Error('config.transferModel must include url and truncationLayer');
    }

    // For the local model the browser URL points at our own static route; the server
    // can load it much more efficiently straight from disk.
    if (name === 'mobilenet_local') {
      const localPath = path.resolve(__dirname, '../../../mobilenet/model.json');
      url = `file://${localPath}`;
    }

    notify(5, `Loading transfer model (${name || modelFormat})...`);
    if (!transferModelCache[url]) {
      if (modelFormat === 'graph') {
        transferModelCache[url] = await tf.loadGraphModel(url);
      } else {
        const fullModel = await tf.loadLayersModel(url);
        const layer = fullModel.getLayer(truncationLayer);
        transferModelCache[url] = tf.model({ inputs: fullModel.inputs, outputs: layer.output });
      }
      console.log(`[server] Transfer model loaded and cached: ${url}`);
    }
    const transferModel = transferModelCache[url];
    notify(20, 'Transfer model ready');

    // 2. Process images: data URL / file path → tensor, extract features immediately
    const labels = Object.keys(imageMap).sort();
    const numClasses = labels.length;

    // Count total first for accurate progress reporting
    let totalImages = 0;
    for (const label of labels) totalImages += imageMap[label].length;
    notify(25, `Processing ${totalImages} images...`);

    const labelVectors = [];
    const activationList = [];
    let processed = 0;

    // 3. Extract MobileNet features — dispose each image tensor immediately to halve peak RAM
    for (const label of labels) {
      const oneHot = labels.map(l => (l === label ? 1 : 0));
      for (const input of imageMap[label]) {
        const imgTensor = await loadImageTensor(tf, input);
        const batched = tf.expandDims(imgTensor, 0);
        const activation = modelFormat === 'graph'
          ? transferModel.execute(batched, truncationLayer)
          : transferModel.predict(batched);
        activationList.push(tf.squeeze(activation, [0]));
        batched.dispose();
        activation.dispose();
        imgTensor.dispose(); // freed immediately — peak RAM is just activations
        labelVectors.push(oneHot);
        processed++;
        if (processed % 50 === 0) {
          const pct = 25 + Math.round((processed / totalImages) * 20);
          notify(pct, `Extracted features: ${processed}/${totalImages}`);
          await new Promise(r => setImmediate(r)); // yield to HTTP polling
        }
      }
    }

    notify(45, `Feature extraction complete (${totalImages} images)`);

    const activations = tf.stack(activationList);
    const labelsTensor = tf.tensor2d(labelVectors, [totalImages, numClasses]);

    notify(55, 'Building custom model...');

    // 4. Build custom classification head
    const [, h, w, depth] = activations.shape; // [N, 7, 7, 256]
    const customModel = buildCustomModel(tf, h, w, depth, numClasses, config);

    // 5. Train
    const epochs = config.epochs || 20;
    const batchSizeFraction = config.batchSizeFraction || 0.4;
    const batchSize = Math.max(1, Math.floor(totalImages * batchSizeFraction));

    notify(65, 'Training...');

    await customModel.fit(activations, labelsTensor, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const epochProgress = 65 + Math.round(((epoch + 1) / epochs) * 30);
          notify(epochProgress,
            `Epoch ${epoch + 1}/${epochs} — loss: ${logs.loss.toFixed(5)}`
          );
          // Yield to the Node.js event loop so the HTTP server can respond
          // to polling requests from the browser between epochs.
          await new Promise(r => setImmediate(r));
        }
      }
    });

    activations.dispose();
    labelsTensor.dispose();

    notify(96, 'Serializing model...');

    // 6. Capture model artifacts
    let capturedArtifacts = null;
    await customModel.save({
      save: async (artifacts) => {
        capturedArtifacts = artifacts;
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
      }
    });

    customModel.dispose();

    const weightDataBase64 = Buffer.from(capturedArtifacts.weightData).toString('base64');

    const modelFiles = {
      modelTopology: capturedArtifacts.modelTopology,
      weightSpecs: capturedArtifacts.weightSpecs,
      weightDataBase64,
      labelIndex: labels
    };

    notify(100, 'Done');
    jobs[jobId] = { status: 'done', progress: 100, message: 'Training complete', modelFiles };
  } catch (err) {
    console.error(`[${jobId}] Training error:`, err);
    jobs[jobId] = { status: 'error', progress: 0, message: err.message, error: err.message };
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function loadImageTensor(tf, input) {
  let buffer;

  if (typeof input === 'string' && input.startsWith('data:')) {
    // Base64 data URL
    const base64Data = input.split(',')[1];
    buffer = Buffer.from(base64Data, 'base64');
  } else {
    // File path (from session import)
    buffer = await fs.readFile(input);
  }

  // Decode with sharp → raw RGB
  const { data, info } = await sharp(buffer)
    .resize(TARGET_SIZE, TARGET_SIZE)
    .toColorspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixelData = new Uint8Array(data);
  const raw = tf.tensor3d(pixelData, [info.height, info.width, 3], 'int32');

  // Normalize: (x / 127) - 1 → [-1, 1]
  const normalized = raw.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  raw.dispose();
  return normalized;
}

function buildCustomModel(tf, h, w, depth, numClasses, config) {
  const learningRate = config.learningRate || 0.0001;
  const optimizerType = config.optimizer || 'adam';

  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [h, w, depth],
    filters: 5,
    kernelSize: 5,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: 100,
    activation: 'relu',
    kernelInitializer: 'varianceScaling',
    useBias: true
  }));
  model.add(tf.layers.dense({
    units: numClasses,
    activation: 'softmax',
    kernelInitializer: 'varianceScaling',
    useBias: false
  }));

  const optimizer = optimizerType === 'sgd'
    ? tf.train.sgd(learningRate)
    : tf.train.adam(learningRate);

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

module.exports = router;
