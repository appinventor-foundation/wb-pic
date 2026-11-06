# Benchmark: MobileNet v1 vs v3 Small

Reproducible benchmark of the two transfer models the app ships, run **in the browser** with the app's own code
(`src/ml`): same preprocessing, same frozen-backbone + conv-head training, same WebGL backend. Written as a
[Notebook Kit](https://observablehq.github.io/notebook-kit/) notebook on top of a plain-JS core.

Not a clinical evaluation. It measures a teaching tool.

## Run it

```bash
npm install --legacy-peer-deps      # once (react-mic has a pre-existing React peer conflict)
DATASET_DIR=/path/to/flat_benign_vs_malignant npm run benchmark
```

Open <http://localhost:5173/>, keep the tab in the foreground, and press *Run benchmark*. The run uses all
images with 5-fold cross-validation and 3 head seeds, and includes the learning curve: 54 head-training runs on top
of one feature extraction per model.

To render a saved run without re-running, open `http://localhost:5173/?results=<file>.json` (files in
`benchmark/results/`).

The dataset folder must contain `benign/` and `malignant/` image folders. The dev server serves it at `/dataset`
(`vite.js`); the models are served from `public/models` (`npm run benchmark` runs `copy-models` first).

Results are written to `benchmark/results/benchmark-<date>.json`. The
notebook's report cells read that JSON, so a report can be regenerated without re-running anything.

Tests for the pure-JS core (splits, metrics, aggregation): `npm run benchmark:test`.

## What is measured

- **Quality** (5-fold stratified CV, folds shared by both models): accuracy, balanced accuracy, sensitivity
  (malignant recall), specificity, precision, F1, ROC-AUC, PR-AUC, sensitivity at 90/95% specificity, Brier, ECE.
  Mean ± std across folds; head seeds are averaged within a fold.
- **Per class:** precision, recall and F1 with benign and with malignant as the positive class, plus macro averages
  (`perClassRows`, computed from the stored out-of-fold scores), and pooled confusion matrices.
- **Training curves:** training loss and accuracy per epoch (no validation set in the app's training).
- **Training:** the app's own settings for both models (20 epochs, Adam at learning rate 1e-4, batch size 40% of the
  training set). There are no other training variants.
- **Data efficiency:** learning curve over training images per class.
- **Speed and size:** model load, download size, parameters in the downloaded files, single-image latency
  (backbone only and end to end, p50/p95 after warm-up), images per second, GPU memory after load.
- **Environment:** tfjs version, backend, GPU renderer, browser, git commit, config hash, file-list hash.

## How it works

```
core/dataset.js   manifest -> samples; class order fixed, malignant = positive (label 1)
core/extract.js   streams images through the frozen backbone one at a time (DataProcessor.convertImage +
                  the same truncation as ImageClassifier); features kept on the CPU, cached in IndexedDB
core/head.js      the app's conv head via its ModelBuilder, seeded init and epoch order
core/splits.js    seeded stratified k-fold and nested per-class subsampling
core/metrics.js   metrics, ROC, calibration (all pure)
core/aggregate.js results JSON -> report tables (pure)
core/speed.js     latency / size benchmark
core/bench.js     orchestration, checkpointing, results schema
notebooks/index.html   run + report
vite.js, serve.js   dev server (adds /dataset, /models, /results, /env.json)
```

Extraction is the expensive step and the backbone is frozen, so features are extracted once per model and every fold,
seed and learning-curve size trains only the small head. Finished runs and partial extraction are
checkpointed in IndexedDB, so an interrupted run resumes when you reload the page and press *Run benchmark*. Changing the code that
produces a run record means bumping `RUN_VERSION` in `core/bench.js` (and `PREPROCESSING_VERSION` in
`core/extract.js` for preprocessing) so stale checkpoints are not reused.

## Reproducibility notes

- Splits, head initialisation and per-epoch shuffling are seeded. `fit()`'s own shuffle and the string
  initialisers in `defaults.js` are not seeded, so the benchmark replaces them with seeded equivalents. Hyperparameters
  are otherwise the app's.
- WebGL numerics can differ slightly across GPUs and drivers, so exact metrics may move in the last digits.
  Timing depends on the machine, browser and tab state, so keep the tab fronted and compare numbers only within a run.
- `WEBGL_DELETE_TEXTURE_THRESHOLD` is capped (`core/tf.js`). The dataset has ~300 distinct image sizes (up to
  6000×4000), and with the tfjs default every new size left a texture in the pool until extraction slowed and stalled.
  The app has the same exposure when users upload large images of many sizes.
- The notebook's Plot/Inputs/htl libraries are loaded unpinned from a CDN by Notebook Kit. They affect charts only.
  tfjs comes from `package-lock.json`.

## Caveats

- The images appear to be ISIC files: several images can come from one lesion or patient, and duplicates exist. Splits
  are per image, so scores may be optimistic. Without lesion/patient metadata this is not controlled.
- The app's "v1" is the 0.25-width network (7×7×256 features); v3 Small is width 1.0 (7×7×576). The comparison mixes
  architecture and capacity.
- Preprocessing is the app's: centre-crop to square, then resize to 224×224.
