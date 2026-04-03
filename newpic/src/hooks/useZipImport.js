import { useCallback } from 'react';
import JSZip from 'jszip';

const SUPPORTED_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'gif', 'webp']);
const MIME_MAP = {
  jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
  gif: 'image/gif', webp: 'image/webp'
};
const MAX_IMAGES_PER_CLASS = 300;

// Yield to the browser event loop so the UI stays responsive between extractions
const yieldToMain = () => new Promise(r => setTimeout(r, 0));

/**
 * Custom hook for importing a ZIP dataset.
 *
 * When serverUrl is provided, the raw ZIP is uploaded to the server which
 * extracts it there — no browser decompression, no beach ball.
 *
 * Falls back to in-browser JSZip when serverUrl is not set (small datasets only).
 *
 * Expected ZIP structure:
 *   className1/image1.jpg
 *   className2/image1.jpg
 *   ...
 *
 * @param {Function} onImport   - Called with { className: [url, ...] }
 * @param {Function} onStart    - Called when import begins (optional)
 * @param {Function} onComplete - Called when import finishes (optional)
 * @param {string}   serverUrl  - If set, upload ZIP to server for extraction
 * @param {Function} onProgress - Called with progress info (optional)
 */
export const useZipImport = (onImport, onStart, onComplete, serverUrl, onProgress) => {
  const handleZipImport = useCallback(async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';

    onStart?.();

    if (serverUrl) {
      await importViaServer(file, serverUrl, onImport, onComplete, onProgress);
    } else {
      await importInBrowser(file, onImport, onComplete, onProgress);
    }
  }, [onImport, onStart, onComplete, serverUrl, onProgress]);

  return { handleZipImport };
};

// ─── Server-side import ───────────────────────────────────────────────────────
// Streams the raw ZIP to the server; server extracts and returns image URLs.

async function importViaServer(file, serverUrl, onImport, onComplete, onProgress) {
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        onProgress?.({ phase: 'upload', uploadPct: pct, filename: file.name, fileSize: file.size });
      }
    };

    xhr.upload.onload = () => {
      onProgress?.({ phase: 'extract', done: 0, total: 0, filename: file.name });
    };

    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        alert(`Import failed: Server error ${xhr.status}: ${xhr.responseText}`);
        onComplete?.(); resolve(); return;
      }
      let result;
      try { result = JSON.parse(xhr.responseText); }
      catch { alert('Import failed: Invalid server response'); onComplete?.(); resolve(); return; }
      const { importJobId } = result;
      if (!importJobId) {
        alert('Import failed: No job ID returned from server');
        onComplete?.(); resolve(); return;
      }
      // Poll for extraction progress
      const intervalId = setInterval(async () => {
        let job;
        try {
          const resp = await fetch(`${serverUrl}/api/import-status/${importJobId}`);
          job = await resp.json();
        } catch {
          clearInterval(intervalId);
          alert('Import failed: Lost connection to server');
          onComplete?.(); resolve(); return;
        }
        if (job.status === 'running') {
          onProgress?.({ phase: 'extract', done: job.filesProcessed, total: job.totalFiles, filename: file.name });
        } else if (job.status === 'done') {
          clearInterval(intervalId);
          const { imageMap } = job;
          if (!imageMap || Object.keys(imageMap).length === 0) {
            alert('No valid class folders found in ZIP. Expected: className/image.jpg');
            onComplete?.(); resolve(); return;
          }
          onImport(imageMap);
          onComplete?.();
          resolve();
        } else if (job.status === 'error') {
          clearInterval(intervalId);
          alert(`Import failed: ${job.error}`);
          onComplete?.(); resolve();
        }
      }, 500);
    };

    xhr.onerror = () => { alert('Import failed: Network error'); onComplete?.(); resolve(); };
    xhr.open('POST', `${serverUrl}/api/import-zip`);
    xhr.setRequestHeader('Content-Type', 'application/zip');
    xhr.send(file);
  });
}

// ─── Browser-side import (fallback for small datasets) ────────────────────────

async function importInBrowser(file, onImport, onComplete, onProgress) {
  let zip;
  try {
    zip = await JSZip.loadAsync(file);
  } catch (err) {
    onComplete?.();
    alert(`Failed to read ZIP file: ${err.message}`);
    return;
  }

  const newImageMap = {};
  let truncated = false;

  const entries = Object.entries(zip.files).sort(([a], [b]) => a.localeCompare(b));

  // Detect a single wrapper folder — see server/routes/train.js for explanation.
  const topLevelNames = new Set();
  let hasDeepFiles = false;
  for (const [entryPath, zipEntry] of entries) {
    if (zipEntry.dir) continue;
    const p = entryPath.split('/');
    if (p[0] === '__MACOSX') continue;
    if (p.length < 2) continue;
    topLevelNames.add(p[0]);
    if (p.length >= 3) hasDeepFiles = true;
  }
  const pathOffset = (topLevelNames.size === 1 && hasDeepFiles) ? 1 : 0;

  // Pre-filter to image entries only so progress counts are accurate
  const imageEntries = entries.filter(([path, zipEntry]) => {
    if (zipEntry.dir) return false;
    const parts = path.split('/');
    if (parts[0] === '__MACOSX') return false;
    if (parts.length < 2 + pathOffset) return false;
    const ext = path.split('.').pop().toLowerCase();
    return SUPPORTED_EXTENSIONS.has(ext);
  });

  onProgress?.({ phase: 'extract', filePct: 0, done: 0, total: imageEntries.length, filename: file.name });

  for (let i = 0; i < imageEntries.length; i++) {
    const [path, zipEntry] = imageEntries[i];

    const parts = path.split('/');
    const className = parts[pathOffset];
    const ext = path.split('.').pop().toLowerCase();

    if (!newImageMap[className]) newImageMap[className] = [];
    if (newImageMap[className].length >= MAX_IMAGES_PER_CLASS) {
      truncated = true;
      onProgress?.({ phase: 'extract', filePct: Math.round(((i + 1) / imageEntries.length) * 100), done: i + 1, total: imageEntries.length, filename: file.name });
      continue;
    }

    const blob = await zipEntry.async('blob');
    const mimeType = MIME_MAP[ext] || `image/${ext}`;
    const typed = new Blob([blob], { type: mimeType });
    newImageMap[className].push(URL.createObjectURL(typed));

    onProgress?.({ phase: 'extract', filePct: Math.round(((i + 1) / imageEntries.length) * 100), done: i + 1, total: imageEntries.length, filename: file.name });

    // Yield every 10 files so the browser can process events
    if (i % 10 === 9) await yieldToMain();
  }

  if (Object.keys(newImageMap).length === 0) {
    onComplete?.();
    alert('No valid class folders found in ZIP.\nExpected: className/image.jpg');
    return;
  }

  if (truncated) {
    alert(`Some classes had more than ${MAX_IMAGES_PER_CLASS} images — extras were skipped.`);
  }

  onImport(newImageMap);
  onComplete?.();
}
