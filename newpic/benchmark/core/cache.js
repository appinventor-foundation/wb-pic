// Tiny IndexedDB key/value store for extracted features and finished runs. Every call is
// best-effort: if storage is unavailable (private window, quota) the benchmark still runs, uncached.
const DB = 'pic-benchmark', STORE = 'kv';

const open = () => new Promise((resolve, reject) => {
  const req = indexedDB.open(DB, 1);
  req.onupgradeneeded = () => req.result.createObjectStore(STORE);
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

const run = async (mode, fn) => {
  const db = await open();
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = fn(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req && req.result);
      tx.onerror = tx.onabort = () => reject(tx.error);
    });
  } finally { db.close(); }
};

export const cacheGet = (key) => run('readonly', s => s.get(key)).catch(() => undefined);
export const cachePut = (key, value) => run('readwrite', s => s.put(value, key)).then(() => true, () => false);
export const cacheKeys = (prefix) => run('readonly', s => s.getAllKeys()).then(k => k.filter(x => x.startsWith(prefix)), () => []);
export const cacheClear = () => run('readwrite', s => s.clear()).catch(() => undefined);
