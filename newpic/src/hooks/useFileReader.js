/**
 * Reads files as base64 data URLs using FileReader.
 * Returns Promise<string[]>.
 *
 * Data URLs (vs blob: URLs) are required for WebGL compatibility —
 * tf.browser.fromPixels / texSubImage2D rejects blob: URLs in Chrome
 * with a "tainted canvas" error even though they are same-origin.
 */
export const readFilesAsDataURLs = (files) => {
  return Promise.all(
    Array.from(files).map((file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(new Error(`Failed to read ${file.name}: ${e.target.error}`));
        reader.readAsDataURL(file);
      });
    })
  );
};
