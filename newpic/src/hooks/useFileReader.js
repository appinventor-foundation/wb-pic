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

/**
 * Downscales a data URL to a small centre-cropped square JPEG data URL.
 * Used for history thumbnails so many entries stay cheap to keep in memory.
 * Resolves null if the image cannot be loaded.
 */
export const createThumbnail = (dataUrl, size = 160) => {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const side = Math.min(img.width, img.height);
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      canvas.getContext('2d').drawImage(
        img, (img.width - side) / 2, (img.height - side) / 2, side, side, 0, 0, size, size
      );
      resolve(canvas.toDataURL('image/jpeg', 0.7));
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
};
