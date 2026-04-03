import * as JSZip from 'jszip';
import * as FileSaver from 'file-saver';
import { EnvironmentAdapter } from './EnvironmentAdapter';

export class BrowserAdapter extends EnvironmentAdapter {
  async loadImage(imageSource) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      // http(s) URLs are cross-origin — must set crossOrigin before src so the
      // browser performs a CORS fetch. The server already returns Access-Control-Allow-Origin: *
      if (/^https?:\/\//.test(imageSource)) {
        img.crossOrigin = 'anonymous';
      }
      img.onload = () => {
        // Draw onto a canvas we own so tf.browser.fromPixels always receives an
        // origin-clean source. Without this, Chrome's WebGL2 backend throws
        // "Tainted canvases may not be loaded" for both blob: URLs and cross-origin
        // http: URLs even when the server returns CORS headers.
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.getContext('2d').drawImage(img, 0, 0);
        resolve(canvas);
      };
      img.onerror = () => reject(new Error(`Failed to load image: ${imageSource}`));
      img.src = imageSource;
    });
  }

  async createZipArchive(files) {
    // Uses JSZip (from TestView.js lines 230-243)
    // files: { 'filename': content (Blob or string) }
    const zip = new JSZip();
    for (const [filename, content] of Object.entries(files)) {
      zip.file(filename, content);
    }
    return await zip.generateAsync({ type: 'blob' });
  }

  async saveFile(data, filename) {
    // Uses FileSaver (from TestView.js line 242)
    FileSaver.saveAs(data, filename);
  }

  async readZipArchive(data) {
    // Uses JSZip (from LabelView.js lines 120-124, 138-141)
    const zip = new JSZip();
    const archive = await zip.loadAsync(data);
    return archive.files;
  }
}
