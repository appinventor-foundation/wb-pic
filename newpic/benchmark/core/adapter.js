import { BrowserAdapter } from '../../src/ml/adapters/BrowserAdapter.js';

// Same decode path as the app's BrowserAdapter (HTMLImageElement drawn on a canvas at natural size), but it reuses
// one canvas and releases each decoded image immediately. The app's adapter allocates a fresh full-size canvas per
// image, which adds up over thousands of images. The pixels handed to tf.browser.fromPixels are identical.
export class ReusableCanvasAdapter extends BrowserAdapter {
  constructor() {
    super();
    this.canvas = document.createElement('canvas');
  }

  loadImage(imageSource) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        this.canvas.width = img.naturalWidth;
        this.canvas.height = img.naturalHeight;
        this.canvas.getContext('2d').drawImage(img, 0, 0);
        img.onload = img.onerror = null;
        img.src = '';
        resolve(this.canvas);
      };
      img.onerror = () => reject(new Error(`Failed to load image: ${imageSource}`));
      img.src = imageSource;
    });
  }
}
