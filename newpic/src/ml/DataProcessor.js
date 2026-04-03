import * as tf from '@tensorflow/tfjs';
import { ImageProcessingError, ValidationError } from './utils/errors';

export class DataProcessor {
  constructor(config, adapter) {
    this.config = config;
    this.adapter = adapter;
  }

  async convertImage(imageSource) {
    // Extract from TestView.js lines 82-97
    try {
      const img = await this.adapter.loadImage(imageSource);
      const { targetSize } = this.config.imageProcessing;

      // Convert to tensor
      // NodeAdapter returns { tensor } pre-loaded; BrowserAdapter returns an HTMLImageElement
      let tensor;
      if (img && img.tensor instanceof tf.Tensor) {
        tensor = img.tensor; // Node.js path: tensor already loaded by adapter
      } else {
        tensor = tf.browser.fromPixels(img); // Browser path
      }

      // Center-crop to square (matches Sharp's fit:'cover' and Android's cropImage())
      const size = Math.min(tensor.shape[0], tensor.shape[1]);
      const beginHeight = Math.floor((tensor.shape[0] - size) / 2);
      const beginWidth = Math.floor((tensor.shape[1] - size) / 2);
      tensor = tensor.slice([beginHeight, beginWidth, 0], [size, size, 3]);

      // Resize to 224x224
      tensor = tensor.resizeBilinear(targetSize);

      // Normalize: (x/127) - 1 maps [0,255] to [-1,1]
      tensor = tensor.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

      return tensor;
    } catch (error) {
      throw new ImageProcessingError(
        `Failed to convert image: ${error.message}`,
        imageSource
      );
    }
  }

  async generateInputTensors(imageMap) {
    // Extract from TestView.js lines 176-203
    const imageTensors = [];
    const labelTensors = [];

    // Convert all images to tensors
    const imageMapConverted = {};
    for (let label in imageMap) {
      imageMapConverted[label] = [];
    }

    for (let label in imageMap) {
      for (let imgIndex in imageMap[label]) {
        const convertedImg = await this.convertImage(imageMap[label][imgIndex]);
        imageMapConverted[label].push(convertedImg);
      }
    }

    // Create one-hot encoded labels
    const allLabels = Object.keys(imageMapConverted).sort();
    allLabels.forEach(label => {
      imageMapConverted[label].forEach(tensor => {
        imageTensors.push(tensor);
        const labelTensor = allLabels.map(l => (l === label ? 1 : 0));
        labelTensors.push(labelTensor);
      });
    });

    const images = tf.stack(imageTensors);
    const labels = tf.stack(labelTensors);

    // Cleanup intermediate tensors
    if (this.config.memory.autoDisposeIntermediateTensors) {
      imageTensors.forEach(t => t.dispose());
      // labelTensors are plain arrays, not tensors, so no disposal needed
      Object.values(imageMapConverted).flat().forEach(t => t.dispose());
    }

    return { images, labels, labelIndex: allLabels };
  }

  validateImageMap(imageMap) {
    if (!imageMap || typeof imageMap !== 'object') {
      throw new ValidationError('imageMap must be an object', 'imageMap');
    }

    const labels = Object.keys(imageMap);

    if (labels.length < 2) {
      throw new ValidationError(
        'imageMap must contain at least 2 labels',
        'imageMap.labels'
      );
    }

    labels.forEach(label => {
      if (!Array.isArray(imageMap[label])) {
        throw new ValidationError(
          `Images for label "${label}" must be an array`,
          `imageMap.${label}`
        );
      }

      if (imageMap[label].length === 0) {
        throw new ValidationError(
          `Label "${label}" must have at least one image`,
          `imageMap.${label}`
        );
      }
    });
  }
}
