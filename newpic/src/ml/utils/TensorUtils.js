import * as tf from '@tensorflow/tfjs';

export class TensorUtils {
  static disposeMany(...tensors) {
    tensors.forEach(tensor => {
      if (tensor && typeof tensor.dispose === 'function') {
        tensor.dispose();
      }
    });
  }

  // Lets the browser repaint (e.g. a progress bar) during long loops of synchronous tensor work.
  // setTimeout rather than tf.nextFrame: requestAnimationFrame stalls in background tabs.
  static yieldToUI() {
    return new Promise(resolve => setTimeout(resolve, 0));
  }

  static getMemoryInfo() {
    return tf.memory();
  }

  static logMemory(label = '') {
    const info = tf.memory();
    console.log(`[TF Memory${label ? ' - ' + label : ''}]`, {
      numTensors: info.numTensors,
      numBytes: `${(info.numBytes / 1024 / 1024).toFixed(2)} MB`
    });
  }
}
