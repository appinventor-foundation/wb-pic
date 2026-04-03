import * as tf from '@tensorflow/tfjs';

export class TensorUtils {
  static disposeMany(...tensors) {
    tensors.forEach(tensor => {
      if (tensor && typeof tensor.dispose === 'function') {
        tensor.dispose();
      }
    });
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
