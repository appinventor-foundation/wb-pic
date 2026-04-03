import * as tf from '@tensorflow/tfjs';
import { ModelLoadError } from './utils/errors';

export class ModelBuilder {
  constructor(config) {
    this.config = config;
  }

  async getTransferModel() {
    // Extract from TestView.js lines 52-65
    try {
      const { url, truncationLayer, modelFormat = 'layers' } = this.config.transferModel;
      console.log('Loading mobilenet..');

      if (modelFormat === 'graph') {
        // GraphModel format (e.g., MobileNet v2)
        // Cannot truncate upfront - must use execute() during prediction
        const model = await tf.loadGraphModel(url);
        console.log('Loaded MobileNet as GraphModel');
        console.log(`Will extract features from layer: ${truncationLayer}`);
        return model;
      } else {
        // LayersModel format (e.g., MobileNet v1)
        // Can create truncated model upfront
        const model = await tf.loadLayersModel(url);
        const layer = model.getLayer(truncationLayer);
        const truncated = tf.model({
          inputs: model.inputs,
          outputs: layer.output,
          name: 'truncatedMobileNet'
        });
        console.log('Mobilenet model is modified');
        truncated.summary();
        return truncated;
      }
    } catch (error) {
      throw new ModelLoadError(`Failed to load transfer model: ${error.message}`);
    }
  }

  buildCustomModel(numLabels) {
    // Extract from TestView.js lines 148-174
    const { layers, optimizer, loss } = this.config.customModel;
    const model = tf.sequential();

    layers.forEach((layerConfig, index) => {
      const layerDef = { ...layerConfig };

      // Replace 'numLabels' placeholder with actual value
      if (layerDef.units === 'numLabels') {
        layerDef.units = numLabels;
      }

      // Add inputShape to first layer
      if (index === 0) {
        layerDef.inputShape = this.config.transferModel.outputShape;
      }

      // Create layer based on type
      const { type, ...config } = layerDef;
      switch (type) {
        case 'conv2d':
          model.add(tf.layers.conv2d(config));
          break;
        case 'flatten':
          model.add(tf.layers.flatten());
          break;
        case 'dense':
          model.add(tf.layers.dense(config));
          break;
        default:
          throw new Error(`Unknown layer type: ${type}`);
      }
    });

    // Compile model with accuracy metric
    const optimizerInstance = this._buildOptimizer(optimizer);
    model.compile({
      optimizer: optimizerInstance,
      loss,
      metrics: ['accuracy']  // Track accuracy during training
    });

    return model;
  }

  _buildOptimizer(optimizerConfig) {
    const { type, learningRate } = optimizerConfig;
    switch (type) {
      case 'adam':
        return tf.train.adam(learningRate);
      case 'sgd':
        return tf.train.sgd(learningRate);
      case 'adagrad':
        return tf.train.adagrad(learningRate);
      case 'adadelta':
        return tf.train.adadelta(learningRate);
      default:
        throw new Error(`Unknown optimizer: ${type}`);
    }
  }
}
