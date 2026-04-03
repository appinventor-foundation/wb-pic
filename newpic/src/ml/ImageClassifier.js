import * as tf from '@tensorflow/tfjs';
import { DEFAULT_CONFIG } from './config/defaults';
import { ModelBuilder } from './ModelBuilder';
import { DataProcessor } from './DataProcessor';
import { ModelExporter } from './ModelExporter';
import { BrowserAdapter } from './adapters/BrowserAdapter';
import { TensorUtils } from './utils/TensorUtils';
import { TrainingError } from './utils/errors';
import ValidationSplitter from './utils/ValidationSplitter';

export class ImageClassifier {
  constructor(config = {}, adapter = null) {
    // Deep merge config with defaults
    this.config = this._mergeConfig(DEFAULT_CONFIG, config);
    this.adapter = adapter || new BrowserAdapter();

    this.modelBuilder = new ModelBuilder(this.config);
    this.dataProcessor = new DataProcessor(this.config, this.adapter);
    this.modelExporter = new ModelExporter(this.config, this.adapter);

    this.transferModel = null;
    this.customModel = null;
    this.labelIndex = null;
    this.isInitialized = false;
  }

  async train(imageMap, options = {}) {
    // Extract user parameters and progressCallback
    const { progressCallback, learningRate, optimizer, epochs, batchSizeFraction, useValidation, transferModel } = options;

    try {
      // Apply selected transfer model if provided (overrides default config)
      if (transferModel) {
        this.config.transferModel = transferModel;
        this.modelBuilder = new ModelBuilder(this.config);
      }

      // Validate input
      this.dataProcessor.validateImageMap(imageMap);

      // Build merged config object combining user settings with DEFAULT_CONFIG
      const trainingConfig = {
        customModel: {
          ...this.config.customModel,
          optimizer: {
            type: optimizer || this.config.customModel.optimizer.type,
            learningRate: learningRate !== undefined ? learningRate : this.config.customModel.optimizer.learningRate
          }
        },
        training: {
          epochs: epochs !== undefined ? epochs : this.config.training.epochs,
          batchSizeFraction: batchSizeFraction !== undefined ? batchSizeFraction : this.config.training.batchSizeFraction,
          shuffle: this.config.training.shuffle,
          useValidation: useValidation !== undefined ? useValidation : this.config.training.useValidation
        },
        transferModel: this.config.transferModel
      };

      // Load transfer model
      this._notifyProgress(progressCallback, {
        stage: 'loading',
        message: 'Loading mobilenet...',
        progress: 20
      });

      this.transferModel = await this.modelBuilder.getTransferModel();

      this._notifyProgress(progressCallback, {
        stage: 'loading',
        message: 'Truncating mobilenet...',
        progress: 50
      });

      // Generate input tensors
      const { images, labels, labelIndex } = await this.dataProcessor.generateInputTensors(imageMap);
      this.labelIndex = labelIndex;

      // Generate activations
      const activations = this._generateActivations(images);

      // Build custom model with user's optimizer settings
      this._notifyProgress(progressCallback, {
        stage: 'building',
        message: 'Building custom model...',
        progress: 70
      });

      // Create new ModelBuilder with merged config to apply optimizer + learningRate
      const modelBuilder = new ModelBuilder(trainingConfig);
      this.customModel = modelBuilder.buildCustomModel(labelIndex.length);

      // Train model with merged config
      await this._fitModel(activations, labels, trainingConfig.training, progressCallback);

      this.isInitialized = true;

      // Cleanup
      images.dispose();
      labels.dispose();
      activations.dispose();

      return {
        customModel: this.customModel,
        transferModel: this.transferModel,
        labelIndex: this.labelIndex
      };
    } catch (error) {
      throw new TrainingError(`Training failed: ${error.message}`, { originalError: error });
    }
  }

  async predict(imageUrl) {
    // Extract from TestView.js lines 294-324
    if (!this.isInitialized) {
      throw new Error('Model not initialized. Call train() first or load a model with importModel().');
    }

    const convertedImg = await this.dataProcessor.convertImage(imageUrl);
    const activation = this._extractFeatures(tf.stack([convertedImg]));
    const prediction = this.customModel.predict(activation);
    const topK = await prediction.topk(this.labelIndex.length);

    const confidences = await topK.values.data();
    const ranks = await topK.indices.data();

    const predictions = this.labelIndex.map((label, index) => ({
      label,
      confidence: confidences[ranks.indexOf(index)],
      rank: ranks.indexOf(index)
    }));

    // Cleanup
    convertedImg.dispose();
    activation.dispose();
    prediction.dispose();
    topK.values.dispose();
    topK.indices.dispose();

    return { predictions };
  }

  async exportModel() {
    if (!this.isInitialized) {
      throw new Error('Model not initialized. Call train() first or load a model with importModel().');
    }
    return await this.modelExporter.exportModel(this.customModel, this.labelIndex);
  }

  async exportData(imageMap) {
    return await this.modelExporter.exportData(imageMap);
  }

  /**
   * Load a model returned by the training server.
   * @param {Object} serverModelData - { modelTopology, weightSpecs, weightDataBase64, labelIndex }
   */
  async load(serverModelData) {
    const { modelTopology, weightSpecs, weightDataBase64, labelIndex } = serverModelData;

    // Decode base64 weight data to ArrayBuffer
    const binary = atob(weightDataBase64);
    const weightData = new ArrayBuffer(binary.length);
    const view = new Uint8Array(weightData);
    for (let i = 0; i < binary.length; i++) {
      view[i] = binary.charCodeAt(i);
    }

    this.customModel = await tf.loadLayersModel(
      tf.io.fromMemory({ modelTopology, weightSpecs, weightData })
    );
    this.labelIndex = labelIndex;

    // Always reload transfer model to pick up any config changes (e.g. v1 → v2)
    TensorUtils.disposeMany(this.transferModel);
    this.transferModel = await this.modelBuilder.getTransferModel();

    this.isInitialized = true;
  }

  async importModel(modelArchive) {
    const result = await this.modelExporter.importModel(modelArchive);
    this.customModel = result.model;
    this.labelIndex = Object.values(result.labelIndex);

    // Also need to load transfer model for predictions
    this.transferModel = await this.modelBuilder.getTransferModel();

    this.isInitialized = true;
    return result;
  }

  async importData(dataArchive) {
    return await this.modelExporter.importData(dataArchive);
  }

  dispose() {
    TensorUtils.disposeMany(this.transferModel, this.customModel);
    this.transferModel = null;
    this.customModel = null;
    this.labelIndex = null;
    this.isInitialized = false;
  }

  getMemoryInfo() {
    return TensorUtils.getMemoryInfo();
  }

  // Private methods

  _extractFeatures(imageTensor) {
    // Extract features using transfer model
    // Supports both GraphModel (v2) and LayersModel (v1) formats
    const modelFormat = this.config.transferModel.modelFormat || 'layers';

    if (modelFormat === 'graph') {
      // GraphModel: use execute() with truncation layer
      const truncationLayer = this.config.transferModel.truncationLayer;
      return this.transferModel.execute(imageTensor, truncationLayer);
    } else {
      // LayersModel: use predict() on already-truncated model
      return this.transferModel.predict(imageTensor);
    }
  }

  _generateActivations(images) {
    // Extract from TestView.js lines 136-146
    const activations = [];
    tf.unstack(images).forEach(image => {
      const activation = this._extractFeatures(tf.stack([image]));
      activations.push(tf.unstack(activation)[0]);
    });
    return tf.stack(activations);
  }

  async _fitModel(activations, labels, trainingConfig, progressCallback) {
    let trainActivations = activations;
    let trainLabels = labels;
    let valActivations = null;
    let valLabels = null;
    let splitTensors = null;

    try {
      // Handle validation split if enabled
      if (trainingConfig.useValidation) {
        console.log('🎯 ImageClassifier: Validation enabled - performing train/val split');

        this._notifyProgress(progressCallback, {
          stage: 'splitting',
          message: 'Creating validation split...',
          progress: 65
        });

        splitTensors = ValidationSplitter.splitStratified(activations, labels, 0.2, 42);
        trainActivations = splitTensors.trainActivations;
        trainLabels = splitTensors.trainLabels;
        valActivations = splitTensors.valActivations;
        valLabels = splitTensors.valLabels;

        this._notifyProgress(progressCallback, {
          stage: 'splitting',
          message: `Split: ${splitTensors.trainSize} train, ${splitTensors.valSize} validation`,
          progress: 70,
          data: {
            trainSize: splitTensors.trainSize,
            valSize: splitTensors.valSize
          }
        });
      } else {
        console.log('🎯 ImageClassifier: Training without validation (using all data)');
      }

      // Calculate batch size from training set size
      const batchSize = Math.floor(trainActivations.shape[0] * trainingConfig.batchSizeFraction);

      // Prepare fit options
      const fitOptions = {
        batchSize,
        epochs: trainingConfig.epochs,
        shuffle: trainingConfig.shuffle,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch ${epoch + 1} - All log keys:`, Object.keys(logs));
            console.log(`Epoch ${epoch + 1} - Full logs:`, logs);

            // Check for validation metrics with different possible key names
            const valLoss = logs.val_loss || logs.valLoss;
            const valAcc = logs.val_acc || logs.valAcc || logs.val_accuracy || logs.valAccuracy;

            const message = trainingConfig.useValidation
              ? `Epoch ${epoch + 1}/${trainingConfig.epochs} - loss: ${logs.loss.toFixed(5)}, val_loss: ${valLoss?.toFixed(5) || 'N/A'}, val_acc: ${valAcc ? (valAcc * 100).toFixed(1) + '%' : 'N/A'}`
              : `Epoch ${epoch + 1}/${trainingConfig.epochs} - loss: ${logs.loss.toFixed(5)}`;

            this._notifyProgress(progressCallback, {
              stage: 'training',
              message,
              progress: 70 + ((epoch + 1) / trainingConfig.epochs) * 30,
              data: {
                epoch: epoch + 1,
                loss: logs.loss,
                acc: logs.acc,
                val_loss: valLoss,
                val_acc: valAcc,
                type: 'epoch'
              }
            });
          }
        }
      };

      // Add validation data if available
      if (trainingConfig.useValidation && valActivations && valLabels) {
        fitOptions.validationData = [valActivations, valLabels];
        console.log('📊 Training with validation data - metrics will be computed each epoch');
        console.log(`  Validation data shapes: activations=${valActivations.shape}, labels=${valLabels.shape}`);
        console.log('  fitOptions:', fitOptions);
      }

      // Train the model
      console.log(`🚀 Starting model.fit() with ${trainActivations.shape[0]} training samples, batch size: ${batchSize}, epochs: ${trainingConfig.epochs}`);
      console.log('  fitOptions being passed to model.fit():', {
        batchSize: fitOptions.batchSize,
        epochs: fitOptions.epochs,
        shuffle: fitOptions.shuffle,
        hasValidationData: !!fitOptions.validationData,
        hasCallbacks: !!fitOptions.callbacks
      });
      await this.customModel.fit(trainActivations, trainLabels, fitOptions);
      console.log('✅ Training complete');

    } finally {
      // Cleanup validation split tensors if they were created
      if (splitTensors) {
        trainActivations.dispose();
        trainLabels.dispose();
        valActivations.dispose();
        valLabels.dispose();
      }
    }
  }

  _notifyProgress(callback, event) {
    if (callback && typeof callback === 'function') {
      callback(event);
    }
  }

  _mergeConfig(defaults, overrides) {
    // Deep merge configuration objects
    const merged = { ...defaults };

    for (const key in overrides) {
      if (overrides[key] && typeof overrides[key] === 'object' && !Array.isArray(overrides[key])) {
        merged[key] = this._mergeConfig(defaults[key] || {}, overrides[key]);
      } else {
        merged[key] = overrides[key];
      }
    }

    return merged;
  }
}
