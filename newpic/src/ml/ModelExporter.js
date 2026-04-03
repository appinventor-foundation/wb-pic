import * as tf from '@tensorflow/tfjs';

export class ModelExporter {
  constructor(config, adapter) {
    this.config = config;
    this.adapter = adapter;
  }

  async exportModel(model, labelIndex) {
    // Extract from TestView.js lines 205-247
    const { extensionName, truncationLayer, url } = this.config.transferModel;
    if (!extensionName) {
      throw new Error(`The selected transfer model (${this.config.transferModel.name}) is not supported by the App Inventor extension and cannot be exported.`);
    }
    const transferModelInfo = {
      name: extensionName,
      lastLayer: truncationLayer,
      url
    };

    const zipSaver = {
      save: async (modelSpecs) => {
        const weightsBlob = new Blob(
          [modelSpecs.weightData],
          { type: 'application/octet-stream' }
        );

        const weightsManifest = [{
          paths: ['./model.weights.bin'],
          weights: modelSpecs.weightSpecs
        }];

        const modelTopologyBlob = new Blob(
          [JSON.stringify({
            modelTopology: modelSpecs.modelTopology,
            weightsManifest
          })],
          { type: 'application/json' }
        );

        // Create label mapping: {0: "label1", 1: "label2", ...}
        const labels = {};
        labelIndex.forEach((label, i) => {
          labels[i] = label;
        });

        const files = {
          'model.json': modelTopologyBlob,
          'model.weights.bin': weightsBlob,
          'model_labels.json': JSON.stringify(labels),
          'transfer_model.json': JSON.stringify(transferModelInfo)
        };

        return await this.adapter.createZipArchive(files);
      }
    };

    const blob = await model.save(zipSaver);
    await this.adapter.saveFile(blob, this.config.export.modelFileName);

    return blob;
  }

  async exportData(imageMap) {
    // Extract from TestView.js lines 249-258
    const files = {
      'images.json': JSON.stringify(imageMap)
    };

    const blob = await this.adapter.createZipArchive(files);
    await this.adapter.saveFile(blob, this.config.export.dataFileName);

    return blob;
  }

  async importModel(modelArchive) {
    // Extract from LabelView.js lines 118-133
    const files = await this.adapter.readZipArchive(modelArchive);

    const modelJson = await files['model.json'].async('blob');
    const weights = await files['model.weights.bin'].async('blob');
    const labels = JSON.parse(await files['model_labels.json'].async('string'));
    const transferInfo = JSON.parse(await files['transfer_model.json'].async('string'));

    const model = await tf.loadLayersModel(
      tf.io.browserFiles([
        new File([modelJson], 'model.json'),
        new File([weights], 'model.weights.bin')
      ])
    );

    return { model, labelIndex: labels, transferModelInfo: transferInfo };
  }

  async importData(dataArchive) {
    // Extract from LabelView.js lines 135-145
    const files = await this.adapter.readZipArchive(dataArchive);
    const imageMap = JSON.parse(await files['images.json'].async('string'));
    return imageMap;
  }
}
