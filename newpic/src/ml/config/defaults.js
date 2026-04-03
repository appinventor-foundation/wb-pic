export const TRANSFER_MODEL_OPTIONS = {
  mobilenet_v1: {
    name: 'mobilenet_v1',
    label: 'MobileNet v1 (remote)',
    extensionName: 'mobilenet',         // name the App Inventor extension expects
    url: 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json',
    truncationLayer: 'conv_pw_13_relu',
    outputShape: [7, 7, 256],
    modelFormat: 'layers'
  },
  mobilenet_v2: {
    name: 'mobilenet_v2',
    label: 'MobileNet v2 (remote)',
    extensionName: 'mobilenet_v2',      // name the App Inventor extension expects
    url: 'https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json',
    truncationLayer: 'module_apply_default/MobilenetV2/Conv_1/Relu6',
    outputShape: [7, 7, 1280],
    modelFormat: 'graph'
  },
  mobilenet_local: {
    name: 'mobilenet_local',
    label: 'MobileNet (local folder)',
    extensionName: 'mobilenet',         // same architecture as the bundled extension model
    url: 'http://localhost:5000/mobilenet/model.json',
    truncationLayer: 'conv_pw_13_relu',
    outputShape: [7, 7, 256],
    modelFormat: 'layers'
  },
  mobilenet_v2_local: {
    name: 'mobilenet_v2_local',
    label: 'MobileNet v2 (local folder)',
    extensionName: 'mobilenet_v2',      // name the App Inventor extension expects
    url: 'http://localhost:5000/mobilenet_v2/model.json',
    truncationLayer: 'module_apply_default/MobilenetV2/Conv_1/Relu6',
    outputShape: [7, 7, 1280],
    modelFormat: 'graph'
  }
};

export const DEFAULT_CONFIG = {
  // Active transfer model — change the key to switch models
  transferModel: TRANSFER_MODEL_OPTIONS.mobilenet_local,

  // Custom Model Architecture
  customModel: {
    layers: [
      { type: 'conv2d', filters: 5, kernelSize: 5, strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling' },
      { type: 'flatten' },
      { type: 'dense', units: 100, activation: 'relu', kernelInitializer: 'varianceScaling', useBias: true },
      { type: 'dense', units: 'numLabels', activation: 'softmax', kernelInitializer: 'varianceScaling', useBias: false }
    ],
    optimizer: {
      type: 'adam',
      learningRate: 0.0001
    },
    loss: 'categoricalCrossentropy'
  },

  // Training Parameters
  training: {
    epochs: 20,
    batchSizeFraction: 0.4,  // 40% of dataset
    shuffle: true,
    useValidation: false  // Enable 20% validation split
  },

  // Image Processing
  imageProcessing: {
    targetSize: [224, 224],
    normalization: 'mobilenet'  // (x/127) - 1
  },

  // Memory Management
  memory: {
    autoDisposeIntermediateTensors: true
  },

  // Export Settings
  export: {
    modelFileName: 'model.mdl',
    dataFileName: 'data.zip'
  }
};
