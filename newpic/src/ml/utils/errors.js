export class ImageClassifierError extends Error {
  constructor(message, code) {
    super(message);
    this.name = 'ImageClassifierError';
    this.code = code;
  }
}

export class ModelLoadError extends ImageClassifierError {
  constructor(message) {
    super(message, 'MODEL_LOAD_ERROR');
    this.name = 'ModelLoadError';
  }
}

export class ImageProcessingError extends ImageClassifierError {
  constructor(message, imageSource) {
    super(message, 'IMAGE_PROCESSING_ERROR');
    this.name = 'ImageProcessingError';
    this.imageSource = imageSource;
  }
}

export class TrainingError extends ImageClassifierError {
  constructor(message, details) {
    super(message, 'TRAINING_ERROR');
    this.name = 'TrainingError';
    this.details = details;
  }
}

export class ValidationError extends ImageClassifierError {
  constructor(message, field) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.field = field;
  }
}
