import * as tf from '@tensorflow/tfjs';

/**
 * Utility for creating stratified train/validation splits
 * Ensures each class contributes proportionally to both sets
 */
class ValidationSplitter {
  /**
   * Create a stratified train/validation split
   * @param {tf.Tensor} activations - Feature activations tensor [numSamples, features]
   * @param {tf.Tensor} labels - One-hot encoded labels [numSamples, numClasses]
   * @param {number} validationSplit - Fraction of data for validation (default 0.2)
   * @param {number} seed - Random seed for reproducible splits (default 42)
   * @returns {Object} Split tensors: { trainActivations, trainLabels, valActivations, valLabels, trainSize, valSize }
   */
  static splitStratified(activations, labels, validationSplit = 0.2, seed = 42) {
    return tf.tidy(() => {
      // Get class indices for each sample
      const classIndices = labels.argMax(-1).arraySync();
      const numSamples = activations.shape[0];
      const numClasses = labels.shape[1];

      console.log('🔀 ValidationSplitter: Starting stratified split');
      console.log(`  Total samples: ${numSamples}, Classes: ${numClasses}, Split: ${(validationSplit * 100).toFixed(0)}% validation`);

      // Group sample indices by class
      const classSamples = {};
      for (let i = 0; i < numSamples; i++) {
        const classIdx = classIndices[i];
        if (!classSamples[classIdx]) {
          classSamples[classIdx] = [];
        }
        classSamples[classIdx].push(i);
      }

      // Split each class and collect indices
      const trainIndices = [];
      const valIndices = [];

      console.log('  Per-class split:');
      for (let classIdx = 0; classIdx < numClasses; classIdx++) {
        const samples = classSamples[classIdx] || [];
        if (samples.length === 0) continue;

        // Shuffle samples for this class with seed
        const shuffled = this._shuffleWithSeed([...samples], seed + classIdx);

        // Split this class
        const numVal = Math.max(1, Math.floor(samples.length * validationSplit));
        const numTrain = samples.length - numVal;

        console.log(`    Class ${classIdx}: ${samples.length} total → ${numTrain} train + ${numVal} val`);

        trainIndices.push(...shuffled.slice(0, numTrain));
        valIndices.push(...shuffled.slice(numTrain));
      }

      // Shuffle combined indices to avoid class ordering
      const finalTrainIndices = this._shuffleWithSeed(trainIndices, seed);
      const finalValIndices = this._shuffleWithSeed(valIndices, seed + 1);

      // Gather tensors using indices
      const trainActivations = tf.gather(activations, finalTrainIndices);
      const trainLabels = tf.gather(labels, finalTrainIndices);
      const valActivations = tf.gather(activations, finalValIndices);
      const valLabels = tf.gather(labels, finalValIndices);

      console.log(`✅ Split complete: ${finalTrainIndices.length} training samples, ${finalValIndices.length} validation samples`);
      console.log(`  Train/Val ratio: ${(finalTrainIndices.length / numSamples * 100).toFixed(1)}% / ${(finalValIndices.length / numSamples * 100).toFixed(1)}%`);

      // Keep tensors (prevent disposal by tidy)
      return {
        trainActivations: tf.keep(trainActivations),
        trainLabels: tf.keep(trainLabels),
        valActivations: tf.keep(valActivations),
        valLabels: tf.keep(valLabels),
        trainSize: finalTrainIndices.length,
        valSize: finalValIndices.length
      };
    });
  }

  /**
   * Deterministic shuffle using Linear Congruential Generator
   * @param {Array} array - Array to shuffle
   * @param {number} seed - Random seed
   * @returns {Array} Shuffled array
   */
  static _shuffleWithSeed(array, seed) {
    const shuffled = [...array];
    let currentSeed = seed;

    // LCG parameters (from Numerical Recipes)
    const a = 1664525;
    const c = 1013904223;
    const m = Math.pow(2, 32);

    // Fisher-Yates shuffle with seeded random
    for (let i = shuffled.length - 1; i > 0; i--) {
      currentSeed = (a * currentSeed + c) % m;
      const j = Math.floor((currentSeed / m) * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }

    return shuffled;
  }
}

export default ValidationSplitter;
