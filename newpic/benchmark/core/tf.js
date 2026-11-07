// The single tfjs instance for the benchmark. Notebook cells must import tf from here
// (not from `npm:@tensorflow/tfjs`) so they share an engine with src/ml.
import * as tf from '@tensorflow/tfjs';

// The dataset has ~300 distinct image sizes (up to 6000x4000). tfjs keeps every freed WebGL texture for reuse
// unless this is set, so uploading each new size grew GPU memory until extraction slowed down and then stalled.
// Freed textures are deleted once the pool passes this many bytes.
tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 256 * 1024 * 1024);

export { tf };
