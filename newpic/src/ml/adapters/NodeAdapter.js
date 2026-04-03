import { EnvironmentAdapter } from './EnvironmentAdapter';

/**
 * Node.js adapter for server-side use.
 * Requires: sharp, archiver, unzipper, @tensorflow/tfjs-node-gpu
 *
 * This file is NOT bundled for the browser — it is only used when
 * the src/ml/ files are imported from a Node.js environment.
 */
export class NodeAdapter extends EnvironmentAdapter {
  /**
   * Load an image from a data URL or file path.
   * Returns { tensor } where tensor is a tf.Tensor3D [H, W, 3] uint8.
   */
  async loadImage(imageSource) {
    // eslint-disable-next-line no-undef
    const sharp = require('sharp');
    // eslint-disable-next-line no-undef
    const tf = require('@tensorflow/tfjs-node-gpu');

    let buffer;
    if (typeof imageSource === 'string' && imageSource.startsWith('data:')) {
      const base64Data = imageSource.split(',')[1];
      buffer = Buffer.from(base64Data, 'base64');
    } else {
      // eslint-disable-next-line no-undef
      const fsPromises = require('fs/promises');
      buffer = await fsPromises.readFile(imageSource);
    }

    // Decode to raw RGB pixels
    const { data, info } = await sharp(buffer)
      .ensureAlpha(false)
      .toColorspace('srgb')
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Ensure 3-channel output (strip alpha if present)
    let pixelData;
    if (info.channels === 3) {
      pixelData = new Uint8Array(data);
    } else {
      // Convert RGBA → RGB
      pixelData = new Uint8Array(info.width * info.height * 3);
      for (let i = 0; i < info.width * info.height; i++) {
        pixelData[i * 3]     = data[i * info.channels];
        pixelData[i * 3 + 1] = data[i * info.channels + 1];
        pixelData[i * 3 + 2] = data[i * info.channels + 2];
      }
    }

    const tensor = tf.tensor3d(pixelData, [info.height, info.width, 3]);
    return { tensor };
  }

  /**
   * Create a ZIP archive from a map of filename → content (Buffer or string).
   * Returns a Buffer containing the ZIP.
   */
  async createZipArchive(files) {
    // eslint-disable-next-line no-undef
    const archiver = require('archiver');
    // eslint-disable-next-line no-undef
    const { Writable } = require('stream');

    return new Promise((resolve, reject) => {
      const chunks = [];
      const output = new Writable({
        write(chunk, _enc, cb) { chunks.push(chunk); cb(); }
      });

      const archive = archiver('zip', { zlib: { level: 6 } });
      archive.on('error', reject);
      output.on('finish', () => resolve(Buffer.concat(chunks)));

      archive.pipe(output);
      for (const [filename, content] of Object.entries(files)) {
        const buf = Buffer.isBuffer(content) ? content : Buffer.from(content);
        archive.append(buf, { name: filename });
      }
      archive.finalize();
    });
  }

  /**
   * Save data to a file on disk.
   * @param {Buffer|string} data
   * @param {string} filename
   * @param {string} [dir='.'] - Output directory
   */
  async saveFile(data, filename, dir = '.') {
    // eslint-disable-next-line no-undef
    const fsPromises = require('fs/promises');
    // eslint-disable-next-line no-undef
    const path = require('path');
    const outPath = path.join(dir, filename);
    await fsPromises.writeFile(outPath, data);
    return outPath;
  }

  /**
   * Read a ZIP archive and return a JSZip-compatible files map.
   * Each entry has an async(type) method for reading content.
   */
  async readZipArchive(data) {
    // eslint-disable-next-line no-undef
    const unzipper = require('unzipper');

    const buf = Buffer.isBuffer(data) ? data : Buffer.from(data);
    const directory = await unzipper.Open.buffer(buf);

    const files = {};
    for (const entry of directory.files) {
      if (entry.type === 'Directory') continue;
      const name = entry.path;
      files[name] = {
        async: async (type) => {
          const content = await entry.buffer();
          if (type === 'string') return content.toString('utf8');
          if (type === 'base64') return content.toString('base64');
          if (type === 'arraybuffer') return content.buffer.slice(content.byteOffset, content.byteOffset + content.byteLength);
          return content; // default: Buffer
        }
      };
    }
    return files;
  }
}
