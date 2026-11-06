import { tf } from './tf.js';

async function sha256(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2, '0')).join('');
}

function gpuRenderer() {
  try {
    const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  } catch { return null; }
}

// Everything needed to interpret (and compare) timing numbers, plus a hash of the run config.
export async function collectEnvironment(config) {
  const server = await fetch('/env.json').then(r => r.json()).catch(() => ({}));
  const f = tf.env().features;
  return {
    date: new Date().toISOString(),
    backend: tf.getBackend(),
    tfjs: tf.version.tfjs,
    webgl: { version: f.WEBGL_VERSION, floatTextures: f.WEBGL_RENDER_FLOAT32_CAPABLE, deleteTextureThreshold: f.WEBGL_DELETE_TEXTURE_THRESHOLD, renderer: gpuRenderer() },
    userAgent: navigator.userAgent,
    cores: navigator.hardwareConcurrency,
    deviceMemoryGB: navigator.deviceMemory ?? null,
    ...server, // git commit/dirty, node version of the dev server
    configHash: await sha256(JSON.stringify(config))
  };
}
