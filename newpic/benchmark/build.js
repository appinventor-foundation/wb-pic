// Static build of the notebook (no dataset or models are bundled).
import { build } from 'vite';
import { viteConfig } from './vite.js';

await build(viteConfig());
