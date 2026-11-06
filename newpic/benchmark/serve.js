import { createServer } from 'vite';
import { viteConfig } from './vite.js';

const server = await createServer(viteConfig());
await server.listen();
server.printUrls();
if (!process.env.DATASET_DIR) console.warn('DATASET_DIR is not set: /dataset will return 404');
