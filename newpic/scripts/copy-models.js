// Copies the transfer model folders from the repo root into public/models so
// they are served by the dev server and included in `npm run build`.
// The destination names match the paths the Express app serves them under.
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '../..');
const dest = path.resolve(__dirname, '../public/models');

const models = {
  mobilenet: 'mobilenet',
  mobilenetv2: 'mobilenet_v2',
  mobilenetv3: 'mobilenet_v3'
};

fs.rmSync(dest, { recursive: true, force: true });
for (const [source, name] of Object.entries(models)) {
  fs.cpSync(path.join(repoRoot, source), path.join(dest, name), { recursive: true });
}
console.log(`Copied ${Object.keys(models).length} models to ${path.relative(process.cwd(), dest)}`);
