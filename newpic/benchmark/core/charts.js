// Static SVG charts built from a results object: no DOM, no libraries, so the same code runs in Node and the images
// are reproducible from a saved run (see export-charts.js). Each function returns an SVG string.
import { summaryRows, rocRows, lossCurveRows, pooledConfusion } from './aggregate.js';

const FONT = 'Helvetica, Arial, sans-serif';
const INK = '#1b1e23';
const MUTED = '#6b7280';
const GRID = '#e5e7eb';
const COLORS = ['#4e79a7', '#f28e2c', '#59a14f', '#e15759'];

export const MODEL_LABELS = {
  mobilenet_local: 'MobileNet v1',
  mobilenet_v3_small_local: 'MobileNet v3 Small'
};
export const modelLabel = (key) => MODEL_LABELS[key] ?? key;

export const escapeXml = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' }[c]));

const text = (x, y, content, { size = 13, weight = 400, anchor = 'start', fill = INK } = {}) =>
  `<text x="${x}" y="${y}" font-family="${FONT}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}" fill="${fill}">${escapeXml(content)}</text>`;

const svgFrame = (width, height, body) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
  `<rect width="${width}" height="${height}" fill="#ffffff"/>${body}</svg>\n`;

const num = (v) => Math.round(v * 100) / 100;

// Generic line chart. series: [{ label, color, points: [{ x, y }] }]; domains and ticks are explicit so each chart reads cleanly.
export function lineChartSvg({ title, xLabel, yLabel, series, xDomain, yDomain, xTicks, yTicks, diagonal = false, dots = false, width = 640, height = 440 }) {
  const m = { top: 74, right: 28, bottom: 62, left: 70 };
  const w = width - m.left - m.right, h = height - m.top - m.bottom;
  const sx = (v) => m.left + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * w;
  const sy = (v) => m.top + h - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * h;
  const fmt = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
  let body = text(m.left - 46, 30, title, { size: 18, weight: 700 });

  // legend (one row under the title)
  let lx = m.left - 46;
  for (const s of series) {
    body += `<line x1="${lx}" y1="52" x2="${lx + 22}" y2="52" stroke="${s.color}" stroke-width="3"/>` + text(lx + 28, 56, s.label, { size: 13 });
    lx += 28 + s.label.length * 7.4 + 26;
  }

  for (const t of yTicks) {
    body += `<line x1="${m.left}" y1="${num(sy(t))}" x2="${m.left + w}" y2="${num(sy(t))}" stroke="${GRID}"/>` + text(m.left - 10, num(sy(t)) + 4, fmt(t), { size: 12, anchor: 'end', fill: MUTED });
  }
  for (const t of xTicks) {
    body += `<line x1="${num(sx(t))}" y1="${m.top}" x2="${num(sx(t))}" y2="${m.top + h}" stroke="${GRID}"/>` + text(num(sx(t)), m.top + h + 20, fmt(t), { size: 12, anchor: 'middle', fill: MUTED });
  }
  body += `<rect x="${m.left}" y="${m.top}" width="${w}" height="${h}" fill="none" stroke="#9ca3af"/>`;
  if (diagonal) body += `<line x1="${num(sx(xDomain[0]))}" y1="${num(sy(yDomain[0]))}" x2="${num(sx(xDomain[1]))}" y2="${num(sy(yDomain[1]))}" stroke="#9ca3af" stroke-dasharray="5,4"/>`;

  for (const s of series) {
    body += `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round" points="${s.points.map(p => `${num(sx(p.x))},${num(sy(p.y))}`).join(' ')}"/>`;
    if (dots) for (const p of s.points) body += `<circle cx="${num(sx(p.x))}" cy="${num(sy(p.y))}" r="3" fill="#ffffff" stroke="${s.color}" stroke-width="1.8"/>`;
  }

  body += text(m.left + w / 2, height - 16, xLabel, { size: 13, anchor: 'middle' });
  body += `<g transform="translate(18 ${m.top + h / 2}) rotate(-90)">${text(0, 0, yLabel, { size: 13, anchor: 'middle' })}</g>`;
  return svgFrame(width, height, body);
}

const firstCondition = (results) => Object.keys(results.config.conditions)[0];
const seriesColor = (results, model) => COLORS[results.config.models.indexOf(model) % COLORS.length];

export function rocSvg(results) {
  const cond = firstCondition(results);
  const auc = Object.fromEntries(summaryRows(results).map(r => [r.model, r.rocAuc.mean]));
  const rows = rocRows(results, cond);
  return lineChartSvg({
    title: 'ROC curve (mean over folds and seeds)',
    xLabel: 'false positive rate', yLabel: 'true positive rate (sensitivity)',
    xDomain: [0, 1], yDomain: [0, 1], xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1],
    diagonal: true, width: 560, height: 500,
    series: results.config.models.map(model => ({
      label: `${modelLabel(model)} (AUC ${auc[model].toFixed(3)})`, color: seriesColor(results, model),
      points: rows.filter(r => r.model === model).map(r => ({ x: r.fpr, y: r.tpr }))
    }))
  });
}

function trainingSvg(results, key, title, yLabel, yDomain, yTicks) {
  const rows = lossCurveRows(results);
  const epochs = Math.max(...rows.map(r => r.epoch));
  return lineChartSvg({
    title, xLabel: 'epoch', yLabel, dots: true,
    xDomain: [1, epochs], yDomain, xTicks: [1, ...Array.from({ length: Math.floor(epochs / 5) }, (_, i) => (i + 1) * 5)], yTicks,
    series: results.config.models.map(model => ({
      label: modelLabel(model), color: seriesColor(results, model),
      points: rows.filter(r => r.model === model).map(r => ({ x: r.epoch, y: r[key] }))
    }))
  });
}

// Training loss and accuracy are measured on the training images (the app trains without a validation set).
export const trainingLossSvg = (results) => {
  const rows = lossCurveRows(results);
  const top = Math.ceil(Math.max(...rows.map(r => r.loss)) * 10) / 10, bottom = Math.floor(Math.min(...rows.map(r => r.loss)) * 10) / 10;
  const ticks = []; for (let t = bottom; t <= top + 1e-9; t += 0.1) ticks.push(Math.round(t * 10) / 10);
  return trainingSvg(results, 'loss', 'Training loss (mean over folds and seeds)', 'training loss', [bottom, top], ticks);
};
export const trainingAccuracySvg = (results) =>
  trainingSvg(results, 'acc', 'Training accuracy (mean over folds and seeds)', 'training accuracy', [0.4, 1], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]);

export function confusionMatrixSvg(results) {
  const cond = firstCondition(results);
  const panels = results.config.models.map(m => ({ model: m, c: pooledConfusion(results, m, cond) }));
  const panelW = 440, cellW = 150, cellH = 104, rowLabelW = 120;
  const width = 40 + panels.length * panelW, height = 372;
  let body = text(24, 30, 'Confusion matrix', { size: 18, weight: 700 });
  panels.forEach(({ model, c }, i) => {
    const x0 = 24 + i * panelW, gx = x0 + rowLabelW, gy = 118;
    body += text(x0, 66, modelLabel(model), { size: 15, weight: 700 });
    ['predicted benign', 'predicted malignant'].forEach((label, j) => { body += text(gx + j * cellW + cellW / 2, gy - 12, label, { size: 13, anchor: 'middle', fill: MUTED }); });
    ['actual benign', 'actual malignant'].forEach((label, r) => { body += text(gx - 10, gy + r * cellH + cellH / 2 + 5, label, { size: 13, anchor: 'end', fill: MUTED }); });
    for (let r = 0; r < 2; r++) {
      const rowTotal = c.matrix[r][0] + c.matrix[r][1];
      for (let j = 0; j < 2; j++) {
        const correct = r === j, v = c.matrix[r][j];
        const x = gx + j * cellW, y = gy + r * cellH;
        body += `<rect x="${x}" y="${y}" width="${cellW - 4}" height="${cellH - 4}" rx="4" fill="${correct ? '#cfe8e4' : '#f7d9d3'}"/>`;
        body += text(x + (cellW - 4) / 2, y + 50, String(v), { size: 28, weight: 700, anchor: 'middle' });
        body += text(x + (cellW - 4) / 2, y + 74, `${(100 * v / rowTotal).toFixed(0)}% of row`, { size: 13, anchor: 'middle', fill: MUTED });
      }
    }
  });
  body += text(24, height - 18, 'Rows are the true class, columns the prediction. Counts pool all folds, so each image appears once.', { size: 12, fill: MUTED });
  return svgFrame(width, height, body);
}

// name -> svg for the four charts, in the order they are written to disk
export const allCharts = (results) => ({
  'confusion-matrix': confusionMatrixSvg(results),
  'roc-curve': rocSvg(results),
  'training-loss': trainingLossSvg(results),
  'training-accuracy': trainingAccuracySvg(results)
});
