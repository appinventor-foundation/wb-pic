// Binary-classification metrics. `scores` are P(positive); `y` is 0/1. Positive = malignant.

export const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
export const std = (xs) => {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1));
};
export function quantile(xs, q) {
  const s = Array.from(xs).sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
}

export function confusion(scores, y, threshold = 0.5) {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (let i = 0; i < y.length; i++) {
    const pred = scores[i] >= threshold ? 1 : 0;
    if (pred && y[i]) c.tp++;
    else if (pred && !y[i]) c.fp++;
    else if (!pred && !y[i]) c.tn++;
    else c.fn++;
  }
  return c;
}

const ratio = (a, b) => (b === 0 ? 0 : a / b);

export function classificationMetrics(scores, y, threshold = 0.5) {
  const { tp, fp, tn, fn } = confusion(scores, y, threshold);
  const precision = ratio(tp, tp + fp);
  const recall = ratio(tp, tp + fn); // sensitivity
  return {
    accuracy: ratio(tp + tn, y.length),
    precision,
    recall,
    specificity: ratio(tn, tn + fp),
    f1: ratio(2 * precision * recall, precision + recall),
    balancedAccuracy: (recall + ratio(tn, tn + fp)) / 2
  };
}

// ROC points at every distinct score: [{ fpr, tpr }] from (0,0) to (1,1). Ties are grouped.
export function rocCurve(scores, y) {
  const order = Array.from(y.keys()).sort((a, b) => scores[b] - scores[a]);
  const P = y.reduce((s, v) => s + v, 0);
  const N = y.length - P;
  const pts = [{ fpr: 0, tpr: 0 }];
  let tp = 0, fp = 0;
  for (let k = 0; k < order.length; k++) {
    y[order[k]] ? tp++ : fp++;
    if (k === order.length - 1 || scores[order[k + 1]] !== scores[order[k]]) {
      pts.push({ fpr: ratio(fp, N), tpr: ratio(tp, P) });
    }
  }
  return pts;
}

export function rocAuc(scores, y) {
  const pts = rocCurve(scores, y);
  let area = 0;
  for (let i = 1; i < pts.length; i++) area += (pts[i].fpr - pts[i - 1].fpr) * (pts[i].tpr + pts[i - 1].tpr) / 2;
  return area;
}

// Average precision (area under the precision-recall step curve), ties grouped.
export function averagePrecision(scores, y) {
  const order = Array.from(y.keys()).sort((a, b) => scores[b] - scores[a]);
  const P = y.reduce((s, v) => s + v, 0);
  let tp = 0, fp = 0, prevRecall = 0, ap = 0;
  for (let k = 0; k < order.length; k++) {
    y[order[k]] ? tp++ : fp++;
    if (k === order.length - 1 || scores[order[k + 1]] !== scores[order[k]]) {
      const recall = ratio(tp, P);
      ap += (recall - prevRecall) * ratio(tp, tp + fp);
      prevRecall = recall;
    }
  }
  return ap;
}

// TPR on a fixed FPR grid (linear interpolation), so curves from different runs can be averaged.
export function tprOnGrid(scores, y, points = 101) {
  const pts = rocCurve(scores, y);
  const out = [];
  let j = 1;
  for (let g = 0; g < points; g++) {
    const x = g / (points - 1);
    while (j < pts.length - 1 && pts[j].fpr < x) j++;
    const a = pts[j - 1], b = pts[j];
    out.push(b.fpr === a.fpr ? b.tpr : a.tpr + (b.tpr - a.tpr) * (x - a.fpr) / (b.fpr - a.fpr));
  }
  return out;
}

// Best sensitivity among operating points whose specificity is at least `spec`.
export function sensitivityAtSpecificity(scores, y, spec) {
  let best = 0;
  for (const p of rocCurve(scores, y)) if (1 - p.fpr >= spec) best = Math.max(best, p.tpr);
  return best;
}

export const brier = (scores, y) => mean(Array.from(y, (v, i) => (scores[i] - v) ** 2));

// Expected calibration error on the predicted class's confidence, equal-width bins.
export function expectedCalibrationError(scores, y, bins = 10) {
  const sum = Array(bins).fill(0), hit = Array(bins).fill(0), n = Array(bins).fill(0);
  for (let i = 0; i < y.length; i++) {
    const pred = scores[i] >= 0.5 ? 1 : 0;
    const conf = pred ? scores[i] : 1 - scores[i];
    const b = Math.min(bins - 1, Math.floor(conf * bins));
    sum[b] += conf; hit[b] += pred === y[i] ? 1 : 0; n[b]++;
  }
  let ece = 0;
  for (let b = 0; b < bins; b++) if (n[b]) ece += (n[b] / y.length) * Math.abs(hit[b] / n[b] - sum[b] / n[b]);
  return ece;
}

// Precision / recall / F1 / support with each class as the positive class in turn, plus macro averages.
// `scores` are P(class 1); class 0 uses 1 - score, so both views share one threshold of 0.5.
export function perClassMetrics(scores, y, names = ['benign', 'malignant']) {
  const out = {};
  [0, 1].forEach((cls) => {
    const yc = Array.from(y, v => (v === cls ? 1 : 0));
    const sc = Array.from(scores, s => (cls === 1 ? s : 1 - s));
    const { precision, recall, f1 } = classificationMetrics(sc, yc);
    out[names[cls]] = { precision, recall, f1, support: yc.reduce((a, b) => a + b, 0) };
  });
  const [a, b] = names.map(n => out[n]);
  out.macro = { precision: (a.precision + b.precision) / 2, recall: (a.recall + b.recall) / 2, f1: (a.f1 + b.f1) / 2 };
  return out;
}

export function allMetrics(scores, y) {
  return {
    ...classificationMetrics(scores, y),
    rocAuc: rocAuc(scores, y),
    prAuc: averagePrecision(scores, y),
    sensAtSpec90: sensitivityAtSpecificity(scores, y, 0.9),
    sensAtSpec95: sensitivityAtSpecificity(scores, y, 0.95),
    brier: brier(scores, y),
    ece: expectedCalibrationError(scores, y)
  };
}
