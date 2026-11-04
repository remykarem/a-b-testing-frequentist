// SVG histograms. buildModel() turns samples into bins; createPlot() draws one panel
// and returns setProgress(t) so the caller can animate bars growing from 0 to full height.
import { maxOf, normCdf } from './stats.js';

const NS = 'http://www.w3.org/2000/svg';
const W = 520;
const H = 260;
const M = { l: 52, r: 12, t: 12, b: 42 };
const IW = W - M.l - M.r;
const IH = H - M.t - M.b;
const CONTINUOUS_BINS = 30;

function svgEl(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

const niceMax = (v) => {
  const step = v > 0.4 ? 0.1 : v > 0.15 ? 0.05 : v > 0.06 ? 0.02 : 0.01;
  return Math.ceil((v * 1.05) / step) * step;
};
const tickLabel = (v) => String(+v.toPrecision(3));

// samples: [sampleA, sampleB]; params: [paramA, paramB], or null for observed data (no true distribution
// to compare with). Both panels share bins and y-scale.
export function buildModel(dist, samples, params) {
  const top = maxOf(samples.map(maxOf));
  let bins;
  let binOf;
  let expectedFor;
  if (dist.discrete) {
    bins = Array.from({ length: top + 1 }, (_, k) => ({ x0: k, x1: k + 1 }));
    binOf = (x) => x;
    expectedFor = (bin, param) => dist.pmf(bin.x0, param);
  } else {
    const width = top / CONTINUOUS_BINS;
    bins = Array.from({ length: CONTINUOUS_BINS }, (_, i) => ({ x0: i * width, x1: (i + 1) * width }));
    binOf = (x) => Math.min(CONTINUOUS_BINS - 1, Math.floor(x / width));
    expectedFor = (bin, param) => dist.cdf(bin.x1, param) - dist.cdf(bin.x0, param);
  }

  const panels = samples.map((s, i) => {
    const freq = new Array(bins.length).fill(0);
    for (const x of s) freq[binOf(x)]++;
    return { freq: freq.map((c) => c / s.length), expected: params ? bins.map((b) => expectedFor(b, params[i])) : null };
  });
  const ymax = niceMax(maxOf(panels.flatMap((p) => [...p.freq, ...(p.expected ?? [])])));
  return { discrete: dist.discrete, bins, panels, ymax };
}

// Clears the svg and draws the y axis; returns the value -> pixel mapping
function drawFrame(svg, ymax, yLabel = 'relative frequency') {
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const yOf = (v) => M.t + IH - (v / ymax) * IH;
  for (let k = 0; k <= 4; k++) {
    const v = (ymax * k) / 4;
    svgEl('line', { x1: M.l, x2: W - M.r, y1: yOf(v), y2: yOf(v), class: 'grid' }, svg);
    svgEl('text', { x: M.l - 6, y: yOf(v) + 4, class: 'tick', 'text-anchor': 'end' }, svg, v.toFixed(2));
  }
  svgEl('text', { transform: `translate(12 ${M.t + IH / 2}) rotate(-90)`, class: 'axis-title', 'text-anchor': 'middle' }, svg, yLabel);
  return yOf;
}

export function createPlot(svg, model, i, cls) {
  const { bins, discrete, ymax } = model;
  const { freq, expected } = model.panels[i];
  const yOf = drawFrame(svg, ymax);
  const nb = bins.length;
  const slot = IW / nb;
  const gap = discrete ? Math.min(4, slot * 0.2) : 1;
  // x axis
  if (discrete) {
    const every = Math.ceil(nb / 12);
    bins.forEach((b, j) => {
      if (j % every === 0) svgEl('text', { x: M.l + (j + 0.5) * slot, y: M.t + IH + 16, class: 'tick', 'text-anchor': 'middle' }, svg, b.x0);
    });
  } else {
    for (let j = 0; j <= nb; j += nb / 5) {
      const x1 = bins[Math.min(j, nb - 1)];
      const v = j === nb ? bins[nb - 1].x1 : x1.x0;
      svgEl('text', { x: M.l + j * slot, y: M.t + IH + 16, class: 'tick', 'text-anchor': 'middle' }, svg, tickLabel(v));
    }
  }
  svgEl('text', { x: M.l + IW / 2, y: H - 6, class: 'axis-title', 'text-anchor': 'middle' }, svg, 'value');

  // bars (animated) and true-distribution markers
  const rects = bins.map((_, j) => svgEl('rect', { x: M.l + j * slot + gap / 2, width: Math.max(0.5, slot - gap), y: yOf(0), height: 0, class: `bar ${cls}` }, svg));
  (expected ?? []).forEach((v, j) => {
    svgEl('line', { x1: M.l + j * slot + gap / 2, x2: M.l + (j + 1) * slot - gap / 2, y1: yOf(v), y2: yOf(v), class: 'expected' }, svg);
  });
  svgEl('line', { x1: M.l, x2: W - M.r, y1: yOf(0), y2: yOf(0), class: 'axis' }, svg);

  return (t) => {
    freq.forEach((v, j) => {
      const h = (v * t / ymax) * IH;
      rects[j].setAttribute('y', M.t + IH - h);
      rects[j].setAttribute('height', h);
    });
  };
}

// Histogram of simulated sample means over mu +/- 4 SE, with the CLT normal curve N(mu, se^2)
// and a dashed line at `observed` (the mean of the generated sample).
export function createMeansPlot(svg, { means, mu, se, observed }, cls) {
  const NB = 40;
  const sd = se > 0 ? se : 0.1;
  const lo = mu - 4 * sd;
  const width = (8 * sd) / NB;
  const edge = (j) => lo + j * width;
  const freq = new Array(NB).fill(0);
  for (const m of means) {
    const j = Math.floor((m - lo) / width);
    if (j >= 0 && j < NB) freq[j]++;
  }
  const rel = freq.map((c) => c / means.length);
  const expected = rel.map((_, j) => (se > 0 ? normCdf((edge(j + 1) - mu) / sd) - normCdf((edge(j) - mu) / sd) : 0));
  const ymax = niceMax(maxOf([...rel, ...expected]));
  const yOf = drawFrame(svg, ymax);
  const slot = IW / NB;
  const xOf = (v) => M.l + ((v - lo) / (8 * sd)) * IW;

  for (let k = 0; k <= 4; k++) svgEl('text', { x: M.l + (k * IW) / 4, y: M.t + IH + 16, class: 'tick', 'text-anchor': 'middle' }, svg, tickLabel(lo + (k * 8 * sd) / 4));
  svgEl('text', { x: M.l + IW / 2, y: H - 6, class: 'axis-title', 'text-anchor': 'middle' }, svg, 'sample mean');

  const rects = rel.map((_, j) => svgEl('rect', { x: M.l + j * slot + 0.5, width: slot - 1, y: yOf(0), height: 0, class: `bar ${cls}` }, svg));
  if (se > 0) {
    const d = expected.map((v, j) => `${j ? 'L' : 'M'}${M.l + (j + 0.5) * slot},${yOf(v)}`).join(' ');
    svgEl('path', { d, class: 'curve' }, svg);
  }
  if (observed >= lo && observed <= lo + 8 * sd) svgEl('line', { x1: xOf(observed), x2: xOf(observed), y1: M.t, y2: M.t + IH, class: 'observed' }, svg);
  svgEl('line', { x1: M.l, x2: W - M.r, y1: yOf(0), y2: yOf(0), class: 'axis' }, svg);

  return (t) => {
    rel.forEach((v, j) => {
      const h = ((v * t) / ymax) * IH;
      rects[j].setAttribute('y', M.t + IH - h);
      rects[j].setAttribute('height', h);
    });
  };
}

// Histogram of simulated Delta = mean_B - mean_A over many repeated experiments, with:
//  - the sampling-distribution curve N(mu, se^2)
//  - (if `test`) the null distribution of Delta scaled by the test's SE, the rejection region,
//    and a second axis in test-statistic units
//  - the observed delta as a vertical line
export function createDiffPlot(svg, { ds, mu, se, observed, test, tails }) {
  const NB = 50;
  const sd = se > 0 ? se : 1e-9;
  const seN = test ? test.se : 0;
  const lo0 = Math.min(mu - 4 * sd, observed, test ? -4 * seN : Infinity);
  const hi0 = Math.max(mu + 4 * sd, observed, test ? 4 * seN : -Infinity);
  const pad = 0.03 * (hi0 - lo0 || 1);
  const lo = lo0 - pad;
  const hi = hi0 + pad;
  const width = (hi - lo) / NB;
  const edge = (j) => lo + j * width;

  const freq = new Array(NB).fill(0);
  for (const d of ds) {
    const j = Math.floor((d - lo) / width);
    if (j >= 0 && j < NB) freq[j]++;
  }
  const rel = freq.map((c) => c / ds.length);
  const sampling = rel.map((_, j) => normCdf((edge(j + 1) - mu) / sd) - normCdf((edge(j) - mu) / sd));
  const nullCurve = test ? rel.map((_, j) => test.cdf(edge(j + 1) / seN) - test.cdf(edge(j) / seN)) : [];
  const ymax = niceMax(maxOf([...rel, ...sampling, ...nullCurve]));
  const yOf = drawFrame(svg, ymax);
  const xOf = (v) => M.l + ((v - lo) / (hi - lo)) * IW;
  const slot = IW / NB;
  const mid = (j) => M.l + (j + 0.5) * slot;
  const axisY = M.t + IH;

  // rejection region(s) behind everything else
  if (test) {
    const xc = test.crit * seN;
    const regions = tails === 'two' ? [[lo, -xc], [xc, hi]] : [[xc, hi]];
    for (const [a, b] of regions) {
      const x0 = xOf(Math.max(a, lo));
      const x1 = xOf(Math.min(b, hi));
      if (x1 > x0) svgEl('rect', { x: x0, y: M.t, width: x1 - x0, height: IH, class: 'reject' }, svg);
    }
  }

  rel.forEach((v, j) => svgEl('rect', { x: M.l + j * slot + 0.5, width: slot - 1, y: yOf(v), height: IH - (yOf(v) - M.t), class: 'bar d' }, svg));
  const path = (vals, cls) => svgEl('path', { d: vals.map((v, j) => `${j ? 'L' : 'M'}${mid(j)},${yOf(v)}`).join(' '), class: cls }, svg);
  if (se > 0) path(sampling, 'curve');
  if (test) path(nullCurve, 'curve null');
  svgEl('line', { x1: M.l, x2: W - M.r, y1: axisY, y2: axisY, class: 'axis' }, svg);

  // x axes: Delta on the first row, test statistic on the second
  for (let k = 0; k <= 4; k++) svgEl('text', { x: M.l + (k * IW) / 4, y: axisY + 15, class: 'tick', 'text-anchor': 'middle' }, svg, tickLabel(lo + (k * (hi - lo)) / 4));
  svgEl('text', { x: M.l - 6, y: axisY + 15, class: 'axis-title', 'text-anchor': 'end' }, svg, 'Δ');
  if (test) {
    for (let k = -4; k <= 4; k++) {
      const x = k * seN;
      if (x >= lo && x <= hi) svgEl('text', { x: xOf(x), y: axisY + 32, class: 'tick', 'text-anchor': 'middle' }, svg, k);
    }
    svgEl('text', { x: M.l - 6, y: axisY + 32, class: 'axis-title', 'text-anchor': 'end' }, svg, test.sym);
  }

  // observed delta
  const ox = xOf(observed);
  svgEl('line', { x1: ox, x2: ox, y1: M.t, y2: axisY, class: 'obs' }, svg);
  const label = `δ = ${tickLabel(observed)}` + (test ? `, ${test.sym} = ${tickLabel(test.stat)}` : '');
  const right = ox > M.l + IW * 0.6;
  svgEl('text', { x: ox + (right ? -5 : 5), y: M.t + 11, class: 'obs-label', 'text-anchor': right ? 'end' : 'start' }, svg, label);
}

// Null distribution of the test statistic (z or t): the observed statistic, the p-value area
// (at least as extreme as observed) and the rejection region (total area alpha).
export function createStatPlot(svg, { sym, stat, pdf, crit, tails, ci, level }) {
  const reach = [Math.abs(stat), ...(ci ? [ci.lo, ci.hi].filter(isFinite).map(Math.abs) : [])];
  const R = Math.max(4.5, Math.ceil(maxOf(reach) + 0.5));
  const lo = -R;
  const hi = R;
  const ymax = niceMax(pdf(0));
  const yOf = drawFrame(svg, ymax, 'density');
  const xOf = (v) => M.l + ((v - lo) / (hi - lo)) * IW;
  const axisY = M.t + IH;
  const area = (a, b, cls) => {
    a = Math.max(a, lo);
    b = Math.min(b, hi);
    if (b <= a) return;
    const steps = Math.max(2, Math.ceil((b - a) * 40));
    const pts = Array.from({ length: steps + 1 }, (_, k) => {
      const x = a + ((b - a) * k) / steps;
      return `L${xOf(x)},${yOf(pdf(x))}`;
    });
    svgEl('path', { d: `M${xOf(a)},${yOf(0)} ${pts.join(' ')} L${xOf(b)},${yOf(0)} Z`, class: cls }, svg);
  };

  // rejection region, then the p-value area on top of it
  const lowerReject = tails === 'two';
  if (lowerReject) area(lo, -crit, 'reject-area');
  area(crit, hi, 'reject-area');
  if (tails === 'two') {
    area(lo, -Math.abs(stat), 'pval-area');
    area(Math.abs(stat), hi, 'pval-area');
  } else {
    area(stat, hi, 'pval-area');
  }

  const curve = Array.from({ length: 241 }, (_, k) => {
    const x = lo + ((hi - lo) * k) / 240;
    return `${k ? 'L' : 'M'}${xOf(x)},${yOf(pdf(x))}`;
  });
  svgEl('path', { d: curve.join(' '), class: 'curve' }, svg);
  svgEl('line', { x1: M.l, x2: W - M.r, y1: axisY, y2: axisY, class: 'axis' }, svg);

  for (let k = Math.ceil(lo); k <= hi; k++) svgEl('text', { x: xOf(k), y: axisY + 16, class: 'tick', 'text-anchor': 'middle' }, svg, k);
  svgEl('text', { x: M.l + IW / 2, y: H - 6, class: 'axis-title', 'text-anchor': 'middle' }, svg, sym);

  // critical value(s)
  for (const c of lowerReject ? [-crit, crit] : [crit]) {
    svgEl('line', { x1: xOf(c), x2: xOf(c), y1: yOf(pdf(c)), y2: axisY, class: 'crit' }, svg);
  }
  // confidence interval for d, drawn on the statistic axis (ends divided by the SE)
  if (ci) {
    const y = yOf(ymax * 0.1);
    const a = xOf(Math.max(ci.lo, lo));
    const b = xOf(Math.min(ci.hi, hi));
    svgEl('line', { x1: a, x2: b, y1: y, y2: y, class: 'ci' }, svg);
    svgEl('line', { x1: a, x2: a, y1: y - 6, y2: y + 6, class: 'ci' }, svg);
    if (isFinite(ci.hi)) svgEl('line', { x1: b, x2: b, y1: y - 6, y2: y + 6, class: 'ci' }, svg);
    else svgEl('path', { d: `M${b},${y} L${b - 8},${y - 5} L${b - 8},${y + 5} Z`, class: 'ci-arrow' }, svg);
    const range = isFinite(ci.dHi) ? `[${tickLabel(ci.dLo)}, ${tickLabel(ci.dHi)}]` : `[${tickLabel(ci.dLo)}, ∞)`;
    svgEl('text', { x: (a + b) / 2, y: y - 10, class: 'ci-label', 'text-anchor': 'middle' }, svg, `${level}% CI for the difference: ${range}`);
  }

  // observed statistic
  const ox = xOf(stat);
  svgEl('line', { x1: ox, x2: ox, y1: M.t, y2: axisY, class: 'obs' }, svg);
  const right = ox > M.l + IW * 0.6;
  svgEl('text', { x: ox + (right ? -5 : 5), y: M.t + 11, class: 'obs-label', 'text-anchor': right ? 'end' : 'start' }, svg, `${sym} = ${tickLabel(stat)}`);
}
