// Numerics and descriptive statistics. Pure functions: no DOM, no dependencies.

// ---------- random numbers ----------

// mulberry32: small seedable PRNG, returns floats in [0, 1)
export function makeRng(seed) {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- descriptive ----------

export const sum = (a) => a.reduce((s, x) => s + x, 0);
export const mean = (a) => sum(a) / a.length;
export const variance = (a) => {
  const m = mean(a);
  return a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1);
};
export const maxOf = (a) => a.reduce((m, x) => (x > m ? x : m), -Infinity);

// Mid-ranks (1-based) plus sum of (t^3 - t) over tie groups, for the Mann-Whitney tie correction
export function rankWithTies(values) {
  const n = values.length;
  const idx = values.map((_, i) => i).sort((i, j) => values[i] - values[j]);
  const ranks = new Array(n);
  let tieSum = 0;
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[idx[j + 1]] === values[idx[i]]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    const t = j - i + 1;
    tieSum += t ** 3 - t;
    i = j + 1;
  }
  return { ranks, tieSum };
}

// ---------- special functions ----------

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let s = LANCZOS[0];
  for (let i = 1; i < 9; i++) s += LANCZOS[i] / (x + i);
  const t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(s);
}

const EPS = 1e-15;
const TINY = 1e-300;

function gammaSeries(a, x) {
  let ap = a;
  let sum = 1 / a;
  let del = sum;
  for (let n = 0; n < 500; n++) {
    ap++;
    del *= x / ap;
    sum += del;
    if (Math.abs(del) < Math.abs(sum) * EPS) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

function gammaContinuedFraction(a, x) {
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 500; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}

// Regularised upper incomplete gamma Q(a, x)
export function gammaQ(a, x) {
  if (x <= 0) return 1;
  return x < a + 1 ? 1 - gammaSeries(a, x) : gammaContinuedFraction(a, x);
}

// Standard normal survival function P(Z > z)
export function normSf(z) {
  const q = gammaQ(0.5, (z * z) / 2); // = erfc(|z| / sqrt 2)
  return z >= 0 ? 0.5 * q : 1 - 0.5 * q;
}
export const normCdf = (z) => normSf(-z);

function betaContinuedFraction(a, b, x) {
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 500; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + aa / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

// Regularised incomplete beta I_x(a, b)
export function betaI(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2)
    ? (bt * betaContinuedFraction(a, b, x)) / a
    : 1 - (bt * betaContinuedFraction(b, a, 1 - x)) / b;
}

export const normPdf = (x) => Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI);

// Student t density
export const tPdf = (x, df) =>
  Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI) - ((df + 1) / 2) * Math.log(1 + (x * x) / df));

// Student t survival function P(T > t)
export function tSf(t, df) {
  const tail = 0.5 * betaI(df / (df + t * t), df / 2, 0.5);
  return t >= 0 ? tail : 1 - tail;
}

// Upper-tail critical value: x such that sf(x) = alpha (bisection on a decreasing function)
function upperCritical(sf, alpha) {
  let lo = 0;
  let hi = 1;
  while (sf(hi) > alpha) hi *= 2;
  for (let i = 0; i < 200; i++) {
    const mid = (lo + hi) / 2;
    if (sf(mid) > alpha) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}
export const zCrit = (alpha) => upperCritical(normSf, alpha);
export const tCrit = (alpha, df) => upperCritical((x) => tSf(x, df), alpha);
