// Distribution registry. To add a distribution, add one entry here:
// the UI builds its dropdown, parameter labels and plots from this object.
import { lgamma } from './stats.js';

function samplePoisson(lambda, rng) {
  // Knuth's method, applied in chunks of 30 so exp(-lambda) never underflows
  let total = 0;
  let left = lambda;
  while (left > 0) {
    const chunk = Math.min(left, 30);
    left -= chunk;
    const limit = Math.exp(-chunk);
    let k = 0;
    let p = rng();
    while (p > limit) {
      k++;
      p *= rng();
    }
    total += k;
  }
  return total;
}

export const DISTRIBUTIONS = {
  bernoulli: {
    name: 'Bernoulli',
    param: 'p',
    paramHint: 'probability of success, 0 to 1',
    defaults: [0.3, 0.6],
    valid: (p) => p >= 0 && p <= 1,
    sample: (p, rng) => (rng() < p ? 1 : 0),
    mean: (p) => p,
    variance: (p) => p * (1 - p),
    tex: { param: 'p', mean: 'p', variance: 'p(1-p)' },
    validValue: (x) => x === 0 || x === 1,
    valueHint: 'each value must be 0 or 1',
    exampleObs: [
      [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
      [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    ],
    discrete: true,
    pmf: (k, p) => (k === 1 ? p : k === 0 ? 1 - p : 0),
  },
  poisson: {
    name: 'Poisson',
    param: 'λ',
    paramHint: 'mean count, above 0',
    defaults: [4, 5],
    valid: (l) => l > 0,
    sample: samplePoisson,
    mean: (l) => l,
    variance: (l) => l,
    tex: { param: '\\lambda', mean: '\\lambda', variance: '\\lambda' },
    validValue: (x) => Number.isInteger(x) && x >= 0,
    valueHint: 'non-negative whole numbers',
    exampleObs: [
      [3, 5, 2, 4, 6, 3, 4, 7, 2, 4, 5, 3, 4, 1, 6, 4, 3, 5, 4, 2],
      [5, 7, 4, 6, 8, 5, 3, 6, 5, 7, 4, 6, 5, 9, 5, 4, 6, 7, 5, 6],
    ],
    discrete: true,
    pmf: (k, l) => Math.exp(k * Math.log(l) - l - lgamma(k + 1)),
  },
  exponential: {
    name: 'Exponential',
    param: 'λ',
    paramHint: 'rate, above 0 (mean is 1/λ)',
    defaults: [1, 0.8],
    valid: (l) => l > 0,
    sample: (l, rng) => -Math.log(1 - rng()) / l,
    mean: (l) => 1 / l,
    variance: (l) => 1 / l ** 2,
    tex: { param: '\\lambda', mean: '1/\\lambda', variance: '1/\\lambda^2' },
    validValue: (x) => x >= 0,
    valueHint: 'non-negative numbers',
    exampleObs: [
      [0.42, 1.35, 0.08, 2.1, 0.73, 0.31, 1.66, 0.95, 0.19, 3.02, 0.57, 1.18, 0.26, 0.88, 1.94, 0.12, 0.64, 1.41, 0.37, 0.79],
      [0.55, 1.92, 0.27, 3.4, 1.05, 0.48, 2.31, 0.83, 1.67, 0.15, 2.74, 1.28, 0.62, 4.05, 0.97, 1.51, 0.34, 2.08, 0.71, 1.22],
    ],
    discrete: false,
    cdf: (x, l) => 1 - Math.exp(-l * x),
  },
};
