import { test } from 'node:test';
import assert from 'node:assert/strict';
import { makeRng, normSf, tSf, zCrit, tCrit, rankWithTies, mean, variance } from './stats.js';
import { DISTRIBUTIONS } from './distributions.js';
import { TESTS, testsFor } from './tests.js';

const close = (a, b, tol = 1e-6) => assert.ok(Math.abs(a - b) < tol, `${a} not within ${tol} of ${b}`);

test('rng is deterministic per seed', () => {
  const a = makeRng(42);
  const b = makeRng(42);
  for (let i = 0; i < 5; i++) assert.equal(a(), b());
  assert.notEqual(makeRng(1)(), makeRng(2)());
});

test('normal tail probabilities', () => {
  close(normSf(0), 0.5);
  close(normSf(1.96), 0.0249979, 1e-6);
  close(normSf(-1.96), 0.9750021, 1e-6);
  close(normSf(5), 2.8665157e-7, 1e-12);
});

test('t tail probabilities and critical values', () => {
  close(tSf(0, 7), 0.5);
  close(tSf(2.228139, 10), 0.025, 1e-6);
  close(tSf(-2.228139, 10), 0.975, 1e-6);
  close(tCrit(0.025, 10), 2.228139, 1e-5);
  close(zCrit(0.025), 1.959964, 1e-5);
});

test('ranks handle ties', () => {
  const { ranks, tieSum } = rankWithTies([10, 20, 20, 30]);
  assert.deepEqual(ranks, [1, 2.5, 2.5, 4]);
  assert.equal(tieSum, 6);
});

test('samplers match their theoretical means', () => {
  const rng = makeRng(7);
  const n = 40000;
  for (const [key, param] of [['bernoulli', 0.3], ['poisson', 4], ['poisson', 75], ['exponential', 2]]) {
    const d = DISTRIBUTIONS[key];
    const xs = Array.from({ length: n }, () => d.sample(param, rng));
    close(mean(xs), d.mean(param), 0.05 * Math.max(1, d.mean(param)));
  }
});

test("Welch t-test matches a hand calculation", () => {
  const a = [1, 2, 3, 4, 5];
  const b = [3, 4, 5, 6, 8];
  const res = TESTS.welch.run(a, b, { alpha: 0.05, tails: 'two' });
  // d = 2.2, s2_a = 2.5, s2_b = 3.7 -> SE = sqrt(1.24) -> t = 1.9757, df = 7.711
  close(variance(a), 2.5);
  close(res.p, 0.0850, 2e-4);
  assert.equal(res.reject, false);
});

test('z-test: a clear difference is significant, none is not', () => {
  const a = [...Array(30).fill(1), ...Array(70).fill(0)];
  const b = [...Array(60).fill(1), ...Array(40).fill(0)];
  assert.ok(TESTS.ztest.run(a, b, { alpha: 0.05, tails: 'two' }).reject);
  assert.ok(!TESTS.ztest.run(a, a, { alpha: 0.05, tails: 'two' }).reject);
  // one-sided in the wrong direction never rejects
  assert.ok(!TESTS.ztest.run(b, a, { alpha: 0.05, tails: 'greater' }).reject);
});

test('Mann-Whitney detects a shift and handles identical data', () => {
  const rng = makeRng(3);
  const a = Array.from({ length: 100 }, () => DISTRIBUTIONS.exponential.sample(1, rng));
  const b = Array.from({ length: 100 }, () => DISTRIBUTIONS.exponential.sample(0.3, rng));
  assert.ok(TESTS.mannwhitney.run(a, b, { alpha: 0.05, tails: 'two' }).reject);
  const same = TESTS.mannwhitney.run([1, 1, 1], [1, 1, 1], { alpha: 0.05, tails: 'two' });
  assert.ok(same.degenerate);
});

test('only Bernoulli offers the z-test', () => {
  assert.deepEqual(testsFor('bernoulli'), ['ztest', 'welch', 'mannwhitney']);
  assert.deepEqual(testsFor('poisson'), ['welch', 'mannwhitney']);
});

test('densities', async () => {
  const { normPdf, tPdf } = await import('./stats.js');
  close(normPdf(0), 0.3989423, 1e-6);
  close(tPdf(0, 10), 0.3891084, 1e-6);
  close(tPdf(1.5, 4), 0.12288, 1e-6); // (3/8)(1 + 1.5^2/4)^(-5/2)
});

test('observed data: validation hints and no ground truth', async () => {
  const { conclude } = await import('./tests.js');
  const { buildModel } = await import('./plot.js');
  assert.ok(DISTRIBUTIONS.bernoulli.validValue(1) && !DISTRIBUTIONS.bernoulli.validValue(2));
  assert.ok(DISTRIBUTIONS.poisson.validValue(3) && !DISTRIBUTIONS.poisson.validValue(2.5));
  assert.ok(DISTRIBUTIONS.exponential.validValue(0.2) && !DISTRIBUTIONS.exponential.validValue(-1));
  const res = TESTS.welch.run([1, 2, 3, 4, 5], [3, 4, 5, 6, 8], { alpha: 0.05, tails: 'two' });
  assert.equal(conclude(res, { alpha: 0.05, tails: 'two' }).truth, '');
  assert.ok(conclude(res, { alpha: 0.05, tails: 'two', trueMeanA: 1, trueMeanB: 2 }).truth.length > 0);
  const model = buildModel(DISTRIBUTIONS.poisson, [[1, 2, 2], [0, 3, 3]], null);
  assert.equal(model.panels[0].expected, null);
  assert.ok(model.ymax > 0);
});
