// UI wiring: reads the form, generates samples, draws plots, runs and renders the analysis.
import { makeRng, mean, variance, sum } from './stats.js';
import { DISTRIBUTIONS } from './distributions.js';
import { TESTS, testsFor, conclude, f } from './tests.js';
import { buildModel, createPlot, createMeansPlot, createDiffPlot, createStatPlot } from './plot.js';

const $ = (id) => document.getElementById(id);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

let state = null; // { distKey, a, b, params (null for observations), seed, mode, id }
let resampleCount = 0;
let diffResample = 0;
let generation = 0;
let diffCache = null;
let lastRes = null;
let raf = 0;

// ---------- form ----------

function populateDistributions() {
  $('dist').innerHTML = Object.entries(DISTRIBUTIONS)
    .map(([key, d]) => `<option value="${key}">${d.name}</option>`)
    .join('');
}

function syncDistribution() {
  const d = DISTRIBUTIONS[$('dist').value];
  for (const id of ['paramNameA', 'paramNameB']) $(id).textContent = d.param;
  $('pA').value = d.defaults[0];
  $('pB').value = d.defaults[1];
  $('paramHint').textContent = `${d.param}: ${d.paramHint}`;
  $('obsHint').textContent = `${d.name} data: ${d.valueHint}.`;
  $('obsA').value = d.exampleObs[0].join(', ');
  $('obsB').value = d.exampleObs[1].join(', ');
  $('test').innerHTML = testsFor($('dist').value)
    .map((id) => `<option value="${id}">${TESTS[id].name}</option>`)
    .join('');
}

function readSetup() {
  const dist = DISTRIBUTIONS[$('dist').value];
  const params = [Number($('pA').value), Number($('pB').value)];
  const ns = [Number($('nA').value), Number($('nB').value)];
  if (params.some((p) => !isFinite(p) || !dist.valid(p))) throw new Error(`Parameter ${dist.param} must be valid: ${dist.paramHint}.`);
  if (ns.some((n) => !Number.isInteger(n) || n < 2 || n > 200000)) throw new Error('Sample sizes must be whole numbers from 2 to 200,000.');
  const seedText = $('seed').value.trim();
  if (seedText !== '' && !Number.isInteger(Number(seedText))) throw new Error('Seed must be a whole number (or blank for random).');
  const seed = seedText === '' ? Math.floor(Math.random() * 2 ** 32) : Number(seedText);
  return { dist, params, ns, seed };
}

// ---------- generate + plot ----------

function setMode() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  $('form').hidden = mode !== 'simulation';
  $('obsForm').hidden = mode !== 'observation';
  $('dataHeading').textContent = mode === 'simulation' ? '2. Simulate' : '2. Observations';
}

function generate(e) {
  e.preventDefault();
  $('error').textContent = '';
  let setup;
  try {
    setup = readSetup();
  } catch (err) {
    $('error').textContent = err.message;
    return;
  }
  const { dist, params, ns, seed } = setup;
  const rngs = [makeRng(seed), makeRng(seed ^ 0x9e3779b9)];
  const samples = [0, 1].map((i) => Array.from({ length: ns[i] }, () => dist.sample(params[i], rngs[i])));
  state = { distKey: $('dist').value, a: samples[0], b: samples[1], params, seed, mode: 'simulation', id: ++generation };
  $('seedUsed').textContent = `seed ${seed}`;
  showData(dist);
}

function parseObservations(text, label, dist) {
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  const bad = values.findIndex((x) => !isFinite(x) || !dist.validValue(x));
  if (bad >= 0) throw new Error(`Group ${label}: "${tokens[bad]}" is not valid. ${dist.name} data: ${dist.valueHint}.`);
  if (values.length < 2) throw new Error(`Group ${label} needs at least 2 values.`);
  if (values.length > 200000) throw new Error(`Group ${label} has too many values (limit 200,000).`);
  return values;
}

function loadObservations(e) {
  e.preventDefault();
  $('obsError').textContent = '';
  const dist = DISTRIBUTIONS[$('dist').value];
  let a;
  let b;
  try {
    a = parseObservations($('obsA').value, 'A', dist);
    b = parseObservations($('obsB').value, 'B', dist);
  } catch (err) {
    $('obsError').textContent = err.message;
    return;
  }
  // No true parameters for real data. The seed only drives the bootstrap resampling.
  state = { distKey: $('dist').value, a, b, params: null, seed: Math.floor(Math.random() * 2 ** 32), mode: 'observation', id: ++generation };
  showData(dist);
}

// Real data has no "true distribution" to draw from, so only the bootstrap options make sense
function applyDataMode() {
  const obs = state.mode === 'observation';
  for (const name of ['source', 'dsource']) {
    document.querySelector(`input[name="${name}"][value="true"]`).disabled = obs;
    if (obs) document.querySelector(`input[name="${name}"][value="bootstrap"]`).checked = true;
  }
}

// X_A ~ Bernoulli(p = 0.3) for simulations, X_A ~ Bernoulli(p_A) for observations
function renderDataMath(dist) {
  ['A', 'B'].forEach((g, i) => {
    const m = $(`dataMath${g}`);
    m.replaceChildren();
    const arg = state.mode === 'simulation' ? `${dist.tex.param} = ${f(state.params[i])}` : `${dist.tex.param}_${g}`;
    tex(m, `X_${g} \\sim \\text{${dist.name}}(${arg})`);
  });
}

// Sample statistics worked out for the data (simulated or observed): x-bar, s^2 and s
function renderStatsMath(dist) {
  ['A', 'B'].forEach((g, i) => {
    const m = $(`statsMath${g}`);
    m.replaceChildren();
    const x = i ? state.b : state.a;
    const n = x.length;
    const xbar = mean(x);
    const ss = x.reduce((acc, v) => acc + (v - xbar) ** 2, 0);
    const v = ss / (n - 1);
    const hat = dist.name === 'Bernoulli' ? `= \\hat p_${g} ` : '';
    tex(m, `\\bar x_${g} = \\dfrac{1}{n_${g}}\\sum_{i=1}^{n_${g}} x_i = \\dfrac{${f(sum(x))}}{${n}} ${hat}= ${f(xbar)}`);
    tex(m, `s_${g}^2 = \\dfrac{1}{n_${g}-1}\\sum_{i=1}^{n_${g}} (x_i - \\bar x_${g})^2 = \\dfrac{${f(ss)}}{${n - 1}} = ${f(v)}`);
    tex(m, `s_${g} = \\sqrt{s_${g}^2} = \\sqrt{${f(v)}} = ${f(Math.sqrt(v))}`);
  });
}

function showData(dist) {
  $('plots').hidden = false;
  $('sampling').hidden = false;
  $('analysis').hidden = false;

  const samples = [state.a, state.b];
  const model = buildModel(dist, samples, state.params);
  const plots = [createPlot($('plotA'), model, 0, 'a'), createPlot($('plotB'), model, 1, 'b')];
  ['A', 'B'].forEach((g, i) => {
    const s = samples[i];
    $(`summary${g}`).textContent = `n = ${s.length}, mean = ${+mean(s).toPrecision(4)}, sd = ${+Math.sqrt(variance(s)).toPrecision(4)}` + (dist.name === 'Bernoulli' ? `, successes = ${sum(s)}` : '');
  });
  renderDataMath(dist);
  renderStatsMath(dist);
  $('plotsNote').textContent = state.mode === 'simulation'
    ? 'Bars are the observed relative frequencies. The short horizontal lines mark what the true distribution predicts.'
    : 'Bars are the observed relative frequencies of your data.';
  applyDataMode();
  animate(plots);
  runSampling(false);
  runAnalysis();
}

function animate(plots) {
  cancelAnimationFrame(raf);
  if (reducedMotion) return plots.forEach((p) => p(1));
  const start = performance.now();
  const duration = 900;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - (1 - t) ** 3;
    plots.forEach((p) => p(eased));
    if (t < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// ---------- sampling distribution of the mean ----------

function runSampling(animated) {
  if (!state) return;
  const n = Number($('sn').value);
  const reps = Number($('sreps').value);
  const err = $('samplingError');
  err.textContent = '';
  if (!Number.isInteger(n) || n < 1 || n > 5000) return void (err.textContent = 'Sample size must be a whole number from 1 to 5,000.');
  if (!Number.isInteger(reps) || reps < 100 || reps > 10000) return void (err.textContent = 'Repeats must be a whole number from 100 to 10,000.');
  if (n * reps > 5e6) return void (err.textContent = 'Sample size × repeats must be at most 5,000,000.');

  const dist = DISTRIBUTIONS[state.distKey];
  const boot = document.querySelector('input[name="source"]:checked').value === 'bootstrap';
  const plots = ['A', 'B'].map((g, i) => {
    const param = state.params?.[i];
    const data = i ? state.b : state.a;
    const rng = makeRng(state.seed ^ (i ? 0xc2b2ae35 : 0x85ebca6b) ^ (resampleCount * 0x9e3779b1));
    const draw = boot ? () => data[Math.floor(rng() * data.length)] : () => dist.sample(param, rng);
    const means = Array.from({ length: reps }, () => {
      let s = 0;
      for (let k = 0; k < n; k++) s += draw();
      return s / n;
    });
    // CLT target: true (mu, sigma^2) when simulating, observed (x-bar, s^2) when bootstrapping
    const mu = boot ? mean(data) : dist.mean(param);
    const v = boot ? variance(data) : dist.variance(param);
    const se = Math.sqrt(v / n);
    const m = $(`meansMath${g}`);
    m.replaceChildren();
    if (boot) {
      tex(m, `X^*_i \\sim \\text{drawn with replacement from the observed sample } (n_${g} = ${data.length}),\\quad i = 1,\\dots,${n}`);
      tex(m, `\\bar x_${g} = ${f(mu)},\\qquad s_${g}^2 = ${f(v)}`);
      tex(m, `\\bar X^*_${g} = \\dfrac{1}{n}\\sum_{i=1}^{n} X^*_i \\;\\approx\\; N\\!\\left(\\bar x,\\ \\dfrac{s^2}{n}\\right) = N\\!\\left(${f(mu)},\\ \\dfrac{${f(v)}}{${n}}\\right) = N(${f(mu)},\\ ${f(se * se)}),\\quad SE = \\dfrac{s}{\\sqrt n} = ${f(se)}`);
    } else {
      const t = dist.tex;
      tex(m, `X_i \\sim \\text{${dist.name}}(${t.param} = ${f(param)}),\\quad i = 1,\\dots,${n}`);
      tex(m, `\\mu = ${t.mean} = ${f(mu)},\\qquad \\sigma^2 = ${t.variance} = ${f(v)}`);
      tex(m, `\\bar X_${g} = \\dfrac{1}{n}\\sum_{i=1}^{n} X_i \\;\\approx\\; N\\!\\left(\\mu,\\ \\dfrac{\\sigma^2}{n}\\right) = N\\!\\left(${f(mu)},\\ \\dfrac{${f(v)}}{${n}}\\right) = N(${f(mu)},\\ ${f(se * se)}),\\quad SE = \\dfrac{\\sigma}{\\sqrt n} = ${f(se)}`);
    }
    $(`meansSummary${g}`).textContent = `mean of sample means = ${+mean(means).toPrecision(4)} (${boot ? 'observed sample mean' : 'true mean'} ${+mu.toPrecision(4)}), sd of sample means = ${+Math.sqrt(variance(means)).toPrecision(4)} (theory ${boot ? 's' : 'σ'}/√n = ${+se.toPrecision(4)})`;
    return createMeansPlot($(`meansPlot${g}`), { means, mu, se, observed: mean(i ? state.b : state.a) }, g.toLowerCase());
  });
  // Non-animated redraws must not cancel an animation still running elsewhere (e.g. the section 1 bars)
  if (animated) animate(plots);
  else plots.forEach((p) => p(1));
}

// ---------- test statistic and p-value ----------

function runStat(res, alpha, tails) {
  const svg = $('statPlot');
  const m = $('statMath');
  m.replaceChildren();
  const sd = res.statDist;
  if (!sd) {
    svg.replaceChildren();
    $('statSummary').textContent = 'No test statistic is available for this test on these samples.';
    return;
  }
  const level = +((1 - alpha) * 100).toFixed(4);
  createStatPlot(svg, { sym: sd.sym, stat: sd.stat, pdf: sd.pdf, crit: sd.crit, tails, ci: sd.ci, level });
  tex(m, `\\text{Under } H_0:\\ ${sd.sym} \\sim ${sd.ref},\\qquad ${sd.sym}_{obs} = ${f(sd.stat)}`);
  tex(m, tails === 'two'
    ? `p = 2\\,P(${sd.sym === 'z' ? 'Z' : 'T'} \\ge |${sd.sym}_{obs}|) = ${f(res.p)}`
    : `p = P(${sd.sym === 'z' ? 'Z' : 'T'} \\ge ${sd.sym}_{obs}) = ${f(res.p)}`);
  const ci = sd.ci;
  let ciNote = '';
  if (ci) {
    const closing = isFinite(ci.dHi) ? ']' : ')';
    tex(m, `${level}\\%\\ \\text{CI for the difference}:\\ [${f(ci.dLo)},\\ ${f(ci.dHi)}${closing}\\quad\\Longrightarrow\\quad \\text{in } ${sd.sym} \\text{ units: } [${f(ci.lo)},\\ ${f(ci.hi)}${closing}`);
    const contains0 = ci.dLo <= 0 && ci.dHi >= 0;
    ciNote = ` The thick bar is the ${level}% confidence interval for the difference μ_B − μ_A, drawn here by dividing its ends by the SE, so it is centred on the observed ${sd.sym}. It ${contains0 ? 'contains' : 'excludes'} 0, which matches ${contains0 === !res.reject ? 'the test decision' : 'the p-value only approximately (the proportions z-test uses a different SE for its interval)'}.`;
  }
  $('statSummary').textContent =
    `The purple area is the p-value: how likely a statistic at least this extreme is if H₀ is true. ` +
    `The red area is the rejection region, with total area α = ${alpha}; its boundary is the critical value ${tails === 'two' ? '±' : ''}${+sd.crit.toPrecision(4)}. ` +
    (res.reject ? `The observed ${sd.sym} lands in the rejection region (p ≤ α), so we reject H₀.` : `The observed ${sd.sym} is outside the rejection region (p > α), so we fail to reject H₀.`) + ciNote;
}

// ---------- sampling distribution of the difference ----------

// Repeat the whole experiment `reps` times (n_A and n_B as generated) and record Delta = mean_B - mean_A.
// Cached so that changing the test / alpha / tails only redraws.
function simulateDiffs(source, reps) {
  const key = [state.id, state.seed, source, reps, diffResample].join('|');
  if (diffCache?.key === key) return diffCache;
  const dist = DISTRIBUTIONS[state.distKey];
  const boot = source === 'bootstrap';
  const rng = makeRng(state.seed ^ 0x27d4eb2f ^ (diffResample * 0x9e3779b1));
  const group = (i) => {
    const data = i ? state.b : state.a;
    const param = state.params?.[i];
    const n = data.length;
    return {
      n,
      draw: boot ? () => data[Math.floor(rng() * n)] : () => dist.sample(param, rng),
      mu: boot ? mean(data) : dist.mean(param),
      v: boot ? variance(data) : dist.variance(param),
    };
  };
  const [gA, gB] = [group(0), group(1)];
  const avg = (g) => {
    let s = 0;
    for (let k = 0; k < g.n; k++) s += g.draw();
    return s / g.n;
  };
  const ds = Array.from({ length: reps }, () => {
    const a = avg(gA);
    return avg(gB) - a;
  });
  diffCache = {
    key, ds, boot,
    mu: gB.mu - gA.mu,
    se: Math.sqrt(gA.v / gA.n + gB.v / gB.n),
    vA: gA.v, vB: gB.v,
    observed: mean(state.b) - mean(state.a),
  };
  return diffCache;
}

function runDiff() {
  if (!state || !lastRes) return;
  const err = $('diffError');
  err.textContent = '';
  const reps = Number($('dreps').value);
  if (!Number.isInteger(reps) || reps < 100 || reps > 10000) return void (err.textContent = 'Repeats must be a whole number from 100 to 10,000.');
  if ((state.a.length + state.b.length) * reps > 2e7) return void (err.textContent = 'Too many draws: (n_A + n_B) × repeats must be at most 20,000,000.');

  const source = document.querySelector('input[name="dsource"]:checked').value;
  const sim = simulateDiffs(source, reps);
  const tails = document.querySelector('input[name="tails"]:checked').value;
  const test = lastRes.diff;
  createDiffPlot($('diffPlot'), { ds: sim.ds, mu: sim.mu, se: sim.se, observed: sim.observed, test, tails });

  const nA = state.a.length;
  const nB = state.b.length;
  const m = $('diffMath');
  m.replaceChildren();
  const [centre, vA, vB] = sim.boot ? [`\\bar x_B - \\bar x_A`, 's_A^2', 's_B^2'] : [`\\mu_B - \\mu_A`, '\\sigma_A^2', '\\sigma_B^2'];
  tex(m, `\\Delta = \\bar X_B - \\bar X_A \\;\\approx\\; N\\!\\left(${centre},\\ \\dfrac{${vA}}{n_A} + \\dfrac{${vB}}{n_B}\\right) = N\\!\\left(${f(sim.mu)},\\ \\dfrac{${f(sim.vA)}}{${nA}} + \\dfrac{${f(sim.vB)}}{${nB}}\\right) = N(${f(sim.mu)},\\ ${f(sim.se ** 2)})`);
  let summary = `mean of simulated Δ = ${+mean(sim.ds).toPrecision(4)}, sd = ${+Math.sqrt(variance(sim.ds)).toPrecision(4)} (theory ${+sim.se.toPrecision(4)}).`;
  if (test) {
    const xc = test.crit * test.se;
    const inside = sim.ds.filter((d) => (tails === 'two' ? Math.abs(d) >= xc : d >= xc)).length / sim.ds.length;
    const sym = test.sym;
    tex(m, `\\text{Under } H_0:\\ \\dfrac{\\Delta}{SE} \\sim ${test.ref},\\qquad SE = ${f(test.se)}`);
    tex(m, `${sym}_{obs} = \\dfrac{\\delta}{SE} = \\dfrac{${f(test.d)}}{${f(test.se)}} = ${f(test.stat)}`);
    tex(m, tails === 'two'
      ? `\\text{Reject } H_0 \\text{ if } |${sym}| \\ge ${f(test.crit)}\\ \\ (\\text{i.e. } |\\delta| \\ge ${f(xc)})`
      : `\\text{Reject } H_0 \\text{ if } ${sym} \\ge ${f(test.crit)}\\ \\ (\\text{i.e. } \\delta \\ge ${f(xc)})`);
    summary += ` ${+(inside * 100).toFixed(1)}% of the simulated Δ values land in the rejection region` +
      (sim.boot ? '.' : ', which approximates the power of this test at these sample sizes (a Type I error rate if the true groups are equal).');
  } else {
    summary += ' This test does not use d, so there is no null curve or test statistic to draw.';
  }
  $('diffSummary').textContent = summary;
}

// ---------- analysis ----------

function tex(parent, latex) {
  const div = document.createElement('div');
  div.className = 'tex';
  if (window.katex) window.katex.render(latex, div, { displayMode: true, throwOnError: false });
  else div.textContent = latex;
  parent.appendChild(div);
}

function el(tag, className, text, parent) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

// Delta = mean_B - mean_A (random), delta = its observed value; every test here is built around them
function renderDelta() {
  const m = $('deltaMath');
  m.replaceChildren();
  const [xA, xB] = [mean(state.a), mean(state.b)];
  tex(m, '\\Delta = \\bar X_B - \\bar X_A');
  tex(m, `\\delta = \\bar x_B - \\bar x_A = ${f(xB)} - ${f(xA)} = ${f(xB - xA)}`);
}

function runAnalysis() {
  if (!state) return;
  const alpha = Number($('alpha').value);
  $('alphaError').textContent = '';
  if (!(alpha > 0 && alpha < 1)) {
    $('alphaError').textContent = 'α must be between 0 and 1.';
    return;
  }
  const tails = document.querySelector('input[name="tails"]:checked').value;
  renderDelta();
  const res = TESTS[$('test').value].run(state.a, state.b, { alpha, tails });
  lastRes = res;
  runStat(res, alpha, tails);

  const hyp = $('hyp');
  hyp.replaceChildren();
  tex(hyp, `H_0:\\ ${res.hypotheses.h0}`);
  tex(hyp, `H_1:\\ ${res.hypotheses.h1}`);

  const steps = $('steps');
  steps.replaceChildren();
  res.steps.forEach((s, i) => {
    const box = el('div', 'step', undefined, steps);
    el('h4', '', `${i + 1}. ${s.title}`, box);
    s.lines.forEach((l) => tex(box, l));
    if (s.note) el('p', 'note', s.note, box);
  });

  const dist = DISTRIBUTIONS[state.distKey];
  const c = conclude(res, { alpha, tails, trueMeanA: state.params && dist.mean(state.params[0]), trueMeanB: state.params && dist.mean(state.params[1]) });
  const out = $('concl');
  out.replaceChildren();
  el('p', 'headline', c.headline, out);
  el('p', '', c.detail, out);
  if (c.truth) el('p', 'note', c.truth, out);
  runDiff();
}

// ---------- init ----------

populateDistributions();
syncDistribution();
$('dist').addEventListener('change', syncDistribution);
$('form').addEventListener('submit', generate);
$('obsForm').addEventListener('submit', loadObservations);
document.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener('change', setMode));
for (const id of ['test', 'alpha']) $(id).addEventListener('input', runAnalysis);
const debounce = (fn, ms) => {
  let id;
  return () => {
    clearTimeout(id);
    id = setTimeout(fn, ms);
  };
};
const resample = debounce(() => runSampling(false), 200);
$('sn').addEventListener('input', () => {
  $('snRange').value = $('sn').value;
  resample();
});
$('snRange').addEventListener('input', () => {
  $('sn').value = $('snRange').value;
  resample();
});
$('sreps').addEventListener('input', resample);
$('dreps').addEventListener('input', debounce(runDiff, 200));
document.querySelectorAll('input[name="dsource"]').forEach((r) => r.addEventListener('change', runDiff));
$('dresample').addEventListener('click', () => {
  diffResample++;
  runDiff();
});
document.querySelectorAll('input[name="source"]').forEach((r) => r.addEventListener('change', () => runSampling(false)));
$('resample').addEventListener('click', () => {
  resampleCount++;
  runSampling(true);
});
document.querySelectorAll('input[name="tails"]').forEach((r) => r.addEventListener('change', runAnalysis));
