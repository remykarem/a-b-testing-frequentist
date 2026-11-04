// Hypothesis tests. Each test takes two samples and returns:
//   { hypotheses: {h0, h1}, steps: [{title, lines, note?}], p, reject, degenerate? }
// where hypotheses and step lines are LaTeX strings (rendered with KaTeX).
// To add a test, write one function and register it in TESTS.
import { mean, variance, sum, normSf, normCdf, normPdf, tSf, tPdf, zCrit, tCrit, rankWithTies } from './stats.js';

// number -> LaTeX string with `d` significant digits
export const f = (x, d = 4) => {
  if (!isFinite(x)) return x > 0 ? '\\infty' : x < 0 ? '-\\infty' : '\\text{NaN}';
  if (x !== 0 && Math.abs(x) < 1e-4) {
    const [m, e] = x.toExponential(d - 1).split('e');
    return `${m}\\times10^{${+e}}`;
  }
  return String(+x.toPrecision(d));
};

const pct = (alpha) => `${+((1 - alpha) * 100).toFixed(4)}%`;

// Confidence interval for d, plus the same interval in test-statistic units (divided by the test's SE)
const makeCI = (d, half, scale, tails) => {
  const dLo = d - half;
  const dHi = tails === 'two' ? d + half : Infinity;
  return { dLo, dHi, lo: dLo / scale, hi: dHi / scale };
};

function hypotheses(sym, tails) {
  const A = `${sym}_A`;
  const B = `${sym}_B`;
  return tails === 'two'
    ? { h0: `${A} = ${B}`, h1: `${A} \\neq ${B}` }
    : { h0: `${B} \\le ${A}`, h1: `${B} > ${A}` };
}

function decisionStep(p, alpha) {
  const reject = p <= alpha;
  return {
    title: 'Decision',
    lines: [`p = ${f(p)} ${reject ? '\\le' : '>'} \\alpha = ${f(alpha)}`],
    note: reject ? 'p ≤ α, so reject H₀.' : 'p > α, so fail to reject H₀.',
  };
}

function degenerate(hyp, steps, reason) {
  return {
    hypotheses: hyp,
    steps: [...steps, { title: 'Cannot compute', lines: [], note: reason }],
    p: NaN,
    reject: false,
    degenerate: true,
  };
}

// ---------- Welch's t-test ----------

function welch(a, b, { alpha, tails }) {
  const nA = a.length;
  const nB = b.length;
  const mA = mean(a);
  const mB = mean(b);
  const vA = variance(a);
  const vB = variance(b);
  const hyp = hypotheses('\\mu', tails);
  const d = mB - mA;
  const qA = vA / nA;
  const qB = vB / nB;
  const se = Math.sqrt(qA + qB);

  const steps = [
    {
      title: 'Sample summaries',
      lines: [
        `\\bar x_A = ${f(mA)},\\quad s_A^2 = ${f(vA)},\\quad n_A = ${nA}`,
        `\\bar x_B = ${f(mB)},\\quad s_B^2 = ${f(vB)},\\quad n_B = ${nB}`,
      ],
    },
    { title: 'Observed difference', lines: [`\\delta = \\bar x_B - \\bar x_A = ${f(mB)} - ${f(mA)} = ${f(d)}`] },
  ];
  if (!(se > 0)) return degenerate(hyp, steps, 'Both samples have zero variance, so the standard error is 0.');

  const df = (qA + qB) ** 2 / (qA ** 2 / (nA - 1) + qB ** 2 / (nB - 1));
  const t = d / se;
  const p = tails === 'two' ? 2 * tSf(Math.abs(t), df) : tSf(t, df);

  steps.push(
    {
      title: 'Standard error',
      lines: [
        `SE = \\sqrt{\\dfrac{s_A^2}{n_A} + \\dfrac{s_B^2}{n_B}} = \\sqrt{\\dfrac{${f(vA)}}{${nA}} + \\dfrac{${f(vB)}}{${nB}}} = ${f(se)}`,
      ],
    },
    { title: 'Test statistic', lines: [`t = \\dfrac{\\delta}{SE} = \\dfrac{${f(d)}}{${f(se)}} = ${f(t)}`] },
    {
      title: 'Degrees of freedom (Welch–Satterthwaite)',
      lines: [
        `\\nu = \\dfrac{\\left(\\frac{s_A^2}{n_A} + \\frac{s_B^2}{n_B}\\right)^2}{\\frac{(s_A^2/n_A)^2}{n_A-1} + \\frac{(s_B^2/n_B)^2}{n_B-1}} = ${f(df)}`,
      ],
    },
    {
      title: 'p-value',
      lines:
        tails === 'two'
          ? [`p = 2\\,P(T_{\\nu} \\ge |t|) = 2\\,P(T_{${f(df, 3)}} \\ge ${f(Math.abs(t))}) = ${f(p)}`]
          : [`p = P(T_{\\nu} \\ge t) = P(T_{${f(df, 3)}} \\ge ${f(t)}) = ${f(p)}`],
    },
  );

  if (tails === 'two') {
    const crit = tCrit(alpha / 2, df);
    steps.push({
      title: `${pct(alpha)} confidence interval for the difference`,
      lines: [
        `\\delta \\pm t_{\\alpha/2,\\nu}\\,SE = ${f(d)} \\pm ${f(crit)}\\times${f(se)} = [${f(d - crit * se)},\\ ${f(d + crit * se)}]`,
      ],
    });
  } else {
    const crit = tCrit(alpha, df);
    steps.push({
      title: `${pct(alpha)} one-sided confidence bound`,
      lines: [`\\delta - t_{\\alpha,\\nu}\\,SE = ${f(d)} - ${f(crit)}\\times${f(se)} \\Rightarrow [${f(d - crit * se)},\\ \\infty)`],
    });
  }
  steps.push(decisionStep(p, alpha));
  // `diff` describes the null distribution of Delta = mean_B - mean_A so it can be plotted
  const diff = {
    d, se, stat: t, sym: 't', ref: `t_{${f(df, 3)}}`,
    crit: tails === 'two' ? tCrit(alpha / 2, df) : tCrit(alpha, df),
    cdf: (x) => 1 - tSf(x, df),
  };
  // `statDist` is the null distribution of the test statistic, for the p-value plot
  const statDist = { sym: 't', stat: t, ref: diff.ref, crit: diff.crit, pdf: (x) => tPdf(x, df), ci: makeCI(d, diff.crit * se, se, tails) };
  return { hypotheses: hyp, steps, p, reject: p <= alpha, diff, statDist };
}

// ---------- two-proportion z-test (Bernoulli data) ----------

function zTest(a, b, { alpha, tails }) {
  const nA = a.length;
  const nB = b.length;
  const xA = sum(a);
  const xB = sum(b);
  const pA = xA / nA;
  const pB = xB / nB;
  const hyp = hypotheses('p', tails);
  const d = pB - pA;
  const pooled = (xA + xB) / (nA + nB);
  const se = Math.sqrt(pooled * (1 - pooled) * (1 / nA + 1 / nB));

  const steps = [
    {
      title: 'Sample proportions',
      lines: [
        `\\hat p_A = \\dfrac{x_A}{n_A} = \\dfrac{${xA}}{${nA}} = ${f(pA)}`,
        `\\hat p_B = \\dfrac{x_B}{n_B} = \\dfrac{${xB}}{${nB}} = ${f(pB)}`,
        `\\delta = \\hat p_B - \\hat p_A = ${f(d)}`,
      ],
    },
    {
      title: 'Pooled proportion (assumes H₀ is true)',
      lines: [`\\hat p = \\dfrac{x_A + x_B}{n_A + n_B} = \\dfrac{${xA + xB}}{${nA + nB}} = ${f(pooled)}`],
    },
  ];
  if (!(se > 0)) return degenerate(hyp, steps, 'Every observation is identical, so the pooled standard error is 0.');

  const z = d / se;
  const p = tails === 'two' ? 2 * normSf(Math.abs(z)) : normSf(z);
  const seCI = Math.sqrt((pA * (1 - pA)) / nA + (pB * (1 - pB)) / nB);
  const small = Math.min(nA * pooled, nA * (1 - pooled), nB * pooled, nB * (1 - pooled)) < 5;

  steps.push(
    {
      title: 'Standard error under H₀',
      lines: [
        `SE = \\sqrt{\\hat p(1-\\hat p)\\left(\\dfrac{1}{n_A} + \\dfrac{1}{n_B}\\right)}`,
          `= \\sqrt{${f(pooled)}\\times${f(1 - pooled)}\\times\\left(\\dfrac{1}{${nA}} + \\dfrac{1}{${nB}}\\right)}`,
        `= ${f(se)}`
      ],
      note: small ? 'Warning: an expected count is below 5, so the normal approximation may be poor.' : undefined,
    },
    { title: 'Test statistic', lines: [`z = \\dfrac{\\delta}{SE} = \\dfrac{${f(d)}}{${f(se)}} = ${f(z)}`] },
    {
      title: 'p-value',
      lines:
        tails === 'two'
          ? [`p = 2\\,P(Z \\ge |z|) = 2\\,P(Z \\ge ${f(Math.abs(z))}) = ${f(p)}`]
          : [`p = P(Z \\ge z) = P(Z \\ge ${f(z)}) = ${f(p)}`],
    },
  );

  const seLine = `SE_{CI} = \\sqrt{\\dfrac{\\hat p_A(1-\\hat p_A)}{n_A} + \\dfrac{\\hat p_B(1-\\hat p_B)}{n_B}} = ${f(seCI)}`;
  if (tails === 'two') {
    const crit = zCrit(alpha / 2);
    steps.push({
      title: `${pct(alpha)} confidence interval for the difference`,
      lines: [seLine, `\\delta \\pm z_{\\alpha/2}\\,SE_{CI} = ${f(d)} \\pm ${f(crit)}\\times${f(seCI)} = [${f(d - crit * seCI)},\\ ${f(d + crit * seCI)}]`],
      note: 'The interval uses the unpooled standard error, since it makes no assumption that H₀ holds.',
    });
  } else {
    const crit = zCrit(alpha);
    steps.push({
      title: `${pct(alpha)} one-sided confidence bound`,
      lines: [seLine, `\\delta - z_{\\alpha}\\,SE_{CI} = ${f(d)} - ${f(crit)}\\times${f(seCI)} \\Rightarrow [${f(d - crit * seCI)},\\ \\infty)`],
    });
  }
  steps.push(decisionStep(p, alpha));
  const diff = {
    d, se, stat: z, sym: 'z', ref: 'N(0,1)',
    crit: tails === 'two' ? zCrit(alpha / 2) : zCrit(alpha),
    cdf: normCdf,
  };
  const statDist = { sym: 'z', stat: z, ref: diff.ref, crit: diff.crit, pdf: normPdf, ci: makeCI(d, diff.crit * seCI, se, tails) };
  return { hypotheses: hyp, steps, p, reject: p <= alpha, diff, statDist };
}

// ---------- Mann-Whitney U (normal approximation, tie-corrected) ----------

function mannWhitney(a, b, { alpha, tails }) {
  const nA = a.length;
  const nB = b.length;
  const N = nA + nB;
  const hyp =
    tails === 'two'
      ? { h0: 'F_A = F_B', h1: 'F_A \\neq F_B' }
      : { h0: '\\text{B is not stochastically larger than A}', h1: 'P(B > A) > P(A > B)' };

  const { ranks, tieSum } = rankWithTies([...a, ...b]);
  const rA = sum(ranks.slice(0, nA));
  const rB = sum(ranks.slice(nA));
  const uB = rB - (nB * (nB + 1)) / 2;
  const muU = (nA * nB) / 2;
  const varU = ((nA * nB) / 12) * (N + 1 - tieSum / (N * (N - 1)));
  const sd = Math.sqrt(varU);

  const steps = [
    {
      title: 'Rank all observations together',
      lines: [`R_A = ${f(rA)},\\quad R_B = ${f(rB)}\\qquad (R_A + R_B = N(N+1)/2 = ${(N * (N + 1)) / 2})`],
      note: 'Tied values receive the average of the ranks they span.',
    },
    {
      title: 'U statistic for B',
      lines: [
        `U_B = R_B - \\dfrac{n_B(n_B+1)}{2} = ${f(rB)} - ${(nB * (nB + 1)) / 2} = ${f(uB)}`,
        `\\dfrac{U_B}{n_A n_B} = ${f(uB / (nA * nB))} \\quad\\text{(estimate of } P(B>A) + \\tfrac12 P(B=A)\\text{)}`,
      ],
    },
  ];
  if (!(sd > 0)) return degenerate(hyp, steps, 'Every observation is tied, so the variance of U is 0.');

  const diff = uB - muU;
  const cc = tails === 'two' ? 0.5 * Math.sign(diff) : 0.5;
  const z = (diff - cc) / sd;
  const p = tails === 'two' ? Math.min(1, 2 * normSf(Math.abs(z))) : normSf(z);

  steps.push(
    {
      title: 'Mean and standard deviation of U under H₀',
      lines: [
        `\\mu_U = \\dfrac{n_A n_B}{2} = ${f(muU)}`,
        `\\sigma_U^2 = \\dfrac{n_A n_B}{12}\\left(N+1 - \\dfrac{\\sum(t^3-t)}{N(N-1)}\\right) = \\dfrac{${nA}\\times${nB}}{12}\\left(${N + 1} - \\dfrac{${tieSum}}{${N * (N - 1)}}\\right) = ${f(varU)}`,
        `\\sigma_U = ${f(sd)}`,
      ],
      note: 'The sum runs over groups of tied values, where t is the group size.',
    },
    {
      title: 'Test statistic (with continuity correction)',
      lines: [
        tails === 'two'
          ? `z = \\dfrac{U_B - \\mu_U - \\tfrac12\\operatorname{sign}(U_B - \\mu_U)}{\\sigma_U} = \\dfrac{${f(uB)} - ${f(muU)} - ${f(cc)}}{${f(sd)}} = ${f(z)}`
          : `z = \\dfrac{U_B - \\mu_U - \\tfrac12}{\\sigma_U} = \\dfrac{${f(uB)} - ${f(muU)} - 0.5}{${f(sd)}} = ${f(z)}`,
      ],
    },
    {
      title: 'p-value',
      lines:
        tails === 'two'
          ? [`p = 2\\,P(Z \\ge |z|) = 2\\,P(Z \\ge ${f(Math.abs(z))}) = ${f(p)}`]
          : [`p = P(Z \\ge z) = P(Z \\ge ${f(z)}) = ${f(p)}`],
    },
    decisionStep(p, alpha),
  );
  const statDist = { sym: 'z', stat: z, ref: 'N(0,1)', crit: tails === 'two' ? zCrit(alpha / 2) : zCrit(alpha), pdf: normPdf };
  return { hypotheses: hyp, steps, p, reject: p <= alpha, statDist };
}

// ---------- registry ----------

export const TESTS = {
  ztest: { name: 'Two-proportion z-test', applies: (dist) => dist === 'bernoulli', run: zTest },
  welch: { name: "Welch's t-test", applies: () => true, run: welch },
  mannwhitney: { name: 'Mann–Whitney U test', applies: () => true, run: mannWhitney },
};

export const testsFor = (dist) => Object.keys(TESTS).filter((id) => TESTS[id].applies(dist));

// Plain-English conclusion, plus a check against the simulation's ground truth
export function conclude(res, { alpha, tails, trueMeanA, trueMeanB }) {
  const claim = tails === 'two' ? 'the two groups differ' : 'B is greater than A';
  let headline;
  let detail;
  if (res.degenerate) {
    headline = 'The test could not be computed.';
    detail = 'See the last step above for why.';
  } else if (res.reject) {
    headline = 'Reject the null hypothesis.';
    detail = `At α = ${alpha}, p = ${+res.p.toPrecision(3)} ≤ α, so there is statistically significant evidence that ${claim}.`;
  } else {
    headline = 'Fail to reject the null hypothesis.';
    detail = `At α = ${alpha}, p = ${+res.p.toPrecision(3)} > α, so the data do not provide statistically significant evidence that ${claim}. This is not proof that they are equal: the test may lack power at these sample sizes.`;
  }

  if (trueMeanA == null) return { headline, detail, truth: '' }; // observed data: no ground truth

  const effectExists = tails === 'two' ? trueMeanA !== trueMeanB : trueMeanB > trueMeanA;
  let verdict;
  if (res.degenerate) verdict = '';
  else if (res.reject) verdict = effectExists ? 'The test made the right call (true positive).' : 'This is a Type I error (false positive): H₀ is actually true.';
  else verdict = effectExists ? 'This is a Type II error (missed a real effect).' : 'The test made the right call (true negative).';

  const truth = `Ground truth, known because the data is simulated: true mean of A = ${+trueMeanA.toPrecision(4)}, true mean of B = ${+trueMeanB.toPrecision(4)}. ${verdict}`;
  return { headline, detail, truth };
}
