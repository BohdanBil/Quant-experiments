'use strict';
/* Portfolio Risk Lab: draws results/dashboard.json and runs what-if simulations through serve.py. */

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d', 'toggleSpikelines',
    'hoverClosestCartesian', 'hoverCompareCartesian'],
  toImageButtonOptions: {format: 'png', scale: 2},
};
const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
const THEMES = ['auto', 'light', 'dark'];
const redraws = new Map();          // chart id -> function that (re)draws it with current colours

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'style') node.style.cssText = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child != null) node.append(child instanceof Node ? child : document.createTextNode(child));
  }
  return node;
}

// numbers always in one format (the browser's own locale might group digits differently)
const num = (x) => Number(x).toLocaleString('en-AU');
const aud = (x, digits = 0) => x == null ? '–' : `${x < 0 ? '−' : ''}A$${Math.abs(x).toLocaleString('en-AU',
  {minimumFractionDigits: digits, maximumFractionDigits: digits})}`;
const pct = (x, digits = 1) => x == null ? '–' : `${(x * 100).toFixed(digits)}%`;
const pval = (p) => p == null ? '–' : p < 0.001 ? '< 0.001' : p.toFixed(3);
const plural = (n, word) => `${num(n)} ${word}${n === 1 ? '' : 's'}`;
const passes = (p) => p == null || p >= 0.05;

function rgba(hex, alpha) {
  const v = hex.replace('#', '');
  const n = parseInt(v.length === 3 ? v.split('').map((c) => c + c).join('') : v, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function merge(base, extra) {
  const out = {...base};
  for (const [key, value] of Object.entries(extra || {})) {
    const nested = value && typeof value === 'object' && !Array.isArray(value)
      && base[key] && typeof base[key] === 'object' && !Array.isArray(base[key]);
    out[key] = nested ? merge(base[key], value) : value;
  }
  return out;
}

/* Colour follows the entity: each model and each stock keeps its colour in every chart. */
function modelVar(name) {
  if (/HAR/.test(name)) return '--s1';
  if (/^Original/.test(name)) return '--s2';
  if (/^Normal/.test(name)) return '--s3';
  if (/^Student/.test(name)) return '--s4';
  if (/GJR/.test(name)) return '--s5';
  if (/^Historical/.test(name)) return '--s6';
  if (/EWMA/.test(name)) return '--s7';
  return '--muted';
}
const modelColor = (name) => css(modelVar(name));
const stockVar = (i) => SERIES[i % SERIES.length];
const shortModel = (name) => (/^Original/.test(name) ? 'First version (normal + drift)'
  : name.replace(/ \(\d+d\)$/, ''));
const dayLabel = (day) => plural(day, 'day');       // "10 days"
const dayAdj = (day) => `${day}-day`;               // "10-day VaR"

/* ---------- Plotly plumbing ---------- */
function layout(extra) {
  const axis = {
    gridcolor: css('--grid'), linecolor: css('--axis'), zerolinecolor: css('--axis'),
    tickfont: {color: css('--muted'), size: 11}, title: {font: {color: css('--ink-2'), size: 12}},
    automargin: true,
  };
  return merge({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: {family: css('--font'), color: css('--ink-2'), size: 12},
    margin: {l: 10, r: 14, t: 10, b: 10, pad: 2},
    xaxis: axis, yaxis: axis,
    legend: {orientation: 'h', x: 0, xanchor: 'left', y: 1.01, yanchor: 'bottom',
      font: {color: css('--ink-2'), size: 12}, bgcolor: 'rgba(0,0,0,0)'},
    hoverlabel: {bgcolor: css('--surface'), bordercolor: css('--axis'),
      font: {color: css('--ink'), family: css('--font')}},
    hovermode: 'closest',
  }, extra);
}

function plot(id, build) {
  const draw = () => {
    const node = $(id);
    if (!node || !window.Plotly) return;
    const figure = build();
    Plotly.react(node, figure.data, layout(figure.layout), PLOT_CONFIG);
  };
  redraws.set(id, draw);
  draw();
  return draw;
}

const redrawAll = () => redraws.forEach((draw) => draw());
const vline = (x, color, dash = 'dash') => ({type: 'line', x0: x, x1: x, yref: 'paper', y0: 0, y1: 1,
  line: {color, width: 1.3, dash}});
const hline = (y, color, dash = 'dash') => ({type: 'line', xref: 'paper', x0: 0, x1: 1, y0: y, y1: y,
  line: {color, width: 1.3, dash}});
const label = (props) => ({showarrow: false, font: {size: 11, color: css('--ink-2')}, ...props});

/* ---------- shared UI pieces ---------- */
function kpi(title, value, sub, extra) {
  return h('div', {class: 'kpi'},
    h('div', {class: 'kpi-label'}, title),
    h('div', {class: 'kpi-value'}, value),
    h('div', {class: 'kpi-sub'}, sub),
    extra ? h('div', {style: 'margin-top:6px'}, extra) : null);
}

const pill = (ok, text) => h('span', {class: `pill ${ok ? 'pill-good' : 'pill-bad'}`},
  ok ? '✓' : '✗', text ? ` ${text}` : '');

function zonePill(count) {
  const zone = count <= 4 ? 'green' : count <= 9 ? 'yellow' : 'red';
  const cls = {green: 'pill-good', yellow: 'pill-warn', red: 'pill-bad'}[zone];
  const icon = {green: '●', yellow: '▲', red: '■'}[zone];
  return h('span', {class: `pill ${cls}`}, `${icon} ${zone}`);
}

const table = (head, rows) => h('table', {},
  h('thead', {}, h('tr', {}, head.map((x) => h('th', {}, x)))), h('tbody', {}, rows));

function options(select, items, selected) {
  select.replaceChildren(...items.map(([value, text]) =>
    h('option', {value, selected: String(value) === String(selected)}, text)));
}

/* ---------- theme ---------- */
function applyTheme(theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  $('theme-toggle').textContent = `Theme: ${theme}`;
  try { localStorage.setItem('risk-lab-theme', theme); } catch (error) { /* storage may be blocked */ }
  redrawAll();
}

function setupTheme() {
  let theme = new URLSearchParams(window.location.search).get('theme');     // e.g. ?theme=dark
  if (!theme) {
    try { theme = localStorage.getItem('risk-lab-theme') || 'auto'; } catch (error) { theme = 'auto'; }
  }
  applyTheme(THEMES.includes(theme) ? theme : 'auto');
  $('theme-toggle').addEventListener('click', () => {
    const current = document.documentElement.dataset.theme || 'auto';
    applyTheme(THEMES[(THEMES.indexOf(current) + 1) % THEMES.length]);
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
}

/* ---------- start ---------- */
function fatal(message) {
  const node = $('fatal');
  node.textContent = message;
  node.hidden = false;
  $('portfolio-line').textContent = '';
}

async function init() {
  setupTheme();
  if (!window.Plotly) {
    fatal('The charting library could not be loaded (it comes from cdn.jsdelivr.net). '
      + 'Check your internet connection and reload the page.');
    return;
  }
  let data;
  try {
    const response = await fetch('/api/results', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
  } catch (error) {
    fatal(`Could not load the results (${error.message}). Start the dashboard with `
      + '"python serve.py" in the project folder and open the address it prints.');
    return;
  }
  renderOverview(data);
  renderFutures(data);
  renderModels(data);
  if (data.backtest) renderBacktest(data.backtest);
  else $('backtest').hidden = true;
  renderDrivers(data);
  renderMarket(data);
  setupWhatIf(data);
  const m = data.meta;
  $('footer-line').textContent = `Data: Yahoo Finance, ${m.data_start} to ${m.data_end} `
    + `(${num(m.trading_days)} trading days, ${m.bad_opens} bad opening prices cleaned). `
    + `Report generated ${m.generated}, ${num(m.paths)} paths per model. `
    + 'A study, not investment advice.';
}
document.addEventListener('DOMContentLoaded', init);

/* ---------- overview ---------- */
function renderOverview(d) {
  const H = d.horizon;
  const main = d.measures[d.main_model];
  const original = d.measures[d.meta.original_model];
  const tickers = d.meta.tickers;
  $('portfolio-line').textContent = `${aud(d.value)} portfolio of ${tickers.length} ASX stocks · bought and `
    + `held · data to ${d.meta.data_end} · main model: ${d.main_model}`;
  $('headline').replaceChildren(`Over the next ${H} trading days there is a 1-in-100 chance of losing more than `,
    h('span', {class: 'num'}, aud(main[H].var99)), '.');
  $('subheadline').textContent = `In the worst 2.5% of outcomes the loss averages ${aud(main[H].es975)}. `
    + `The first version of this project put the 1-in-100 loss at only ${aud(original[H].var99)}.`;
  $('weights-strip').replaceChildren(...tickers.map((t, i) => h('span', {class: 'chip'},
    h('span', {class: 'dot', style: `background: var(${stockVar(i)})`}), h('b', {}, t), pct(d.weights[t], 1))));

  const tiles = [];
  for (const day of d.days) {
    tiles.push(kpi(`${dayAdj(day)} VaR 99%`, aud(main[day].var99),
      day === H ? `first version: ${aud(original[day].var99)}` : `${pct(main[day].var99 / d.value)} of the portfolio`));
  }
  tiles.push(kpi(`${dayAdj(H)} Expected Shortfall`, aud(main[H].es975), 'average of the worst 2.5%'));
  tiles.push(kpi('Volatility now', pct(d.volatility.portfolio_now),
    `per year; two-year average ${pct(d.volatility.portfolio_average)}`));
  if (d.backtest) {
    const row = d.backtest.summary.find((r) => r.model === d.backtest.main_model);
    const ok = passes(row['95% Kupiec p']) && passes(row['99% Kupiec p']) && passes(row['99% clustering p']);
    tiles.push(kpi('Backtest, 99% VaR', pct(row['99% breaches'], 2), 'of days breached; target 1%',
      pill(ok, ok ? 'passes all tests' : 'fails a test')));
  }
  $('kpis').replaceChildren(...tiles);
}

/* ---------- simulated futures ---------- */
function fanFigure({fan, main, compare, value, samples}) {
  const f = fan[main];
  const x = f.p50.map((_, i) => i);
  const color = modelColor(main);
  const band = (lo, hi, alpha, name) => [
    {x, y: lo, type: 'scatter', mode: 'lines', line: {width: 0}, hoverinfo: 'skip', showlegend: false},
    {x, y: hi, type: 'scatter', mode: 'lines', line: {width: 0}, fill: 'tonexty',
      fillcolor: rgba(color, alpha), name, hoverinfo: 'skip'},
  ];
  const traces = [];
  (samples || []).forEach((path, i) => traces.push({x, y: path, type: 'scatter', mode: 'lines',
    line: {width: 0.7, color: rgba(css('--muted'), 0.45)}, hoverinfo: 'skip', name: 'Sample paths',
    legendgroup: 'samples', showlegend: i === 0}));
  traces.push(...band(f.p1, f.p99, 0.10, '98% of paths'), ...band(f.p5, f.p95, 0.17, '90% of paths'),
    ...band(f.p25, f.p75, 0.28, '50% of paths'));
  traces.push({x, y: f.p50, type: 'scatter', mode: 'lines', line: {width: 2.4, color}, name: 'Median',
    hovertemplate: 'Median A$%{y:,.0f}<extra></extra>'});
  traces.push({x, y: f.p1, type: 'scatter', mode: 'lines', line: {width: 1.6, color, dash: 'dot'},
    name: `${main}: worst 1 in 100`, hovertemplate: `${main}, worst 1 in 100: A$%{y:,.0f}<extra></extra>`});
  if (compare && fan[compare]) {
    traces.push({x, y: fan[compare].p1, type: 'scatter', mode: 'lines',
      line: {width: 2, color: modelColor(compare)}, name: `${compare}: worst 1 in 100`,
      hovertemplate: `${compare}, worst 1 in 100: A$%{y:,.0f}<extra></extra>`});
  }
  return {data: traces, layout: {
    hovermode: 'x unified', legend: {traceorder: 'normal'},
    xaxis: {title: {text: 'Trading days ahead'}, range: [0, x.length - 1]},
    yaxis: {tickprefix: 'A$', tickformat: ',.0f', title: {text: 'Portfolio value'}},
    shapes: [hline(value, css('--axis'), 'solid')],
  }};
}

function distFigure({histograms, measures, main, day, models, logScale}) {
  const hist = histograms[day];
  const centers = hist.edges.slice(0, -1).map((e, i) => (e + hist.edges[i + 1]) / 2);
  const traces = models.filter((m) => hist.share[m]).map((m) => ({
    x: centers, y: hist.share[m].map((v) => (logScale && v <= 0 ? null : v)), type: 'scatter', mode: 'lines',
    line: {shape: 'hvh', width: m === main ? 2.2 : 1.6, color: modelColor(m)},
    fill: m === main && !logScale ? 'tozeroy' : 'none', fillcolor: rgba(modelColor(m), 0.12), name: m,
    hovertemplate: `${m}: %{y:.2~p} of paths<extra></extra>`,
  }));
  const m = measures[main][day];
  const lines = [['VaR 99%', m.var99, 0.99], ['ES 97.5%', m.es975, 0.78]];
  return {data: traces, layout: {
    hovermode: 'x unified',
    xaxis: {title: {text: `Profit or loss after ${dayLabel(day)}`}, tickprefix: 'A$', tickformat: ',.0f'},
    yaxis: {title: {text: 'Share of paths'}, type: logScale ? 'log' : 'linear', tickformat: '.1~p'},
    shapes: lines.map(([, v]) => vline(-v, css('--ink-2'))),
    annotations: lines.map(([name, v, y]) => label({x: -v, yref: 'paper', y, xanchor: 'right', xshift: -6,
      align: 'right', text: `${name}<br>${aud(v)} loss`})),
  }};
}

function renderFutures(d) {
  const models = Object.keys(d.fan);
  const compare = $('fan-compare');
  options(compare, [['', 'nothing'], ...models.filter((m) => m !== d.main_model).map((m) => [m, m])],
    d.meta.comparison_model);
  const drawFan = plot('fan-chart', () => fanFigure({fan: d.fan, main: d.main_model, compare: compare.value,
    value: d.value, samples: $('fan-samples').checked ? d.sample_paths : null}));
  compare.addEventListener('change', drawFan);
  $('fan-samples').addEventListener('change', drawFan);

  const horizon = $('dist-horizon');
  options(horizon, d.days.map((day) => [day, dayLabel(day)]), d.horizon);
  const shown = new Set([d.main_model, d.meta.comparison_model]);
  const drawDist = plot('dist-chart', () => distFigure({histograms: d.histograms, measures: d.measures,
    main: d.main_model, day: +horizon.value, models: models.filter((m) => shown.has(m)),
    logScale: $('dist-log').checked}));
  $('dist-models').replaceChildren(...models.map((m) => {
    const button = h('button', {type: 'button', class: 'chip-toggle', 'aria-pressed': String(shown.has(m))},
      h('span', {class: 'dot', style: `background: var(${modelVar(m)})`}), m);
    button.addEventListener('click', () => {
      if (shown.has(m)) shown.delete(m); else shown.add(m);
      button.setAttribute('aria-pressed', String(shown.has(m)));
      drawDist();
    });
    return button;
  }));
  horizon.addEventListener('change', drawDist);
  $('dist-log').addEventListener('change', drawDist);
}

/* ---------- model comparison ---------- */
const MEASURE_NAMES = {var99: 'VaR 99%', es975: 'Expected Shortfall 97.5%', var95: 'VaR 95%'};

function modelsFigure(d, day, key) {
  const rows = Object.entries(d.measures).map(([name, byDay]) =>
    ({name, v: byDay[day][key], se: byDay[day][`${key}_se`]}));
  rows.push({name: 'Closed form (normal)', v: d.closed_form[day][key], se: 0});
  if (day === 1) rows.push({name: 'Historical, last 500 days', v: d.historical_1d[key], se: 0});
  rows.sort((a, b) => a.v - b.v);
  const color = (name) => (name === d.main_model ? css('--s1')
    : name === d.meta.original_model ? css('--s2') : css('--deemph'));
  return {data: [{
    type: 'bar', orientation: 'h', y: rows.map((r) => r.name), x: rows.map((r) => r.v),
    marker: {color: rows.map((r) => color(r.name))},
    error_x: {type: 'data', array: rows.map((r) => 2 * r.se), color: css('--ink-2'), thickness: 1, width: 3},
    text: rows.map((r) => aud(r.v)), textposition: 'outside', cliponaxis: false,
    textfont: {color: css('--ink-2')}, hovertemplate: '%{y}: A$%{x:,.0f}<extra></extra>',
  }], layout: {
    showlegend: false, bargap: 0.45,
    xaxis: {tickprefix: 'A$', tickformat: ',.0f', title: {text: `${dayAdj(day)} ${MEASURE_NAMES[key]}`},
      range: [0, Math.max(...rows.map((r) => r.v)) * 1.2]},
  }};
}

function renderModels(d) {
  const horizon = $('models-horizon');
  const measure = $('models-measure');
  options(horizon, d.days.map((day) => [day, dayLabel(day)]), d.horizon);
  const draw = plot('models-chart', () => modelsFigure(d, +horizon.value, measure.value));
  horizon.addEventListener('change', draw);
  measure.addEventListener('change', draw);

  const c = d.convergence;
  plot('convergence-chart', () => {
    const color = css('--s1');
    return {data: [
      {x: c.sizes, y: c.var99.p5, type: 'scatter', mode: 'lines', line: {width: 0}, hoverinfo: 'skip',
        showlegend: false},
      {x: c.sizes, y: c.var99.p95, type: 'scatter', mode: 'lines', line: {width: 0}, fill: 'tonexty',
        fillcolor: rgba(color, 0.16), name: '90% of repeated runs', hoverinfo: 'skip'},
      {x: c.sizes, y: c.var99.p50, type: 'scatter', mode: 'lines+markers', name: 'Typical run',
        line: {color, width: 2}, marker: {size: 8, color, line: {color: css('--surface'), width: 2}},
        customdata: c.sizes.map((_, i) => [c.var99.p5[i], c.var99.p95[i]]),
        hovertemplate: '%{x:,} paths: A$%{customdata[0]:,.0f} to A$%{customdata[1]:,.0f}<extra></extra>'},
    ], layout: {
      xaxis: {type: 'log', title: {text: 'Simulated paths'}, tickvals: c.sizes, ticktext: c.sizes.map(num)},
      yaxis: {tickprefix: 'A$', tickformat: ',.0f', title: {text: `${dayAdj(c.day)} VaR 99%`}},
      shapes: [hline(c.var99.all, css('--ink-2'))],
      annotations: [label({xref: 'paper', x: 1, y: c.var99.all, xanchor: 'right', yanchor: 'bottom',
        text: `all ${num(c.paths)} paths: ${aud(c.var99.all)}`})],
    }};
  });
  $('convergence-note').textContent = `With 100 paths, as in the first version, 90% of runs put the `
    + `${dayAdj(c.day)} VaR 99% anywhere from ${aud(c.var99.p5[0])} to ${aud(c.var99.p95[0])}.`;

  const rows = [];
  for (const day of d.days) {
    rows.push(h('tr', {}, h('td', {class: 'group', colspan: 4}, dayLabel(day))));
    const entries = [['Closed form (normal)', d.closed_form[day], false]];
    if (day === 1) entries.push(['Historical, last 500 days', d.historical_1d, false]);
    for (const [name, byDay] of Object.entries(d.measures)) entries.push([name, byDay[day], true]);
    for (const [name, v, withSe] of entries) {
      const cell = (k) => h('td', {}, aud(v[k]),
        withSe ? h('span', {class: 'kpi-sub'}, ` ±${v[`${k}_se`].toFixed(1)}`) : null);
      rows.push(h('tr', {class: name === d.main_model ? 'main-row' : null}, h('td', {}, name),
        cell('var95'), cell('var99'), cell('es975')));
    }
  }
  $('risk-table').replaceChildren(table(['Model', 'VaR 95%', 'VaR 99%', 'ES 97.5%'], rows));
}

/* ---------- backtest ---------- */
function backtestFigure(b, model, losses) {
  const v = b.var99[model].map((x) => x * 100);
  const hitX = [];
  const hitY = [];
  losses.forEach((loss, i) => {
    if (loss > v[i]) { hitX.push(b.dates[i]); hitY.push(loss); }
  });
  return {data: [
    {type: 'bar', x: b.dates, y: losses.map((l) => Math.max(l, 0)), marker: {color: css('--deemph')},
      name: 'Daily loss', hovertemplate: 'loss %{y:.2f}%<extra></extra>'},
    {type: 'scatter', mode: 'lines', x: b.dates, y: v, line: {color: modelColor(model), width: 1.3},
      name: 'VaR 99% forecast', hovertemplate: 'VaR %{y:.2f}%<extra></extra>'},
    {type: 'scatter', mode: 'markers', x: hitX, y: hitY, name: 'Breach: loss above VaR',
      marker: {color: css('--critical'), size: 7, line: {color: css('--surface'), width: 1.5}},
      hovertemplate: 'breach %{y:.2f}%<extra></extra>'},
  ], layout: {
    uirevision: 'backtest', bargap: 0, hovermode: 'x unified',
    yaxis: {ticksuffix: '%', title: {text: 'Daily loss'}, rangemode: 'tozero'},
    xaxis: {type: 'date', rangeslider: {visible: true, thickness: 0.07, bgcolor: css('--surface-2'),
      bordercolor: css('--border')}},
  }};
}

function ratesFigure(b) {
  const rows = b.summary.map((r) => {
    const problems = [];
    if (!passes(r['99% Kupiec p'])) problems.push('wrong rate');
    if (!passes(r['99% clustering p'])) problems.push('clusters');
    return {name: shortModel(r.model), rate: r['99% breaches'], problems};
  });
  return {data: [{
    type: 'bar', orientation: 'h', y: rows.map((r) => r.name), x: rows.map((r) => r.rate),
    marker: {color: rows.map((r) => (r.problems.length ? css('--critical') : css('--good')))},
    text: rows.map((r) => `${pct(r.rate, 2)}  ${r.problems.length ? `✗ ${r.problems.join(', ')}` : '✓ passes'}`),
    textposition: 'outside', cliponaxis: false, textfont: {color: css('--ink-2')},
    hovertemplate: '%{y}: %{x:.2%} of days<extra></extra>',
  }], layout: {
    showlegend: false, bargap: 0.4, margin: {t: 26},
    yaxis: {autorange: 'reversed'},
    xaxis: {tickformat: '.1%', title: {text: '99% VaR breaches (target 1%)'},
      range: [0, Math.max(...rows.map((r) => r.rate)) * 1.9]},
    shapes: [vline(0.01, css('--ink'))],
    annotations: [label({x: 0.01, yref: 'paper', y: 1, yanchor: 'bottom', text: 'target'})],
  }};
}

function renderBacktest(b) {
  const select = $('bt-model');
  options(select, Object.keys(b.var99).map((m) => [m, shortModel(m)]), b.main_model);
  const losses = b.losses.map((x) => x * 100);
  const draw = plot('bt-chart', () => backtestFigure(b, select.value, losses));
  const zoom = (range) => Plotly.relayout('bt-chart', range ? {'xaxis.range': range} : {'xaxis.autorange': true});
  select.closest('.controls').append(
    h('button', {type: 'button', class: 'ghost-button', onclick: () => zoom(['2007-07-01', '2009-12-31'])}, '2008 crisis'),
    h('button', {type: 'button', class: 'ghost-button', onclick: () => zoom(['2020-01-01', '2020-12-31'])}, 'COVID 2020'),
    h('button', {type: 'button', class: 'ghost-button', onclick: () => zoom(null)}, 'All years'));
  const stats = () => {
    const row = b.summary.find((r) => r.model === select.value);
    const pairs = b.back_to_back[select.value] || [];
    $('bt-stats').replaceChildren(
      h('span', {}, 'Breaches ', h('b', {}, pct(row['99% breaches'], 2)), ' (target 1%) ',
        pill(passes(row['99% Kupiec p']))),
      h('span', {}, 'Clustering test p = ', h('b', {}, pval(row['99% clustering p'])), ' ',
        pill(passes(row['99% clustering p']))),
      h('span', {}, 'Back-to-back breaches: ', h('b', {}, String(pairs.length)),
        pairs.length ? ` (${pairs.join(', ')})` : ''));
  };
  select.addEventListener('change', () => { draw(); stats(); });
  stats();
  plot('bt-rates', () => ratesFigure(b));

  const rows = b.summary.map((r) => h('tr', {class: r.model === b.main_model ? 'main-row' : null},
    h('td', {}, shortModel(r.model)),
    h('td', {}, pct(r['95% breaches'], 2)), h('td', {}, pval(r['95% Kupiec p']), ' ', pill(passes(r['95% Kupiec p']))),
    h('td', {}, pct(r['99% breaches'], 2)), h('td', {}, pval(r['99% Kupiec p']), ' ', pill(passes(r['99% Kupiec p']))),
    h('td', {}, pval(r['99% clustering p']), ' ', pill(passes(r['99% clustering p']))),
    h('td', {}, `${r['worst 250d']} `, zonePill(r['worst 250d'])),
    h('td', {}, r['ES 97.5% ratio'].toFixed(2))));
  $('bt-table').replaceChildren(table(['Model', '95% breaches', 'Kupiec p', '99% breaches', 'Kupiec p',
    'Clustering p', 'Worst 250 days', 'ES ratio'], rows));

  const regimes = [...new Set(b.regimes.map((r) => r.regime))];
  const models = [...new Set(b.regimes.map((r) => r.model))];
  const regimeRows = models.map((m) => h('tr', {class: m === b.main_model ? 'main-row' : null},
    h('td', {}, shortModel(m)),
    ...regimes.map((g) => {
      const r = b.regimes.find((x) => x.model === m && x.regime === g);
      const ok = passes(r.kupiec_p) && passes(r.clustering_p);
      return h('td', {}, `${pct(r.breaches, 2)} `, pill(ok));
    })));
  $('bt-regimes').replaceChildren(table(['Model', ...regimes], regimeRows));
}

/* ---------- risk drivers ---------- */
function contribFigure(c) {
  const order = c.tickers.map((_, i) => i).sort((a, b) => c.share[a] - c.share[b]);
  const y = order.map((i) => c.tickers[i]);
  const bar = (name, values, color, what) => ({type: 'bar', orientation: 'h', name, y,
    x: order.map((i) => values[i]), marker: {color}, text: order.map((i) => pct(values[i], 0)),
    textposition: 'outside', cliponaxis: false, textfont: {color: css('--ink-2')},
    hovertemplate: `%{y}: %{x:.1%} of the ${what}<extra></extra>`});
  return {data: [
    bar('Share of money', c.weights, css('--deemph'), 'money'),
    bar(`Share of risk (${dayAdj(c.day)} ES)`, c.share, css('--s1'), 'risk'),
  ], layout: {
    barmode: 'group', bargap: 0.3, bargroupgap: 0.12,
    xaxis: {tickformat: '.0%', range: [0, Math.max(...c.share, ...c.weights) * 1.25]},
  }};
}

function stressFigure(stress, measures, days) {
  const names = stress.map((s) => (s.start ? `${s.name}<br>${s.start} to ${s.end}` : `${s.name}<br>ending ${s.end}`));
  const refs = days.filter((day) => day >= 10);
  return {data: [{
    type: 'bar', orientation: 'h', y: names, x: stress.map((s) => s.loss_aud), marker: {color: css('--s2')},
    text: stress.map((s) => `${aud(s.loss_aud)} (${pct(s.loss)})`), textposition: 'outside',
    cliponaxis: false, textfont: {color: css('--ink-2')},
    hovertemplate: '%{y}<br>loss A$%{x:,.0f}<extra></extra>',
  }], layout: {
    showlegend: false, bargap: 0.45, margin: {t: 28},
    yaxis: {autorange: 'reversed'},
    xaxis: {tickprefix: 'A$', tickformat: ',.0f', title: {text: 'Loss on today\'s portfolio'},
      range: [0, Math.max(...stress.map((s) => s.loss_aud)) * 1.4]},
    shapes: refs.map((day) => vline(measures[day].es975, css('--s1'))),
    annotations: refs.map((day, i) => label({x: measures[day].es975, yref: 'paper', y: 1, yanchor: 'bottom',
      xanchor: i % 2 ? 'left' : 'right', text: `model ${dayAdj(day)} ES`})),
  }};
}

function drawdownFigure(pr) {
  const e = pr.drawdown_hist.edges;
  const centers = pr.drawdown_hist.share.map((_, i) => (e[i] + e[i + 1]) / 2);
  return {data: [{type: 'bar', x: centers, y: pr.drawdown_hist.share, marker: {color: css('--s1')},
    hovertemplate: 'largest fall %{x:.1%}: %{y:.1%} of paths<extra></extra>'}], layout: {
    showlegend: false, bargap: 0.06, margin: {t: 28},
    xaxis: {tickformat: '.0%', title: {text: 'Largest peak-to-trough fall within the horizon'}},
    yaxis: {tickformat: '.0%', title: {text: 'Share of paths'}},
    shapes: [vline(pr.median_drawdown, css('--ink-2')), vline(pr.p95_drawdown, css('--ink-2'))],
    annotations: [
      label({x: pr.median_drawdown, yref: 'paper', y: 1, yanchor: 'bottom', text: `median ${pct(pr.median_drawdown)}`}),
      label({x: pr.p95_drawdown, yref: 'paper', y: 1, yanchor: 'bottom', text: `1 in 20: ${pct(pr.p95_drawdown)}`}),
    ],
  }};
}

function putFigure(put) {
  const names = ['VaR 99%', 'ES 97.5%'];
  const bar = (name, m, color) => ({type: 'bar', name, x: names, y: [m.var99, m.es975], marker: {color},
    text: [aud(m.var99), aud(m.es975)], textposition: 'outside', cliponaxis: false,
    textfont: {color: css('--ink-2')}, hovertemplate: `${name}, %{x}: A$%{y:,.0f}<extra></extra>`});
  return {data: [bar('Without the put', put.before, css('--deemph')), bar('With the put', put.after, css('--s1'))],
    layout: {barmode: 'group', bargap: 0.45, bargroupgap: 0.1,
      yaxis: {tickprefix: 'A$', tickformat: ',.0f', title: {text: 'Loss'},
        range: [0, Math.max(put.before.var99, put.before.es975) * 1.2]}}};
}

function renderDrivers(d) {
  plot('contrib-chart', () => contribFigure(d.contributions));
  const mv = d.min_variance;
  $('minvar-note').textContent = `Long-only minimum-variance weights (${d.meta.tickers.map((t, i) =>
    `${t} ${pct(mv.weights[i], 0)}`).join(', ')}) would cut the ${dayAdj(mv.day)} ES from `
    + `${aud(mv.es_current)} to ${aud(mv.es)}, measured on the same data they were fitted to.`;
  plot('stress-chart', () => stressFigure(d.stress, d.measures[d.main_model], d.days));
  const pr = d.path_risk[d.main_model];
  plot('drawdown-chart', () => drawdownFigure(pr));
  $('drawdown-note').textContent = `There is a ${pct(pr.p_down_any)} chance of being down `
    + `${pct(d.stop_loss, 0)} at some point within ${plural(d.horizon, 'day')}, but only ${pct(pr.p_down_end)} `
    + 'of being down that much on the last day. A single end-date VaR hides the difference.';
  plot('put-chart', () => putFigure(d.put));
  $('put-note').textContent = `A ${dayAdj(d.horizon)} put struck at ${pct(d.put.strike, 0)} of today's value `
    + `costs about ${aud(d.put.premium)} (the model's fair price). It caps the loss, a kinked payoff that `
    + 'simulation handles and a closed-form normal VaR cannot.';
}

/* ---------- market ---------- */
function renderMarket(d) {
  const m = d.market;
  const tickers = d.meta.tickers;
  const drawGrowth = plot('growth-chart', () => {
    const logScale = $('growth-log').checked;
    const traces = tickers.map((t, i) => ({type: 'scatter', mode: 'lines', x: m.weekly_dates, y: m.growth[t],
      name: t, line: {color: css(stockVar(i)), width: 1.5}, hovertemplate: `${t} A$%{y:,.0f}<extra></extra>`}));
    traces.push({type: 'scatter', mode: 'lines', x: m.weekly_dates, y: m.portfolio_growth,
      name: 'Equal-weight portfolio', line: {color: css('--ink'), width: 2.6},
      hovertemplate: 'Portfolio A$%{y:,.0f}<extra></extra>'});
    const logTicks = [25, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
    return {data: traces, layout: {hovermode: 'x unified', xaxis: {type: 'date'},
      yaxis: logScale ? {type: 'log', tickvals: logTicks, ticktext: logTicks.map((v) => `A$${num(v)}`),
        title: {text: 'Value of A$100'}}
        : {type: 'linear', tickprefix: 'A$', tickformat: ',.0f', title: {text: 'Value of A$100'}}}};
  });
  $('growth-log').addEventListener('change', drawGrowth);

  plot('vol-chart', () => {
    const crises = [['2007-11-01', '2009-03-31', 'financial crisis'], ['2020-02-20', '2020-06-30', 'COVID']];
    return {data: [{type: 'scatter', mode: 'lines', x: m.vol_dates, y: m.portfolio_vol,
      line: {color: css('--s1'), width: 1.8}, hovertemplate: '%{x}: %{y:.1%} a year<extra></extra>'}],
    layout: {
      showlegend: false, xaxis: {type: 'date'},
      yaxis: {tickformat: '.0%', title: {text: 'Volatility per year'}, rangemode: 'tozero'},
      shapes: [...crises.map(([x0, x1]) => ({type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1,
        fillcolor: rgba(css('--muted'), 0.16), line: {width: 0}, layer: 'below'})),
      hline(d.volatility.portfolio_now, css('--ink-2'))],
      annotations: [...crises.map(([x0, , text]) => label({x: x0, yref: 'paper', y: 1, xanchor: 'left',
        yanchor: 'top', xshift: 4, text})),
      label({xref: 'paper', x: 1, y: d.volatility.portfolio_now, xanchor: 'right', yanchor: 'bottom',
        yshift: 2, bgcolor: css('--surface'), borderpad: 2,
        text: `today's forecast ${pct(d.volatility.portfolio_now)}`})],
    }};
  });

  const c = m.correlation;
  const period = $('corr-window');
  options(period, [['recent', `last ${num(c.days)} days`], ['full', `since ${c.since.slice(0, 4)}`]], 'recent');
  const drawCorr = plot('corr-chart', () => {
    const matrix = period.value === 'full' ? c.matrix_full : c.matrix;
    return {data: [{type: 'heatmap', z: matrix, x: c.tickers, y: c.tickers, zmin: -0.2, zmax: 1,
      colorscale: [[0, css('--seq-100')], [0.4, css('--seq-300')], [0.7, css('--seq-500')], [1, css('--seq-700')]],
      text: matrix.map((row) => row.map((v) => v.toFixed(2))), texttemplate: '%{text}',
      hovertemplate: '%{y} and %{x}: correlation %{z:.2f}<extra></extra>', xgap: 2, ygap: 2,
      colorbar: {thickness: 10, outlinewidth: 0, tickfont: {color: css('--muted')}}}],
    layout: {yaxis: {autorange: 'reversed'}}};
  });
  period.addEventListener('change', drawCorr);
  const at = (matrix, a, b) => matrix[c.tickers.indexOf(a)][c.tickers.indexOf(b)].toFixed(2);
  if (['CBA', 'BHP', 'WBC'].every((t) => c.tickers.includes(t))) {
    $('corr-note').textContent = `Banks still move together (CBA and WBC: ${at(c.matrix, 'CBA', 'WBC')}), but `
      + `banks and miners have drifted apart: CBA and BHP ${at(c.matrix, 'CBA', 'BHP')} recently, against `
      + `${at(c.matrix_full, 'CBA', 'BHP')} since ${c.since.slice(0, 4)}. Lower correlation means more diversification.`;
  }

  plot('stockvol-chart', () => {
    const v = d.volatility;
    const series = [['now (HAR)', '--s1', 'Now: HAR (main model)'], ['now (GJR-GARCH)', '--s5', 'Now: GJR-GARCH'],
      ['now (EWMA)', '--s7', 'Now: EWMA'], ['10-year average', '--deemph', '10-year average']];
    return {data: series.map(([key, color, name]) => ({type: 'bar', name, x: v.tickers, y: v[key],
      marker: {color: css(color)}, hovertemplate: `%{x}, ${name}: %{y:.1%}<extra></extra>`})),
    layout: {barmode: 'group', bargap: 0.28, bargroupgap: 0.08,
      yaxis: {tickformat: '.0%', title: {text: 'Volatility per year'}}}};
  });
}

/* ---------- what-if ---------- */
function setupWhatIf(d) {
  const tickers = d.meta.tickers;
  const fields = {};
  const status = (text) => { $('whatif-status').textContent = text; };
  const updateTotal = () => {
    const total = tickers.reduce((sum, t) => sum + (Number(fields[t].number.value) || 0), 0);
    $('whatif-total').textContent = Math.abs(total - 100) < 0.5 ? 'Weights add up to 100%.'
      : `Weights add up to ${total.toFixed(1)}%; they will be scaled to 100%.`;
  };
  $('whatif-weights').replaceChildren(...tickers.map((t, i) => {
    const start = (d.weights[t] * 100).toFixed(1);
    const range = h('input', {type: 'range', min: 0, max: 100, step: 0.1, value: start, 'aria-label': `${t} weight`});
    const number = h('input', {type: 'number', min: 0, max: 100, step: 0.1, value: start,
      'aria-label': `${t} weight in percent`});
    range.addEventListener('input', () => { number.value = range.value; updateTotal(); });
    number.addEventListener('input', () => { range.value = number.value; updateTotal(); });
    fields[t] = {range, number};
    return h('div', {class: 'weight-row'},
      h('label', {}, h('span', {class: 'dot', style: `background: var(${stockVar(i)})`}), t), range, number);
  }));
  const setWeights = (weightOf) => {
    tickers.forEach((t, i) => {
      const v = (weightOf(t, i) * 100).toFixed(1);
      fields[t].range.value = v;
      fields[t].number.value = v;
    });
    updateTotal();
  };
  document.querySelectorAll('[data-preset]').forEach((button) => button.addEventListener('click', () => {
    if (button.dataset.preset === 'equal') setWeights(() => 1 / tickers.length);
    else setWeights((t, i) => d.min_variance.weights[i]);
  }));
  updateTotal();

  $('whatif-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const weights = Object.fromEntries(tickers.map((t) => [t, Math.max(0, Number(fields[t].number.value) || 0)]));
    if (Object.values(weights).every((v) => v === 0)) {
      status('Give at least one stock a positive weight.');
      return;
    }
    const body = {weights, horizon: Number($('whatif-horizon').value), paths: Number($('whatif-paths').value),
      value: Number($('whatif-value').value) || d.value};
    const button = $('whatif-run');
    button.disabled = true;
    status(`Simulating ${num(body.paths)} paths…`);
    const started = performance.now();
    try {
      const response = await fetch('/api/whatif', {method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)});
      const result = await response.json();
      if (!response.ok || result.error) throw new Error(result.error || `HTTP ${response.status}`);
      renderWhatIf(result);
      const query = new URLSearchParams({weights: tickers.map((t) => `${t}:${weights[t]}`).join(','),
        horizon: body.horizon, paths: body.paths, value: body.value});
      history.replaceState(null, '', `?${query.toString().replaceAll('%3A', ':').replaceAll('%2C', ',')}#whatif`);
      status(`Done in ${((performance.now() - started) / 1000).toFixed(1)} s. The address bar now links to this scenario.`);
    } catch (error) {
      status(`Could not run the simulation: ${error.message}. Is "python serve.py" still running?`);
    } finally {
      button.disabled = false;
    }
  });

  // shareable links: ?weights=CBA:40,BHP:20,...&horizon=20&paths=10000 fills in the form and runs it
  const params = new URLSearchParams(window.location.search);
  if (params.has('weights')) {
    const given = Object.fromEntries(params.get('weights').split(',')
      .map((item) => item.split(':')).map(([t, v]) => [t.trim().toUpperCase(), Number(v)]));
    setWeights((t) => (given[t] || 0) / 100);
    for (const [key, id] of [['horizon', 'whatif-horizon'], ['paths', 'whatif-paths'], ['value', 'whatif-value']]) {
      if (params.has(key)) $(id).value = params.get(key);
    }
    $('whatif-form').requestSubmit();
  }
}

function renderWhatIf(w) {
  $('whatif-results').hidden = false;
  const H = w.horizon;
  const main = w.measures[w.main_model];
  const normalName = Object.keys(w.measures).find((k) => k !== w.main_model);
  const normal = w.measures[normalName];
  const tiles = w.days.map((day) => kpi(`${dayAdj(day)} VaR 99%`, aud(main[day].var99),
    `${normalName} model: ${aud(normal[day].var99)}`));
  tiles.push(kpi(`${dayAdj(H)} Expected Shortfall`, aud(main[H].es975), 'average of the worst 2.5%'));
  const pr = w.path_risk[w.main_model];
  tiles.push(kpi(`Chance of a ${pct(w.stop_loss, 0)} drop`, pct(pr.p_down_any),
    `at any point in ${plural(H, 'day')}; ${pct(pr.p_down_end)} on the last day`));
  $('whatif-kpis').replaceChildren(...tiles);
  plot('whatif-fan', () => fanFigure({fan: w.fan, main: w.main_model, compare: normalName, value: w.value,
    samples: null}));
  plot('whatif-contrib', () => contribFigure(w.contributions));
  plot('whatif-stress', () => stressFigure(w.stress, main, w.days));
  $('whatif-results').scrollIntoView({behavior: 'smooth', block: 'start'});
}
