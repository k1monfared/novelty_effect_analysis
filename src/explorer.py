"""Generate a self-contained per-series novelty explorer at docs/index.html.

Embeds the daily lifts and the fitted transient for every experiment-metric as
JSON and draws them with a small amount of vanilla JavaScript and inline SVG.
No external assets, no build step, works offline. No emojis.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config, model
from .pipeline import load_series


def _build_payload(cfg) -> dict:
    gt, series = load_series(cfg)
    gt_by = {(g["experiment_id"], g["metric"]): g for g in gt}
    K = cfg["estimator"]["decision_day"]

    out = []
    for (exp, metric), (days, y, se, true_eff) in sorted(series.items()):
        fit = model.fit(days, y, se, cfg)
        g = gt_by[(exp, metric)]
        # Sample the fitted curve on the integer day grid for plotting.
        dd = days
        curve = (fit.L + fit.A * np.exp(-dd / fit.tau)) if fit.tau > 0 else np.full_like(dd, fit.L)
        # Naive early average uses only the first K days, matching the decision point.
        kmask = days < K
        wk = 1.0 / np.maximum(se[kmask] ** 2, 1e-12)
        naive = float(np.sum(wk * y[kmask]) / np.sum(wk))
        out.append({
            "id": f"{exp} / {metric}",
            "exp": exp, "metric": metric,
            "days": [int(x) for x in days],
            "lift": [round(float(v) * 100, 4) for v in y],
            "se": [round(float(v) * 100, 4) for v in se],
            "fit": [round(float(v) * 100, 4) for v in curve],
            "L": round(fit.L * 100, 4),
            "L_true": round(float(g["L_true"]) * 100, 4),
            "naive": round(naive * 100, 4),
            "cat_true": g["category"],
            "cat_pred_note": "",
            "K": K,
        })
    return {"series": out, "K": K}


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Novelty-effect explorer</title>
<style>
  :root {
    --ink:#212529; --muted:#6c757d; --line:#dee2e6;
    --obs:#4C6EF5; --fit:#E8590C; --asym:#2B8A3E; --naive:#7048e8; --true:#495057;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         color:var(--ink); background:#f8f9fa; }
  header { padding:20px 24px; background:#fff; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 4px; font-size:19px; }
  .sub { color:var(--muted); font-size:13px; }
  main { max-width:960px; margin:22px auto; padding:0 16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:18px; }
  .controls { display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  select { padding:7px 10px; font-size:14px; border:1px solid var(--line); border-radius:8px; background:#fff; }
  .stats { display:flex; gap:22px; flex-wrap:wrap; margin:12px 0 4px; font-size:13px; }
  .stat b { display:block; font-size:16px; }
  .stat span { color:var(--muted); }
  .legend { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:10px; }
  .key { display:inline-flex; align-items:center; gap:6px; }
  .swatch { width:16px; height:3px; border-radius:2px; display:inline-block; }
  .pill { padding:2px 8px; border-radius:999px; font-size:12px; background:#f1f3f5; }
  .disclaimer { margin:10px 0 0; padding:10px 14px; border-left:4px solid var(--asym);
                background:#f1f3f5; font-weight:700; font-size:14px; color:var(--ink); }
  footer { color:var(--muted); font-size:12px; text-align:center; margin:26px 0; }
</style>
</head>
<body>
<header>
  <h1>Novelty-effect explorer</h1>
  <blockquote class="disclaimer">This is a portfolio demonstration built on synthetic data.</blockquote>
</header>
<main>
  <div class="card">
    <div class="controls">
      <label>Series
        <select id="pick"></select>
      </label>
      <span class="pill" id="cat"></span>
    </div>
    <div id="chart"></div>
    <div class="legend">
      <span class="key"><span class="swatch" style="background:var(--obs)"></span>observed daily lift</span>
      <span class="key"><span class="swatch" style="background:var(--fit)"></span>fitted transient</span>
      <span class="key"><span class="swatch" style="background:var(--asym)"></span>debiased long-term</span>
      <span class="key"><span class="swatch" style="background:var(--naive)"></span>naive early average</span>
      <span class="key"><span class="swatch" style="background:var(--true)"></span>true long-term</span>
    </div>
    <div class="stats" id="stats"></div>
  </div>
  <footer>Reproduce with <code>python scripts/run_demo.py</code>. Every number is from the actual run, fixed seed.</footer>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const pick = document.getElementById('pick');
DATA.series.forEach((s,i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = s.id; pick.appendChild(o);
});

const W = 880, H = 380, P = {t:16, r:16, b:40, l:52};
function draw(s) {
  const xs = s.days, n = xs.length;
  const allY = s.lift.concat(s.fit, [s.L, s.L_true, s.naive],
     s.lift.map((v,i)=>v+1.96*s.se[i]), s.lift.map((v,i)=>v-1.96*s.se[i]));
  let ymin = Math.min(...allY), ymax = Math.max(...allY);
  const pad = (ymax - ymin) * 0.08 || 1; ymin -= pad; ymax += pad;
  const xmax = Math.max(...xs);
  const X = d => P.l + (d/xmax) * (W-P.l-P.r);
  const Y = v => H-P.b - ((v-ymin)/(ymax-ymin)) * (H-P.t-P.b);
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width','100%');
  function line(x1,y1,x2,y2,stroke,w,dash){
    const l=document.createElementNS(NS,'line');
    l.setAttribute('x1',x1);l.setAttribute('y1',y1);l.setAttribute('x2',x2);l.setAttribute('y2',y2);
    l.setAttribute('stroke',stroke);l.setAttribute('stroke-width',w);
    if(dash) l.setAttribute('stroke-dasharray',dash); svg.appendChild(l);
  }
  function text(x,y,t,anchor,fill){
    const e=document.createElementNS(NS,'text');
    e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('font-size','11');
    e.setAttribute('text-anchor',anchor||'middle');e.setAttribute('fill',fill||'#6c757d');
    e.textContent=t; svg.appendChild(e);
  }
  // axes
  line(P.l,Y(0),W-P.r,Y(0),'#ced4da',1);
  line(P.l,P.t,P.l,H-P.b,'#adb5bd',1);
  // y ticks
  for(let k=0;k<=4;k++){
    const v = ymin + (ymax-ymin)*k/4;
    line(P.l-4,Y(v),P.l,Y(v),'#adb5bd',1);
    text(P.l-8,Y(v)+3,v.toFixed(1),'end');
  }
  text((P.l+W-P.r)/2, H-8, 'day');
  // x ticks
  for(let d=0; d<=xmax; d+=7){ line(X(d),H-P.b,X(d),H-P.b+4,'#adb5bd',1); text(X(d),H-P.b+16,d); }
  // reference lines
  line(P.l,Y(s.L_true),W-P.r,Y(s.L_true),'var(--true)',1.5,'2 3');
  line(P.l,Y(s.L),W-P.r,Y(s.L),'var(--asym)',2,'6 4');
  line(P.l,Y(s.naive),X(s.K),Y(s.naive),'var(--naive)',2,'2 3');
  line(X(s.K),P.t,X(s.K),H-P.b,'var(--naive)',0.8,'2 4');
  // error bars + points
  for(let i=0;i<n;i++){
    const lo=s.lift[i]-1.96*s.se[i], hi=s.lift[i]+1.96*s.se[i];
    line(X(xs[i]),Y(lo),X(xs[i]),Y(hi),'#ced4da',1);
    const c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',X(xs[i]));c.setAttribute('cy',Y(s.lift[i]));c.setAttribute('r',2.8);
    c.setAttribute('fill','var(--obs)');c.setAttribute('opacity','0.85');svg.appendChild(c);
  }
  // fitted curve
  let dpath='';
  for(let i=0;i<n;i++){ dpath += (i?'L':'M') + X(xs[i]) + ' ' + Y(s.fit[i]) + ' '; }
  const path=document.createElementNS(NS,'path');
  path.setAttribute('d',dpath);path.setAttribute('fill','none');
  path.setAttribute('stroke','var(--fit)');path.setAttribute('stroke-width',2.4);svg.appendChild(path);
  const chart=document.getElementById('chart'); chart.innerHTML=''; chart.appendChild(svg);

  document.getElementById('cat').textContent = 'category: ' + s.cat_true;
  document.getElementById('stats').innerHTML =
    `<div class="stat"><b>${s.naive.toFixed(2)}pp</b><span>naive @ day ${s.K}</span></div>`+
    `<div class="stat"><b>${s.L.toFixed(2)}pp</b><span>debiased long-term</span></div>`+
    `<div class="stat"><b>${s.L_true.toFixed(2)}pp</b><span>true long-term</span></div>`+
    `<div class="stat"><b>${(s.naive-s.L_true).toFixed(2)}pp</b><span>naive error</span></div>`+
    `<div class="stat"><b>${(s.L-s.L_true).toFixed(2)}pp</b><span>debiased error</span></div>`;
}
pick.addEventListener('change', () => draw(DATA.series[pick.value]));
// Start on the tabs_revamp review_completions series if present.
let start = DATA.series.findIndex(s => s.exp==='tabs_revamp' && s.metric==='review_completions');
if (start < 0) start = 0;
pick.value = start; draw(DATA.series[start]);
</script>
</body>
</html>
"""


def write_explorer(cfg) -> str:
    config.ensure_dirs()
    payload = _build_payload(cfg)
    html = _HTML.replace("__DATA__", json.dumps(payload))
    path = os.path.join(config.ROOT, "docs", "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path
