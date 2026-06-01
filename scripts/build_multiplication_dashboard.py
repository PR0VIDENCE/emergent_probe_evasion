"""
Build a self-contained HTML dashboard for the multiplication-probe experiments.

Auto-discovers every probes_* / probe_*_strategy folder under
data/concepts/multiplication_pilot/, joins per-sample probe scores with the
matching rollouts.jsonl when available, and emits a single HTML file with
three tabs:

  1. Overview      — per-regime mean probe score + accuracy table per experiment
  2. Layer sweep   — heatmap of (position × layer) probe scores per regime
                     (uses ood_per_probe.csv if present)
  3. Explorer      — filterable table of per-rollout scores; click a row to
                     see the question, thinking, and response

Usage:
  uv run python scripts/build_multiplication_dashboard.py
  open data/concepts/multiplication_pilot/dashboard.html
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PILOT_ROOT = PROJECT_ROOT / "data" / "concepts" / "multiplication_pilot"
OUTPUT_PATH = PILOT_ROOT / "dashboard.html"


def latest_file(dirpath: Path, pattern: str) -> Path | None:
    """Pick the newest file matching `pattern` in dirpath, handling cases like
    'ood_eval (3).csv' (downloaded with numeric suffixes) by preferring the
    largest suffix number (typically the most recent)."""
    files = sorted(dirpath.glob(pattern))
    if not files:
        return None
    # Score by trailing number-in-parens if present; otherwise mtime.
    def keyfn(p: Path):
        m = re.search(r"\((\d+)\)", p.name)
        if m:
            return (1, int(m.group(1)))
        return (0, p.stat().st_mtime)
    files.sort(key=keyfn, reverse=True)
    return files[0]


def read_csv_rows(path: Path) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def read_jsonl(path: Path) -> list[dict]:
    if path is None or not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def coerce_numbers(rows: list[dict], numeric_fields: list[str]) -> list[dict]:
    """Convert specified fields from str to float, treating empty as None."""
    out = []
    for row in rows:
        new = dict(row)
        for field in numeric_fields:
            if field in new:
                v = new[field]
                if v == "" or v is None:
                    new[field] = None
                else:
                    try:
                        new[field] = float(v)
                    except (TypeError, ValueError):
                        pass
        out.append(new)
    return out


def load_regime_system_prompts() -> dict[str, str]:
    """Import REGIMES dict from multiplication_evasion.py so the explorer
    can show the system prompt that produced each rollout. The dict lives in
    one place (the evasion script) — we just read it out."""
    scripts_dir = PROJECT_ROOT / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import importlib
        mod = importlib.import_module("multiplication_evasion")
        regimes = getattr(mod, "REGIMES", {})
        return {name: cfg.get("system_prompt", "") for name, cfg in regimes.items()}
    except Exception as e:
        print(f"  WARN: couldn't load REGIMES from multiplication_evasion.py: {e}",
              file=sys.stderr)
        return {}


def discover_experiments() -> list[dict]:
    """Find probe-output folders and load their CSVs / JSONLs."""
    experiments = []
    # Look for any folder under pilot whose name starts with 'probes' or
    # 'probe_' — these contain ood_eval.csv, ood_per_probe.csv, etc.
    for dirpath in sorted(PILOT_ROOT.iterdir()):
        if not dirpath.is_dir():
            continue
        if not (dirpath.name.startswith("probes") or
                dirpath.name.startswith("probe_")):
            continue
        ood_eval_path = latest_file(dirpath, "ood_eval*.csv")
        if ood_eval_path is None:
            continue  # not a probe-output folder we care about
        ood_per_probe_path = latest_file(dirpath, "ood_per_probe*.csv")
        per_sample_path = latest_file(dirpath, "ood_per_sample*.jsonl")

        ood_eval = coerce_numbers(read_csv_rows(ood_eval_path),
                                  ["n", "mean_best", "mean_combiner", "auc_best",
                                   "auc_combiner", "accuracy"])
        ood_per_probe = coerce_numbers(read_csv_rows(ood_per_probe_path),
                                       ["n", "layer", "mean", "min", "max"])
        per_sample = read_jsonl(per_sample_path)

        experiments.append({
            "name": dirpath.name,
            "ood_eval": ood_eval,
            "ood_per_probe": ood_per_probe,
            "per_sample": per_sample,
        })
    return experiments


def discover_rollouts() -> dict[str, dict]:
    """Index all rollouts.jsonl files under multiplication_pilot/, keyed by
    problem_id. Each entry includes the full rollout + which experiment it
    came from (helpful for tracing)."""
    by_id: dict[str, dict] = {}
    for dirpath in sorted(PILOT_ROOT.rglob("rollouts*.jsonl")):
        if "activations" in dirpath.parts:
            continue  # skip activation-side meta files
        exp_name = dirpath.parent.parent.name  # e.g. evasion_strategies_v1
        for r in read_jsonl(dirpath):
            pid = r.get("problem_id")
            if pid is None:
                continue
            # First seen wins (so original rollout context isn't overwritten
            # by re-runs in adjacent experiments).
            if pid not in by_id:
                by_id[pid] = {
                    "experiment": exp_name,
                    "user_text": r.get("user_text", ""),
                    "thinking": r.get("thinking", "") or "",
                    "response": r.get("response", "") or "",
                    "n_gen_tokens": r.get("n_gen_tokens"),
                    "extracted_answer": r.get("extracted_answer"),
                    "true_answer": r.get("true_answer"),
                    "correct": r.get("correct"),
                    "operation": r.get("operation"),
                    "regime": r.get("regime"),
                    "source": r.get("source"),
                    "a": r.get("a"), "b": r.get("b"),
                }
    return by_id


def build_html(experiments: list[dict], rollouts: dict) -> str:
    payload = {
        "experiments": experiments,
        "rollouts": rollouts,
        "regime_system_prompts": load_regime_system_prompts(),
    }
    payload_json = json.dumps(payload, separators=(",", ":"), default=str)

    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multiplication Probe Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #fafafa;
    --panel: #ffffff;
    --border: #e1e4e8;
    --text: #24292e;
    --muted: #6a737d;
    --accent: #0366d6;
    --pos: #5cb85c;
    --neg: #d9534f;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: var(--bg); color: var(--text); }
  header { padding: 16px 24px; background: var(--panel);
           border-bottom: 1px solid var(--border); }
  header h1 { margin: 0 0 4px 0; font-size: 18px; }
  header .meta { color: var(--muted); font-size: 13px; }
  nav { padding: 0 24px; background: var(--panel);
        border-bottom: 1px solid var(--border); display: flex; gap: 24px; }
  nav button { background: none; border: none; padding: 12px 4px; cursor: pointer;
               font-size: 14px; color: var(--muted); border-bottom: 2px solid transparent; }
  nav button.active { color: var(--text); border-bottom-color: var(--accent); }
  main { padding: 20px 24px; }
  .tab { display: none; }
  .tab.active { display: block; }
  table { border-collapse: collapse; font-size: 13px; width: 100%; }
  th, td { padding: 6px 10px; border-bottom: 1px solid var(--border);
           text-align: left; }
  th { background: #f6f8fa; font-weight: 600; font-size: 12px;
       text-transform: uppercase; color: var(--muted); }
  tr:hover { background: #f6f8fa; }
  .num { font-variant-numeric: tabular-nums; text-align: right; }
  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 6px; margin-bottom: 20px; }
  .card-header { padding: 12px 16px; border-bottom: 1px solid var(--border);
                 font-weight: 600; font-size: 14px; }
  .card-body { padding: 12px 16px; overflow-x: auto; }
  .controls { display: flex; gap: 10px; margin-bottom: 12px;
              flex-wrap: wrap; align-items: center; }
  .controls input, .controls select {
    padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px;
    font-size: 13px; background: white;
  }
  .controls label { font-size: 12px; color: var(--muted); margin-right: 4px; }
  .heatmap { border-collapse: collapse; }
  .heatmap th, .heatmap td { padding: 4px 6px; font-size: 11px;
    text-align: center; min-width: 36px; }
  .heatmap th { background: #f6f8fa; }
  .heatmap td.cell { font-family: monospace; }
  .regime-pill { display: inline-block; padding: 2px 6px; border-radius: 3px;
                 font-size: 11px; background: #eaecef; color: var(--text); }
  .check-yes { color: var(--pos); font-weight: 600; }
  .check-no { color: var(--neg); font-weight: 600; }
  .rollout-row { cursor: pointer; }
  .rollout-detail { background: #fafafa; padding: 12px 16px;
                    border-left: 3px solid var(--accent); margin: 4px 0;
                    font-family: ui-monospace, monospace; font-size: 12px;
                    white-space: pre-wrap; max-height: 600px; overflow-y: auto; }
  .rollout-detail h4 { font-family: -apple-system, sans-serif;
                       font-size: 13px; margin: 8px 0 4px 0; color: var(--accent); }
  .rollout-detail .meta-row { color: var(--muted); font-family: -apple-system, sans-serif; }
  .pill-mult { background: #cdeac0; }
  .pill-non-mult { background: #fbd1d1; }
  .pill-borderline { background: #fff5b8; }
  .small { font-size: 11px; color: var(--muted); }
  details { margin: 4px 0; }
  summary { cursor: pointer; }
  .empty { color: var(--muted); padding: 20px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Multiplication Probe Dashboard</h1>
  <div class="meta">Probe-evasion experiments across regimes, digit sizes, and strategy prompts.</div>
</header>
<nav>
  <button class="tab-btn active" data-tab="overview">Overview</button>
  <button class="tab-btn" data-tab="trends">Trends</button>
  <button class="tab-btn" data-tab="heatmap">Layer Sweep</button>
  <button class="tab-btn" data-tab="explorer">Explorer</button>
</nav>
<main>
  <section id="overview" class="tab active"></section>
  <section id="trends" class="tab"></section>
  <section id="heatmap" class="tab"></section>
  <section id="explorer" class="tab"></section>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const PAYLOAD = JSON.parse(document.getElementById('payload').textContent);
const experiments = PAYLOAD.experiments;
const rollouts = PAYLOAD.rollouts;
const REGIME_SYSTEM_PROMPTS = PAYLOAD.regime_system_prompts || {};

// --- helpers ---
function fmt(v, digits=4) {
  if (v === null || v === undefined || v === '') return '—';
  if (typeof v === 'number') return v.toFixed(digits);
  return String(v);
}
function pct(v) {
  if (v === null || v === undefined || v === '') return '—';
  return (Number(v) * 100).toFixed(1) + '%';
}
function viridis(t) {
  // Cheap colormap approximation 0..1 -> [r,g,b] hex (purple->yellow)
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * (0.267 + t * (0.992 - 0.267) * t));
  const g = Math.round(255 * (0.005 + t * 0.906));
  const b = Math.round(255 * (0.329 + (1 - t) * 0.207));
  return `rgb(${r},${g},${b})`;
}
function regimeLabel(s) {
  if (!s) return '';
  return s.replace(/^regime_/, '').replace(/_/g, ' ');
}

// --- Overview tab ---
function renderOverview() {
  const root = document.getElementById('overview');
  if (!experiments.length) {
    root.innerHTML = '<div class="empty">No experiments found.</div>';
    return;
  }
  let html = '<div class="small" style="margin-bottom:12px;">Headline metrics: <b>reason_pool</b> and <b>answer_pool</b> are mean probe scores averaged across all 15 layers at the reasoning_mean_pool and answer_mean_pool positions — more robust than the val-best single probe.</div>';
  for (const exp of experiments) {
    const rows = exp.ood_eval.filter(r => r.slice && r.slice.startsWith('type:'));
    if (!rows.length) continue;
    // Detect whether this experiment has the new pool columns
    const hasReasonPool = rows.some(r => r.mean_reasoning_mean_pool !== undefined && r.mean_reasoning_mean_pool !== null && r.mean_reasoning_mean_pool !== '');
    const hasAnswerPool = rows.some(r => r.mean_answer_mean_pool !== undefined && r.mean_answer_mean_pool !== null && r.mean_answer_mean_pool !== '');
    html += `<div class="card"><div class="card-header">${exp.name}</div><div class="card-body"><table>`;
    html += `<thead><tr><th>Slice</th><th class="num">n</th>`;
    if (hasReasonPool) html += `<th class="num">reason_pool</th>`;
    if (hasAnswerPool) html += `<th class="num">answer_pool</th>`;
    html += `<th class="num">mean_best</th><th class="num">mean_combiner</th><th class="num">auc_best</th><th class="num">accuracy</th></tr></thead><tbody>`;
    for (const r of rows) {
      const slice = r.slice.replace(/^type:/, '');
      html += `<tr><td>${slice}</td>`;
      html += `<td class="num">${r.n ?? '—'}</td>`;
      if (hasReasonPool) html += `<td class="num">${fmt(r.mean_reasoning_mean_pool)}</td>`;
      if (hasAnswerPool) html += `<td class="num">${fmt(r.mean_answer_mean_pool)}</td>`;
      html += `<td class="num">${fmt(r.mean_best)}</td>`;
      html += `<td class="num">${fmt(r.mean_combiner)}</td>`;
      html += `<td class="num">${fmt(r.auc_best)}</td>`;
      html += `<td class="num">${pct(r.accuracy)}</td></tr>`;
    }
    html += `</tbody></table></div></div>`;
  }
  root.innerHTML = html || '<div class="empty">No "type:" rows in any ood_eval.csv.</div>';
}

// --- Layer Sweep tab ---
function renderHeatmap() {
  const root = document.getElementById('heatmap');
  let html = '';
  let any = false;
  for (const exp of experiments) {
    if (!exp.ood_per_probe || !exp.ood_per_probe.length) continue;
    any = true;
    // Group by source then by (position, layer)
    const bySource = {};
    const layers = new Set(), positions = new Set();
    for (const r of exp.ood_per_probe) {
      if (!bySource[r.source]) bySource[r.source] = {};
      bySource[r.source][`${r.position}|${r.layer}`] = r;
      layers.add(r.layer); positions.add(r.position);
    }
    const layerList = Array.from(layers).map(Number).sort((a,b)=>a-b);
    const posList = Array.from(positions);
    const sources = Object.keys(bySource).sort();

    html += `<div class="card"><div class="card-header">${exp.name} — probe score per (position × layer)</div><div class="card-body">`;
    html += `<div class="small" style="margin-bottom: 8px;">color: mean probe score across rollouts of that source. Hover a cell for value.</div>`;
    for (const source of sources) {
      html += `<div style="margin-bottom: 14px;"><div style="font-weight:600;font-size:12px;margin:6px 0;">${regimeLabel(source)}</div>`;
      html += `<table class="heatmap"><thead><tr><th></th>`;
      for (const l of layerList) html += `<th>L${l}</th>`;
      html += `</tr></thead><tbody>`;
      for (const p of posList) {
        html += `<tr><th style="text-align:right;">${p}</th>`;
        for (const l of layerList) {
          const r = bySource[source][`${p}|${l}`];
          if (!r) { html += `<td>—</td>`; continue; }
          const v = Number(r.mean);
          const bg = viridis(v);
          const textColor = v > 0.55 ? 'black' : 'white';
          html += `<td class="cell" title="mean=${fmt(v)} min=${fmt(r.min)} max=${fmt(r.max)} n=${r.n}" style="background:${bg};color:${textColor};">${v.toFixed(2)}</td>`;
        }
        html += `</tr>`;
      }
      html += `</tbody></table></div>`;
    }
    html += `</div></div>`;
  }
  if (!any) {
    root.innerHTML = '<div class="empty">No ood_per_probe.csv files found. Re-run multiplication_train_probes.py after the layer-sweep edit.</div>';
    return;
  }
  root.innerHTML = html;
}

// --- Explorer tab ---
function renderExplorer() {
  const root = document.getElementById('explorer');

  // Flatten per_sample rows across experiments, attach experiment label
  const samples = [];
  for (const exp of experiments) {
    for (const s of (exp.per_sample || [])) {
      samples.push({
        experiment: exp.name,
        problem_id: s.problem_id,
        regime: (s.source || '').replace(/^regime_/, ''),
        source: s.source,
        label: s.label,
        score_best: s.score_best,
        score_combiner: s.score_combiner,
        score_reasoning_pool: s.score_reasoning_mean_pool,
        score_answer_pool: s.score_answer_mean_pool,
        correct: s.correct,
        uses_op_keyword: s.uses_op_keyword,
      });
    }
  }

  if (!samples.length) {
    root.innerHTML = '<div class="empty">No per-sample data found.</div>';
    return;
  }

  // Build filter UI
  const allExps = Array.from(new Set(samples.map(s => s.experiment))).sort();
  const allRegimes = Array.from(new Set(samples.map(s => s.regime).filter(Boolean))).sort();

  // Detect which score columns are present in this data
  const hasReasonPool = samples.some(s => s.score_reasoning_pool !== undefined && s.score_reasoning_pool !== null);
  const hasAnswerPool = samples.some(s => s.score_answer_pool !== undefined && s.score_answer_pool !== null);

  root.innerHTML = `
    <div class="card"><div class="card-header">Rollout Explorer (${samples.length} rollouts across ${allExps.length} experiments)</div><div class="card-body">
      <div class="controls">
        <label>Experiment <select id="exp-filter"><option value="">all</option>${allExps.map(e => `<option>${e}</option>`).join('')}</select></label>
        <label>Regime / source <select id="regime-filter"><option value="">all</option>${allRegimes.map(r => `<option>${r}</option>`).join('')}</select></label>
        <label>Correct <select id="correct-filter"><option value="">any</option><option value="true">correct</option><option value="false">wrong</option></select></label>
        <label>Filter score by <select id="score-metric-select">
          <option value="score_best">best single</option>
          ${hasReasonPool ? '<option value="score_reasoning_pool" selected>reason_pool</option>' : ''}
          ${hasAnswerPool ? '<option value="score_answer_pool">answer_pool</option>' : ''}
        </select></label>
        <label>≥ <input id="score-min" type="number" min="0" max="1" step="0.05" style="width:60px;"></label>
        <label>≤ <input id="score-max" type="number" min="0" max="1" step="0.05" style="width:60px;"></label>
        <label>Search prompt <input id="prompt-search" type="text" placeholder="text contains..." style="width:160px;"></label>
        <span id="explorer-count" class="small"></span>
      </div>
      <table id="explorer-table">
        <thead><tr>
          <th>experiment</th><th>regime</th><th>problem_id</th>
          ${hasReasonPool ? '<th class="num">reason_pool</th>' : ''}
          ${hasAnswerPool ? '<th class="num">answer_pool</th>' : ''}
          <th class="num">best</th><th>correct</th><th>prompt</th>
        </tr></thead>
        <tbody id="explorer-body"></tbody>
      </table>
    </div></div>
  `;

  const tbody = document.getElementById('explorer-body');
  const count = document.getElementById('explorer-count');

  function applyFilters() {
    const expFilter = document.getElementById('exp-filter').value;
    const regimeFilter = document.getElementById('regime-filter').value;
    const correctFilter = document.getElementById('correct-filter').value;
    const scoreMetric = document.getElementById('score-metric-select').value;
    const scoreMin = parseFloat(document.getElementById('score-min').value);
    const scoreMax = parseFloat(document.getElementById('score-max').value);
    const promptSearch = document.getElementById('prompt-search').value.toLowerCase();

    let filtered = samples;
    if (expFilter) filtered = filtered.filter(s => s.experiment === expFilter);
    if (regimeFilter) filtered = filtered.filter(s => s.regime === regimeFilter);
    if (correctFilter) {
      const want = correctFilter === 'true';
      filtered = filtered.filter(s => Boolean(s.correct) === want);
    }
    const getScore = (s) => s[scoreMetric];
    if (!isNaN(scoreMin)) filtered = filtered.filter(s => {
      const v = getScore(s); return v !== undefined && v !== null && v >= scoreMin;
    });
    if (!isNaN(scoreMax)) filtered = filtered.filter(s => {
      const v = getScore(s); return v !== undefined && v !== null && v <= scoreMax;
    });
    if (promptSearch) {
      filtered = filtered.filter(s => {
        const r = rollouts[s.problem_id];
        if (!r) return false;
        return (r.user_text || '').toLowerCase().includes(promptSearch);
      });
    }
    // Sort by the chosen metric descending so the most-firing examples are first
    filtered.sort((a, b) => (getScore(b) ?? 0) - (getScore(a) ?? 0));
    count.textContent = `(${filtered.length} matching, sorted by ${scoreMetric})`;
    renderRows(filtered.slice(0, 500));
    if (filtered.length > 500) count.textContent += ' — showing first 500';
  }

  function renderRows(rows) {
    const colspan = 4 + (hasReasonPool ? 1 : 0) + (hasAnswerPool ? 1 : 0) + 2;
    const html = rows.map((s, i) => {
      const r = rollouts[s.problem_id];
      const prompt = r ? (r.user_text || '').split('\n')[0].slice(0, 120) : '(rollout not local)';
      const cell = (v) => {
        if (v === null || v === undefined) return '<td class="num">—</td>';
        const bg = viridis(v);
        const tc = v > 0.55 ? 'black' : 'white';
        return `<td class="num" style="background:${bg};color:${tc};">${fmt(v, 3)}</td>`;
      };
      return `
        <tr class="rollout-row" data-idx="${i}">
          <td><span class="small">${s.experiment}</span></td>
          <td><span class="regime-pill">${s.regime || s.source || '?'}</span></td>
          <td><span class="small">${s.problem_id || ''}</span></td>
          ${hasReasonPool ? cell(s.score_reasoning_pool) : ''}
          ${hasAnswerPool ? cell(s.score_answer_pool) : ''}
          ${cell(s.score_best)}
          <td>${s.correct === true ? '<span class="check-yes">✓</span>' : s.correct === false ? '<span class="check-no">✗</span>' : '—'}</td>
          <td><span class="small">${prompt}</span></td>
        </tr>
        <tr class="detail-row" style="display:none;" data-idx="${i}">
          <td colspan="${colspan}"></td>
        </tr>
      `;
    }).join('');
    tbody.innerHTML = html;

    // Wire click-to-expand
    tbody.querySelectorAll('.rollout-row').forEach((tr, i) => {
      const sample = rows[i];
      tr.addEventListener('click', () => {
        const detail = tbody.querySelector(`.detail-row[data-idx="${i}"]`);
        if (detail.style.display === 'none') {
          const r = rollouts[sample.problem_id];
          let detailHtml = '';
          detailHtml += `<div class="rollout-detail">`;
          detailHtml += `<div class="meta-row">score_best=${fmt(sample.score_best)} score_combiner=${fmt(sample.score_combiner)} correct=${sample.correct}`;
          if (r) detailHtml += ` n_gen_tokens=${r.n_gen_tokens} extracted=${r.extracted_answer} true=${r.true_answer}`;
          detailHtml += `</div>`;
          // Look up system prompt by the regime name (with or without
          // "regime_" prefix). REGIME_SYSTEM_PROMPTS is keyed by the
          // unprefixed regime name as it appears in REGIMES.
          const regimeKey = (sample.regime || '').trim();
          const sysPrompt = REGIME_SYSTEM_PROMPTS[regimeKey]
            || REGIME_SYSTEM_PROMPTS[regimeKey.replace(/^regime_/, '')]
            || '';
          if (sysPrompt) {
            detailHtml += `<h4>System prompt (${regimeKey})</h4>${escapeHtml(sysPrompt)}`;
          } else if (regimeKey) {
            detailHtml += `<h4>System prompt (${regimeKey})</h4><span class="meta-row">— not found in REGIMES dict —</span>`;
          }
          if (!r) {
            detailHtml += `<div class="meta-row" style="margin-top:8px;">No local rollouts.jsonl found for this experiment — only probe scores are available. Sync the rollouts file to see the user prompt, thinking, and response.</div>`;
          } else {
            detailHtml += `<h4>User prompt</h4>${escapeHtml(r.user_text)}`;
            detailHtml += `<h4>Thinking (${(r.thinking || '').length} chars)</h4>${escapeHtml((r.thinking || '').slice(0, 8000))}`;
            if ((r.thinking || '').length > 8000) detailHtml += `\n[...truncated]`;
            detailHtml += `<h4>Response (${(r.response || '').length} chars)</h4>${escapeHtml((r.response || '').slice(0, 4000))}`;
          }
          detailHtml += `</div>`;
          detail.firstElementChild.innerHTML = detailHtml;
          detail.style.display = '';
        } else {
          detail.style.display = 'none';
        }
      });
    });
  }

  function escapeHtml(s) {
    return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  ['exp-filter','regime-filter','correct-filter','score-metric-select','score-min','score-max','prompt-search']
    .forEach(id => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('input', applyFilters);
      el.addEventListener('change', applyFilters);
    });
  applyFilters();
}

// --- Trends tab (line charts) ---
// Consistent color per regime so the same line is the same color across charts.
const REGIME_COLORS = {
  'regime_baseline':                 '#666666',
  'baseline':                        '#666666',
  'regime_awareness_only':           '#a55194',
  'awareness_only':                  '#a55194',
  'regime_technical_disclosure':     '#756bb1',
  'technical_disclosure':            '#756bb1',
  'regime_technical_extended':       '#3182bd',
  'technical_extended':              '#3182bd',
  'regime_strategy_repeated_addition': '#3182bd',
  'strategy_repeated_addition':        '#3182bd',
  'regime_strategy_russian_peasant': '#d62728',
  'strategy_russian_peasant':        '#d62728',
  'regime_strategy_hard_switch':     '#ff7f0e',
  'strategy_hard_switch':            '#ff7f0e',
  'regime_strategy_verbal_misdirection': '#2ca02c',
  'strategy_verbal_misdirection':        '#2ca02c',
  'regime_strategy_place_value':     '#bcbd22',
  'strategy_place_value':            '#bcbd22',
};
function colorFor(regime) {
  return REGIME_COLORS[regime] || '#888888';
}

const _chartInstances = [];
function destroyCharts() {
  for (const c of _chartInstances) try { c.destroy(); } catch(e){}
  _chartInstances.length = 0;
}

function renderTrends() {
  const root = document.getElementById('trends');
  destroyCharts();
  let html = '';

  // ===== Chart A: cross-experiment summary =====
  // For each regime, draw a line: x = experiment, y = chosen metric.
  // User can switch among: mean_reasoning_mean_pool, mean_answer_mean_pool,
  // mean_best, mean_combiner. The pool metrics are the more robust headlines.
  const SUMMARY_METRICS = [
    {key: 'mean_reasoning_mean_pool', label: 'reasoning_mean_pool (layer-averaged)'},
    {key: 'mean_answer_mean_pool',    label: 'answer_mean_pool (layer-averaged)'},
    {key: 'mean_best',                label: 'best single probe (val-selected)'},
    {key: 'mean_combiner',            label: 'top-K combiner'},
  ];
  const allRegimesGlobal = new Set();
  const expNamesGlobal = [];
  for (const exp of experiments) {
    let hasRegimeRows = false;
    for (const r of (exp.ood_eval || [])) {
      if (r.slice && r.slice.startsWith('type:regime_')) {
        hasRegimeRows = true;
        allRegimesGlobal.add(r.slice.replace('type:', ''));
      }
    }
    if (hasRegimeRows) expNamesGlobal.push(exp.name);
  }
  expNamesGlobal.sort();
  if (expNamesGlobal.length && allRegimesGlobal.size) {
    html += `<div class="card"><div class="card-header">Cross-experiment summary — probe score per regime</div><div class="card-body">`;
    html += `<div class="controls"><label>Metric `;
    html += `<select id="summary-metric-select">`;
    for (const m of SUMMARY_METRICS) {
      html += `<option value="${m.key}">${m.label}</option>`;
    }
    html += `</select></label>`;
    html += `<span class="small">x = experiment (digit-size / difficulty), y = mean probe score across rollouts in that regime</span></div>`;
    html += `<canvas id="chart-summary" height="120"></canvas>`;
    html += `</div></div>`;
  }

  // ===== Chart B: per-experiment layer curves =====
  // For each experiment with ood_per_probe data, render a chart per position
  // with one line per regime: x=layer, y=mean_best (averaged across rollouts
  // of that regime at that layer/position).
  for (const exp of experiments) {
    if (!exp.ood_per_probe || !exp.ood_per_probe.length) continue;
    // Group ood_per_probe by source, then by position->[(layer, mean)]
    const byPosBySrc = {};  // position -> source -> [{layer, mean}]
    const positions = new Set();
    for (const r of exp.ood_per_probe) {
      positions.add(r.position);
      if (!byPosBySrc[r.position]) byPosBySrc[r.position] = {};
      if (!byPosBySrc[r.position][r.source]) byPosBySrc[r.position][r.source] = [];
      byPosBySrc[r.position][r.source].push({layer: r.layer, mean: r.mean});
    }
    const posList = Array.from(positions);
    html += `<div class="card"><div class="card-header">${exp.name} — probe score by layer, per position</div><div class="card-body">`;
    html += `<div class="small" style="margin-bottom:8px;">Each chart: one line per regime. x = layer index, y = mean probe score across rollouts of that regime at this position.</div>`;
    html += `<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap:16px;">`;
    for (const p of posList) {
      html += `<div><div style="font-weight:600;font-size:12px;margin-bottom:4px;">${p}</div><canvas id="chart-${exp.name}-${p}" height="200"></canvas></div>`;
    }
    html += `</div></div></div>`;
  }

  if (!html) {
    root.innerHTML = '<div class="empty">No trend data. Probably need to rerun multiplication_train_probes.py with the layer-sweep edit so ood_per_probe.csv files exist.</div>';
    return;
  }
  root.innerHTML = html;

  // Now wire up actual Chart.js instances (after DOM insert)
  // Chart A — cross-experiment summary, with metric selector
  if (expNamesGlobal.length && allRegimesGlobal.size) {
    let summaryChart = null;
    const regimeList = Array.from(allRegimesGlobal).sort();
    function drawSummary(metricKey) {
      if (summaryChart) {
        try { summaryChart.destroy(); } catch(e) {}
        const idx = _chartInstances.indexOf(summaryChart);
        if (idx >= 0) _chartInstances.splice(idx, 1);
      }
      const ctx = document.getElementById('chart-summary');
      // Rebuild expSummary for this metric
      const expSummary = {};
      for (const exp of experiments) {
        const byRegime = {};
        for (const r of (exp.ood_eval || [])) {
          if (!r.slice || !r.slice.startsWith('type:regime_')) continue;
          const regime = r.slice.replace('type:', '');
          const v = r[metricKey];
          if (v !== undefined && v !== null && v !== '') {
            byRegime[regime] = Number(v);
          }
        }
        if (Object.keys(byRegime).length) expSummary[exp.name] = byRegime;
      }
      const names = Object.keys(expSummary).sort();
      const datasets = regimeList.map(regime => ({
        label: regime.replace(/^regime_/, ''),
        data: names.map(name => expSummary[name]?.[regime] ?? null),
        borderColor: colorFor(regime),
        backgroundColor: colorFor(regime) + '33',
        borderWidth: 2, pointRadius: 4, tension: 0.15, spanGaps: true,
      }));
      summaryChart = new Chart(ctx, {
        type: 'line',
        data: { labels: names, datasets },
        options: {
          responsive: true,
          scales: {
            y: { min: 0, max: 1, title: { display: true, text: 'mean probe score' } },
            x: { title: { display: true, text: 'experiment' } },
          },
          plugins: { legend: { position: 'right' } },
        },
      });
      _chartInstances.push(summaryChart);
    }
    drawSummary('mean_reasoning_mean_pool');  // default to the robust headline
    document.getElementById('summary-metric-select').addEventListener('change', (e) => {
      drawSummary(e.target.value);
    });
  }

  // Chart B (per-experiment, per-position layer curves)
  for (const exp of experiments) {
    if (!exp.ood_per_probe || !exp.ood_per_probe.length) continue;
    const byPosBySrc = {};
    const positions = new Set();
    for (const r of exp.ood_per_probe) {
      positions.add(r.position);
      if (!byPosBySrc[r.position]) byPosBySrc[r.position] = {};
      if (!byPosBySrc[r.position][r.source]) byPosBySrc[r.position][r.source] = [];
      byPosBySrc[r.position][r.source].push({layer: r.layer, mean: r.mean});
    }
    for (const pos of positions) {
      const canvasId = `chart-${exp.name}-${pos}`;
      const ctx = document.getElementById(canvasId);
      if (!ctx) continue;
      const sources = Object.keys(byPosBySrc[pos]).sort();
      // Common layer axis (union, sorted ascending)
      const layerSet = new Set();
      sources.forEach(s => byPosBySrc[pos][s].forEach(pt => layerSet.add(pt.layer)));
      const layers = Array.from(layerSet).map(Number).sort((a,b)=>a-b);
      const datasets = sources.map(source => {
        const pts = byPosBySrc[pos][source];
        const byLayer = Object.fromEntries(pts.map(p => [Number(p.layer), p.mean]));
        return {
          label: source.replace(/^regime_/, ''),
          data: layers.map(l => byLayer[l] ?? null),
          borderColor: colorFor(source),
          backgroundColor: colorFor(source) + '33',
          borderWidth: 2, pointRadius: 3, tension: 0.15, spanGaps: true,
        };
      });
      _chartInstances.push(new Chart(ctx, {
        type: 'line',
        data: { labels: layers.map(l => `L${l}`), datasets },
        options: {
          responsive: true,
          scales: {
            y: { min: 0, max: 1, title: { display: true, text: 'mean probe score' } },
            x: { title: { display: true, text: 'layer' } },
          },
          plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } },
        },
      }));
    }
  }
}

// --- tab switcher ---
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});

// Render all tabs on load
renderOverview();
renderTrends();
renderHeatmap();
renderExplorer();
</script>
</body>
</html>
""".replace("__PAYLOAD__", payload_json)


def main():
    if not PILOT_ROOT.exists():
        print(f"ERROR: {PILOT_ROOT} not found", file=sys.stderr)
        return 1

    print(f"Discovering experiments under {PILOT_ROOT}...")
    experiments = discover_experiments()
    print(f"  found {len(experiments)} experiment folders:")
    for e in experiments:
        n_eval = len(e["ood_eval"])
        n_probe = len(e["ood_per_probe"])
        n_sample = len(e["per_sample"])
        print(f"    {e['name']:<30} eval_rows={n_eval}  per_probe_rows={n_probe}  per_sample_rows={n_sample}")

    print(f"\nIndexing rollouts...")
    rollouts = discover_rollouts()
    print(f"  indexed {len(rollouts)} rollouts")

    html = build_html(experiments, rollouts)
    with open(OUTPUT_PATH, "w") as f:
        f.write(html)
    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print(f"Open with: open {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
