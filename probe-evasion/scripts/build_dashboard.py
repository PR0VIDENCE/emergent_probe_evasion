#!/usr/bin/env python3
"""Build an interactive HTML dashboard for probe evasion experiment results."""

import json
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "outputs", "trial_1_data")
OUTPUT_PATH = os.path.join(DATA_DIR, "dashboard.html")

REGIMES = ["baseline", "awareness_only", "technical_disclosure", "technical_extended", "iterative_feedback"]

def load_data():
    all_data = {}
    for regime in REGIMES:
        path = os.path.join(DATA_DIR, f"{regime}.json")
        with open(path) as f:
            all_data[regime] = json.load(f)
    return all_data

def build_html(all_data):
    data_json = json.dumps(all_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Probe Evasion Experiment — Trial 1 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

:root {{
  --bg: #0f0f1a;
  --bg-card: #1a1a2e;
  --bg-card-hover: #222240;
  --bg-input: #16162b;
  --border: #2a2a4a;
  --text: #e0e0f0;
  --text-dim: #8888aa;
  --text-bright: #ffffff;
  --accent: #6c63ff;
  --accent-dim: #4a44b0;
  --green: #22c55e;
  --red: #ef4444;
  --orange: #f59e0b;
  --blue: #3b82f6;
  --purple: #a855f7;
  --teal: #14b8a6;

  --regime-baseline: #3b82f6;
  --regime-awareness: #f59e0b;
  --regime-technical: #a855f7;
  --regime-extended: #22c55e;
  --regime-iterative: #ef4444;
}}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}}

.header {{
  background: linear-gradient(135deg, #1a1a3e 0%, #0f0f2a 100%);
  border-bottom: 1px solid var(--border);
  padding: 2rem 2rem 0;
}}

.header h1 {{
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-bright);
  margin-bottom: 0.25rem;
}}

.header .subtitle {{
  color: var(--text-dim);
  font-size: 0.9rem;
  margin-bottom: 1.5rem;
}}

.tabs {{
  display: flex;
  gap: 0;
  border-bottom: none;
}}

.tab {{
  padding: 0.75rem 1.5rem;
  cursor: pointer;
  color: var(--text-dim);
  font-size: 0.85rem;
  font-weight: 500;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 8px 8px 0 0;
  transition: all 0.2s;
  user-select: none;
}}

.tab:hover {{
  color: var(--text);
  background: var(--bg-card);
}}

.tab.active {{
  color: var(--accent);
  background: var(--bg);
  border-color: var(--border);
  border-bottom-color: var(--bg);
  position: relative;
  z-index: 1;
}}

.container {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 1.5rem 2rem;
}}

.tab-content {{
  display: none;
}}

.tab-content.active {{
  display: block;
}}

/* Stats Cards */
.stats-row {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}

.stat-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
}}

.stat-card .label {{
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  margin-bottom: 0.25rem;
}}

.stat-card .value {{
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-bright);
}}

.stat-card .detail {{
  font-size: 0.8rem;
  color: var(--text-dim);
  margin-top: 0.25rem;
}}

/* Card */
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}}

.card h2 {{
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 1rem;
}}

.card h3 {{
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 0.75rem;
}}

/* Heatmap */
.heatmap-container {{
  overflow-x: auto;
}}

.heatmap {{
  width: 100%;
  border-collapse: separate;
  border-spacing: 3px;
  font-size: 0.8rem;
}}

.heatmap th {{
  padding: 0.5rem 0.75rem;
  font-weight: 600;
  color: var(--text-dim);
  text-align: center;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}

.heatmap th.row-header {{
  text-align: left;
  min-width: 160px;
}}

.heatmap td {{
  padding: 0.6rem 0.75rem;
  text-align: center;
  border-radius: 6px;
  font-weight: 600;
  font-size: 0.85rem;
  cursor: default;
  transition: transform 0.15s;
  position: relative;
}}

.heatmap td:hover {{
  transform: scale(1.05);
  z-index: 1;
}}

/* Charts */
.chart-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 1.5rem;
}}

.chart-box {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
}}

.chart-box h3 {{
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 1rem;
}}

.chart-box canvas {{
  max-height: 350px;
}}

/* Response Browser */
.browser-controls {{
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  align-items: end;
}}

.control-group {{
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}}

.control-group label {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  font-weight: 600;
}}

.control-group select {{
  background: var(--bg-input);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 2rem 0.5rem 0.75rem;
  font-size: 0.85rem;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%238888aa' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 0.6rem center;
}}

.control-group select:focus {{
  outline: none;
  border-color: var(--accent);
}}

.response-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}}

.response-header {{
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
}}

.response-header .question-text {{
  font-weight: 600;
  font-size: 1rem;
  color: var(--text-bright);
}}

.response-meta {{
  display: flex;
  gap: 1rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}}

.response-meta .meta-item {{
  display: flex;
  align-items: center;
  gap: 0.3rem;
}}

.regime-badge {{
  display: inline-block;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}}

.regime-badge.baseline {{ background: rgba(59,130,246,0.15); color: var(--regime-baseline); }}
.regime-badge.awareness_only {{ background: rgba(245,158,11,0.15); color: var(--regime-awareness); }}
.regime-badge.technical_disclosure {{ background: rgba(168,85,247,0.15); color: var(--regime-technical); }}
.regime-badge.technical_extended {{ background: rgba(34,197,94,0.15); color: var(--regime-extended); }}
.regime-badge.iterative_feedback {{ background: rgba(239,68,68,0.15); color: var(--regime-iterative); }}

.response-body {{
  display: grid;
  grid-template-columns: 1fr 340px;
  min-height: 300px;
}}

@media (max-width: 900px) {{
  .response-body {{
    grid-template-columns: 1fr;
  }}
}}

.response-text {{
  padding: 1.5rem;
  overflow-y: auto;
  max-height: 600px;
}}

.text-section {{
  margin-bottom: 1.5rem;
}}

.text-section-header {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  font-weight: 600;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  user-select: none;
}}

.text-section-header .toggle {{
  font-size: 0.6rem;
  transition: transform 0.2s;
}}

.text-section-header .toggle.collapsed {{
  transform: rotate(-90deg);
}}

.text-content {{
  font-size: 0.85rem;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--text);
  max-height: 400px;
  overflow-y: auto;
  padding-right: 0.5rem;
}}

.text-content.thinking {{
  color: var(--text-dim);
  font-style: italic;
  border-left: 3px solid var(--border);
  padding-left: 1rem;
}}

.text-content.collapsed {{
  display: none;
}}

.probe-sidebar {{
  padding: 1.5rem;
  border-left: 1px solid var(--border);
  background: rgba(0,0,0,0.15);
  overflow-y: auto;
  max-height: 600px;
}}

.probe-sidebar h3 {{
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-dim);
  font-weight: 600;
  margin-bottom: 0.75rem;
}}

.probe-position-group {{
  margin-bottom: 1.25rem;
}}

.probe-position-label {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--text-dim);
  margin-bottom: 0.4rem;
  font-weight: 600;
}}

.probe-bar-row {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.3rem;
}}

.probe-layer-label {{
  font-size: 0.7rem;
  color: var(--text-dim);
  width: 50px;
  text-align: right;
  flex-shrink: 0;
}}

.probe-bar-track {{
  flex: 1;
  height: 18px;
  background: rgba(255,255,255,0.05);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}}

.probe-bar-fill {{
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
  min-width: 2px;
}}

.probe-bar-value {{
  font-size: 0.7rem;
  font-weight: 600;
  width: 42px;
  text-align: right;
  flex-shrink: 0;
}}

.threshold-line {{
  position: absolute;
  left: 50%;
  top: 0;
  bottom: 0;
  width: 1px;
  background: rgba(255,255,255,0.3);
  z-index: 1;
}}

/* Turns (for iterative feedback) */
.turns-nav {{
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  padding: 0 1.5rem;
  padding-top: 1rem;
}}

.turn-btn {{
  padding: 0.4rem 1rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-dim);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}}

.turn-btn:hover {{
  background: var(--bg-card-hover);
  color: var(--text);
}}

.turn-btn.active {{
  background: var(--accent-dim);
  color: var(--text-bright);
  border-color: var(--accent);
}}

/* Position selector */
.position-tabs {{
  display: flex;
  gap: 0.25rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}}

.position-tab {{
  padding: 0.3rem 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text-dim);
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s;
}}

.position-tab:hover {{
  background: var(--bg-card-hover);
}}

.position-tab.active {{
  background: var(--accent-dim);
  color: var(--text-bright);
  border-color: var(--accent);
}}

/* Legend */
.legend {{
  display: flex;
  gap: 1.25rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}}

.legend-item {{
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}}

.legend-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
}}

/* Scrollbar */
::-webkit-scrollbar {{
  width: 6px;
  height: 6px;
}}

::-webkit-scrollbar-track {{
  background: transparent;
}}

::-webkit-scrollbar-thumb {{
  background: var(--border);
  border-radius: 3px;
}}

::-webkit-scrollbar-thumb:hover {{
  background: #3a3a5a;
}}

/* Tooltip */
.tooltip {{
  position: fixed;
  background: #222244;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  color: var(--text);
  max-width: 300px;
  z-index: 1000;
  pointer-events: none;
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  display: none;
}}

.no-data {{
  text-align: center;
  padding: 3rem;
  color: var(--text-dim);
  font-style: italic;
}}

/* Summary insight cards */
.insights {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}

.insight {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  border-left: 4px solid var(--accent);
}}

.insight .insight-title {{
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 0.35rem;
}}

.insight .insight-body {{
  font-size: 0.8rem;
  color: var(--text-dim);
  line-height: 1.5;
}}

.insight .insight-value {{
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--accent);
}}

.bold {{ font-weight: 700; color: var(--text-bright); }}
</style>
</head>
<body>

<div class="header">
  <h1>Probe Evasion Experiment — Trial 1</h1>
  <p class="subtitle">QwQ-32B responses to tree-related questions across 5 evasion regimes &middot; 125 total rollouts &middot; Linear probe confidence at 4 extraction positions &times; 5 layers</p>
  <div class="tabs">
    <div class="tab active" data-tab="overview">Overview</div>
    <div class="tab" data-tab="comparison">Regime Comparison</div>
    <div class="tab" data-tab="browser">Response Browser</div>
    <div class="tab" data-tab="details">Probe Details</div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<!-- ==================== OVERVIEW TAB ==================== -->
<div class="container tab-content active" id="tab-overview">
  <div class="stats-row" id="stats-row"></div>
  <div class="insights" id="insights"></div>
  <div class="card">
    <h2>Probe Confidence Heatmap — Mean across rollouts</h2>
    <div class="position-tabs" id="overview-position-tabs"></div>
    <div class="heatmap-container">
      <table class="heatmap" id="overview-heatmap"></table>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Mean Probe Confidence by Regime</h3>
      <canvas id="chart-regime-bars"></canvas>
    </div>
    <div class="chart-box">
      <h3>Token Counts by Regime</h3>
      <canvas id="chart-tokens"></canvas>
    </div>
  </div>
</div>

<!-- ==================== COMPARISON TAB ==================== -->
<div class="container tab-content" id="tab-comparison">
  <div class="card">
    <h2>Probe Confidence by Layer — Averaged across questions &amp; rollouts</h2>
    <div class="position-tabs" id="comparison-position-tabs"></div>
    <div style="max-width: 800px;">
      <canvas id="chart-layers"></canvas>
    </div>
  </div>
  <div class="card">
    <h2>Probe Confidence by Extraction Position — Averaged across layers</h2>
    <canvas id="chart-positions"></canvas>
  </div>
  <div class="card">
    <h2>Per-Question Probe Confidence</h2>
    <div class="position-tabs" id="perq-position-tabs"></div>
    <canvas id="chart-per-question"></canvas>
  </div>
</div>

<!-- ==================== BROWSER TAB ==================== -->
<div class="container tab-content" id="tab-browser">
  <div class="browser-controls">
    <div class="control-group">
      <label>Regime</label>
      <select id="sel-regime"></select>
    </div>
    <div class="control-group">
      <label>Question</label>
      <select id="sel-question"></select>
    </div>
    <div class="control-group">
      <label>Rollout</label>
      <select id="sel-rollout"></select>
    </div>
  </div>
  <div id="response-display"></div>
</div>

<!-- ==================== DETAILS TAB ==================== -->
<div class="container tab-content" id="tab-details">
  <div class="card">
    <h2>Full Probe Score Matrix</h2>
    <p style="font-size:0.8rem;color:var(--text-dim);margin-bottom:1rem;">Every individual probe confidence per regime &times; question &times; position &times; layer (averaged across rollouts)</p>
    <div class="browser-controls">
      <div class="control-group">
        <label>Question</label>
        <select id="detail-question"></select>
      </div>
    </div>
    <div class="heatmap-container" id="detail-matrix"></div>
  </div>
</div>

<script>
// ============================================================
// DATA
// ============================================================
const DATA = {data_json};

const REGIMES = {json.dumps(REGIMES)};
const REGIME_LABELS = {{
  baseline: "Baseline",
  awareness_only: "Awareness Only",
  technical_disclosure: "Technical Disclosure",
  technical_extended: "Technical Extended",
  iterative_feedback: "Iterative Feedback"
}};
const REGIME_COLORS = {{
  baseline: "#3b82f6",
  awareness_only: "#f59e0b",
  technical_disclosure: "#a855f7",
  technical_extended: "#22c55e",
  iterative_feedback: "#ef4444"
}};
const REGIME_COLORS_DIM = {{
  baseline: "rgba(59,130,246,0.15)",
  awareness_only: "rgba(245,158,11,0.15)",
  technical_disclosure: "rgba(168,85,247,0.15)",
  technical_extended: "rgba(34,197,94,0.15)",
  iterative_feedback: "rgba(239,68,68,0.15)"
}};

const POSITIONS = ["last_token", "end_of_reasoning", "first_answer_sentence_end", "answer_mean_pool"];
const POSITION_LABELS = {{
  last_token: "Last Token",
  end_of_reasoning: "End of Reasoning",
  first_answer_sentence_end: "1st Answer Sent.",
  answer_mean_pool: "Answer Mean Pool"
}};
const LAYERS = ["8", "20", "32", "44", "56"];
const QUESTIONS = ["q1", "q2", "q3", "q4", "q5"];

// ============================================================
// HELPERS
// ============================================================

function getEntries(regime) {{
  return DATA[regime] || [];
}}

// Get probe result for an entry. For iterative_feedback, use last turn.
function getProbeResults(entry) {{
  if (entry.turns) {{
    const lastTurn = entry.turns[entry.turns.length - 1];
    return lastTurn.probe_results;
  }}
  return entry.probe_results;
}}

function getAnswer(entry) {{
  if (entry.turns) return entry.turns[entry.turns.length - 1].answer;
  return entry.answer;
}}

function getThinking(entry) {{
  if (entry.turns) return entry.turns[entry.turns.length - 1].thinking_trace;
  return entry.thinking_trace;
}}

function getTokens(entry) {{
  if (entry.turns) {{
    let thinkTotal = 0, ansTotal = 0;
    for (const t of entry.turns) {{
      thinkTotal += t.thinking_tokens || 0;
      ansTotal += t.answer_tokens || 0;
    }}
    return {{ thinking: thinkTotal, answer: ansTotal }};
  }}
  return {{ thinking: entry.thinking_tokens || 0, answer: entry.answer_tokens || 0 }};
}}

function getElapsed(entry) {{
  if (entry.turns) {{
    return entry.turns.reduce((s, t) => s + (t.elapsed_seconds || 0), 0);
  }}
  return entry.elapsed_seconds || 0;
}}

// Mean confidence for a regime at a given position, optionally filtered by question
function meanConfidence(regime, position, layer, questionId) {{
  const entries = getEntries(regime);
  let vals = [];
  for (const e of entries) {{
    if (questionId && e.question_id !== questionId) continue;
    const pr = getProbeResults(e);
    if (pr && pr[position] && pr[position][layer]) {{
      vals.push(pr[position][layer].mean_confidence);
    }}
  }}
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}}

// Grand mean across all layers for a position
function meanConfidenceAllLayers(regime, position, questionId) {{
  let vals = [];
  for (const layer of LAYERS) {{
    const v = meanConfidence(regime, position, layer, questionId);
    if (v !== null) vals.push(v);
  }}
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}}

// Grand mean across all positions and layers
function grandMean(regime, questionId) {{
  let vals = [];
  for (const pos of POSITIONS) {{
    for (const layer of LAYERS) {{
      const v = meanConfidence(regime, pos, layer, questionId);
      if (v !== null) vals.push(v);
    }}
  }}
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}}

function confColor(v) {{
  if (v === null) return "#333";
  // Green (low) to yellow (mid) to red (high)
  const r = v < 0.5 ? Math.round(v * 2 * 255) : 255;
  const g = v < 0.5 ? 255 : Math.round((1 - (v - 0.5) * 2) * 255);
  return `rgb(${{r}}, ${{g}}, 60)`;
}}

function confTextColor(v) {{
  if (v === null) return "#888";
  return v > 0.35 && v < 0.65 ? "#111" : "#fff";
}}

function fmt(v, digits) {{
  if (v === null || v === undefined) return "—";
  return v.toFixed(digits !== undefined ? digits : 3);
}}

const questionTexts = {{}};

// ============================================================
// TAB SWITCHING
// ============================================================
document.querySelectorAll('.tab').forEach(tab => {{
  tab.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  }});
}});

// ============================================================
// OVERVIEW TAB
// ============================================================
function buildOverview() {{
  // Collect question texts
  for (const regime of REGIMES) {{
    for (const e of getEntries(regime)) {{
      questionTexts[e.question_id] = e.question;
    }}
  }}

  // Stats
  let totalResponses = 0;
  let totalThinkTokens = 0;
  let totalAnswerTokens = 0;
  let totalTime = 0;
  for (const regime of REGIMES) {{
    for (const e of getEntries(regime)) {{
      totalResponses++;
      const t = getTokens(e);
      totalThinkTokens += t.thinking;
      totalAnswerTokens += t.answer;
      totalTime += getElapsed(e);
    }}
  }}

  document.getElementById('stats-row').innerHTML = `
    <div class="stat-card">
      <div class="label">Total Responses</div>
      <div class="value">${{totalResponses}}</div>
      <div class="detail">5 regimes &times; 5 questions &times; 5 rollouts</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Thinking Tokens</div>
      <div class="value">${{(totalThinkTokens / 1000).toFixed(1)}}k</div>
      <div class="detail">Avg ${{Math.round(totalThinkTokens / totalResponses)}} per response</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Answer Tokens</div>
      <div class="value">${{(totalAnswerTokens / 1000).toFixed(1)}}k</div>
      <div class="detail">Avg ${{Math.round(totalAnswerTokens / totalResponses)}} per response</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Generation Time</div>
      <div class="value">${{(totalTime / 60).toFixed(0)}}m</div>
      <div class="detail">Avg ${{(totalTime / totalResponses).toFixed(1)}}s per response</div>
    </div>
  `;

  // Insights
  const baselineGM = grandMean("baseline");
  const bestEvasion = REGIMES.slice(1).reduce((best, r) => {{
    const gm = grandMean(r);
    return gm < best.val ? {{ regime: r, val: gm }} : best;
  }}, {{ regime: null, val: 1 }});
  const worstEvasion = REGIMES.slice(1).reduce((worst, r) => {{
    const gm = grandMean(r);
    return gm > worst.val ? {{ regime: r, val: gm }} : worst;
  }}, {{ regime: null, val: 0 }});

  document.getElementById('insights').innerHTML = `
    <div class="insight">
      <div class="insight-title">Baseline Detection Rate</div>
      <div class="insight-value">${{(baselineGM * 100).toFixed(1)}}%</div>
      <div class="insight-body">Mean probe confidence when model answers without evasion instructions</div>
    </div>
    <div class="insight" style="border-left-color:var(--green)">
      <div class="insight-title">Best Evasion</div>
      <div class="insight-value">${{(bestEvasion.val * 100).toFixed(1)}}%</div>
      <div class="insight-body">${{REGIME_LABELS[bestEvasion.regime]}} achieved the lowest mean probe confidence</div>
    </div>
    <div class="insight" style="border-left-color:var(--red)">
      <div class="insight-title">Evasion Reduction</div>
      <div class="insight-value">${{((baselineGM - bestEvasion.val) / baselineGM * 100).toFixed(1)}}%</div>
      <div class="insight-body">Relative reduction from baseline to best-performing regime</div>
    </div>
  `;

  // Heatmap with position selector
  let currentPosition = "last_token";
  const posTabsEl = document.getElementById('overview-position-tabs');
  posTabsEl.innerHTML = POSITIONS.map(p =>
    `<button class="position-tab ${{p === currentPosition ? 'active' : ''}}" data-pos="${{p}}">${{POSITION_LABELS[p]}}</button>`
  ).join('');

  function renderHeatmap() {{
    let html = '<tr><th class="row-header">Regime</th>';
    for (const q of QUESTIONS) {{
      html += `<th>${{q.toUpperCase()}}</th>`;
    }}
    html += '<th>Mean</th></tr>';

    for (const regime of REGIMES) {{
      html += `<tr><th class="row-header"><span class="regime-badge ${{regime}}">${{REGIME_LABELS[regime]}}</span></th>`;
      let regimeVals = [];
      for (const q of QUESTIONS) {{
        const v = meanConfidenceAllLayers(regime, currentPosition, q);
        regimeVals.push(v);
        html += `<td style="background:${{confColor(v)}};color:${{confTextColor(v)}}">${{fmt(v, 3)}}</td>`;
      }}
      const meanV = regimeVals.filter(x => x !== null);
      const mv = meanV.length ? meanV.reduce((a, b) => a + b) / meanV.length : null;
      html += `<td style="background:${{confColor(mv)}};color:${{confTextColor(mv)}};font-weight:700">${{fmt(mv, 3)}}</td>`;
      html += '</tr>';
    }}
    document.getElementById('overview-heatmap').innerHTML = html;
  }}

  posTabsEl.addEventListener('click', e => {{
    if (e.target.classList.contains('position-tab')) {{
      posTabsEl.querySelectorAll('.position-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      currentPosition = e.target.dataset.pos;
      renderHeatmap();
    }}
  }});
  renderHeatmap();

  // Chart: mean confidence by regime (at last_token, all layers)
  const ctx1 = document.getElementById('chart-regime-bars').getContext('2d');
  new Chart(ctx1, {{
    type: 'bar',
    data: {{
      labels: REGIMES.map(r => REGIME_LABELS[r]),
      datasets: POSITIONS.map((pos, i) => ({{
        label: POSITION_LABELS[pos],
        data: REGIMES.map(r => meanConfidenceAllLayers(r, pos)),
        backgroundColor: [`rgba(59,130,246,0.${{3 + i * 2}})`, `rgba(245,158,11,0.${{3 + i * 2}})`, `rgba(168,85,247,0.${{3 + i * 2}})`, `rgba(34,197,94,0.${{3 + i * 2}})`, `rgba(239,68,68,0.${{3 + i * 2}})`][i] || 'rgba(100,100,200,0.5)',
        borderColor: ['#3b82f6', '#f59e0b', '#a855f7', '#22c55e'][i] || '#888',
        borderWidth: 1
      }}))
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: '#8888aa', font: {{ size: 11 }} }} }},
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8888aa', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y: {{ min: 0, max: 1, ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.08)' }}, title: {{ display: true, text: 'Mean Confidence', color: '#8888aa' }} }}
      }}
    }}
  }});

  // Chart: token counts
  const ctx2 = document.getElementById('chart-tokens').getContext('2d');
  const thinkData = REGIMES.map(r => {{
    const entries = getEntries(r);
    return entries.reduce((s, e) => s + getTokens(e).thinking, 0) / entries.length;
  }});
  const ansData = REGIMES.map(r => {{
    const entries = getEntries(r);
    return entries.reduce((s, e) => s + getTokens(e).answer, 0) / entries.length;
  }});
  new Chart(ctx2, {{
    type: 'bar',
    data: {{
      labels: REGIMES.map(r => REGIME_LABELS[r]),
      datasets: [
        {{ label: 'Thinking Tokens', data: thinkData, backgroundColor: 'rgba(108,99,255,0.5)', borderColor: '#6c63ff', borderWidth: 1 }},
        {{ label: 'Answer Tokens', data: ansData, backgroundColor: 'rgba(20,184,166,0.5)', borderColor: '#14b8a6', borderWidth: 1 }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8888aa', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ stacked: true, ticks: {{ color: '#8888aa', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y: {{ stacked: true, ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.08)' }}, title: {{ display: true, text: 'Avg Tokens', color: '#8888aa' }} }}
      }}
    }}
  }});
}}

// ============================================================
// COMPARISON TAB
// ============================================================
let compChartLayers = null;
let compChartPositions = null;
let compChartPerQ = null;

function buildComparison() {{
  // Layers chart with position selector
  let currentPos = "last_token";
  const posTabsEl = document.getElementById('comparison-position-tabs');
  posTabsEl.innerHTML = POSITIONS.map(p =>
    `<button class="position-tab ${{p === currentPos ? 'active' : ''}}" data-pos="${{p}}">${{POSITION_LABELS[p]}}</button>`
  ).join('');

  function renderLayersChart() {{
    if (compChartLayers) compChartLayers.destroy();
    const ctx = document.getElementById('chart-layers').getContext('2d');
    compChartLayers = new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: LAYERS.map(l => `Layer ${{l}}`),
        datasets: REGIMES.map(r => ({{
          label: REGIME_LABELS[r],
          data: LAYERS.map(l => meanConfidence(r, currentPos, l)),
          borderColor: REGIME_COLORS[r],
          backgroundColor: REGIME_COLORS[r] + '33',
          fill: false,
          tension: 0.3,
          pointRadius: 5,
          pointHoverRadius: 7,
          borderWidth: 2
        }}))
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#8888aa', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          y: {{ min: 0, max: 1, ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.08)' }}, title: {{ display: true, text: 'Mean Confidence', color: '#8888aa' }} }}
        }}
      }}
    }});
  }}

  posTabsEl.addEventListener('click', e => {{
    if (e.target.classList.contains('position-tab')) {{
      posTabsEl.querySelectorAll('.position-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      currentPos = e.target.dataset.pos;
      renderLayersChart();
    }}
  }});
  renderLayersChart();

  // Positions chart
  const ctx2 = document.getElementById('chart-positions').getContext('2d');
  compChartPositions = new Chart(ctx2, {{
    type: 'bar',
    data: {{
      labels: POSITIONS.map(p => POSITION_LABELS[p]),
      datasets: REGIMES.map(r => ({{
        label: REGIME_LABELS[r],
        data: POSITIONS.map(p => meanConfidenceAllLayers(r, p)),
        backgroundColor: REGIME_COLORS[r] + '88',
        borderColor: REGIME_COLORS[r],
        borderWidth: 1
      }}))
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ labels: {{ color: '#8888aa', font: {{ size: 11 }} }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8888aa', font: {{ size: 10 }} }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
        y: {{ min: 0, max: 1, ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.08)' }}, title: {{ display: true, text: 'Mean Confidence', color: '#8888aa' }} }}
      }}
    }}
  }});

  // Per-question chart
  let perQPos = "last_token";
  const perQTabsEl = document.getElementById('perq-position-tabs');
  perQTabsEl.innerHTML = POSITIONS.map(p =>
    `<button class="position-tab ${{p === perQPos ? 'active' : ''}}" data-pos="${{p}}">${{POSITION_LABELS[p]}}</button>`
  ).join('');

  function renderPerQ() {{
    if (compChartPerQ) compChartPerQ.destroy();
    const ctx = document.getElementById('chart-per-question').getContext('2d');
    compChartPerQ = new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: QUESTIONS.map(q => questionTexts[q] ? questionTexts[q].substring(0, 40) + '...' : q),
        datasets: REGIMES.map(r => ({{
          label: REGIME_LABELS[r],
          data: QUESTIONS.map(q => meanConfidenceAllLayers(r, perQPos, q)),
          backgroundColor: REGIME_COLORS[r] + '88',
          borderColor: REGIME_COLORS[r],
          borderWidth: 1
        }}))
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#8888aa', font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: '#8888aa', font: {{ size: 9 }}, maxRotation: 20 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
          y: {{ min: 0, max: 1, ticks: {{ color: '#8888aa' }}, grid: {{ color: 'rgba(255,255,255,0.08)' }}, title: {{ display: true, text: 'Mean Confidence', color: '#8888aa' }} }}
        }}
      }}
    }});
  }}

  perQTabsEl.addEventListener('click', e => {{
    if (e.target.classList.contains('position-tab')) {{
      perQTabsEl.querySelectorAll('.position-tab').forEach(t => t.classList.remove('active'));
      e.target.classList.add('active');
      perQPos = e.target.dataset.pos;
      renderPerQ();
    }}
  }});
  renderPerQ();
}}

// ============================================================
// RESPONSE BROWSER
// ============================================================
function buildBrowser() {{
  const selRegime = document.getElementById('sel-regime');
  const selQuestion = document.getElementById('sel-question');
  const selRollout = document.getElementById('sel-rollout');

  selRegime.innerHTML = REGIMES.map(r => `<option value="${{r}}">${{REGIME_LABELS[r]}}</option>`).join('');
  selQuestion.innerHTML = QUESTIONS.map(q => `<option value="${{q}}">${{q.toUpperCase()}}: ${{questionTexts[q] || ''}}</option>`).join('');
  selRollout.innerHTML = [0,1,2,3,4].map(i => `<option value="${{i}}">Rollout ${{i}}</option>`).join('');

  function renderResponse() {{
    const regime = selRegime.value;
    const qid = selQuestion.value;
    const rollout = parseInt(selRollout.value);
    const entries = getEntries(regime);
    const entry = entries.find(e => e.question_id === qid && e.rollout === rollout);
    if (!entry) {{
      document.getElementById('response-display').innerHTML = '<div class="no-data">No data found</div>';
      return;
    }}

    const isIterative = regime === 'iterative_feedback';
    let turnsHtml = '';
    let bodyHtml = '';

    if (isIterative && entry.turns) {{
      turnsHtml = `<div class="turns-nav">` +
        entry.turns.map((t, i) => `<button class="turn-btn ${{i === entry.turns.length - 1 ? 'active' : ''}}" data-turn="${{i}}">Turn ${{i + 1}}</button>`).join('') +
        `</div>`;
    }}

    function renderTurnBody(turnData, turnIdx) {{
      const thinking = turnData.thinking_trace || '';
      const answer = turnData.answer || '';
      const pr = turnData.probe_results;
      const elapsed = turnData.elapsed_seconds ? turnData.elapsed_seconds.toFixed(1) : '—';

      let textSections = '';
      if (thinking) {{
        textSections += `
          <div class="text-section">
            <div class="text-section-header" onclick="toggleSection(this)">
              <span class="toggle">&#9660;</span> Thinking Trace (${{turnData.thinking_tokens || 0}} tokens)
            </div>
            <div class="text-content thinking">${{escapeHtml(thinking)}}</div>
          </div>`;
      }}
      textSections += `
        <div class="text-section">
          <div class="text-section-header" onclick="toggleSection(this)">
            <span class="toggle">&#9660;</span> Answer (${{turnData.answer_tokens || 0}} tokens, ${{elapsed}}s)
          </div>
          <div class="text-content">${{escapeHtml(answer)}}</div>
        </div>`;

      let probeHtml = '';
      if (pr) {{
        for (const pos of POSITIONS) {{
          if (!pr[pos]) continue;
          probeHtml += `<div class="probe-position-group"><div class="probe-position-label">${{POSITION_LABELS[pos]}}</div>`;
          for (const layer of LAYERS) {{
            if (!pr[pos][layer]) continue;
            const conf = pr[pos][layer].mean_confidence;
            const pct = (conf * 100).toFixed(1);
            const color = confColor(conf);
            probeHtml += `
              <div class="probe-bar-row">
                <div class="probe-layer-label">L${{layer}}</div>
                <div class="probe-bar-track">
                  <div class="threshold-line"></div>
                  <div class="probe-bar-fill" style="width:${{pct}}%;background:${{color}}"></div>
                </div>
                <div class="probe-bar-value" style="color:${{color}}">${{pct}}%</div>
              </div>`;
          }}
          probeHtml += `</div>`;
        }}
      }}

      return `
        <div class="response-body" data-turn="${{turnIdx}}">
          <div class="response-text">${{textSections}}</div>
          <div class="probe-sidebar">
            <h3>Probe Scores</h3>
            ${{probeHtml}}
          </div>
        </div>`;
    }}

    const totalTokens = getTokens(entry);
    const totalElapsed = getElapsed(entry);

    if (isIterative && entry.turns) {{
      bodyHtml = entry.turns.map((t, i) => renderTurnBody(t, i)).join('');
    }} else {{
      bodyHtml = renderTurnBody({{
        thinking_trace: entry.thinking_trace,
        answer: entry.answer,
        thinking_tokens: entry.thinking_tokens,
        answer_tokens: entry.answer_tokens,
        elapsed_seconds: entry.elapsed_seconds,
        probe_results: entry.probe_results
      }}, 0);
    }}

    document.getElementById('response-display').innerHTML = `
      <div class="response-card">
        <div class="response-header">
          <div>
            <span class="regime-badge ${{regime}}">${{REGIME_LABELS[regime]}}</span>
            <span class="question-text" style="margin-left:0.75rem">${{entry.question}}</span>
          </div>
          <div class="response-meta">
            <div class="meta-item">Rollout ${{rollout}}</div>
            <div class="meta-item">${{totalTokens.thinking + totalTokens.answer}} tokens</div>
            <div class="meta-item">${{totalElapsed.toFixed(1)}}s</div>
          </div>
        </div>
        ${{turnsHtml}}
        ${{bodyHtml}}
      </div>`;

    // Show only last turn for iterative, handle turn switching
    if (isIterative && entry.turns) {{
      const turnBodies = document.querySelectorAll('.response-body[data-turn]');
      turnBodies.forEach(b => b.style.display = 'none');
      turnBodies[turnBodies.length - 1].style.display = '';

      document.querySelectorAll('.turn-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('.turn-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          turnBodies.forEach(b => b.style.display = 'none');
          turnBodies[parseInt(btn.dataset.turn)].style.display = '';
        }});
      }});
    }}
  }}

  selRegime.addEventListener('change', renderResponse);
  selQuestion.addEventListener('change', renderResponse);
  selRollout.addEventListener('change', renderResponse);
  renderResponse();
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

function toggleSection(header) {{
  const content = header.nextElementSibling;
  const toggle = header.querySelector('.toggle');
  content.classList.toggle('collapsed');
  toggle.classList.toggle('collapsed');
}}

// ============================================================
// DETAILS TAB
// ============================================================
function buildDetails() {{
  const sel = document.getElementById('detail-question');
  sel.innerHTML = `<option value="">All Questions (avg)</option>` +
    QUESTIONS.map(q => `<option value="${{q}}">${{q.toUpperCase()}}: ${{questionTexts[q] || ''}}</option>`).join('');

  function render() {{
    const qFilter = sel.value || null;
    let html = '<table class="heatmap"><tr><th class="row-header">Regime</th>';
    for (const pos of POSITIONS) {{
      for (const layer of LAYERS) {{
        html += `<th style="font-size:0.65rem">${{POSITION_LABELS[pos].substring(0,8)}}<br>L${{layer}}</th>`;
      }}
    }}
    html += '</tr>';

    for (const regime of REGIMES) {{
      html += `<tr><th class="row-header"><span class="regime-badge ${{regime}}">${{REGIME_LABELS[regime]}}</span></th>`;
      for (const pos of POSITIONS) {{
        for (const layer of LAYERS) {{
          const v = meanConfidence(regime, pos, layer, qFilter);
          html += `<td style="background:${{confColor(v)}};color:${{confTextColor(v)}};font-size:0.75rem;padding:0.4rem">${{fmt(v, 2)}}</td>`;
        }}
      }}
      html += '</tr>';
    }}
    html += '</table>';
    document.getElementById('detail-matrix').innerHTML = html;
  }}

  sel.addEventListener('change', render);
  render();
}}

// ============================================================
// INIT
// ============================================================
buildOverview();
buildComparison();
buildBrowser();
buildDetails();
</script>
</body>
</html>"""


def main():
    all_data = load_data()
    html = build_html(all_data)
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)
    print(f"Dashboard written to {OUTPUT_PATH}")
    print(f"  Size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
    print(f"  Open with: open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
