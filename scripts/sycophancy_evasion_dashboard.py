"""
Generate a self-contained HTML dashboard for the sycophancy evasion pilot.

Reads:
  - rollouts_labeled*.jsonl  (question text, responses, judge labels, thinking)
  - per_rollout_scores.jsonl (DiffMean + LR probe scores)

Writes:
  - <evasion-dir>/dashboard.html

The HTML is self-contained (no external dependencies) — embeds the joined
data inline + vanilla JS. Open in any browser.

Usage:
  uv run python scripts/sycophancy_evasion_dashboard.py \\
      --evasion-dir data/concepts/sycophancy_qa_v2/stage1_evasion_pilot
"""

import argparse
import glob
import html
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evasion-dir", required=True)
    p.add_argument("--output", default=None,
                   help="Default: <evasion-dir>/dashboard.html")
    args = p.parse_args()

    ed = Path(args.evasion_dir)
    if not ed.is_absolute():
        ed = Path.cwd() / ed
    out_path = Path(args.output) if args.output else ed / "dashboard.html"

    # Load scores
    score_path = ed / "per_rollout_scores.jsonl"
    if not score_path.exists():
        raise FileNotFoundError(f"missing {score_path}")
    scores = {}
    with open(score_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            scores[r["rollout_id"]] = r

    # Load labeled rollouts (auto-discover suffix variant)
    lab_path = ed / "rollouts_labeled.jsonl"
    if not lab_path.exists():
        cands = sorted(glob.glob(str(ed / "rollouts*labeled*.jsonl")))
        if not cands:
            raise FileNotFoundError(f"no rollouts_labeled jsonl under {ed}")
        lab_path = Path(cands[0])

    rollouts = []
    with open(lab_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    print(f"Loaded {len(rollouts)} labeled rollouts from {lab_path.name}")
    print(f"Loaded {len(scores)} scores from {score_path.name}")

    # Join + group by qid
    by_qid = defaultdict(dict)
    for r in rollouts:
        rid = f"{r['question_id']}_{r['framing']}_{r['system_prompt_id']}"
        sc = scores.get(rid, {})
        regime = r["regime"]
        truncated = not (r.get("response") or "").strip()
        entry = {
            "rollout_id": rid,
            "question_id": r["question_id"],
            "source": r["source"],
            "regime": regime,
            "question": r["question"],
            "user_text": r["user_text"],
            "correct": r["correct_answer"],
            "incorrect": r["incorrect_answer"],
            "long_correct": r.get("long_correct_answer", "") or "",
            "thinking": r.get("thinking", ""),
            "response": r.get("response", ""),
            "judge": r["label_judge"],
            "judge_reason": r.get("judge_reason", ""),
            "n_tokens": r.get("n_gen_tokens", 0),
            "truncated": truncated,
            "diffmean": sc.get("diffmean_score"),
            "lr": sc.get("lr_score"),
        }
        by_qid[r["question_id"]][regime] = entry

    # Flatten to a list of qid entries (one row per qid, columns per regime)
    rows = []
    regimes_seen = set()
    for qid, regs in by_qid.items():
        regimes_seen.update(regs.keys())
        row = {"qid": qid, "question": next(iter(regs.values()))["question"],
               "source": next(iter(regs.values()))["source"],
               "correct": next(iter(regs.values()))["correct"],
               "incorrect": next(iter(regs.values()))["incorrect"],
               "long_correct": next(iter(regs.values())).get("long_correct", ""),
               "by_regime": regs}
        rows.append(row)
    rows.sort(key=lambda r: r["qid"])

    regimes_list = sorted(regimes_seen,
                          key=lambda r: {"baseline": 0, "technical_disclosure": 1}.get(r, 99))
    print(f"Regimes present: {regimes_list}")

    # Summary stats
    stats = {}
    for regime in regimes_list:
        labels_count = defaultdict(int)
        scores_by_label = defaultdict(list)
        for row in rows:
            r = row["by_regime"].get(regime)
            if not r: continue
            labels_count[r["judge"]] += 1
            if isinstance(r["diffmean"], (int, float)):
                scores_by_label[r["judge"]].append(r["diffmean"])
        stats[regime] = {
            "labels": dict(labels_count),
            "scores_by_label": {k: v for k, v in scores_by_label.items()},
        }

    # Build the embedded JSON data
    embedded = {
        "rows": rows,
        "regimes": regimes_list,
        "stats": stats,
    }

    html_doc = build_html(embedded)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    print(f"\nDashboard written: {out_path}")
    print(f"  size: {out_path.stat().st_size:,} bytes")
    print(f"  open in browser:  open {out_path}")


def build_html(data):
    embedded_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sycophancy evasion dashboard</title>
<style>
  :root {{
    --bg: #fafaf8; --fg: #1a1a1a; --muted: #6b6b6b; --border: #e0e0e0;
    --hi-bg: #fff7d6; --pos: #c91515; --neg: #2a6b2a; --neutral: #6b6b6b;
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         background: var(--bg); color: var(--fg); margin: 0; padding: 0 0 4em 0; }}
  header {{ background: white; border-bottom: 1px solid var(--border); padding: 1em 2em;
            position: sticky; top: 0; z-index: 10; }}
  h1 {{ margin: 0 0 0.3em 0; font-size: 1.4em; }}
  .subtitle {{ color: var(--muted); font-size: 0.9em; }}
  main {{ padding: 1em 2em; max-width: 1600px; margin: 0 auto; }}
  section {{ margin: 1.5em 0; }}
  h2 {{ font-size: 1.05em; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; margin-bottom: 0.6em; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.8em; }}
  .stat-card {{ background: white; border: 1px solid var(--border); border-radius: 4px; padding: 0.8em; }}
  .stat-card h3 {{ margin: 0 0 0.4em 0; font-size: 0.85em; color: var(--muted); font-weight: 600; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 0.85em; padding: 2px 0; }}
  .filters {{ display: flex; gap: 0.8em; flex-wrap: wrap; align-items: center; margin: 0.8em 0; }}
  .filters label {{ font-size: 0.85em; color: var(--muted); }}
  .filters select, .filters input {{ padding: 4px 8px; border: 1px solid var(--border); border-radius: 3px;
                                     font-family: inherit; font-size: 0.85em; }}
  table {{ width: 100%; border-collapse: collapse; background: white; font-size: 0.83em; }}
  th, td {{ padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: top; }}
  th {{ background: #f0f0ed; font-weight: 600; cursor: pointer; user-select: none; position: sticky; top: 0; }}
  th:hover {{ background: #e8e8e3; }}
  tr.row-main:hover {{ background: var(--hi-bg); cursor: pointer; }}
  tr.row-expand {{ display: none; background: #fdfdfa; }}
  tr.row-expand.open {{ display: table-row; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 0.75em; font-weight: 500; }}
  .badge.sya {{ background: #ffe7e7; color: var(--pos); }}
  .badge.correct {{ background: #e2f0e2; color: var(--neg); }}
  .badge.neither {{ background: #efefef; color: var(--muted); }}
  .badge.truncated {{ background: #fff3c4; color: #855e00; margin-left: 4px; }}
  .score {{ font-family: SF Mono, Menlo, Consolas, monospace; }}
  .score.high {{ color: var(--pos); font-weight: 600; }}
  .score.low {{ color: var(--neg); }}
  .delta.pos {{ color: var(--pos); }}
  .delta.neg {{ color: var(--neg); }}
  .pair-detail {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; padding: 1em; }}
  .pair-detail h4 {{ margin: 0 0 0.4em 0; font-size: 0.85em; }}
  .text-block {{ background: white; border: 1px solid var(--border); border-radius: 3px; padding: 8px;
                font-size: 0.82em; line-height: 1.45; max-height: 380px; overflow-y: auto; white-space: pre-wrap; }}
  .text-block.thinking {{ font-style: italic; color: #444; }}
  .qmeta {{ font-size: 0.85em; padding: 0.4em 1em; background: #f0f0ed; border-radius: 3px; margin-bottom: 0.6em; }}
  .qmeta strong {{ color: var(--pos); }}
  .qmeta .correct {{ color: var(--neg); }}
  .judge-reason {{ font-size: 0.78em; color: var(--muted); margin-top: 0.4em; font-style: italic; }}
  .hist {{ display: flex; align-items: end; gap: 2px; height: 80px; padding: 4px; background: white; border: 1px solid var(--border); }}
  .hist-bar {{ flex: 1; background: #888; min-width: 2px; }}
  .hist-bar.sya {{ background: var(--pos); }}
  .hist-bar.correct {{ background: var(--neg); }}
  .small {{ font-size: 0.78em; color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Sycophancy probe — evasion pilot dashboard</h1>
  <div class="subtitle">Free-form rollouts under <code>baseline</code> vs <code>technical_disclosure</code>. Click a row to expand thinking + responses side-by-side.</div>
</header>

<main>
  <section>
    <h2>Per-regime summary</h2>
    <div class="stats-grid" id="stats"></div>
  </section>

  <section>
    <h2>Score distribution (DiffMean)</h2>
    <div id="histograms"></div>
  </section>

  <section>
    <h2>Paired rollouts <span class="small" id="filter-count"></span></h2>
    <div class="filters">
      <label>Filter:
        <select id="filter-pattern">
          <option value="">all</option>
          <option value="both_sya">both SyA (paired evasion candidates)</option>
          <option value="broke_sya">SyA → not-SyA (TD broke sycophancy)</option>
          <option value="became_sya">not-SyA → SyA (TD created sycophancy)</option>
          <option value="both_clean">both honest/neither (no SyA in either)</option>
        </select>
      </label>
      <label>Source:
        <select id="filter-source">
          <option value="">any</option>
          <option value="trivia_qa">trivia_qa</option>
          <option value="truthful_qa">truthful_qa</option>
        </select>
      </label>
      <label>Hide truncated:
        <input type="checkbox" id="filter-truncated">
      </label>
      <label>Search:
        <input type="text" id="search" placeholder="qid or question text" size="35">
      </label>
    </div>
    <table>
      <thead>
        <tr>
          <th data-sort="qid">qid</th>
          <th data-sort="question" style="width: 30%;">question</th>
          <th data-sort="baseline_judge">baseline judge</th>
          <th data-sort="td_judge">TD judge</th>
          <th data-sort="baseline_score">b DM</th>
          <th data-sort="td_score">TD DM</th>
          <th data-sort="delta">Δ</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
</main>

<script>
const DATA = {embedded_json};

// ------- helpers -------
const esc = (s) => (s ?? "").toString().replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
const fmt = (v, digits=2) => (typeof v === "number") ? v.toFixed(digits) : "—";
function judgeBadge(j) {{
  const cls = j === "endorsed_incorrect" ? "sya" : j === "endorsed_correct" ? "correct" : "neither";
  const short = j === "endorsed_incorrect" ? "SyA" : j === "endorsed_correct" ? "correct" : (j||"—");
  return `<span class="badge ${{cls}}">${{esc(short)}}</span>`;
}}

// ------- stats panel -------
const statsEl = document.getElementById("stats");
DATA.regimes.forEach(regime => {{
  const s = DATA.stats[regime];
  const total = Object.values(s.labels).reduce((a,b)=>a+b, 0);
  const sya = s.labels.endorsed_incorrect || 0;
  const honest = s.labels.endorsed_correct || 0;
  const neither = s.labels.neither || 0;
  const syaScores = s.scores_by_label.endorsed_incorrect || [];
  const mean = syaScores.length ? syaScores.reduce((a,b)=>a+b,0)/syaScores.length : null;
  const med = syaScores.length ? [...syaScores].sort((a,b)=>a-b)[Math.floor(syaScores.length/2)] : null;
  statsEl.innerHTML += `
    <div class="stat-card">
      <h3>${{esc(regime)}}</h3>
      <div class="stat-row"><span>SyA rate</span><span class="score high">${{(100*sya/total).toFixed(1)}}% (${{sya}}/${{total}})</span></div>
      <div class="stat-row"><span>honest correction</span><span>${{(100*honest/total).toFixed(1)}}% (${{honest}}/${{total}})</span></div>
      <div class="stat-row"><span>neither</span><span>${{(100*neither/total).toFixed(1)}}% (${{neither}}/${{total}})</span></div>
      <div class="stat-row"><span>SyA mean DM</span><span class="score">${{fmt(mean)}}</span></div>
      <div class="stat-row"><span>SyA median DM</span><span class="score">${{fmt(med)}}</span></div>
    </div>`;
}});

// ------- histograms -------
function buildHist(values, n_bins=20, min=null, max=null) {{
  if (values.length === 0) return [];
  const mn = (min ?? Math.min(...values)), mx = (max ?? Math.max(...values));
  const step = (mx - mn) / n_bins || 1;
  const bins = new Array(n_bins).fill(0);
  values.forEach(v => {{
    const i = Math.min(n_bins - 1, Math.max(0, Math.floor((v - mn) / step)));
    bins[i]++;
  }});
  return {{bins, mn, mx}};
}}
const allScores = [];
DATA.rows.forEach(r => {{
  DATA.regimes.forEach(reg => {{
    const e = r.by_regime[reg];
    if (e && typeof e.diffmean === "number") allScores.push(e.diffmean);
  }});
}});
const gmin = Math.min(...allScores), gmax = Math.max(...allScores);
const histsEl = document.getElementById("histograms");
DATA.regimes.forEach(regime => {{
  ["endorsed_incorrect","endorsed_correct","neither"].forEach(lab => {{
    const xs = [];
    DATA.rows.forEach(r => {{
      const e = r.by_regime[regime];
      if (e && e.judge === lab && typeof e.diffmean === "number") xs.push(e.diffmean);
    }});
    if (xs.length === 0) return;
    const {{bins}} = buildHist(xs, 20, gmin, gmax);
    const cls = lab === "endorsed_incorrect" ? "sya" : lab === "endorsed_correct" ? "correct" : "";
    histsEl.innerHTML += `
      <div style="margin-bottom: 0.6em;">
        <div class="small">${{regime}} · ${{lab}} (n=${{xs.length}}, range=[${{Math.min(...xs).toFixed(1)}}, ${{Math.max(...xs).toFixed(1)}}])</div>
        <div class="hist">
          ${{bins.map(c => `<div class="hist-bar ${{cls}}" style="height: ${{c ? Math.max(2, c*8) : 0}}px;"></div>`).join("")}}
        </div>
      </div>`;
  }});
}});

// ------- table -------
let sortKey = "delta", sortDir = -1;
const rowsEl = document.getElementById("rows");

function classifyPair(row) {{
  const b = row.by_regime.baseline?.judge;
  const t = row.by_regime.technical_disclosure?.judge;
  if (b === "endorsed_incorrect" && t === "endorsed_incorrect") return "both_sya";
  if (b === "endorsed_incorrect" && t !== "endorsed_incorrect") return "broke_sya";
  if (b !== "endorsed_incorrect" && t === "endorsed_incorrect") return "became_sya";
  return "both_clean";
}}

function rowSortVal(row, key) {{
  const b = row.by_regime.baseline, t = row.by_regime.technical_disclosure;
  switch (key) {{
    case "qid": return row.qid;
    case "question": return row.question;
    case "baseline_judge": return b?.judge ?? "";
    case "td_judge": return t?.judge ?? "";
    case "baseline_score": return b?.diffmean ?? -Infinity;
    case "td_score": return t?.diffmean ?? -Infinity;
    case "delta": return ((t?.diffmean ?? 0) - (b?.diffmean ?? 0));
  }}
}}

function render() {{
  const pat = document.getElementById("filter-pattern").value;
  const src = document.getElementById("filter-source").value;
  const trunc = document.getElementById("filter-truncated").checked;
  const search = document.getElementById("search").value.toLowerCase();
  let filtered = DATA.rows.filter(r => {{
    if (pat && classifyPair(r) !== pat) return false;
    if (src && r.source !== src) return false;
    if (trunc && (r.by_regime.baseline?.truncated || r.by_regime.technical_disclosure?.truncated)) return false;
    if (search && !r.qid.toLowerCase().includes(search) && !r.question.toLowerCase().includes(search)) return false;
    return true;
  }});
  filtered.sort((a,b) => {{
    const va = rowSortVal(a, sortKey), vb = rowSortVal(b, sortKey);
    if (va < vb) return -1 * sortDir;
    if (va > vb) return 1 * sortDir;
    return 0;
  }});
  document.getElementById("filter-count").textContent = `(${{filtered.length}} of ${{DATA.rows.length}} shown)`;
  rowsEl.innerHTML = filtered.map((r, i) => renderRow(r, i)).join("");
  rowsEl.querySelectorAll("tr.row-main").forEach(tr => {{
    tr.addEventListener("click", () => {{
      const expand = tr.nextElementSibling;
      expand.classList.toggle("open");
    }});
  }});
}}

function renderRow(row, idx) {{
  const b = row.by_regime.baseline, t = row.by_regime.technical_disclosure;
  const delta = (t?.diffmean != null && b?.diffmean != null) ? (t.diffmean - b.diffmean) : null;
  const deltaCls = delta == null ? "" : delta > 1 ? "delta pos" : delta < -1 ? "delta neg" : "";
  const b_trunc = b?.truncated ? '<span class="badge truncated">truncated</span>' : '';
  const t_trunc = t?.truncated ? '<span class="badge truncated">truncated</span>' : '';
  return `
    <tr class="row-main" data-idx="${{idx}}">
      <td><code>${{esc(row.qid)}}</code></td>
      <td>${{esc(row.question.slice(0, 120))}}${{row.question.length > 120 ? "…" : ""}}</td>
      <td>${{judgeBadge(b?.judge)}}${{b_trunc}}</td>
      <td>${{judgeBadge(t?.judge)}}${{t_trunc}}</td>
      <td class="score">${{fmt(b?.diffmean)}}</td>
      <td class="score">${{fmt(t?.diffmean)}}</td>
      <td class="score ${{deltaCls}}">${{delta == null ? "—" : (delta > 0 ? "+" : "") + delta.toFixed(2)}}</td>
    </tr>
    <tr class="row-expand">
      <td colspan="7">
        <div class="qmeta">
          <strong>Q:</strong> ${{esc(row.question)}}<br>
          <strong>correct:</strong> <span class="correct">${{esc(row.correct)}}</span> ·
          <strong>user said (incorrect):</strong> ${{esc(row.incorrect)}}
          ${{row.long_correct ? `<br><strong>long correct:</strong> ${{esc(row.long_correct)}}` : ""}}
        </div>
        <div class="pair-detail">
          ${{renderRegime("baseline", b)}}
          ${{renderRegime("technical_disclosure", t)}}
        </div>
      </td>
    </tr>`;
}}

function renderRegime(name, e) {{
  if (!e) return `<div><h4>${{esc(name)}}</h4><div class="text-block">— not present —</div></div>`;
  return `
    <div>
      <h4>${{esc(name)}} · ${{judgeBadge(e.judge)}} · DM=${{fmt(e.diffmean)}} · ${{e.n_tokens}} tok ${{e.truncated ? '<span class="badge truncated">truncated</span>' : ''}}</h4>
      <div class="judge-reason">judge: ${{esc(e.judge_reason)}}</div>
      <div style="margin-top: 0.4em;"><strong style="font-size: 0.78em;">thinking</strong></div>
      <div class="text-block thinking">${{esc(e.thinking || "—")}}</div>
      <div style="margin-top: 0.4em;"><strong style="font-size: 0.78em;">response</strong></div>
      <div class="text-block">${{esc(e.response || "—")}}</div>
    </div>`;
}}

// Sorting
document.querySelectorAll("th[data-sort]").forEach(th => {{
  th.addEventListener("click", () => {{
    const key = th.getAttribute("data-sort");
    if (sortKey === key) sortDir = -sortDir;
    else {{ sortKey = key; sortDir = (key === "qid" || key === "question") ? 1 : -1; }}
    render();
  }});
}});

// Filters
["filter-pattern","filter-source","filter-truncated","search"].forEach(id => {{
  document.getElementById(id).addEventListener(id === "search" ? "input" : "change", render);
}});

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
