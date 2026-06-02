"""
Stage 7 — build the single self-contained HTML dashboard + summary.json.

Cross-concept summary table (universal regimes only) at the top; per-concept
drill-down below (validation verdict, per-regime stats, score histograms, an
interactive rollout explorer, and a multiplication-only digits×phrasing
heatmap). Failed concepts are rendered with a clear failure banner.

  uv run python scripts/eval_stage_dashboard.py --config configs/eval/qwq32b_fast.yaml
"""

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import common

UNIVERSAL_ORDER = ["baseline", "awareness_only", "technical_disclosure", "technical_extended"]
MAX_THINK = 1500
MAX_RESP = 2000


def _load_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def collect(config) -> dict:
    """Gather everything the dashboard needs into one embeddable dict."""
    concepts = {}
    for concept in config["concepts"]:
        cc = config["_concept_configs"][concept]
        paths = common.stage_paths(config, concept)
        validation = _load_json(paths["validation"]) or {}
        scores_summary = _load_json(paths["ev_dir"] / "scores_summary.json") or {}
        status = common.read_status(config, concept)

        rollouts = []
        for s in common.read_jsonl(paths["ev_scores"]):
            rollouts.append(s)
        # Join response/thinking/question text from labeled evasion rollouts.
        text_by_id = {}
        for r in common.read_jsonl(paths["ev_labeled"]):
            text_by_id[r["rollout_id"]] = {
                "user_text": r["input"]["user_text"][:600],
                "thinking": (r["output"].get("thinking") or "")[:MAX_THINK],
                "response": (r["output"].get("response") or "")[:MAX_RESP],
                "judge_reason": r.get("labels", {}).get("judge_reason", ""),
            }
        for s in rollouts:
            s.update(text_by_id.get(s["rollout_id"], {}))

        concepts[concept] = {
            "category": cc.get("category", ""),
            "canonical_position": cc["probe"].get("canonical_position", "answer_mean_pool"),
            "validation": validation,
            "scores_summary": scores_summary,
            "rollouts": rollouts,
            "is_multiplication": cc["data"]["handler"] == "multiplication_programmatic",
            "eval_axes": cc.get("eval_axes", {}),
            "status": status.get("validation"),
            "error": status.get("error"),
        }
    return {
        "run_id": config["_run_id"],
        "model": config["_model_name"],
        "fast": bool(config.get("fast")),
        "concepts": concepts,
        "universal_order": UNIVERSAL_ORDER,
    }


def build_summary_json(data: dict) -> dict:
    out = {"run_id": data["run_id"], "model": data["model"], "fast": data["fast"], "concepts": {}}
    for concept, c in data["concepts"].items():
        ss = c.get("scores_summary", {})
        regimes = ss.get("canonical_regimes") or ss.get("regimes", {})
        uni = {}
        for r in UNIVERSAL_ORDER:
            st = regimes.get(r)
            if st:
                uni[r] = {
                    "retention_rate": st.get("retention_rate"),
                    "tpr@5": st.get("tpr", {}).get("tpr@5"),
                    "mean_score": st.get("mean_score"),
                    "mean_delta_vs_baseline": st.get("mean_delta_vs_baseline"),
                }
        out["concepts"][concept] = {
            "validation": c["validation"].get("passed"),
            "val_auroc": c["validation"].get("val_auroc"),
            "test_auroc": c["validation"].get("test_auroc"),
            "category": c["category"],
            "universal_regimes": uni,
        }
    return out


def run(config):
    data = collect(config)
    rd = common.run_dir(config)
    rd.mkdir(parents=True, exist_ok=True)
    summary = build_summary_json(data)
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    html_doc = build_html(data)
    (rd / "dashboard.html").write_text(html_doc)
    print(f"  dashboard → {rd / 'dashboard.html'}  ({(rd / 'dashboard.html').stat().st_size:,} bytes)")
    print(f"  summary   → {rd / 'summary.json'}")


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def build_html(data: dict) -> str:
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace("__DATA__", embedded)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Probe-evasion eval dashboard</title>
<style>
:root{--bg:#fafaf8;--fg:#1a1a1a;--muted:#6b6b6b;--border:#e0e0e0;--hi:#fff7d6;--pos:#c91515;--neg:#2a6b2a;--fail:#b00020;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:0 0 4em;}
header{background:#fff;border-bottom:1px solid var(--border);padding:1em 2em;position:sticky;top:0;z-index:10;}
h1{margin:0 0 .2em;font-size:1.4em;}h2{font-size:1.1em;border-bottom:2px solid var(--border);padding-bottom:.3em;margin:1.4em 0 .6em;}
h3{font-size:.95em;margin:.8em 0 .4em;}
.subtitle{color:var(--muted);font-size:.9em;}
main{padding:1em 2em;max-width:1500px;margin:0 auto;}
table{width:100%;border-collapse:collapse;background:#fff;font-size:.84em;margin-bottom:1em;}
th,td{padding:6px 9px;text-align:left;border-bottom:1px solid var(--border);vertical-align:top;}
th{background:#f0f0ed;font-weight:600;}
td.num,th.num{text-align:right;font-family:SF Mono,Menlo,Consolas,monospace;}
.fail-banner{background:#ffe7ea;color:var(--fail);font-weight:600;padding:6px 9px;border-radius:4px;}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.75em;font-weight:600;}
.badge.pass{background:#e2f0e2;color:var(--neg);}.badge.fail{background:#ffe7ea;color:var(--fail);}
.concept-card{background:#fff;border:1px solid var(--border);border-radius:6px;margin:1em 0;padding:0 1em 1em;}
.concept-head{cursor:pointer;padding:.8em 0;font-size:1.05em;font-weight:600;display:flex;justify-content:space-between;align-items:center;}
.concept-body{display:none;}.concept-body.open{display:block;}
.hist{display:flex;align-items:end;gap:2px;height:70px;padding:4px;background:#fff;border:1px solid var(--border);}
.hist-bar{flex:1;min-width:2px;background:#888;}.hist-bar.beh{background:var(--pos);}
.small{font-size:.78em;color:var(--muted);}
.heatcell{padding:5px 8px;text-align:center;font-family:monospace;font-size:.8em;border:1px solid var(--border);}
.filters{margin:.6em 0;}.filters select,.filters input{padding:3px 7px;border:1px solid var(--border);border-radius:3px;font-size:.85em;}
tr.rollout-main:hover{background:var(--hi);cursor:pointer;}tr.rollout-exp{display:none;background:#fdfdfa;}tr.rollout-exp.open{display:table-row;}
.text-block{background:#fff;border:1px solid var(--border);border-radius:3px;padding:8px;font-size:.82em;line-height:1.45;max-height:340px;overflow-y:auto;white-space:pre-wrap;}
.text-block.think{font-style:italic;color:#444;}
.score.hi{color:var(--pos);font-weight:600;}.score.lo{color:var(--neg);}
code{background:#f0f0ed;padding:1px 4px;border-radius:3px;}
</style></head><body>
<header><h1>Probe-evasion eval dashboard</h1>
<div class="subtitle" id="subtitle"></div></header>
<main>
  <section><h2>Cross-concept summary <span class="small">(universal regimes only · TPR@FPR=5%)</span></h2>
    <div id="xsummary"></div>
    <div class="small">Retention = fraction of rollouts where the behavior was expressed (judge). TPR@5% = fraction of behaviorally-relevant rollouts the probe catches at the threshold admitting 5% of training negatives. Δ = mean paired probe-score change vs baseline.</div>
  </section>
  <section><h2>Per-concept drill-down</h2><div id="concepts"></div></section>
</main>
<script>
const DATA = __DATA__;
const esc=s=>(s??"").toString().replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt=(v,d=2)=>typeof v==="number"?v.toFixed(d):"—";
const pct=v=>typeof v==="number"?(v*100).toFixed(0)+"%":"—";
document.getElementById("subtitle").textContent =
  `run: ${DATA.run_id} · model: ${DATA.model} · mode: ${DATA.fast?"fast":"full"}`;

// ---- cross-concept summary ----
function xsummary(){
  const regs=DATA.universal_order;
  let h='<table><thead><tr><th>concept</th><th>cat</th><th>probe</th><th class="num">val AUROC</th><th class="num">OOD AUROC</th>';
  regs.forEach(r=>h+=`<th class="num">${r}<br><span class="small">ret / TPR@5 / Δ</span></th>`);
  h+='</tr></thead><tbody>';
  for(const [name,c] of Object.entries(DATA.concepts)){
    const v=c.validation||{};
    if(!v.passed){
      h+=`<tr><td><b>${esc(name)}</b></td><td>${esc(c.category)}</td>`+
         `<td colspan="${regs.length+3}"><span class="fail-banner">PROBE FAILED VALIDATION (val AUROC = ${v.val_auroc ?? "—"}, OOD AUROC = ${v.test_auroc ?? "—"}; ${esc(v.reason||"")})</span></td></tr>`;
      continue;
    }
    h+=`<tr><td><b>${esc(name)}</b></td><td>${esc(c.category)}</td>`+
       `<td><span class="badge pass">pass</span></td><td class="num">${fmt(v.val_auroc,3)}</td><td class="num">${fmt(v.test_auroc,3)}</td>`;
    const R=(c.scores_summary||{}).canonical_regimes||(c.scores_summary||{}).regimes||{};
    regs.forEach(r=>{
      const st=R[r];
      if(!st){h+='<td class="num">—</td>';return;}
      const tpr=st.tpr?st.tpr["tpr@5"]:null;
      h+=`<td class="num">${pct(st.retention_rate)} / ${pct(tpr)} / ${st.mean_delta_vs_baseline!=null?(st.mean_delta_vs_baseline>0?"+":"")+fmt(st.mean_delta_vs_baseline):"—"}</td>`;
    });
    h+='</tr>';
  }
  h+='</tbody></table>';
  document.getElementById("xsummary").innerHTML=h;
}

// ---- per-concept ----
function buildHist(values,nb=20,mn=null,mx=null){
  if(!values.length)return null;
  mn=mn??Math.min(...values);mx=mx??Math.max(...values);
  const step=(mx-mn)/nb||1;const bins=new Array(nb).fill(0);
  values.forEach(v=>{const i=Math.min(nb-1,Math.max(0,Math.floor((v-mn)/step)));bins[i]++;});
  return {bins,mn,mx};
}

function regimeStatsTable(c){
  const R=(c.scores_summary||{}).regimes||{};
  const names=Object.keys(R).sort((a,b)=>{
    const o={baseline:0,awareness_only:1,technical_disclosure:2,technical_extended:3};
    return (o[a]??99)-(o[b]??99)||a.localeCompare(b);});
  let h='<table><thead><tr><th>regime</th><th class="num">total</th><th class="num">behavioral</th><th class="num">retention</th><th class="num">mean score</th><th class="num">TPR@5</th><th class="num">TPR@10</th><th class="num">TPR@20</th><th class="num">Δ baseline (n)</th></tr></thead><tbody>';
  names.forEach(n=>{const st=R[n];const t=st.tpr||{};
    h+=`<tr><td>${esc(n)}${st.universal?"":' <span class="small">(extra)</span>'}</td><td class="num">${st.n_total}</td><td class="num">${st.n_behavioral}</td><td class="num">${pct(st.retention_rate)}</td><td class="num">${fmt(st.mean_score)}</td>`+
       `<td class="num">${pct(t["tpr@5"])}</td><td class="num">${pct(t["tpr@10"])}</td><td class="num">${pct(t["tpr@20"])}</td>`+
       `<td class="num">${st.mean_delta_vs_baseline!=null?(st.mean_delta_vs_baseline>0?"+":"")+fmt(st.mean_delta_vs_baseline):"—"} (${st.n_paired})</td></tr>`;});
  h+='</tbody></table>';return h;
}

function histPanel(c){
  const all=c.rollouts.filter(r=>typeof r.score==="number").map(r=>r.score);
  if(!all.length)return"";
  const gmin=Math.min(...all),gmax=Math.max(...all);
  const regs=[...new Set(c.rollouts.map(r=>r.regime))];
  let h='<h3>Score distributions (red = behavior expressed)</h3>';
  regs.forEach(reg=>{
    ["beh","non"].forEach(kind=>{
      const xs=c.rollouts.filter(r=>r.regime===reg&&typeof r.score==="number"&&(kind==="beh"?r.behaviorally_relevant:!r.behaviorally_relevant)).map(r=>r.score);
      if(!xs.length)return;
      const hb=buildHist(xs,20,gmin,gmax);
      h+=`<div style="margin-bottom:.4em"><div class="small">${esc(reg)} · ${kind==="beh"?"behavioral":"non-behavioral"} (n=${xs.length})</div><div class="hist">`+
         hb.bins.map(b=>`<div class="hist-bar ${kind==="beh"?"beh":""}" style="height:${b?Math.max(2,b*8):0}px"></div>`).join("")+`</div></div>`;
    });
  });
  return h;
}

function multHeatmap(c){
  if(!c.is_multiplication)return"";
  const axes=c.eval_axes||{};const digits=axes.digits||[];const phr=axes.phrasing||[];
  const regs=[...new Set(c.rollouts.map(r=>r.regime))].sort();
  // mean score on behaviorally-relevant (correct) rollouts per (regime,digits,phrasing)
  let h='<h3>Difficulty sweep — mean probe score on correct rollouts (digits × phrasing)</h3>';
  regs.forEach(reg=>{
    h+=`<div class="small" style="margin-top:.5em"><b>${esc(reg)}</b></div><table style="width:auto"><thead><tr><th></th>`+
       phr.map(p=>`<th class="num">${esc(p)}</th>`).join("")+'</tr></thead><tbody>';
    digits.forEach(d=>{
      h+=`<tr><td><b>d=${d}</b></td>`;
      phr.forEach(p=>{
        const xs=c.rollouts.filter(r=>r.regime===reg&&r.digits===d&&r.phrasing===p&&r.behaviorally_relevant&&typeof r.score==="number").map(r=>r.score);
        const m=xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;
        const bg=m==null?"#f5f5f5":`hsl(${Math.max(0,120-Math.min(120,(m+5)*12))},70%,88%)`;
        h+=`<td class="heatcell" style="background:${bg}">${m==null?"—":m.toFixed(1)}<br><span class="small">n=${xs.length}</span></td>`;
      });
      h+='</tr>';
    });
    h+='</tbody></table>';
  });
  return h;
}

function rolloutExplorer(c,cid){
  const regs=[...new Set(c.rollouts.map(r=>r.regime))].sort();
  let h=`<h3>Rollout explorer</h3><div class="filters">regime: <select id="rf_${cid}"><option value="">all</option>`+
        regs.map(r=>`<option>${esc(r)}</option>`).join("")+`</select> behavioral only: <input type="checkbox" id="rb_${cid}"></div>`+
        `<table><thead><tr><th>unit</th><th>regime</th><th>judge</th><th class="num">score</th></tr></thead><tbody id="rt_${cid}"></tbody></table>`;
  return h;
}

function renderRolloutRows(c,cid){
  const reg=document.getElementById("rf_"+cid).value;
  const behOnly=document.getElementById("rb_"+cid).checked;
  let rows=c.rollouts.filter(r=>(!reg||r.regime===reg)&&(!behOnly||r.behaviorally_relevant));
  rows.sort((a,b)=>(b.score??-1e9)-(a.score??-1e9));
  rows=rows.slice(0,300);
  const tb=document.getElementById("rt_"+cid);
  tb.innerHTML=rows.map((r,i)=>{
    const sc=typeof r.score==="number"?`<span class="score ${r.score>0?"hi":"lo"}">${r.score.toFixed(2)}</span>`:"—";
    return `<tr class="rollout-main" data-c="${cid}" data-i="${i}"><td>${esc(r.unit_id)}</td><td>${esc(r.regime)}</td><td>${esc(r.judge_label||"")}${r.behaviorally_relevant?" ★":""}</td><td class="num">${sc}</td></tr>`+
      `<tr class="rollout-exp"><td colspan="4"><div class="small">${esc(r.user_text||"")}</div>`+
      `<div class="small" style="margin-top:.3em">judge: ${esc(r.judge_reason||"")}</div>`+
      `<div style="margin-top:.3em"><b class="small">thinking</b></div><div class="text-block think">${esc(r.thinking||"—")}</div>`+
      `<div style="margin-top:.3em"><b class="small">response</b></div><div class="text-block">${esc(r.response||"—")}</div></td></tr>`;
  }).join("");
  tb.querySelectorAll("tr.rollout-main").forEach(tr=>tr.addEventListener("click",()=>tr.nextElementSibling.classList.toggle("open")));
}

function renderConcepts(){
  const el=document.getElementById("concepts");let i=0;
  for(const [name,c] of Object.entries(DATA.concepts)){
    const cid="c"+(i++);const v=c.validation||{};
    const badge=v.passed?'<span class="badge pass">probe passed</span>':'<span class="badge fail">probe failed</span>';
    let body='';
    body+=`<div class="small">category: ${esc(c.category)} · canonical position: <code>${esc(c.canonical_position)}</code> · val AUROC: ${fmt(v.val_auroc,3)} (threshold ${v.threshold ?? "—"}) · OOD/test AUROC: ${fmt(v.test_auroc,3)} (floor ${v.ood_threshold ?? "—"}) · ${esc(v.reason||"")}</div>`;
    if(c.error)body+=`<div class="fail-banner" style="margin-top:.5em">orchestrator error: ${esc(c.error)}</div>`;
    if(v.passed && c.rollouts.length){
      body+=regimeStatsTable(c)+multHeatmap(c)+histPanel(c)+rolloutExplorer(c,cid);
    } else if(v.passed){
      body+='<div class="small">No evasion rollouts found.</div>';
    } else {
      body+='<div class="fail-banner" style="margin-top:.5em">Evasion skipped — probe did not validate.</div>';
    }
    el.innerHTML+=`<div class="concept-card"><div class="concept-head" data-cid="${cid}"><span>${esc(name)} ${badge}</span><span class="small">▼</span></div><div class="concept-body" id="body_${cid}">${body}</div></div>`;
  }
  // wire up collapsibles + explorers
  i=0;
  for(const [name,c] of Object.entries(DATA.concepts)){
    const cid="c"+(i++);
    document.querySelector(`.concept-head[data-cid="${cid}"]`).addEventListener("click",()=>{
      document.getElementById("body_"+cid).classList.toggle("open");
    });
    if((c.validation||{}).passed && c.rollouts.length){
      const sel=document.getElementById("rf_"+cid),cb=document.getElementById("rb_"+cid);
      if(sel){sel.addEventListener("change",()=>renderRolloutRows(c,cid));cb.addEventListener("change",()=>renderRolloutRows(c,cid));renderRolloutRows(c,cid);}
    }
  }
}
xsummary();renderConcepts();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    config = common.load_for_cli(args.config, args.fast, args.mock, args.concepts)
    run(config)


if __name__ == "__main__":
    main()
