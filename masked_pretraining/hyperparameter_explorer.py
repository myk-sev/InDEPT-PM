"""Build an interactive explorer for the 72-run reconstruction sweep."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .masking import ALL_STAGES, STAGES
from .post_bridge_test import expected_runs


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_ROOT = ROOT / "inference" / "metrics"
DEFAULT_OUTPUT = ROOT / "inference" / "reports" / "all_excl_fine_t_hp_explorer.html"
LOSS_FIELDS = ("train_loss", "validation_loss")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def read_run_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["global_epoch"] = int(row["global_epoch"])
        row["stage_epoch"] = int(row["stage_epoch"])
        for field in LOSS_FIELDS:
            row[field] = float(row[field])
    return rows


def summarize_stage(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ordered = sorted(rows, key=lambda row: (row["stage_epoch"], row["global_epoch"]))
    final = ordered[-1]
    best_train = min(ordered, key=lambda row: row["train_loss"])
    best_test = min(ordered, key=lambda row: row["validation_loss"])
    return {
        "epochs": len(ordered),
        "final_train": final["train_loss"],
        "best_train": best_train["train_loss"],
        "best_train_epoch": best_train["stage_epoch"],
        "final_test": final["validation_loss"],
        "best_test": best_test["validation_loss"],
        "best_test_epoch": best_test["stage_epoch"],
    }


def run_record(run: object, metrics_root: Path) -> dict[str, object]:
    path = metrics_root / f"{run.artifact_name}.csv"
    record = {
        "number": run.number,
        "artifact": run.artifact_name,
        "model": run.model_name,
        "sweep": run.sweep,
        "value": run.sweep_value,
        "learning_rate": run.learning_rate,
        "model_dim": run.model_dim,
        "layers": run.layers,
        "heads": run.heads,
        "metrics_path": str(path.resolve()),
        "stages": {},
        "error": "",
    }
    if not path.is_file():
        record["status"] = "missing"
        return record
    try:
        rows = read_run_metrics(path)
        record["stages"] = {
            stage: summarize_stage([row for row in rows if row["stage"] == stage])
            for stage in ALL_STAGES
            if any(row["stage"] == stage for row in rows)
        }
    except (KeyError, TypeError, ValueError, OSError) as problem:
        record["status"] = "invalid"
        record["error"] = str(problem)
        return record
    completed = set(record["stages"])
    if set(ALL_STAGES).issubset(completed):
        record["status"] = "bridge complete"
    elif set(STAGES).issubset(completed):
        record["status"] = "base complete"
    else:
        record["status"] = "partial"
    return record


def build_payload(metrics_root: Path) -> dict[str, object]:
    runs = [run_record(run, metrics_root) for run in expected_runs()]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_root": str(metrics_root.resolve()),
        "stages": list(ALL_STAGES),
        "base_stages": list(STAGES),
        "runs": runs,
    }


def write_explorer(output: Path, payload: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    output.write_text(HTML.replace("__EXPLORER_DATA__", data), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = build_payload(args.metrics_root)
    write_explorer(args.output, payload)
    available = sum(run["status"] != "missing" for run in payload["runs"])
    print(f"runs_with_metrics={available}/{len(payload['runs'])}")
    print(f"explorer={args.output.resolve()}")


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>72-run reconstruction explorer</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--surface:#fff;--text:#172033;--muted:#667085;--line:#d9dee8;--accent:#3157d5;--accent-soft:#e8edff;--train:#16856b;--test:#b64b31;--warn:#9b6800;--shadow:0 12px 34px #17203314;font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#11151d;--surface:#191f2a;--text:#eef2f8;--muted:#a9b2c2;--line:#323b4c;--accent:#9aafff;--accent-soft:#273254;--train:#63cfb0;--test:#ff9a82;--warn:#f0c86d;--shadow:none}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text)}button,select,input{font:inherit;color:inherit}button,select,input[type=search]{min-height:40px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:.5rem .7rem}button{cursor:pointer}button:hover{border-color:var(--accent)}main{width:min(1500px,100%);margin:auto;padding:28px clamp(16px,3vw,44px) 64px}h1{font-size:clamp(1.55rem,3vw,2.25rem);line-height:1.15;margin:0}h2{font-size:1.1rem;margin:0 0 14px}p{margin:.35rem 0;color:var(--muted)}.header{display:flex;gap:24px;justify-content:space-between;align-items:end;margin-bottom:24px}.timestamp{text-align:right;font-size:.86rem}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}.stat,.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.stat{padding:14px 16px}.stat strong{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:.86rem}.controls{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:12px;padding:16px;margin-bottom:18px}.controls label{display:grid;gap:5px;color:var(--muted);font-size:.82rem}.controls label:last-child{grid-column:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}.panel{padding:18px;min-width:0}.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:8px}.panel-head p{font-size:.85rem}.chart{width:100%;min-height:320px;display:block}.chart text{fill:var(--muted);font-size:12px}.chart .axis{stroke:var(--line);stroke-width:1}.chart .bar{fill:var(--accent)}.chart .train{stroke:var(--train);fill:var(--train)}.chart .test{stroke:var(--test);fill:var(--test)}.chart .guide{stroke:var(--line);stroke-dasharray:5 4}.chart .point{cursor:pointer;stroke:var(--surface);stroke-width:2}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:.82rem}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.swatch.train{background:var(--train)}.swatch.test{background:var(--test)}.empty{display:grid;place-items:center;min-height:280px;text-align:center;color:var(--muted)}.table-wrap{overflow:auto;max-height:620px}.ranking{width:100%;border-collapse:collapse;font-size:.86rem}.ranking th,.ranking td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.ranking th{position:sticky;top:0;background:var(--surface);z-index:1}.ranking th button{border:0;padding:0;min-height:0;background:none;font-weight:650}.ranking td.num{text-align:right;font-variant-numeric:tabular-nums}.ranking tr[data-selected=true]{background:var(--accent-soft)}.ranking tbody tr{cursor:pointer}.ranking tbody tr:hover{background:var(--accent-soft)}.status{color:var(--muted)}.status.invalid{color:var(--test)}.status.partial{color:var(--warn)}.detail{display:flex;justify-content:space-between;gap:18px;align-items:start}.detail dl{display:grid;grid-template-columns:auto auto;gap:7px 18px;margin:0}.detail dt{color:var(--muted)}.detail dd{margin:0;font-variant-numeric:tabular-nums}.path{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.78rem;overflow-wrap:anywhere}.tooltip{position:fixed;pointer-events:none;z-index:4;background:var(--text);color:var(--bg);padding:7px 9px;border-radius:7px;font-size:.78rem;max-width:280px;box-shadow:var(--shadow)}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.controls label{min-width:0}.controls select,.controls input{width:100%;min-width:0}
@media(max-width:950px){.controls{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){main{padding-inline:12px}.header,.detail{display:block}.timestamp{text-align:left;margin-top:10px}.controls{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}.panel{padding:14px}.chart{min-height:280px}}
</style>
</head>
<body>
<main>
  <header class="header">
    <div><h1>72-run reconstruction explorer</h1><p>Rank final and best training or test (validation) loss for every curriculum stage.</p></div>
    <p class="timestamp" id="timestamp"></p>
  </header>
  <section class="stats" aria-label="Sweep status">
    <div class="stat"><strong id="available">0</strong><span>Runs with metrics</span></div>
    <div class="stat"><strong id="complete">0</strong><span>Base-complete runs</span></div>
    <div class="stat"><strong id="bridge">0</strong><span>Bridge-complete runs</span></div>
    <div class="stat"><strong id="leader">—</strong><span>Best visible score</span></div>
  </section>
  <section class="panel controls" aria-label="Explorer controls">
    <label>Stage<select id="stage"></select></label>
    <label>Rank score<select id="metric"><option value="best_test">Best test (validation)</option><option value="final_test">Final test (validation)</option><option value="best_train">Best training</option><option value="final_train">Final training</option></select></label>
    <label>Model<select id="model"><option value="">All models</option></select></label>
    <label>Sweep<select id="sweep"><option value="">All hyperparameters</option></select></label>
    <label>Find run<input id="search" type="search" placeholder="Model, sweep, or value"></label>
  </section>
  <section class="grid">
    <article class="panel"><div class="panel-head"><div><h2>Top ranked runs</h2><p id="ranking-caption"></p></div></div><svg id="ranking-chart" class="chart" role="img" aria-label="Top ranked model runs"></svg></article>
    <article class="panel"><div class="panel-head"><div><h2>Training–test relationship</h2><p>Lower-left is better; distance above the diagonal is the generalization gap.</p></div></div><svg id="scatter" class="chart" role="img" aria-label="Training versus test loss"></svg></article>
  </section>
  <article class="panel" style="margin-bottom:18px">
    <div class="panel-head"><div><h2>Sortable results</h2><p id="result-count" aria-live="polite"></p></div></div>
    <div class="table-wrap"><table class="ranking"><thead><tr id="headers"></tr></thead><tbody id="rows"></tbody></table></div>
  </article>
  <section class="grid">
    <article class="panel"><div class="panel-head"><div><h2>Selected run across stages</h2><p>Final and best loss show convergence and stage-to-stage stability.</p></div></div><svg id="profile" class="chart" role="img" aria-label="Selected run loss by stage"></svg><div class="legend"><span><i class="swatch train"></i>Training</span><span><i class="swatch test"></i>Test (validation)</span><span>Solid: best · dashed: final</span></div></article>
    <article class="panel"><div class="panel-head"><div><h2>Run detail</h2><p>Configuration, completion, and the selected stage.</p></div></div><div id="detail" class="empty">Select a run with recorded metrics.</div></article>
  </section>
</main>
<div id="tooltip" class="tooltip" hidden></div>
<script>
const DATA=__EXPLORER_DATA__;
const stageLabels={points:'Points',short_blocks:'Short blocks',mixed_blocks:'Mixed blocks',cross_channel:'Cross-channel',suffix_3:'Suffix 3 h',suffix_6:'Suffix 6 h',suffix_12:'Suffix 12 h',tempo_bridge_50:'Bridge 50%',tempo_bridge_70:'Bridge 70%',tempo_bridge_86:'Bridge 86%'};
const metricLabels={best_test:'Best test',final_test:'Final test',best_train:'Best training',final_train:'Final training'};
const state={stage:DATA.stages[0],metric:'best_test',model:'',sweep:'',search:'',sort:'best_test',ascending:true,selected:null};
const $=id=>document.getElementById(id),fmt=value=>Number.isFinite(value)?Number(value).toPrecision(5):'—';
const escapeHtml=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function setup(){
 $('timestamp').textContent=`Generated ${new Date(DATA.generated_at).toLocaleString()}`;
 $('available').textContent=DATA.runs.filter(run=>run.status!=='missing').length;
 $('complete').textContent=DATA.runs.filter(run=>['base complete','bridge complete'].includes(run.status)).length;
 $('bridge').textContent=DATA.runs.filter(run=>run.status==='bridge complete').length;
 DATA.stages.forEach(stage=>$('stage').add(new Option(`${stageLabels[stage]}${DATA.base_stages.includes(stage)?'':' · bridge'}`,stage)));
 [...new Set(DATA.runs.map(run=>run.model))].forEach(value=>$('model').add(new Option(value,value)));
 [...new Set(DATA.runs.map(run=>run.sweep))].forEach(value=>$('sweep').add(new Option(value,value)));
 for(const id of ['stage','metric','model','sweep'])$(id).onchange=event=>{state[id]=event.target.value;if(id==='metric'){state.sort=state.metric;state.ascending=true}render()};
 $('search').oninput=event=>{state.search=event.target.value.trim().toLowerCase();render()};
 renderHeaders();render();
}
function metric(run,key=state.metric){return run.stages[state.stage]?.[key]}
function visible(){const query=state.search;return DATA.runs.filter(run=>(!state.model||run.model===state.model)&&(!state.sweep||run.sweep===state.sweep)&&(!query||`${run.artifact} ${run.model} ${run.sweep} ${run.value}`.toLowerCase().includes(query)))}
function compare(a,b){const av=columnValue(a,state.sort),bv=columnValue(b,state.sort);if(typeof av==='string'&&typeof bv==='string')return (state.ascending?1:-1)*av.localeCompare(bv)||a.number-b.number;const aValid=Number.isFinite(av),bValid=Number.isFinite(bv);if(aValid!==bValid)return aValid?-1:1;if(!aValid)return a.number-b.number;return (state.ascending?1:-1)*(av-bv)||a.number-b.number}
function columnValue(run,key){if(['final_train','best_train','final_test','best_test'].includes(key))return metric(run,key);if(key==='gap'){const stage=run.stages[state.stage],prefix=state.metric.startsWith('best')?'best':'final';return stage?stage[`${prefix}_test`]-stage[`${prefix}_train`]:NaN}return run[key]??''}
const columns=[['number','#'],['model','Model'],['sweep','Sweep'],['value','Value'],['status','Status'],['final_train','Final train'],['best_train','Best train'],['final_test','Final test'],['best_test','Best test'],['gap','Test − train']];
function renderHeaders(){$('headers').innerHTML=columns.map(([key,label])=>`<th scope="col"><button data-sort="${key}">${label}<span aria-hidden="true">${state.sort===key?(state.ascending?' ↑':' ↓'):''}</span></button></th>`).join('');$('headers').querySelectorAll('button').forEach(button=>button.onclick=()=>{const key=button.dataset.sort;if(state.sort===key)state.ascending=!state.ascending;else{state.sort=key;state.ascending=true}renderHeaders();render()})}
function render(){const runs=visible().sort(compare),scored=runs.filter(run=>Number.isFinite(metric(run)));if(!state.selected||!runs.some(run=>run.number===state.selected))state.selected=scored[0]?.number??null;$('leader').textContent=scored.length?fmt(Math.min(...scored.map(run=>metric(run)))):'—';$('ranking-caption').textContent=`${stageLabels[state.stage]} · ${metricLabels[state.metric]} loss`;$('result-count').textContent=`${runs.length} expected runs; ${scored.length} have this stage.`;renderTable(runs);drawRanking(scored);drawScatter(scored);drawProfile();renderDetail()}
function renderTable(runs){$('rows').innerHTML=runs.map(run=>{const stage=run.stages[state.stage],trainKey=state.metric.startsWith('best')?'best_train':'final_train',testKey=state.metric.startsWith('best')?'best_test':'final_test',gap=stage?stage[testKey]-stage[trainKey]:NaN;return `<tr data-run="${run.number}" data-selected="${run.number===state.selected}"><td class="num">${run.number}</td><td>${escapeHtml(run.model)}</td><td>${escapeHtml(run.sweep)}</td><td>${escapeHtml(run.value)}</td><td class="status ${run.status}">${escapeHtml(run.status)}</td><td class="num">${fmt(stage?.final_train)}</td><td class="num">${fmt(stage?.best_train)}</td><td class="num">${fmt(stage?.final_test)}</td><td class="num">${fmt(stage?.best_test)}</td><td class="num">${fmt(gap)}</td></tr>`}).join('');$('rows').querySelectorAll('tr').forEach(row=>row.onclick=()=>{state.selected=Number(row.dataset.run);render()})}
function svgSize(svg){const width=Math.max(320,svg.clientWidth||640),height=320;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);return {width,height,left:64,right:18,top:14,bottom:54}}
function node(name,attrs={},text=''){const item=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([key,value])=>item.setAttribute(key,value));item.textContent=text;return item}
function domain(values){const min=Math.min(...values),max=Math.max(...values),pad=(max-min||Math.abs(max)||1)*.08;return [Math.max(0,min-pad),max+pad]}
function scale(value,a,b,c,d){return c+(value-a)/(b-a)*(d-c)}
function axes(svg,box,xDomain,yDomain,xLabel,yLabel){const {width,height,left,right,top,bottom}=box;for(let i=0;i<5;i++){const x=left+i*(width-left-right)/4,y=top+i*(height-top-bottom)/4;svg.append(node('line',{class:'axis',x1:left,x2:width-right,y1:y,y2:y}));svg.append(node('text',{x:left-8,y:y+4,'text-anchor':'end'},fmt(yDomain[1]-i*(yDomain[1]-yDomain[0])/4)));svg.append(node('text',{x,y:height-bottom+20,'text-anchor':i===0?'start':i===4?'end':'middle'},fmt(xDomain[0]+i*(xDomain[1]-xDomain[0])/4)))}svg.append(node('text',{x:(left+width-right)/2,y:height-7,'text-anchor':'middle'},xLabel));svg.append(node('text',{transform:`translate(15 ${(top+height-bottom)/2}) rotate(-90)`,'text-anchor':'middle'},yLabel))}
function emptyChart(svg,message){svg.replaceChildren();const box=svgSize(svg);svg.append(node('text',{x:box.width/2,y:box.height/2,'text-anchor':'middle'},message))}
function showTip(event,text){const tip=$('tooltip');tip.textContent=text;tip.hidden=false;tip.style.left=`${Math.min(innerWidth-300,event.clientX+12)}px`;tip.style.top=`${event.clientY+12}px`}function hideTip(){$('tooltip').hidden=true}
function drawRanking(runs){const svg=$('ranking-chart'),ranked=[...runs].sort((a,b)=>metric(a)-metric(b)).slice(0,12);if(!ranked.length)return emptyChart(svg,'No recorded metrics for this selection.');svg.replaceChildren();const box=svgSize(svg),values=ranked.map(run=>metric(run)),max=Math.max(...values)*1.05||1,row=(box.height-box.top-box.bottom)/ranked.length;ranked.forEach((run,index)=>{const y=box.top+index*row+row*.16,h=row*.66,w=(box.width-box.left-box.right)*metric(run)/max;svg.append(node('rect',{class:'bar',x:box.left,y,width:w,height:h,rx:3}));svg.append(node('text',{x:box.left-8,y:y+h*.72,'text-anchor':'end'},`#${run.number}`));svg.append(node('text',{x:box.left+w+6,y:y+h*.72},fmt(metric(run))));const hit=node('rect',{x:box.left,y,width:Math.max(w,44),height:h,fill:'transparent',tabindex:0,'aria-label':`Run ${run.number}, ${metricLabels[state.metric]} ${fmt(metric(run))}`});hit.onclick=()=>{state.selected=run.number;render()};hit.onmousemove=e=>showTip(e,`${run.model} · ${run.sweep} ${run.value} · ${fmt(metric(run))}`);hit.onmouseleave=hideTip;svg.append(hit)})}
function drawScatter(runs){const svg=$('scatter'),best=state.metric.startsWith('best'),trainKey=best?'best_train':'final_train',testKey=best?'best_test':'final_test',pairs=runs.map(run=>({run,x:run.stages[state.stage]?.[trainKey],y:run.stages[state.stage]?.[testKey]})).filter(point=>Number.isFinite(point.x)&&Number.isFinite(point.y));if(!pairs.length)return emptyChart(svg,'No paired training and test scores.');svg.replaceChildren();const box=svgSize(svg),both=pairs.flatMap(point=>[point.x,point.y]),d=domain(both),x=value=>scale(value,d[0],d[1],box.left,box.width-box.right),y=value=>scale(value,d[0],d[1],box.height-box.bottom,box.top);axes(svg,box,d,d,`${best?'Best':'Final'} training loss`,`${best?'Best':'Final'} test loss`);svg.append(node('line',{class:'guide',x1:x(d[0]),y1:y(d[0]),x2:x(d[1]),y2:y(d[1])}));pairs.forEach(({run,x:train,y:test})=>{const point=node('circle',{class:'point',cx:x(train),cy:y(test),r:run.number===state.selected?7:5,fill:'var(--accent)',tabindex:0,'aria-label':`Run ${run.number}, training ${fmt(train)}, test ${fmt(test)}`});point.onclick=()=>{state.selected=run.number;render()};point.onmousemove=e=>showTip(e,`#${run.number} ${run.model} · train ${fmt(train)} · test ${fmt(test)} · gap ${fmt(test-train)}`);point.onmouseleave=hideTip;svg.append(point)})}
function drawProfile(){const svg=$('profile'),run=DATA.runs.find(item=>item.number===state.selected);if(!run||!Object.keys(run.stages).length)return emptyChart(svg,'Select a run with recorded metrics.');svg.replaceChildren();const box=svgSize(svg),stages=DATA.stages.filter(stage=>run.stages[stage]),values=stages.flatMap(stage=>['final_train','best_train','final_test','best_test'].map(key=>run.stages[stage][key])),d=domain(values),x=index=>box.left+index*(box.width-box.left-box.right)/Math.max(stages.length-1,1),y=value=>scale(value,d[0],d[1],box.height-box.bottom,box.top);for(let i=0;i<5;i++){const gy=box.top+i*(box.height-box.top-box.bottom)/4;svg.append(node('line',{class:'axis',x1:box.left,x2:box.width-box.right,y1:gy,y2:gy}));svg.append(node('text',{x:box.left-8,y:gy+4,'text-anchor':'end'},fmt(d[1]-i*(d[1]-d[0])/4)))}stages.forEach((stage,index)=>svg.append(node('text',{x:x(index),y:box.height-box.bottom+18,'text-anchor':'end',transform:`rotate(-28 ${x(index)} ${box.height-box.bottom+18})`},stageLabels[stage])));for(const [key,klass,dash] of [['best_train','train',''],['final_train','train','5 4'],['best_test','test',''],['final_test','test','5 4']]){const points=stages.map((stage,index)=>`${x(index)},${y(run.stages[stage][key])}`).join(' ');svg.append(node('polyline',{class:klass,points,fill:'none','stroke-width':key.startsWith('best')?2.5:1.5,'stroke-dasharray':dash}));stages.forEach((stage,index)=>svg.append(node('circle',{class:klass,cx:x(index),cy:y(run.stages[stage][key]),r:3})))} }
function renderDetail(){const run=DATA.runs.find(item=>item.number===state.selected),target=$('detail');if(!run||!Object.keys(run.stages).length){target.className='empty';target.textContent='Select a run with recorded metrics.';return}const stage=run.stages[state.stage];target.className='detail';target.innerHTML=`<dl><dt>Run</dt><dd>#${run.number}</dd><dt>Model</dt><dd>${escapeHtml(run.model)}</dd><dt>Sweep</dt><dd>${escapeHtml(run.sweep)} = ${escapeHtml(run.value)}</dd><dt>Learning rate</dt><dd>${escapeHtml(run.learning_rate)}</dd><dt>Model dimension</dt><dd>${run.model_dim}</dd><dt>Depth</dt><dd>${run.layers}</dd><dt>Heads</dt><dd>${run.heads}</dd><dt>Status</dt><dd>${escapeHtml(run.status)}</dd>${stage?`<dt>Stage epochs</dt><dd>${stage.epochs}</dd><dt>Best test epoch</dt><dd>${stage.best_test_epoch}</dd><dt>Best train epoch</dt><dd>${stage.best_train_epoch}</dd>`:''}</dl><div><p>Metrics source</p><p class="path">${escapeHtml(run.metrics_path)}</p>${run.error?`<p class="status invalid">${escapeHtml(run.error)}</p>`:''}</div>`}
let resizeTimer;new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,50)}).observe(document.querySelector('main'));setup();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
