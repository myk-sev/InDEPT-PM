"""Build an interactive explorer for the 72-run reconstruction sweep."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from math import isclose
from pathlib import Path

from .masking import ALL_STAGES, STAGES
from .post_bridge_test import expected_runs


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_ROOT = ROOT / "inference" / "metrics"
DEFAULT_FORGETTING_REPORT = (
    ROOT
    / "inference"
    / "reports"
    / "all_excl_fine_t_hp_post_bridge_reconstruction_loss.csv"
)
DEFAULT_FORECAST_REPORT_ROOT = ROOT / "inference" / "reports"
DEFAULT_OUTPUT = ROOT / "inference" / "reports" / "all_excl_fine_t_hp_explorer.html"
LOSS_FIELDS = ("train_loss", "validation_loss")
FORECAST_HORIZONS = (3, 6, 12, 24, 36)
FORECAST_SPLITS = ("validation", "temporal_test", "location_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument(
        "--forgetting-report", type=Path, default=DEFAULT_FORGETTING_REPORT
    )
    parser.add_argument(
        "--forecast-report-root", type=Path, default=DEFAULT_FORECAST_REPORT_ROOT
    )
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


def read_forgetting_report(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"artifact_name", "comparison", "split", *STAGES}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("forgetting report has an incomplete header")
    indexed = {
        (row["artifact_name"], row["split"], row["comparison"]): row for row in rows
    }
    results: dict[str, dict[str, dict[str, object]]] = {}
    for artifact, split, _ in indexed:
        keys = [
            (artifact, split, comparison)
            for comparison in (
                "pre_bridge_recorded",
                "post_bridge_inference",
                "difference_post_minus_pre",
            )
        ]
        if not all(key in indexed for key in keys):
            continue
        pre, post, reported = (indexed[key] for key in keys)
        stages = {}
        for stage in STAGES:
            before, after, difference = map(float, (pre[stage], post[stage], reported[stage]))
            if not isclose(after - before, difference, abs_tol=1e-8):
                raise ValueError(f"forgetting difference mismatch for {artifact} {split} {stage}")
            stages[stage] = {
                "pre": before,
                "post": after,
                "difference": difference,
                "percent": difference / before * 100 if before else None,
            }
        results.setdefault(artifact, {})[split] = stages
    return results


def read_forecast_report(path: Path) -> dict[str, dict[str, dict[str, object]]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "model_name",
        "split",
        "horizon_hours",
        "samples",
        "values",
        "model_rmse_1_to_h_ug_m3",
        "model_rmse_at_h_ug_m3",
        "model_mae_1_to_h_ug_m3",
        "model_bias_1_to_h_ug_m3",
        "persistence_rmse_1_to_h_ug_m3",
        "persistence_rmse_at_h_ug_m3",
        "rmse_skill_vs_persistence_pct",
        "selected_epoch",
        "normalized_validation_loss",
        "checkpoint_path",
        "checkpoint_sha256",
        "training_data_path",
        "training_data_sha256",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("forecast report has an incomplete header")
    results: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        split = row["split"]
        if split not in FORECAST_SPLITS:
            raise ValueError(f"forecast report has an unknown split: {split}")
        horizon = int(row["horizon_hours"])
        if horizon < 1:
            raise ValueError("forecast report horizons must be positive")
        key = str(horizon)
        if key in results.setdefault(split, {}):
            raise ValueError(f"duplicate forecast row for {split} {horizon}h")
        skill = row["rmse_skill_vs_persistence_pct"].strip()
        results[split][key] = {
            "model_name": row["model_name"],
            "samples": int(row["samples"]),
            "values": int(row["values"]),
            "rmse": float(row["model_rmse_1_to_h_ug_m3"]),
            "lead_rmse": float(row["model_rmse_at_h_ug_m3"]),
            "mae": float(row["model_mae_1_to_h_ug_m3"]),
            "bias": float(row["model_bias_1_to_h_ug_m3"]),
            "persistence_rmse": float(row["persistence_rmse_1_to_h_ug_m3"]),
            "skill": float(skill) if skill else None,
        }
    return results


def forecast_artifact_name(run: object) -> str:
    prefix = run.artifact_name.removesuffix(run.model_name)
    return f"{prefix}bridge-forecast-{run.model_name}-pretrained"


def run_record(
    run: object,
    metrics_root: Path,
    forgetting: dict[str, dict[str, dict[str, object]]],
    forecast_report_root: Path,
) -> dict[str, object]:
    path = metrics_root / f"{run.artifact_name}.csv"
    forecast_path = forecast_report_root / f"{forecast_artifact_name(run)}.csv"
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
        "forgetting": forgetting.get(run.artifact_name, {}),
        "forecast": {},
        "forecast_report_path": str(forecast_path.resolve()),
        "forecast_status": "missing",
        "forecast_error": "",
        "error": "",
    }
    if forecast_path.is_file():
        try:
            record["forecast"] = read_forecast_report(forecast_path)
            model_names = {
                item["model_name"]
                for split in record["forecast"].values()
                for item in split.values()
            }
            expected_model = f"bridge-forecast-{run.model_name}"
            if model_names != {expected_model}:
                raise ValueError(
                    f"forecast report model mismatch: expected {expected_model}"
                )
            complete = all(
                str(horizon) in record["forecast"].get(split, {})
                for split in FORECAST_SPLITS
                for horizon in FORECAST_HORIZONS
            )
            record["forecast_status"] = "complete" if complete else "partial"
        except (KeyError, TypeError, ValueError, OSError) as problem:
            record["forecast"] = {}
            record["forecast_status"] = "invalid"
            record["forecast_error"] = str(problem)
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


def build_payload(
    metrics_root: Path,
    forgetting_report: Path,
    forecast_report_root: Path = DEFAULT_FORECAST_REPORT_ROOT,
) -> dict[str, object]:
    forgetting = {}
    forgetting_error = ""
    if forgetting_report.is_file():
        try:
            forgetting = read_forgetting_report(forgetting_report)
        except (KeyError, TypeError, ValueError, OSError) as problem:
            forgetting_error = str(problem)
    runs = [
        run_record(run, metrics_root, forgetting, forecast_report_root)
        for run in expected_runs()
    ]
    forecast_horizons = sorted(
        {
            *FORECAST_HORIZONS,
            *(
                int(horizon)
                for run in runs
                for split in run["forecast"].values()
                for horizon in split
            ),
        }
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_root": str(metrics_root.resolve()),
        "forgetting_report": {
            "path": str(forgetting_report.resolve()),
            "exists": forgetting_report.is_file(),
            "error": forgetting_error,
            "runs": sum(bool(run["forgetting"]) for run in runs),
        },
        "forecast_report_root": {
            "path": str(forecast_report_root.resolve()),
            "exists": forecast_report_root.is_dir(),
            "runs": sum(bool(run["forecast"]) for run in runs),
            "invalid_runs": sum(
                run["forecast_status"] == "invalid" for run in runs
            ),
        },
        "forecast_horizons": forecast_horizons,
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
    payload = build_payload(
        args.metrics_root, args.forgetting_report, args.forecast_report_root
    )
    write_explorer(args.output, payload)
    available = sum(run["status"] != "missing" for run in payload["runs"])
    print(f"runs_with_metrics={available}/{len(payload['runs'])}")
    print(f"runs_with_forgetting={payload['forgetting_report']['runs']}/{len(payload['runs'])}")
    print(
        "runs_with_forecast_reports="
        f"{payload['forecast_report_root']['runs']}/{len(payload['runs'])}"
    )
    print(f"explorer={args.output.resolve()}")


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>72-run training pipeline explorer</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--surface:#fff;--text:#172033;--muted:#667085;--line:#d9dee8;--accent:#3157d5;--accent-soft:#e8edff;--train:#16856b;--test:#b64b31;--warn:#9b6800;--shadow:0 12px 34px #17203314;font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#11151d;--surface:#191f2a;--text:#eef2f8;--muted:#a9b2c2;--line:#323b4c;--accent:#9aafff;--accent-soft:#273254;--train:#63cfb0;--test:#ff9a82;--warn:#f0c86d;--shadow:none}}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0;background:var(--bg);color:var(--text)}button,select,input{font:inherit;color:inherit}button,select,input[type=search]{min-height:40px;border:1px solid var(--line);border-radius:8px;background:var(--surface);padding:.5rem .7rem}button{cursor:pointer}button:hover{border-color:var(--accent)}main{width:min(1500px,100%);margin:auto;padding:28px clamp(16px,3vw,44px) 64px}h1{font-size:clamp(1.55rem,3vw,2.25rem);line-height:1.15;margin:0}h2{font-size:1.1rem;margin:0 0 14px}p{margin:.35rem 0;color:var(--muted)}.header{display:flex;gap:24px;justify-content:space-between;align-items:end;margin-bottom:24px}.timestamp{text-align:right;font-size:.86rem}.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-bottom:18px}.stat,.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}.stat{padding:14px 16px}.stat strong{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums}.stat span{color:var(--muted);font-size:.86rem}.controls{display:grid;grid-template-columns:repeat(5,minmax(140px,1fr));gap:12px;padding:16px;margin-bottom:18px}.controls label{display:grid;gap:5px;color:var(--muted);font-size:.82rem}.controls label:last-child{grid-column:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}.panel{padding:18px;min-width:0}.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:start;margin-bottom:8px}.panel-head p{font-size:.85rem}.chart{width:100%;min-height:320px;display:block}.chart text{fill:var(--muted);font-size:12px}.chart .axis{stroke:var(--line);stroke-width:1}.chart .bar{fill:var(--accent)}.chart .train{stroke:var(--train);fill:var(--train)}.chart .test{stroke:var(--test);fill:var(--test)}.chart .guide{stroke:var(--line);stroke-dasharray:5 4}.chart .point{cursor:pointer;stroke:var(--surface);stroke-width:2}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:.82rem}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.swatch.train{background:var(--train)}.swatch.test{background:var(--test)}.empty{display:grid;place-items:center;min-height:280px;text-align:center;color:var(--muted)}.table-wrap{overflow:auto;max-height:620px}.ranking{width:100%;border-collapse:collapse;font-size:.86rem}.ranking th,.ranking td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.ranking th{position:sticky;top:0;background:var(--surface);z-index:1}.ranking th button{border:0;padding:0;min-height:0;background:none;font-weight:650}.ranking td.num{text-align:right;font-variant-numeric:tabular-nums}.ranking tr[data-selected=true]{background:var(--accent-soft)}.ranking tbody tr{cursor:pointer}.ranking tbody tr:hover{background:var(--accent-soft)}.status{color:var(--muted)}.status.invalid{color:var(--test)}.status.partial{color:var(--warn)}.detail{display:flex;justify-content:space-between;gap:18px;align-items:start}.detail dl{display:grid;grid-template-columns:auto auto;gap:7px 18px;margin:0}.detail dt{color:var(--muted)}.detail dd{margin:0;font-variant-numeric:tabular-nums}.path{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.78rem;overflow-wrap:anywhere}.tooltip{position:fixed;pointer-events:none;z-index:4;background:var(--text);color:var(--bg);padding:7px 9px;border-radius:7px;font-size:.78rem;max-width:280px;box-shadow:var(--shadow)}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
.controls label{min-width:0}.controls select,.controls input{width:100%;min-width:0}
.analysis-section{margin-top:18px}.analysis-section .empty{overflow-wrap:anywhere}.analysis-controls{display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:12px;margin:16px 0}.analysis-controls label{display:grid;gap:5px;min-width:0;color:var(--muted);font-size:.82rem}.analysis-controls select{width:100%;min-width:0}.analysis-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:16px}.analysis-stats div{padding:10px 0;border-bottom:1px solid var(--line)}.analysis-stats strong{display:block;font-size:1.2rem;font-variant-numeric:tabular-nums}.analysis-stats span{color:var(--muted);font-size:.82rem}.chart .pre,.chart .model{stroke:var(--accent);fill:var(--accent)}.chart .post,.chart .forgetting,.chart .persistence{stroke:var(--test);fill:var(--test)}.chart .improvement{stroke:var(--train);fill:var(--train)}.positive{color:var(--test)}.negative{color:var(--train)}
@media(max-width:950px){.controls{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){main{padding-inline:12px}.header,.detail{display:block}.timestamp{text-align:left;margin-top:10px}.controls,.analysis-controls{grid-template-columns:1fr}.stats,.analysis-stats{grid-template-columns:1fr 1fr}.panel{padding:14px}.chart{min-height:280px}}
</style>
</head>
<body>
<main>
  <header class="header">
    <div><h1>72-run training pipeline explorer</h1><p>Compare reconstruction, synthetic bridge, and supervised forecast stages.</p></div>
    <p class="timestamp" id="timestamp"></p>
  </header>
  <section class="stats" aria-label="Sweep status">
    <div class="stat"><strong id="available">0</strong><span>Runs with metrics</span></div>
    <div class="stat"><strong id="complete">0</strong><span>Base-complete runs</span></div>
    <div class="stat"><strong id="bridge">0</strong><span>Bridge-complete runs</span></div>
    <div class="stat"><strong id="forecast-available">0</strong><span>Forecast reports</span></div>
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
  <section class="panel analysis-section">
    <div class="panel-head"><div><h2>Forecast stages</h2><p>Final forecast-training reports at 3, 6, 12, 24, and 36 hours. Errors are in µg/m³; skill is relative to persistence.</p></div></div>
    <div id="forecast-empty" class="empty"></div>
    <div id="forecast-content" hidden>
      <div class="analysis-controls" aria-label="Forecast analysis controls">
        <label>Forecast horizon<select id="forecast-horizon"></select></label>
        <label>Data split<select id="forecast-split"><option value="validation">Validation</option><option value="temporal_test">Temporal test</option><option value="location_test">Location test</option></select></label>
        <label>Rank measure<select id="forecast-metric"><option value="rmse">Model RMSE, 1–H</option><option value="lead_rmse">Model RMSE at H</option><option value="mae">Model MAE, 1–H</option><option value="abs_bias">Absolute bias, 1–H</option><option value="skill">RMSE skill vs persistence</option></select></label>
      </div>
      <div class="analysis-stats" aria-live="polite">
        <div><strong id="forecast-count">0</strong><span>Reported runs</span></div>
        <div><strong id="forecast-median">—</strong><span>Median selected measure</span></div>
        <div><strong id="forecast-best">—</strong><span>Best selected measure</span></div>
        <div><strong id="forecast-skilled">0</strong><span>Beating persistence</span></div>
      </div>
      <section class="grid">
        <article><h3>Top forecast runs</h3><svg id="forecast-ranking" class="chart" role="img" aria-label="Forecast result ranking"></svg></article>
        <article><h3>Selected run across forecast stages</h3><svg id="forecast-profile" class="chart" role="img" aria-label="Model and persistence RMSE by forecast horizon"></svg><div class="legend"><span><i class="swatch" style="background:var(--accent)"></i>Model RMSE</span><span><i class="swatch test"></i>Persistence RMSE</span></div></article>
      </section>
      <div class="table-wrap"><table class="ranking"><thead><tr id="forecast-headers"></tr></thead><tbody id="forecast-rows"></tbody></table></div>
      <p class="path" id="forecast-path"></p>
    </div>
  </section>
  <section class="panel analysis-section">
    <div class="panel-head"><div><h2>Post-bridge forgetting</h2><p>Re-evaluation of each final bridge checkpoint on the seven original reconstruction masks. Positive change means loss increased.</p></div></div>
    <div id="forgetting-empty" class="empty"></div>
    <div id="forgetting-content" hidden>
      <div class="analysis-controls" aria-label="Forgetting analysis controls">
        <label>Reconstruction stage<select id="forget-stage"></select></label>
        <label>Data split<select id="forget-split"><option value="validation">Test (validation)</option><option value="training">Training</option></select></label>
        <label>Rank measure<select id="forget-metric"><option value="difference">Loss change</option><option value="percent">Percent change</option><option value="post">Post-bridge loss</option><option value="pre">Pre-bridge loss</option></select></label>
      </div>
      <div class="analysis-stats" aria-live="polite">
        <div><strong id="forget-count">0</strong><span>Analyzed runs</span></div>
        <div><strong id="forget-median">—</strong><span>Median loss change</span></div>
        <div><strong id="forget-worst">—</strong><span>Largest loss increase</span></div>
        <div><strong id="forget-improved">0</strong><span>Unchanged or improved</span></div>
      </div>
      <section class="grid">
        <article><h3>Largest changes</h3><svg id="forget-ranking" class="chart" role="img" aria-label="Post-bridge forgetting ranking"></svg></article>
        <article><h3>Selected run across reconstruction stages</h3><svg id="forget-profile" class="chart" role="img" aria-label="Pre-bridge and post-bridge reconstruction loss"></svg><div class="legend"><span><i class="swatch" style="background:var(--accent)"></i>Pre-bridge</span><span><i class="swatch test"></i>Post-bridge</span></div></article>
      </section>
      <div class="table-wrap"><table class="ranking"><thead><tr id="forget-headers"></tr></thead><tbody id="forget-rows"></tbody></table></div>
      <p class="path" id="forget-path"></p>
    </div>
  </section>
</main>
<div id="tooltip" class="tooltip" hidden></div>
<script>
const DATA=__EXPLORER_DATA__;
const stageLabels={points:'Points',short_blocks:'Short blocks',mixed_blocks:'Mixed blocks',cross_channel:'Cross-channel',suffix_3:'Suffix 3 h',suffix_6:'Suffix 6 h',suffix_12:'Suffix 12 h',tempo_bridge_50:'Bridge 50%',tempo_bridge_70:'Bridge 70%',tempo_bridge_86:'Bridge 86%'};
const metricLabels={best_test:'Best test',final_test:'Final test',best_train:'Best training',final_train:'Final training'};
const forecastMetricLabels={rmse:'Model RMSE, 1–H',lead_rmse:'Model RMSE at H',mae:'Model MAE, 1–H',abs_bias:'Absolute bias, 1–H',skill:'RMSE skill vs persistence'};
const state={stage:DATA.stages[0],metric:'best_test',model:'',sweep:'',search:'',sort:'best_test',ascending:true,selected:null,forecastHorizon:String(DATA.forecast_horizons[0]),forecastSplit:'validation',forecastMetric:'rmse',forecastSort:'rmse',forecastAscending:true,forgetStage:DATA.base_stages[0],forgetSplit:'validation',forgetMetric:'difference',forgetSort:'difference',forgetAscending:false};
const $=id=>document.getElementById(id),fmt=value=>Number.isFinite(value)?Number(value).toPrecision(5):'—';
const escapeHtml=value=>String(value).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
function setup(){
 $('timestamp').textContent=`Generated ${new Date(DATA.generated_at).toLocaleString()}`;
 $('available').textContent=DATA.runs.filter(run=>run.status!=='missing').length;
 $('complete').textContent=DATA.runs.filter(run=>['base complete','bridge complete'].includes(run.status)).length;
 $('bridge').textContent=DATA.runs.filter(run=>run.status==='bridge complete').length;
 $('forecast-available').textContent=DATA.forecast_report_root.runs;
 DATA.stages.forEach(stage=>$('stage').add(new Option(`${stageLabels[stage]}${DATA.base_stages.includes(stage)?'':' · bridge'}`,stage)));
 DATA.base_stages.forEach(stage=>$('forget-stage').add(new Option(stageLabels[stage],stage)));
 DATA.forecast_horizons.forEach(horizon=>$('forecast-horizon').add(new Option(`${horizon} hours`,horizon)));
 [...new Set(DATA.runs.map(run=>run.model))].forEach(value=>$('model').add(new Option(value,value)));
 [...new Set(DATA.runs.map(run=>run.sweep))].forEach(value=>$('sweep').add(new Option(value,value)));
 for(const id of ['stage','metric','model','sweep'])$(id).onchange=event=>{state[id]=event.target.value;if(id==='metric'){state.sort=state.metric;state.ascending=true}render()};
 for(const [id,key] of [['forecast-horizon','forecastHorizon'],['forecast-split','forecastSplit'],['forecast-metric','forecastMetric']])$(id).onchange=event=>{state[key]=event.target.value;if(key==='forecastMetric'){state.forecastSort=state.forecastMetric;state.forecastAscending=state.forecastMetric!=='skill'}renderForecast()};
 for(const [id,key] of [['forget-stage','forgetStage'],['forget-split','forgetSplit'],['forget-metric','forgetMetric']])$(id).onchange=event=>{state[key]=event.target.value;if(key==='forgetMetric'){state.forgetSort=state.forgetMetric;state.forgetAscending=!['difference','percent'].includes(state.forgetMetric)}renderForgetting()};
 $('search').oninput=event=>{state.search=event.target.value.trim().toLowerCase();render()};
 renderHeaders();renderForecastHeaders();renderForgetHeaders();setupForecast();setupForgetting();render();
}
function metric(run,key=state.metric){return run.stages[state.stage]?.[key]}
function visible(){const query=state.search;return DATA.runs.filter(run=>(!state.model||run.model===state.model)&&(!state.sweep||run.sweep===state.sweep)&&(!query||`${run.artifact} ${run.model} ${run.sweep} ${run.value}`.toLowerCase().includes(query)))}
function compare(a,b){const av=columnValue(a,state.sort),bv=columnValue(b,state.sort);if(typeof av==='string'&&typeof bv==='string')return (state.ascending?1:-1)*av.localeCompare(bv)||a.number-b.number;const aValid=Number.isFinite(av),bValid=Number.isFinite(bv);if(aValid!==bValid)return aValid?-1:1;if(!aValid)return a.number-b.number;return (state.ascending?1:-1)*(av-bv)||a.number-b.number}
function columnValue(run,key){if(['final_train','best_train','final_test','best_test'].includes(key))return metric(run,key);if(key==='gap'){const stage=run.stages[state.stage],prefix=state.metric.startsWith('best')?'best':'final';return stage?stage[`${prefix}_test`]-stage[`${prefix}_train`]:NaN}return run[key]??''}
const columns=[['number','#'],['model','Model'],['sweep','Sweep'],['value','Value'],['status','Status'],['final_train','Final train'],['best_train','Best train'],['final_test','Final test'],['best_test','Best test'],['gap','Test − train']];
function renderHeaders(){$('headers').innerHTML=columns.map(([key,label])=>`<th scope="col"><button data-sort="${key}">${label}<span aria-hidden="true">${state.sort===key?(state.ascending?' ↑':' ↓'):''}</span></button></th>`).join('');$('headers').querySelectorAll('button').forEach(button=>button.onclick=()=>{const key=button.dataset.sort;if(state.sort===key)state.ascending=!state.ascending;else{state.sort=key;state.ascending=true}renderHeaders();render()})}
function render(){const runs=visible().sort(compare),scored=runs.filter(run=>Number.isFinite(metric(run)));if(!state.selected||!runs.some(run=>run.number===state.selected))state.selected=scored[0]?.number??null;$('leader').textContent=scored.length?fmt(Math.min(...scored.map(run=>metric(run)))):'—';$('ranking-caption').textContent=`${stageLabels[state.stage]} · ${metricLabels[state.metric]} loss`;$('result-count').textContent=`${runs.length} expected runs; ${scored.length} have this stage.`;renderTable(runs);drawRanking(scored);drawScatter(scored);drawProfile();renderDetail();renderForecast();renderForgetting()}
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
const forecastColumns=[['number','#'],['model','Model'],['sweep','Sweep'],['value','Value'],['forecast_status','Report'],['samples','Samples'],['values','Values'],['rmse','Model RMSE, 1–H'],['lead_rmse','Model RMSE at H'],['mae','Model MAE, 1–H'],['bias','Bias, 1–H'],['persistence_rmse','Persistence RMSE'],['skill','Skill %']];
function setupForecast(){const report=DATA.forecast_report_root,empty=$('forecast-empty'),content=$('forecast-content');if(!report.exists){empty.textContent=`Forecast report directory not found: ${report.path}`;return}if(!report.runs){empty.textContent=report.invalid_runs?`${report.invalid_runs} forecast report(s) could not be read under ${report.path}.`:`No matching forecast-training reports found under ${report.path}. Expected the 72 bridge-forecast-*-pretrained CSV reports.`;return}empty.hidden=true;content.hidden=false}
function forecast(run){return run.forecast?.[state.forecastSplit]?.[state.forecastHorizon]}
function forecastValue(run,key=state.forecastMetric){const item=forecast(run);if(key==='number')return run.number;if(['model','sweep','value','forecast_status'].includes(key))return run[key];if(key==='abs_bias')return item?Math.abs(item.bias):NaN;return item?.[key]}
function compareForecast(a,b){const av=forecastValue(a,state.forecastSort),bv=forecastValue(b,state.forecastSort);if(typeof av==='string'&&typeof bv==='string')return (state.forecastAscending?1:-1)*av.localeCompare(bv)||a.number-b.number;const aValid=Number.isFinite(av),bValid=Number.isFinite(bv);if(aValid!==bValid)return aValid?-1:1;if(!aValid)return a.number-b.number;return (state.forecastAscending?1:-1)*(av-bv)||a.number-b.number}
function renderForecastHeaders(){$('forecast-headers').innerHTML=forecastColumns.map(([key,label])=>`<th scope="col"><button data-forecast-sort="${key}">${label}<span aria-hidden="true">${state.forecastSort===key?(state.forecastAscending?' ↑':' ↓'):''}</span></button></th>`).join('');$('forecast-headers').querySelectorAll('button').forEach(button=>button.onclick=()=>{const key=button.dataset.forecastSort;if(state.forecastSort===key)state.forecastAscending=!state.forecastAscending;else{state.forecastSort=key;state.forecastAscending=key!=='skill'}renderForecastHeaders();renderForecast()})}
function renderForecast(){if(!DATA.forecast_report_root.runs)return;const runs=visible().filter(run=>forecast(run)).sort(compareForecast),values=runs.map(run=>forecastValue(run)).filter(Number.isFinite).sort((a,b)=>a-b);if(runs.length&&!runs.some(run=>run.number===state.selected))state.selected=runs[0].number;$('forecast-count').textContent=runs.length;$('forecast-median').textContent=values.length?`${fmt((values[Math.floor((values.length-1)/2)]+values[Math.ceil((values.length-1)/2)])/2)}${state.forecastMetric==='skill'?'%':''}`:'—';$('forecast-best').textContent=values.length?`${fmt(state.forecastMetric==='skill'?Math.max(...values):Math.min(...values))}${state.forecastMetric==='skill'?'%':''}`:'—';$('forecast-skilled').textContent=runs.filter(run=>forecast(run).skill>0).length;$('forecast-rows').innerHTML=runs.map(run=>{const item=forecast(run);return `<tr data-forecast-run="${run.number}" data-selected="${run.number===state.selected}"><td class="num">${run.number}</td><td>${escapeHtml(run.model)}</td><td>${escapeHtml(run.sweep)}</td><td>${escapeHtml(run.value)}</td><td class="status ${run.forecast_status}">${escapeHtml(run.forecast_status)}</td><td class="num">${item.samples}</td><td class="num">${item.values}</td><td class="num">${fmt(item.rmse)}</td><td class="num">${fmt(item.lead_rmse)}</td><td class="num">${fmt(item.mae)}</td><td class="num">${fmt(item.bias)}</td><td class="num">${fmt(item.persistence_rmse)}</td><td class="num ${item.skill>0?'negative':'positive'}">${Number.isFinite(item.skill)?`${fmt(item.skill)}%`:'—'}</td></tr>`}).join('');$('forecast-rows').querySelectorAll('tr').forEach(row=>row.onclick=()=>{state.selected=Number(row.dataset.forecastRun);render()});const selected=DATA.runs.find(run=>run.number===state.selected);$('forecast-path').textContent=selected&&forecast(selected)?`Selected forecast report: ${selected.forecast_report_path}`:`Forecast report directory: ${DATA.forecast_report_root.path}`;drawForecastRanking(runs);drawForecastProfile()}
function drawForecastRanking(runs){const svg=$('forecast-ranking'),ranked=runs.filter(run=>Number.isFinite(forecastValue(run))).sort((a,b)=>state.forecastMetric==='skill'?forecastValue(b)-forecastValue(a):forecastValue(a)-forecastValue(b)).slice(0,12);if(!ranked.length)return emptyChart(svg,'No forecast results for this selection.');svg.replaceChildren();const box=svgSize(svg),values=ranked.map(run=>forecastValue(run)),signed=state.forecastMetric==='skill',low=signed?Math.min(0,...values):0,high=Math.max(0,...values),pad=(high-low||1)*.05,min=low-pad,max=high+pad,zero=scale(0,min,max,box.left,box.width-box.right),row=(box.height-box.top-box.bottom)/ranked.length;svg.append(node('line',{class:'axis',x1:zero,x2:zero,y1:box.top,y2:box.height-box.bottom}));ranked.forEach((run,index)=>{const value=forecastValue(run),end=scale(value,min,max,box.left,box.width-box.right),y=box.top+index*row+row*.16,h=row*.66,x=Math.min(zero,end),width=Math.max(2,Math.abs(end-zero));svg.append(node('rect',{class:signed?(value>0?'improvement':'forgetting'):'bar',x,y,width,height:h,rx:3}));svg.append(node('text',{x:box.left-8,y:y+h*.72,'text-anchor':'end'},`#${run.number}`));svg.append(node('text',{x:value>=0?end+5:end-5,y:y+h*.72,'text-anchor':value>=0?'start':'end'},`${fmt(value)}${signed?'%':''}`));const hit=node('rect',{x:box.left,y,width:box.width-box.left-box.right,height:h,fill:'transparent',tabindex:0,'aria-label':`Run ${run.number}, ${forecastMetricLabels[state.forecastMetric]} ${fmt(value)}`});hit.onclick=()=>{state.selected=run.number;render()};hit.onmousemove=event=>showTip(event,`#${run.number} ${run.model} · ${state.forecastHorizon} h · ${fmt(value)}${signed?'%':''}`);hit.onmouseleave=hideTip;svg.append(hit)})}
function drawForecastProfile(){const svg=$('forecast-profile'),run=DATA.runs.find(item=>item.number===state.selected),horizons=DATA.forecast_horizons.filter(horizon=>run?.forecast?.[state.forecastSplit]?.[String(horizon)]);if(!horizons.length)return emptyChart(svg,'Select a run with forecast results.');svg.replaceChildren();const box=svgSize(svg),items=horizons.map(horizon=>run.forecast[state.forecastSplit][String(horizon)]),values=items.flatMap(item=>[item.rmse,item.persistence_rmse]),d=domain(values),x=index=>box.left+index*(box.width-box.left-box.right)/Math.max(horizons.length-1,1),y=value=>scale(value,d[0],d[1],box.height-box.bottom,box.top);for(let i=0;i<5;i++){const gy=box.top+i*(box.height-box.top-box.bottom)/4;svg.append(node('line',{class:'axis',x1:box.left,x2:box.width-box.right,y1:gy,y2:gy}));svg.append(node('text',{x:box.left-8,y:gy+4,'text-anchor':'end'},fmt(d[1]-i*(d[1]-d[0])/4)))}horizons.forEach((horizon,index)=>svg.append(node('text',{x:x(index),y:box.height-box.bottom+18,'text-anchor':'middle'},`${horizon} h`)));for(const [key,klass] of [['rmse','model'],['persistence_rmse','persistence']]){svg.append(node('polyline',{class:klass,points:items.map((item,index)=>`${x(index)},${y(item[key])}`).join(' '),fill:'none','stroke-width':2.5}));items.forEach((item,index)=>svg.append(node('circle',{class:klass,cx:x(index),cy:y(item[key]),r:3})))} }
const forgetColumns=[['number','#'],['model','Model'],['sweep','Sweep'],['value','Value'],['pre','Pre-bridge'],['post','Post-bridge'],['difference','Loss change'],['percent','Change %']];
function setupForgetting(){const report=DATA.forgetting_report,empty=$('forgetting-empty'),content=$('forgetting-content');$('forget-path').textContent=`Forgetting report: ${report.path}`;if(report.error){empty.textContent=`Forgetting report could not be read: ${report.error}`;return}if(!report.exists){empty.textContent=`Forgetting report not found. Generate ${report.path} to populate this section.`;return}if(!report.runs){empty.textContent='The forgetting report contains no complete per-run comparison groups yet.';return}empty.hidden=true;content.hidden=false}
function forgetting(run){return run.forgetting?.[state.forgetSplit]?.[state.forgetStage]}
function forgetValue(run,key=state.forgetMetric){return key==='number'?run.number:key==='model'?run.model:key==='sweep'?run.sweep:key==='value'?run.value:forgetting(run)?.[key]}
function compareForgetting(a,b){const av=forgetValue(a,state.forgetSort),bv=forgetValue(b,state.forgetSort);if(typeof av==='string'&&typeof bv==='string')return (state.forgetAscending?1:-1)*av.localeCompare(bv)||a.number-b.number;const aValid=Number.isFinite(av),bValid=Number.isFinite(bv);if(aValid!==bValid)return aValid?-1:1;if(!aValid)return a.number-b.number;return (state.forgetAscending?1:-1)*(av-bv)||a.number-b.number}
function renderForgetHeaders(){$('forget-headers').innerHTML=forgetColumns.map(([key,label])=>`<th scope="col"><button data-forget-sort="${key}">${label}<span aria-hidden="true">${state.forgetSort===key?(state.forgetAscending?' ↑':' ↓'):''}</span></button></th>`).join('');$('forget-headers').querySelectorAll('button').forEach(button=>button.onclick=()=>{const key=button.dataset.forgetSort;if(state.forgetSort===key)state.forgetAscending=!state.forgetAscending;else{state.forgetSort=key;state.forgetAscending=!['difference','percent'].includes(key)}renderForgetHeaders();renderForgetting()})}
function renderForgetting(){if(!DATA.forgetting_report.runs)return;const runs=visible().filter(run=>forgetting(run)).sort(compareForgetting),changes=runs.map(run=>forgetting(run).difference).sort((a,b)=>a-b);if(runs.length&&!runs.some(run=>run.number===state.selected))state.selected=runs[0].number;$('forget-count').textContent=runs.length;$('forget-median').textContent=changes.length?fmt((changes[Math.floor((changes.length-1)/2)]+changes[Math.ceil((changes.length-1)/2)])/2):'—';$('forget-worst').textContent=changes.length?fmt(Math.max(...changes)):'—';$('forget-improved').textContent=changes.filter(value=>value<=0).length;$('forget-rows').innerHTML=runs.map(run=>{const item=forgetting(run),percent=Number.isFinite(item.percent)?`${fmt(item.percent)}%`:'—';return `<tr data-forget-run="${run.number}" data-selected="${run.number===state.selected}"><td class="num">${run.number}</td><td>${escapeHtml(run.model)}</td><td>${escapeHtml(run.sweep)}</td><td>${escapeHtml(run.value)}</td><td class="num">${fmt(item.pre)}</td><td class="num">${fmt(item.post)}</td><td class="num ${item.difference>0?'positive':'negative'}">${fmt(item.difference)}</td><td class="num ${item.percent>0?'positive':'negative'}">${percent}</td></tr>`}).join('');$('forget-rows').querySelectorAll('tr').forEach(row=>row.onclick=()=>{state.selected=Number(row.dataset.forgetRun);render()});drawForgetRanking(runs);drawForgetProfile()}
function drawForgetRanking(runs){const svg=$('forget-ranking'),ranked=runs.filter(run=>Number.isFinite(forgetValue(run))).sort((a,b)=>{const av=forgetValue(a),bv=forgetValue(b);return ['difference','percent'].includes(state.forgetMetric)?bv-av:av-bv}).slice(0,12);if(!ranked.length)return emptyChart(svg,'No forgetting results for this selection.');svg.replaceChildren();const box=svgSize(svg),values=ranked.map(run=>forgetValue(run)),signed=['difference','percent'].includes(state.forgetMetric),low=signed?Math.min(0,...values):0,high=Math.max(0,...values),pad=(high-low||1)*.05,min=low-pad,max=high+pad,zero=scale(0,min,max,box.left,box.width-box.right),row=(box.height-box.top-box.bottom)/ranked.length;svg.append(node('line',{class:'axis',x1:zero,x2:zero,y1:box.top,y2:box.height-box.bottom}));ranked.forEach((run,index)=>{const value=forgetValue(run),end=scale(value,min,max,box.left,box.width-box.right),y=box.top+index*row+row*.16,h=row*.66,x=Math.min(zero,end),width=Math.max(2,Math.abs(end-zero));svg.append(node('rect',{class:signed?(value>0?'forgetting':'improvement'):'pre',x,y,width,height:h,rx:3}));svg.append(node('text',{x:box.left-8,y:y+h*.72,'text-anchor':'end'},`#${run.number}`));svg.append(node('text',{x:value>=0?end+5:end-5,y:y+h*.72,'text-anchor':value>=0?'start':'end'},`${fmt(value)}${state.forgetMetric==='percent'?'%':''}`));const hit=node('rect',{x:box.left,y,width:box.width-box.left-box.right,height:h,fill:'transparent'});hit.onclick=()=>{state.selected=run.number;render()};hit.onmousemove=event=>showTip(event,`#${run.number} ${run.model} · ${stageLabels[state.forgetStage]} · ${fmt(value)}${state.forgetMetric==='percent'?'%':''}`);hit.onmouseleave=hideTip;svg.append(hit)})}
function drawForgetProfile(){const svg=$('forget-profile'),run=DATA.runs.find(item=>item.number===state.selected),stages=DATA.base_stages.filter(stage=>run?.forgetting?.[state.forgetSplit]?.[stage]);if(!stages.length)return emptyChart(svg,'Select a run with forgetting results.');svg.replaceChildren();const box=svgSize(svg),items=stages.map(stage=>run.forgetting[state.forgetSplit][stage]),values=items.flatMap(item=>[item.pre,item.post]),d=domain(values),x=index=>box.left+index*(box.width-box.left-box.right)/Math.max(stages.length-1,1),y=value=>scale(value,d[0],d[1],box.height-box.bottom,box.top);for(let i=0;i<5;i++){const gy=box.top+i*(box.height-box.top-box.bottom)/4;svg.append(node('line',{class:'axis',x1:box.left,x2:box.width-box.right,y1:gy,y2:gy}));svg.append(node('text',{x:box.left-8,y:gy+4,'text-anchor':'end'},fmt(d[1]-i*(d[1]-d[0])/4)))}stages.forEach((stage,index)=>svg.append(node('text',{x:x(index),y:box.height-box.bottom+18,'text-anchor':'end',transform:`rotate(-28 ${x(index)} ${box.height-box.bottom+18})`},stageLabels[stage])));for(const [key,klass] of [['pre','pre'],['post','post']]){svg.append(node('polyline',{class:klass,points:items.map((item,index)=>`${x(index)},${y(item[key])}`).join(' '),fill:'none','stroke-width':2.5}));items.forEach((item,index)=>svg.append(node('circle',{class:klass,cx:x(index),cy:y(item[key]),r:3})))} }
let resizeTimer;new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(render,50)}).observe(document.querySelector('main'));setup();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
