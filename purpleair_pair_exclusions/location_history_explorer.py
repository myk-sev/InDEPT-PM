"""Write the lazy-loaded paired PurpleAir full-history explorer."""

from __future__ import annotations

import json
from pathlib import Path

from purpleair_pair_exclusions.outdoor_quality import OutdoorExclusion


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PurpleAir location history explorer</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#202124;--muted:#666;--line:#ccd2d8;--in:#2878b5;--out:#c43c39;--bad:#f1c75b55;--bad-line:#9a6800} @media(prefers-color-scheme:dark){:root{--bg:#17191c;--fg:#eee;--muted:#aaa;--line:#59616b;--in:#6db7e8;--out:#f08078;--bad:#b9873255;--bad-line:#f1c75b}} *{box-sizing:border-box} body{margin:0 auto;max-width:1200px;padding:1rem;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)} h1{font-size:1.4rem;margin:.2rem 0}.controls{display:flex;gap:.7rem;align-items:end;flex-wrap:wrap;margin:1rem 0}.controls label{display:grid;gap:.25rem}.location{flex:1 1 32rem}input,button{font:inherit;padding:.45rem .6rem}input[type=search]{width:100%}.muted{color:var(--muted)}.error{color:#b3261e}.legend,.reading{display:flex;gap:1.2rem;flex-wrap:wrap;margin:.6rem 0;font-variant-numeric:tabular-nums}.legend span:before{content:"";display:inline-block;width:1.4rem;border-top:3px solid;margin-right:.35rem;vertical-align:middle}.legend .outdoor:before{border-color:var(--out)}.legend .indoor:before{border-color:var(--in)}.legend .invalid:before{border-color:var(--bad-line);border-top-width:8px}.reading .outdoor{color:var(--out)}.reading .indoor{color:var(--in)}.chart-shell{position:relative;height:560px}.chart-shell canvas{position:absolute;inset:0;width:100%;height:100%}#chart{border:1px solid var(--line)}#overlay{cursor:crosshair;touch-action:none}@media(max-width:600px){body{padding:.7rem}.chart-shell{height:480px}.controls>*{flex:1 1 100%}}
</style>
</head>
<body>
<h1>PurpleAir location history explorer</h1>
<div class="controls">
  <label class="location">Location<input id="location" type="search" list="locations" autocomplete="off" aria-describedby="locationHelp"></label>
  <datalist id="locations"></datalist>
  <label>Start UTC<input id="start" type="datetime-local" step="3600"></label>
  <label>End UTC<input id="end" type="datetime-local" step="3600"></label>
  <button id="apply" type="button">Apply range</button>
  <button id="clear" type="button">Entire history</button>
</div>
<p id="locationHelp" class="muted">Search by location name or sensor ID. Blank dates show the entire available history.</p>
<p id="status" class="muted" aria-live="polite"></p>
<p id="error" class="error" role="alert" hidden></p>
<p id="quality" class="error" hidden></p>
<div class="legend"><span class="outdoor">Outdoor PurpleAir PM2.5</span><span class="indoor">Indoor PurpleAir PM2.5</span><span id="invalidLegend" class="invalid" hidden>Known-bad outdoor period</span></div>
<div id="reading" class="reading" aria-live="polite">Select a location to inspect its readings.</div>
<div class="chart-shell">
  <canvas id="chart" role="img" aria-label="Stacked outdoor and indoor PurpleAir PM2.5 histories over the selected UTC time range"></canvas>
  <canvas id="overlay" aria-hidden="true"></canvas>
</div>
<script>
const locations=__LOCATIONS__;
const locationInput=document.getElementById('location'),locationList=document.getElementById('locations'),startInput=document.getElementById('start'),endInput=document.getElementById('end'),status=document.getElementById('status'),error=document.getElementById('error'),quality=document.getElementById('quality'),invalidLegend=document.getElementById('invalidLegend'),reading=document.getElementById('reading'),chart=document.getElementById('chart'),overlay=document.getElementById('overlay');
let current=null,pendingSensor=null,visible=[],geometry=null,overlayContext=null,selectedTime=null,pinned=false;
locations.sort((a,b)=>a.label.localeCompare(b.label)).forEach(location=>{const option=document.createElement('option');option.value=location.label;locationList.append(option)});
function showError(message=''){error.textContent=message;error.hidden=!message}
function utcInput(timestamp){return new Date(timestamp*1000).toISOString().slice(0,16)}
function parseUtc(input){return input.value?Date.parse(`${input.value}Z`)/1000:null}
function formatPm25(value){return Number.isFinite(value)?`${value.toFixed(1)} µg/m³`:'Missing'}
function chooseLocation(){const query=locationInput.value.trim().toLowerCase(),match=locations.find(location=>location.label.toLowerCase()===query)||locations.find(location=>location.label.toLowerCase().includes(query));if(!match){showError('Choose a location from the search results.');return}loadLocation(match)}
function loadLocation(location){if(current?.indoor_sensor_id===location.indoor_sensor_id)return;pendingSensor=location.indoor_sensor_id;current=null;visible=[];selectedTime=null;pinned=false;showError();quality.hidden=true;invalidLegend.hidden=true;status.textContent=`Loading ${location.label}…`;locationInput.value=location.label;const script=document.createElement('script');script.src=`location_history_data/${location.indoor_sensor_id}.js`;script.onload=()=>script.remove();script.onerror=()=>{showError('The location history could not be loaded. Keep the location_history_data folder beside this file.');status.textContent='';script.remove()};document.head.append(script)}
function rangeText(range){if(range.start===null&&range.end===null)return `all downloaded hours — ${range.reason}`;const start=range.start===null?'first reading':new Date(range.start*1000).toISOString(),end=range.end===null?'last reading':`${new Date(range.end*1000).toISOString()} (end exclusive)`;return `${start} to ${end} — ${range.reason}`}
window.__loadPurpleAirLocation=data=>{if(data.indoor_sensor_id!==pendingSensor)return;current=data;const ranges=data.outdoor_exclusions||[];quality.hidden=!ranges.length;invalidLegend.hidden=!ranges.length;quality.textContent=ranges.length?`Excluded from analysis and training: ${ranges.map(rangeText).join('; ')}.`:'';startInput.value='';endInput.value='';startInput.min=endInput.min=utcInput(data.series[0][0]);startInput.max=endInput.max=utcInput(data.series.at(-1)[0]);draw()};
locationInput.addEventListener('change',chooseLocation);locationInput.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();chooseLocation()}});document.getElementById('apply').onclick=draw;document.getElementById('clear').onclick=()=>{startInput.value='';endInput.value='';draw()};
function canvasContext(canvas,width,height){const ratio=devicePixelRatio||1;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;const context=canvas.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);return context}
function color(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}
function nearestIndex(rows,timestamp){let low=0,high=rows.length-1;while(low<high){const middle=Math.floor((low+high)/2);if(rows[middle][0]<timestamp)low=middle+1;else high=middle}return low&&Math.abs(rows[low-1][0]-timestamp)<=Math.abs(rows[low][0]-timestamp)?low-1:low}
function draw(){if(!current)return;showError();const start=parseUtc(startInput),end=parseUtc(endInput);if(start!==null&&!Number.isFinite(start)||end!==null&&!Number.isFinite(end)){showError('Enter valid UTC dates.');return}if(start!==null&&end!==null&&start>end){showError('Start UTC must not be after End UTC.');return}visible=current.series.filter(row=>(start===null||row[0]>=start)&&(end===null||row[0]<=end));if(!visible.length){showError('No readings fall inside that UTC range.');return}const width=chart.parentElement.clientWidth,height=chart.parentElement.clientHeight,context=canvasContext(chart,width,height);overlayContext=canvasContext(overlay,width,height);overlayContext.clearRect(0,0,width,height);const left=72,right=22,top=32,bottom=48,gap=62,panelHeight=(height-top-bottom-gap)/2,indoorTop=top+panelHeight+gap,minTime=visible[0][0],maxTime=visible.at(-1)[0],domainEnd=maxTime===minTime?minTime+3600:maxTime,outdoorMax=Math.max(1,...visible.map(row=>row[2]).filter(Number.isFinite))*1.08,indoorMax=Math.max(1,...visible.map(row=>row[1]).filter(Number.isFinite))*1.08,x=time=>left+(time-minTime)/(domainEnd-minTime)*(width-left-right),y=(value,panelTop,maximum)=>panelTop+panelHeight-value/maximum*panelHeight;geometry={width,height,left,right,top,bottom,gap,panelHeight,indoorTop,minTime,domainEnd,outdoorMax,indoorMax,x,y};context.fillStyle=color('--bad');(current.outdoor_exclusions||[]).forEach(range=>{const rangeStart=Math.max(minTime,range.start??minTime),rangeEnd=Math.min(domainEnd,range.end??domainEnd);if(rangeStart<rangeEnd)context.fillRect(x(rangeStart),top,Math.max(1,x(rangeEnd)-x(rangeStart)),indoorTop+panelHeight-top)});context.font='12px system-ui';context.fillStyle=color('--fg');context.strokeStyle=color('--line');context.lineWidth=1;[['Outdoor PM2.5 (µg/m³)',top,outdoorMax],['Indoor PM2.5 (µg/m³)',indoorTop,indoorMax]].forEach(([label,panelTop,maximum])=>{context.textAlign='left';context.fillText(label,left,panelTop-12);for(let tick=0;tick<=4;tick++){const yy=panelTop+tick*panelHeight/4;context.beginPath();context.moveTo(left,yy);context.lineTo(width-right,yy);context.stroke();context.textAlign='right';context.fillText((maximum*(4-tick)/4).toFixed(maximum<10?1:0),left-8,yy+4)}});const ticks=width<600?3:5;context.textAlign='center';for(let tick=0;tick<ticks;tick++){const time=minTime+(domainEnd-minTime)*tick/(ticks-1),xx=x(time),label=new Date(time*1000).toISOString();context.fillText(domainEnd-minTime>172800?label.slice(0,10):label.slice(5,16).replace('T',' '),xx,height-20)}context.fillText('UTC date and hour',(left+width-right)/2,height-3);drawSeries(context,2,color('--out'),top,outdoorMax);drawSeries(context,1,color('--in'),indoorTop,indoorMax);const fullStart=current.series[0][0],fullEnd=current.series.at(-1)[0],rangeLabel=start===null&&end===null?'entire history':'selected range';status.textContent=`${current.indoor_name} (indoor ${current.indoor_sensor_id}) paired with ${current.outdoor_name} (outdoor ${current.outdoor_sensor_id}); showing ${visible.length.toLocaleString()} of ${current.series.length.toLocaleString()} hourly timestamps across the ${rangeLabel}, ${new Date(minTime*1000).toISOString()} to ${new Date(maxTime*1000).toISOString()}. Full span: ${new Date(fullStart*1000).toISOString()} to ${new Date(fullEnd*1000).toISOString()}.`;if(selectedTime===null||selectedTime<minTime||selectedTime>maxTime)selectedTime=minTime;inspect(selectedTime)}
function drawSeries(context,valueIndex,stroke,panelTop,maximum){context.strokeStyle=stroke;context.lineWidth=1.5;context.beginPath();let active=false,lastTime=null;visible.forEach(row=>{const value=row[valueIndex];if(!Number.isFinite(value)){active=false;return}const xx=geometry.x(row[0]),yy=geometry.y(value,panelTop,maximum);if(!active||lastTime!==null&&row[0]-lastTime>7200)context.moveTo(xx,yy);else context.lineTo(xx,yy);active=true;lastTime=row[0]});context.stroke()}
function inspect(timestamp){if(!geometry||!visible.length)return;const row=visible[nearestIndex(visible,timestamp)],context=overlayContext,xx=geometry.x(row[0]);context.clearRect(0,0,geometry.width,geometry.height);context.strokeStyle=color('--fg');context.setLineDash([4,3]);context.beginPath();context.moveTo(xx,geometry.top);context.lineTo(xx,geometry.indoorTop+geometry.panelHeight);context.stroke();context.setLineDash([]);[[row[2],geometry.top,geometry.outdoorMax,'--out'],[row[1],geometry.indoorTop,geometry.indoorMax,'--in']].forEach(([value,panelTop,maximum,seriesColor])=>{if(!Number.isFinite(value))return;context.fillStyle=color(seriesColor);context.beginPath();context.arc(xx,geometry.y(value,panelTop,maximum),5,0,Math.PI*2);context.fill()});selectedTime=row[0];reading.innerHTML=`<strong>${new Date(row[0]*1000).toISOString()}</strong><span class="indoor">Indoor: ${formatPm25(row[1])}</span><span class="outdoor">Outdoor: ${formatPm25(row[2])}</span>`}
function pointerTime(event){const bounds=overlay.getBoundingClientRect(),xx=Math.max(geometry.left,Math.min(geometry.width-geometry.right,event.clientX-bounds.left));return geometry.minTime+(xx-geometry.left)/(geometry.width-geometry.left-geometry.right)*(geometry.domainEnd-geometry.minTime)}
overlay.addEventListener('pointermove',event=>{if(geometry&&!pinned)inspect(pointerTime(event))});overlay.addEventListener('click',event=>{if(geometry){pinned=true;inspect(pointerTime(event))}});overlay.addEventListener('pointerleave',()=>{if(!pinned){overlayContext.clearRect(0,0,geometry.width,geometry.height);reading.textContent='Move across the chart or click to pin the nearest hourly reading.'}});new ResizeObserver(()=>draw()).observe(chart.parentElement);matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
if(locations.length){locationInput.value=locations[0].label;loadLocation(locations[0])}else{showError('No locations have both indoor and outdoor PurpleAir histories.')}
</script>
</body>
</html>'''


def write_location_history_explorer(
    output: Path,
    pairs: list[dict[str, object]],
    indoor: dict[int, dict[int, float]],
    outdoor: dict[int, dict[int, float]],
    outdoor_exclusions: tuple[OutdoorExclusion, ...] = (),
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "location_history_data"
    data_dir.mkdir(exist_ok=True)
    locations, expected_files = [], set()
    for pair in pairs:
        indoor_id = int(pair["indoor_sensor_id"])
        outdoor_id = int(pair["outdoor_sensor_id"])
        indoor_values = indoor.get(indoor_id, {})
        outdoor_values = outdoor.get(outdoor_id, {})
        if not indoor_values or not outdoor_values:
            continue
        timestamps = sorted(indoor_values.keys() | outdoor_values.keys())
        data = {
            "indoor_sensor_id": indoor_id,
            "indoor_name": pair["indoor_name"],
            "outdoor_sensor_id": outdoor_id,
            "outdoor_name": pair["outdoor_name"],
            "outdoor_exclusions": [
                {
                    "start": exclusion.start,
                    "end": exclusion.end,
                    "reason": exclusion.reason,
                }
                for exclusion in outdoor_exclusions
                if exclusion.sensor_id == outdoor_id
            ],
            "series": [
                [timestamp, indoor_values.get(timestamp), outdoor_values.get(timestamp)]
                for timestamp in timestamps
            ],
        }
        filename = f"{indoor_id}.js"
        expected_files.add(filename)
        payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
        (data_dir / filename).write_text(
            f"window.__loadPurpleAirLocation({payload});\n", encoding="utf-8"
        )
        locations.append(
            {
                "indoor_sensor_id": indoor_id,
                "label": (
                    f"{pair['indoor_name']} — indoor {indoor_id} / outdoor {outdoor_id}"
                ),
            }
        )
    for path in data_dir.glob("*.js"):
        if path.name not in expected_files:
            path.unlink()
    metadata = json.dumps(locations, separators=(",", ":")).replace("</", "<\\/")
    (output / "location_history_explorer.html").write_text(
        HTML.replace("__LOCATIONS__", metadata), encoding="utf-8"
    )
    return len(locations)
