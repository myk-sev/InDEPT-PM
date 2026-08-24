"""Write the lazy-loaded PurpleAir sensor history explorer."""

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
:root{color-scheme:light dark;--bg:#fff;--fg:#202124;--muted:#666;--line:#ccd2d8;--in:#2878b5;--out:#c43c39;--bad:#f1c75b55;--bad-line:#9a6800;--select:#2878b544;--select-line:#185b8d} @media(prefers-color-scheme:dark){:root{--bg:#17191c;--fg:#eee;--muted:#aaa;--line:#59616b;--in:#6db7e8;--out:#f08078;--bad:#b9873255;--bad-line:#f1c75b;--select:#6db7e844;--select-line:#9bd4f5}} *{box-sizing:border-box} body{margin:0 auto;max-width:1200px;padding:1rem;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)} h1{font-size:1.4rem;margin:.2rem 0}.pages{display:flex;gap:.4rem;margin:.8rem 0}.pages a{color:var(--fg);padding:.45rem .7rem;border:1px solid var(--line);border-radius:.25rem;text-decoration:none}.pages a[aria-current=page]{color:var(--bg);background:var(--fg)}.controls{display:flex;gap:.7rem;align-items:end;flex-wrap:wrap;margin:1rem 0}.controls label{display:grid;gap:.25rem}.controls .check{display:flex;align-items:center;gap:.4rem;padding:.45rem 0}.location{flex:1 1 32rem}input,button{font:inherit;padding:.45rem .6rem}input[type=search]{width:100%}.check input{margin:0}button:disabled{opacity:.55}.muted{color:var(--muted)}.error{color:#b3261e}.legend,.reading{display:flex;gap:1.2rem;flex-wrap:wrap;margin:.6rem 0;font-variant-numeric:tabular-nums}.legend span:before{content:"";display:inline-block;width:1.4rem;border-top:3px solid;margin-right:.35rem;vertical-align:middle}.legend .outdoor:before{border-color:var(--out)}.legend .indoor:before{border-color:var(--in)}.legend .invalid:before{border-color:var(--bad-line);border-top-width:8px}.reading .outdoor{color:var(--out)}.reading .indoor{color:var(--in)}.chart-shell{position:relative;height:560px}.chart-shell canvas{position:absolute;inset:0;width:100%;height:100%}#chart{border:1px solid var(--line)}#overlay{cursor:crosshair;touch-action:none}@media(max-width:600px){body{padding:.7rem}.chart-shell{height:480px}.controls>*{flex:1 1 100%}}
</style>
</head>
<body>
<h1>PurpleAir location history explorer</h1>
<nav class="pages" aria-label="History pages"><a id="pairedPage" href="#paired">Paired locations</a><a id="reviewPage" href="#review">1 km review sensors</a><a id="recentPage" href="#recent">Recent-data exclusions</a><a id="unpairedPage" href="#unpaired">Unpaired sensors</a><a id="excludedPage" href="#excluded">Excluded sensors and ranges</a></nav>
<div class="controls">
  <label class="location"><span id="locationLabel">Location</span><input id="location" type="search" list="locations" autocomplete="off" aria-describedby="locationHelp"></label>
  <datalist id="locations"></datalist>
  <label>Start UTC<input id="start" type="datetime-local" step="3600"></label>
  <label>End UTC<input id="end" type="datetime-local" step="3600"></label>
  <label class="check"><input id="k12Only" type="checkbox">K-12 locations only</label>
  <button id="apply" type="button">Apply range</button>
  <button id="clear" type="button">Entire history</button>
  <button id="resetZoom" type="button" disabled>Reset zoom</button>
  <button id="previousLocation" type="button" aria-label="Previous location">&larr; Previous</button>
  <button id="nextLocation" type="button" aria-label="Next location">Next &rarr;</button>
</div>
<p id="locationHelp" class="muted">Search by location name or sensor ID. Blank dates show the entire available history. Drag across either chart and release to zoom; click without dragging to pin a reading.</p>
<p id="status" class="muted" aria-live="polite"></p>
<p id="error" class="error" role="alert" hidden></p>
<p id="quality" class="error" hidden></p>
<div class="legend"><span id="outdoorLegend" class="outdoor">Outdoor PurpleAir PM2.5</span><span id="indoorLegend" class="indoor">Indoor PurpleAir PM2.5</span><span id="invalidLegend" class="invalid" hidden>Excluded period</span></div>
<div id="reading" class="reading" aria-live="polite">Select a location to inspect its readings.</div>
<div class="chart-shell">
  <canvas id="chart" role="img" aria-label="Stacked outdoor and indoor PurpleAir PM2.5 histories over the selected UTC time range"></canvas>
  <canvas id="overlay" aria-hidden="true"></canvas>
</div>
<script>
const pairedLocations=__LOCATIONS__,reviewLocations=__REVIEW_LOCATIONS__,recentExcludedLocations=__RECENT_EXCLUDED_LOCATIONS__,unpairedLocations=__UNPAIRED_LOCATIONS__,excludedLocations=__EXCLUDED_LOCATIONS__;
const locationInput=document.getElementById('location'),locationList=document.getElementById('locations'),locationLabel=document.getElementById('locationLabel'),locationHelp=document.getElementById('locationHelp'),startInput=document.getElementById('start'),endInput=document.getElementById('end'),k12Only=document.getElementById('k12Only'),resetZoom=document.getElementById('resetZoom'),previousLocation=document.getElementById('previousLocation'),nextLocation=document.getElementById('nextLocation'),status=document.getElementById('status'),error=document.getElementById('error'),quality=document.getElementById('quality'),outdoorLegend=document.getElementById('outdoorLegend'),indoorLegend=document.getElementById('indoorLegend'),invalidLegend=document.getElementById('invalidLegend'),reading=document.getElementById('reading'),chart=document.getElementById('chart'),overlay=document.getElementById('overlay'),pairedPage=document.getElementById('pairedPage'),reviewPage=document.getElementById('reviewPage'),recentPage=document.getElementById('recentPage'),unpairedPage=document.getElementById('unpairedPage'),excludedPage=document.getElementById('excludedPage');
let locations=[],page='',current=null,pendingLocation=null,visible=[],geometry=null,overlayContext=null,selectedTime=null,pinned=false,zoomStart=null,zoomEnd=null,drag=null;
[pairedLocations,reviewLocations,recentExcludedLocations,unpairedLocations,excludedLocations].forEach(items=>items.sort((a,b)=>a.label.localeCompare(b.label)));pairedPage.textContent=`Paired locations (${pairedLocations.length})`;reviewPage.textContent=`1 km review sensors (${reviewLocations.length})`;recentPage.textContent=`Recent-data exclusions (${recentExcludedLocations.length})`;unpairedPage.textContent=`Unpaired sensors (${unpairedLocations.length})`;excludedPage.textContent=`Excluded sensors and ranges (${excludedLocations.length})`;
function showError(message=''){error.textContent=message;error.hidden=!message}
function utcInput(timestamp){return new Date(timestamp*1000).toISOString().slice(0,16)}
function parseUtc(input){return input.value?Date.parse(`${input.value}Z`)/1000:null}
function formatPm25(value){return Number.isFinite(value)?`${value.toFixed(1)} µg/m³`:'Missing'}
function chooseLocation(){const query=locationInput.value.trim().toLowerCase(),match=locations.find(location=>location.label.toLowerCase()===query)||locations.find(location=>location.label.toLowerCase().includes(query));if(!match){showError('Choose a location from the search results.');return}loadLocation(match)}
function updateNavigation(){const index=locations.findIndex(location=>location.file===pendingLocation);previousLocation.disabled=index<=0;nextLocation.disabled=index<0||index>=locations.length-1}
function stepLocation(offset){const index=locations.findIndex(location=>location.file===pendingLocation),next=index+offset;if(next>=0&&next<locations.length)loadLocation(locations[next])}
function loadLocation(location){if(current?.data_file===location.file)return;pendingLocation=location.file;updateNavigation();current=null;visible=[];selectedTime=null;pinned=false;zoomStart=zoomEnd=null;resetZoom.disabled=true;showError();quality.hidden=true;invalidLegend.hidden=true;status.textContent=`Loading ${location.label}…`;locationInput.value=location.label;const script=document.createElement('script');script.src=`location_history_data/${location.file}`;script.onload=()=>script.remove();script.onerror=()=>{showError('The location history could not be loaded. Keep the location_history_data folder beside this file.');status.textContent='';script.remove()};document.head.append(script)}
function rangeText(range){if(range.start===null&&range.end===null)return `all downloaded hours — ${range.reason}`;const start=range.start===null?'first reading':new Date(range.start*1000).toISOString(),end=range.end===null?'last reading':`${new Date(range.end*1000).toISOString()} (end exclusive)`;return `${start} to ${end} — ${range.reason}`}
window.__loadPurpleAirLocation=data=>{if(data.data_file!==pendingLocation)return;current=data;const ranges=data.exclusions||[];quality.hidden=!ranges.length;invalidLegend.hidden=!ranges.length;invalidLegend.textContent=data.paired?'Known-bad outdoor period':'Excluded period';quality.textContent=ranges.length?`Excluded from analysis and training: ${ranges.map(rangeText).join('; ')}.`:'';outdoorLegend.hidden=!data.paired&&data.sensor_type!=='outdoor';indoorLegend.hidden=!data.paired&&data.sensor_type!=='indoor';chart.setAttribute('aria-label',data.paired?'Stacked outdoor and indoor PurpleAir PM2.5 histories over the selected UTC time range':`${data.sensor_type} PurpleAir PM2.5 history over the selected UTC time range`);startInput.value='';endInput.value='';startInput.min=endInput.min=utcInput(data.series[0][0]);startInput.max=endInput.max=utcInput(data.series.at(-1)[0]);draw()};
locationInput.addEventListener('change',chooseLocation);locationInput.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();chooseLocation()}});k12Only.onchange=()=>showPage(true);document.getElementById('apply').onclick=()=>{zoomStart=zoomEnd=null;draw()};document.getElementById('clear').onclick=()=>{startInput.value='';endInput.value='';zoomStart=zoomEnd=null;draw()};resetZoom.onclick=()=>{zoomStart=zoomEnd=null;pinned=false;draw()};previousLocation.onclick=()=>stepLocation(-1);nextLocation.onclick=()=>stepLocation(1);
function canvasContext(canvas,width,height){const ratio=devicePixelRatio||1;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);canvas.style.width=`${width}px`;canvas.style.height=`${height}px`;const context=canvas.getContext('2d');context.setTransform(ratio,0,0,ratio,0,0);return context}
function color(name){return getComputedStyle(document.documentElement).getPropertyValue(name).trim()}
function nearestIndex(rows,timestamp){let low=0,high=rows.length-1;while(low<high){const middle=Math.floor((low+high)/2);if(rows[middle][0]<timestamp)low=middle+1;else high=middle}return low&&Math.abs(rows[low-1][0]-timestamp)<=Math.abs(rows[low][0]-timestamp)?low-1:low}
function draw(){if(!current)return;showError();const start=parseUtc(startInput),end=parseUtc(endInput);if(start!==null&&!Number.isFinite(start)||end!==null&&!Number.isFinite(end)){showError('Enter valid UTC dates.');return}if(start!==null&&end!==null&&start>end){showError('Start UTC must not be after End UTC.');return}const ranged=current.series.filter(row=>(start===null||row[0]>=start)&&(end===null||row[0]<=end));visible=ranged.filter(row=>(zoomStart===null||row[0]>=zoomStart)&&(zoomEnd===null||row[0]<=zoomEnd));if(!visible.length){showError('No readings fall inside that UTC range.');return}resetZoom.disabled=zoomStart===null;const paired=current.paired,singleType=current.sensor_type==='outdoor'?'Outdoor':'Indoor',singleColor=current.sensor_type==='outdoor'?'--out':'--in',width=chart.parentElement.clientWidth,height=chart.parentElement.clientHeight,context=canvasContext(chart,width,height);overlayContext=canvasContext(overlay,width,height);overlayContext.clearRect(0,0,width,height);const left=72,right=22,top=32,bottom=48,gap=paired?62:0,panelHeight=(height-top-bottom-gap)/(paired?2:1),indoorTop=paired?top+panelHeight+gap:top,minTime=visible[0][0],maxTime=visible.at(-1)[0],domainEnd=maxTime===minTime?minTime+3600:maxTime,outdoorMax=Math.max(1,...visible.map(row=>row[2]).filter(Number.isFinite))*1.08,indoorMax=Math.max(1,...visible.map(row=>row[1]).filter(Number.isFinite))*1.08,x=time=>left+(time-minTime)/(domainEnd-minTime)*(width-left-right),y=(value,panelTop,maximum)=>panelTop+panelHeight-value/maximum*panelHeight;geometry={width,height,left,right,top,bottom,gap,panelHeight,indoorTop,minTime,domainEnd,outdoorMax,indoorMax,x,y};context.fillStyle=color('--bad');(current.exclusions||[]).forEach(range=>{const rangeStart=Math.max(minTime,range.start??minTime),rangeEnd=Math.min(domainEnd,range.end??domainEnd);if(rangeStart<rangeEnd)context.fillRect(x(rangeStart),top,Math.max(1,x(rangeEnd)-x(rangeStart)),indoorTop+panelHeight-top)});context.font='12px system-ui';context.fillStyle=color('--fg');context.strokeStyle=color('--line');context.lineWidth=1;const panels=paired?[['Outdoor PM2.5 (µg/m³)',top,outdoorMax],['Indoor PM2.5 (µg/m³)',indoorTop,indoorMax]]:[[`${singleType} PM2.5 (µg/m³)`,indoorTop,indoorMax]];panels.forEach(([label,panelTop,maximum])=>{context.textAlign='left';context.fillText(label,left,panelTop-12);for(let tick=0;tick<=4;tick++){const yy=panelTop+tick*panelHeight/4;context.beginPath();context.moveTo(left,yy);context.lineTo(width-right,yy);context.stroke();context.textAlign='right';context.fillText((maximum*(4-tick)/4).toFixed(maximum<10?1:0),left-8,yy+4)}});const ticks=width<600?3:5;context.textAlign='center';for(let tick=0;tick<ticks;tick++){const time=minTime+(domainEnd-minTime)*tick/(ticks-1),xx=x(time),label=new Date(time*1000).toISOString();context.fillText(domainEnd-minTime>172800?label.slice(0,10):label.slice(5,16).replace('T',' '),xx,height-20)}context.fillText('UTC date and hour',(left+width-right)/2,height-3);if(paired)drawSeries(context,2,color('--out'),top,outdoorMax);drawSeries(context,1,color(paired?'--in':singleColor),indoorTop,indoorMax);const fullStart=current.series[0][0],fullEnd=current.series.at(-1)[0],rangeLabel=zoomStart!==null?'drag-selected zoom':start===null&&end===null?'entire history':'selected range',identity=paired?`${current.indoor_name} (indoor ${current.indoor_sensor_id}) paired with ${current.outdoor_name} (outdoor ${current.outdoor_sensor_id})`:current.page==='unpaired'?`Indoor sensor ${current.sensor_id}, not assigned to a pair`:`${current.sensor_name||singleType+' sensor'} (${singleType.toLowerCase()} ${current.sensor_id}), excluded from analysis and training`;status.textContent=`${identity}; showing ${visible.length.toLocaleString()} of ${current.series.length.toLocaleString()} hourly timestamps across the ${rangeLabel}, ${new Date(minTime*1000).toISOString()} to ${new Date(maxTime*1000).toISOString()}. Full span: ${new Date(fullStart*1000).toISOString()} to ${new Date(fullEnd*1000).toISOString()}.`;if(selectedTime===null||selectedTime<minTime||selectedTime>maxTime)selectedTime=minTime;inspect(selectedTime)}
function drawSeries(context,valueIndex,stroke,panelTop,maximum){context.strokeStyle=stroke;context.lineWidth=1.5;context.beginPath();let active=false,lastTime=null;visible.forEach(row=>{const value=row[valueIndex];if(!Number.isFinite(value)){active=false;return}const xx=geometry.x(row[0]),yy=geometry.y(value,panelTop,maximum);if(!active||lastTime!==null&&row[0]-lastTime>7200)context.moveTo(xx,yy);else context.lineTo(xx,yy);active=true;lastTime=row[0]});context.stroke()}
function inspect(timestamp){if(!geometry||!visible.length)return;const row=visible[nearestIndex(visible,timestamp)],context=overlayContext,xx=geometry.x(row[0]),singleOutdoor=current.sensor_type==='outdoor',points=current.paired?[[row[2],geometry.top,geometry.outdoorMax,'--out'],[row[1],geometry.indoorTop,geometry.indoorMax,'--in']]:[[row[1],geometry.indoorTop,geometry.indoorMax,singleOutdoor?'--out':'--in']];context.clearRect(0,0,geometry.width,geometry.height);context.strokeStyle=color('--fg');context.setLineDash([4,3]);context.beginPath();context.moveTo(xx,geometry.top);context.lineTo(xx,geometry.indoorTop+geometry.panelHeight);context.stroke();context.setLineDash([]);points.forEach(([value,panelTop,maximum,seriesColor])=>{if(!Number.isFinite(value))return;context.fillStyle=color(seriesColor);context.beginPath();context.arc(xx,geometry.y(value,panelTop,maximum),5,0,Math.PI*2);context.fill()});selectedTime=row[0];reading.innerHTML=`<strong>${new Date(row[0]*1000).toISOString()}</strong>${current.paired?`<span class="indoor">Indoor: ${formatPm25(row[1])}</span><span class="outdoor">Outdoor: ${formatPm25(row[2])}</span>`:`<span class="${singleOutdoor?'outdoor':'indoor'}">${singleOutdoor?'Outdoor':'Indoor'}: ${formatPm25(row[1])}</span>`}`}
function pointerX(event){const bounds=overlay.getBoundingClientRect();return Math.max(geometry.left,Math.min(geometry.width-geometry.right,event.clientX-bounds.left))}
function timeAtX(xx){return geometry.minTime+(xx-geometry.left)/(geometry.width-geometry.left-geometry.right)*(geometry.domainEnd-geometry.minTime)}
function pointerTime(event){return timeAtX(pointerX(event))}
function drawSelection(xx){const left=Math.min(drag.startX,xx),width=Math.abs(xx-drag.startX),top=geometry.top,height=geometry.indoorTop+geometry.panelHeight-top;overlayContext.clearRect(0,0,geometry.width,geometry.height);overlayContext.fillStyle=color('--select');overlayContext.fillRect(left,top,width,height);overlayContext.strokeStyle=color('--select-line');overlayContext.strokeRect(left+.5,top+.5,Math.max(0,width-1),height-1);reading.textContent=`Release to zoom: ${new Date(timeAtX(left)*1000).toISOString()} to ${new Date(timeAtX(left+width)*1000).toISOString()}.`}
overlay.addEventListener('pointerdown',event=>{if(!geometry||event.button!==0)return;drag={pointerId:event.pointerId,startX:pointerX(event),moved:false};pinned=false;overlay.setPointerCapture(event.pointerId);event.preventDefault()});
overlay.addEventListener('pointermove',event=>{if(!geometry)return;if(!drag){if(!pinned)inspect(pointerTime(event));return}if(event.pointerId!==drag.pointerId)return;const xx=pointerX(event);drag.moved=drag.moved||Math.abs(xx-drag.startX)>=6;if(drag.moved)drawSelection(xx);else inspect(timeAtX(xx))});
overlay.addEventListener('pointerup',event=>{if(!geometry||!drag||event.pointerId!==drag.pointerId)return;const completed=drag,xx=pointerX(event),moved=completed.moved||Math.abs(xx-completed.startX)>=6;drag=null;overlay.releasePointerCapture(event.pointerId);if(!moved){pinned=true;inspect(timeAtX(xx));return}const first=nearestIndex(visible,timeAtX(Math.min(completed.startX,xx))),last=nearestIndex(visible,timeAtX(Math.max(completed.startX,xx)));zoomStart=visible[Math.min(first,last)][0];zoomEnd=visible[Math.max(first,last)][0];selectedTime=zoomStart;pinned=false;draw()});
overlay.addEventListener('pointercancel',()=>{drag=null;if(geometry)inspect(selectedTime??geometry.minTime)});overlay.addEventListener('pointerleave',()=>{if(geometry&&!drag&&!pinned){overlayContext.clearRect(0,0,geometry.width,geometry.height);reading.textContent='Move across the chart, click to pin a reading, or drag to zoom.'}});new ResizeObserver(()=>draw()).observe(chart.parentElement);matchMedia('(prefers-color-scheme: dark)').addEventListener('change',draw);
function showPage(force=false){const hash=window.location.hash,next=hash==='#review'?'review':hash==='#recent'?'recent':hash==='#excluded'?'excluded':hash==='#unpaired'?'unpaired':'paired';if(next===page&&!force)return;page=next;const allLocations=page==='paired'?pairedLocations:page==='review'?reviewLocations:page==='recent'?recentExcludedLocations:page==='unpaired'?unpairedLocations:excludedLocations;locations=k12Only.checked?allLocations.filter(location=>location.k12):allLocations;current=null;pendingLocation=null;visible=[];geometry=null;locationInput.value='';locationList.replaceChildren();locations.forEach(location=>{const option=document.createElement('option');option.value=location.label;locationList.append(option)});const links={paired:pairedPage,review:reviewPage,recent:recentPage,unpaired:unpairedPage,excluded:excludedPage};Object.values(links).forEach(link=>link.removeAttribute('aria-current'));links[page].setAttribute('aria-current','page');locationLabel.textContent=['paired','review'].includes(page)?'Location':'Sensor';locationHelp.textContent=page==='review'?'Review only paired locations represented in the retained 1 km indoor/outdoor sources, including the isolated outdoor archive. Search by location name or sensor ID, or use the arrow buttons one location at a time.':page==='recent'?'These sensors have exclusion ranges added in the most recent reviewed data batch. Search by sensor name or ID; highlighted periods are excluded from analysis and training.':page==='paired'?'Search by location name or sensor ID. Use the arrow buttons to review locations one at a time. Blank dates show the entire available history. Drag across either chart and release to zoom; click without dragging to pin a reading.':page==='unpaired'?'Search by sensor ID or use the arrow buttons to review unpaired sensors one at a time. Blank dates show the entire available history; drag to zoom or click to pin a reading.':'Search by sensor name or ID. Use the arrow buttons to review excluded sensors one at a time. Highlighted periods are excluded from analysis and training.';outdoorLegend.hidden=page==='unpaired';indoorLegend.hidden=false;quality.hidden=true;invalidLegend.hidden=true;startInput.value=endInput.value='';status.textContent='';const empty=k12Only.checked?'No K-12 locations are available on this page.':page==='paired'?'No locations have both indoor and outdoor PurpleAir histories.':page==='review'?'No 1 km review sensors have paired histories.':page==='recent'?'No exclusions were added in the most recent reviewed data batch.':page==='unpaired'?'No unpaired sensors have downloaded indoor histories.':'No excluded sensors have downloaded histories.';reading.textContent=locations.length?'Select a sensor to inspect its readings.':empty;showError();updateNavigation();if(locations.length){locationInput.value=locations[0].label;loadLocation(locations[0])}else showError(empty)}
window.addEventListener('hashchange',()=>showPage());showPage();
</script>
</body>
</html>'''


def write_location_history_explorer(
    output: Path,
    pairs: list[dict[str, object]],
    indoor: dict[int, dict[int, float]],
    outdoor: dict[int, dict[int, float]],
    outdoor_exclusions: tuple[OutdoorExclusion, ...] = (),
    unpaired_sensor_ids: set[int] | tuple[int, ...] = (),
    permanent_exclusions: list[dict[str, object]]
    | tuple[dict[str, object], ...] = (),
    indoor_exclusions: tuple[OutdoorExclusion, ...] = (),
    k12_sensor_ids: set[int] | tuple[int, ...] = (),
    review_outdoor_ids: set[int] | tuple[int, ...] = (),
    review_indoor_ids: set[int] | tuple[int, ...] = (),
) -> int:
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "location_history_data"
    data_dir.mkdir(exist_ok=True)
    (
        locations,
        review_locations,
        recent_locations,
        unpaired_locations,
        excluded_locations,
        expected_files,
    ) = (
        [],
        [],
        [],
        [],
        [],
        set(),
    )
    latest_added_at = max(
        (
            exclusion.added_at_utc
            for exclusion in (*indoor_exclusions, *outdoor_exclusions)
            if exclusion.added_at_utc
        ),
        default="",
    )
    k12_outdoor_ids = {
        int(pair["outdoor_sensor_id"])
        for pair in pairs
        if int(pair["indoor_sensor_id"]) in k12_sensor_ids
    }

    def add_single(
        destination: list[dict[str, object]],
        page: str,
        sensor_type: str,
        sensor_id: int,
        sensor_name: str,
        values: dict[int, float],
        exclusions: tuple[OutdoorExclusion, ...] = (),
    ) -> dict[str, object] | None:
        if not values:
            return None
        filename = f"{page}_{sensor_type}_{sensor_id}.js"
        data = {
            "paired": False,
            "page": page,
            "data_file": filename,
            "sensor_id": sensor_id,
            "sensor_name": sensor_name,
            "sensor_type": sensor_type,
            "exclusions": [
                {
                    "start": exclusion.start,
                    "end": exclusion.end,
                    "reason": exclusion.reason,
                }
                for exclusion in exclusions
            ],
            "series": [
                [timestamp, value, None]
                for timestamp, value in sorted(values.items())
            ],
        }
        expected_files.add(filename)
        payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
        (data_dir / filename).write_text(
            f"window.__loadPurpleAirLocation({payload});\n", encoding="utf-8"
        )
        metadata = {
            "sensor_id": sensor_id,
            "file": filename,
            "k12": sensor_id
            in (k12_outdoor_ids if sensor_type == "outdoor" else k12_sensor_ids),
            "label": (
                f"{sensor_name} — {sensor_type} {sensor_id}"
                if sensor_name
                else f"{sensor_type.title()} sensor {sensor_id}"
            ),
        }
        destination.append(metadata)
        return metadata

    for pair in pairs:
        indoor_id = int(pair["indoor_sensor_id"])
        outdoor_id = int(pair["outdoor_sensor_id"])
        indoor_values = indoor.get(indoor_id, {})
        outdoor_values = outdoor.get(outdoor_id, {})
        if not indoor_values or not outdoor_values:
            continue
        timestamps = sorted(indoor_values.keys() | outdoor_values.keys())
        filename = f"{indoor_id}.js"
        data = {
            "paired": True,
            "page": "paired",
            "data_file": filename,
            "indoor_sensor_id": indoor_id,
            "indoor_name": pair["indoor_name"],
            "outdoor_sensor_id": outdoor_id,
            "outdoor_name": pair["outdoor_name"],
            "exclusions": [
                {
                    "start": exclusion.start,
                    "end": exclusion.end,
                    "reason": exclusion.reason,
                }
                for exclusion in (*indoor_exclusions, *outdoor_exclusions)
                if exclusion.sensor_id in {indoor_id, outdoor_id}
            ],
            "series": [
                [timestamp, indoor_values.get(timestamp), outdoor_values.get(timestamp)]
                for timestamp in timestamps
            ],
        }
        expected_files.add(filename)
        payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
        (data_dir / filename).write_text(
            f"window.__loadPurpleAirLocation({payload});\n", encoding="utf-8"
        )
        location = {
            "indoor_sensor_id": indoor_id,
            "file": filename,
            "k12": indoor_id in k12_sensor_ids,
            "label": (
                f"{pair['indoor_name']} ({float(pair['distance_meters']):.2f} m) — "
                f"indoor {indoor_id} / outdoor {outdoor_id}"
            ),
        }
        locations.append(location)
        if indoor_id in review_indoor_ids or outdoor_id in review_outdoor_ids:
            review_locations.append(location)
    for sensor_id in sorted(unpaired_sensor_ids):
        add_single(
            unpaired_locations,
            "unpaired",
            "indoor",
            sensor_id,
            "",
            indoor.get(sensor_id, {}),
        )
    for row in permanent_exclusions:
        sensor_id = int(row["sensor_id"])
        sensor_name, reason = str(row["sensor_name"]), str(row["reason"])
        add_single(
            excluded_locations,
            "excluded",
            "indoor",
            sensor_id,
            sensor_name,
            indoor.get(sensor_id, {}),
            (OutdoorExclusion(sensor_id, None, None, reason, sensor_name),),
        )
    permanent_ids = {int(row["sensor_id"]) for row in permanent_exclusions}
    ranges_by_indoor: dict[int, list[OutdoorExclusion]] = {}
    for exclusion in indoor_exclusions:
        ranges_by_indoor.setdefault(exclusion.sensor_id, []).append(exclusion)
    for sensor_id, exclusions in sorted(ranges_by_indoor.items()):
        if sensor_id not in permanent_ids:
            metadata = add_single(
                excluded_locations,
                "excluded",
                "indoor",
                sensor_id,
                exclusions[0].sensor_name,
                indoor.get(sensor_id, {}),
                tuple(exclusions),
            )
            if metadata and latest_added_at and any(
                item.added_at_utc == latest_added_at for item in exclusions
            ):
                recent_locations.append(metadata)
    ranges_by_sensor: dict[int, list[OutdoorExclusion]] = {}
    for exclusion in outdoor_exclusions:
        ranges_by_sensor.setdefault(exclusion.sensor_id, []).append(exclusion)
    for sensor_id, exclusions in sorted(ranges_by_sensor.items()):
        metadata = add_single(
            excluded_locations,
            "excluded",
            "outdoor",
            sensor_id,
            exclusions[0].sensor_name,
            outdoor.get(sensor_id, {}),
            tuple(exclusions),
        )
        if metadata and latest_added_at and any(
            item.added_at_utc == latest_added_at for item in exclusions
        ):
            recent_locations.append(metadata)
    for path in data_dir.glob("*.js"):
        if path.name not in expected_files:
            path.unlink()
    metadata = json.dumps(locations, separators=(",", ":")).replace("</", "<\\/")
    review_metadata = json.dumps(
        review_locations, separators=(",", ":")
    ).replace("</", "<\\/")
    recent_metadata = json.dumps(
        recent_locations, separators=(",", ":")
    ).replace("</", "<\\/")
    unpaired_metadata = json.dumps(
        unpaired_locations, separators=(",", ":")
    ).replace("</", "<\\/")
    excluded_metadata = json.dumps(
        excluded_locations, separators=(",", ":")
    ).replace("</", "<\\/")
    (output / "location_history_explorer.html").write_text(
        HTML.replace("__LOCATIONS__", metadata).replace(
            "__REVIEW_LOCATIONS__", review_metadata
        ).replace(
            "__RECENT_EXCLUDED_LOCATIONS__", recent_metadata
        ).replace(
            "__UNPAIRED_LOCATIONS__", unpaired_metadata
        ).replace("__EXCLUDED_LOCATIONS__", excluded_metadata),
        encoding="utf-8",
    )
    return len(locations)
