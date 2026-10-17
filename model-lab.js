"use strict";
(function(){
const $=s=>document.querySelector(s);
const MODEL_ID="onnx-community/SmolLM2-360M-Instruct-ONNX";
const TRANSFORMERS_URL="https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
const SYSTEM_PROMPT='You are a cautious evidence-triage component. Return ONLY one JSON object with exactly these keys: decision, conflict, proposed_closing_time, reason. decision must be one of "review", "consistent", "insufficient". conflict must be a JSON boolean. proposed_closing_time must be a normalized time such as "6 PM" or JSON null. If comparable source records for the same business and effective date disagree, use decision="review", conflict=true, and proposed_closing_time=null. If comparable records agree, use decision="consistent", conflict=false, and proposed_closing_time set to the shared time. If only one comparable record exists or dates cannot be compared, use decision="insufficient", conflict=false, proposed_closing_time=null. Never decide which source is authoritative. Keep reason brief. Never add markdown fences.';
const CASES=[
 {id:"same-date-conflict",label:"Same date · sources disagree",prompt:"Business: Brooklyn North. Effective date: 2026-10-31. Published profile for this exact business and date: closes 8 PM. Manager override for this exact business and date: closes 6 PM. Do the comparable sources disagree?",expected:{decision:"review",conflict:true,proposed_closing_time:null}},
 {id:"same-value",label:"Same date · same time",prompt:"Business: River Market. Effective date: 2026-11-02. Published profile for this exact business and date: closes 6 PM. Manager override for this exact business and date: closes 6 PM. Do the comparable sources disagree?",expected:{decision:"consistent",conflict:false,proposed_closing_time:"6 PM"}},
 {id:"different-dates",label:"Different effective dates",prompt:"Business: Central Point. Published profile: closes 8 PM on 2026-10-20. Manager override: closes 6 PM on 2026-10-21. The target date to evaluate is 2026-10-20. The two values apply to different dates. Are they a confirmed conflict for one effective date?",expected:{decision:"insufficient",conflict:false,proposed_closing_time:null}},
 {id:"single-source",label:"Only one source",prompt:"Business: West Gate. Target effective date: 2026-11-04. Only one source is available: an undated note says closes 7 PM. There is no second source to compare. Is there a confirmed conflict?",expected:{decision:"insufficient",conflict:false,proposed_closing_time:null}},
 {id:"equivalent-time",label:"Equivalent time formats",prompt:"Business: Cedar Square. Effective date: 2026-11-07. Published profile says closes at 18:00. Manager override for the same business and date says closes at 6 PM. These are two written formats for the same time. Do the sources disagree?",expected:{decision:"consistent",conflict:false,proposed_closing_time:"6 PM"}}
];
let generator=null,loadedDevice=null,loadedDtype=null,loadMs=null,busy=false,suiteResults=[],customResult=null,transformersEnv=null,basePageMemoryBytes=null,afterLoadPageMemoryBytes=null,highestPageMemoryBytes=null,memorySamples=[];
function setStatus(selector,text,kind){const e=$(selector);e.textContent=text;e.className="status"+(kind?" "+kind:"")}
function setBusy(v){busy=v;$("#loadBtn").disabled=v;$("#inferBtn").disabled=v||!generator;$("#suiteBtn").disabled=v||!generator;$("#runtimeChoice").disabled=v}
function updateStorage(){
 if(!navigator.storage||!navigator.storage.estimate){$("#storageFact").textContent="Not available";return}
 navigator.storage.estimate().then(x=>{const mb=x.usage?x.usage/1048576:0;$("#storageFact").textContent=typeof x.usage==="number"?mb.toFixed(0)+" MB":"Not available"}).catch(()=>{$("#storageFact").textContent="Not available"});
}
function fmtBytes(bytes){return (bytes/1048576).toFixed(1)+" MB"}
async function samplePageMemory(){
 if(!window.crossOriginIsolated||!window.performance||typeof performance.measureUserAgentSpecificMemory!=="function")return null;
 try{const x=await performance.measureUserAgentSpecificMemory();if(Number.isFinite(x.bytes)){highestPageMemoryBytes=highestPageMemoryBytes===null?x.bytes:Math.max(highestPageMemoryBytes,x.bytes);memorySamples.push({time:new Date().toISOString(),bytes:x.bytes});return x.bytes}}catch(e){}
 return null;
}
function updatePageMemoryFact(currentBytes){
 const el=$("#pageMemoryFact");
 if(!window.crossOriginIsolated){el.textContent="Isolation unavailable";return}
 if(!window.performance||typeof performance.measureUserAgentSpecificMemory!=="function"){el.textContent="API unavailable";return}
 if(basePageMemoryBytes===null){el.textContent="Sampling…";return}
 const current=currentBytes===undefined?afterLoadPageMemoryBytes:currentBytes;
 if(current===null||current===undefined){el.textContent="Sample unavailable";return}
 const delta=current-basePageMemoryBytes;
 el.textContent=fmtBytes(current)+" (Δ "+(delta>=0?"+":"−")+fmtBytes(Math.abs(delta))+")";
}
function initDeviceFacts(){
 $("#webgpuFact").textContent=navigator.gpu?"Available":"Not available";
 if(typeof navigator.deviceMemory==="number")$("#deviceRamFact").textContent="~"+navigator.deviceMemory+" GB";
 updateStorage();
}
function progressCallback(p){
 if(!p||typeof p!=="object")return;
 const total=Number(p.total)||0,loaded=Number(p.loaded)||0,percentage=Number.isFinite(p.progress)?p.progress:(total?loaded/total*100:0);
 if(p.status==="progress"&&total>0){
  $("#loadProgress").style.width=Math.max(0,Math.min(100,percentage))+"%";
  $("#loadStatus").textContent="Downloading/caching "+(p.file||p.name||"model assets")+" · "+Math.max(0,Math.min(100,percentage)).toFixed(0)+"%";
 }else if(p.status==="initiate"){
  $("#loadStatus").textContent="Preparing "+(p.file||p.name||"model asset")+"…";
 }
}
async function loadModel(forceDevice){
 if(busy)return;
 setBusy(true);$("#retryCpuBtn").classList.add("hide");$("#loadProgress").style.width="0%";
 const choice=forceDevice||$("#runtimeChoice").value;
 if(choice==="webgpu"&&!navigator.gpu){
  setStatus("#loadStatus","WebGPU is not exposed by this browser. Select CPU/WASM to load the larger q4 artifact.","warn");setBusy(false);return;
 }
 const device=(choice==="wasm"?"wasm":(choice==="webgpu"?"webgpu":(navigator.gpu?"webgpu":"wasm")));
 const dtype=device==="webgpu"?"q4f16":"q4";
 if(basePageMemoryBytes===null)basePageMemoryBytes=await samplePageMemory();
 const started=performance.now();
 setStatus("#loadStatus","Importing Transformers.js and preparing "+device+" runtime. The first load downloads model files; keep this tab open.","");
 try{
  const lib=await import(TRANSFORMERS_URL);
  transformersEnv=lib.env;
  if(transformersEnv){transformersEnv.useBrowserCache=true;transformersEnv.allowLocalModels=false;}
  generator=await lib.pipeline("text-generation",MODEL_ID,{device,dtype,progress_callback:progressCallback});
  loadedDevice=device;loadedDtype=dtype;loadMs=performance.now()-started;
  $("#readyTag").textContent="MODEL READY";
  $("#readyTag").style.color="var(--mint)";
  $("#readyTag").style.borderColor="#31564e";
  $("#loadProgress").style.width="100%";
  setStatus("#loadStatus","Loaded "+MODEL_ID+" · "+device.toUpperCase()+" · "+dtype+" · initialization "+(loadMs/1000).toFixed(1)+" s. Timer includes download/cache and runtime setup.","good");
  $("#runtimeFact").textContent="Loaded in this tab using "+device+" and "+dtype+".";
  $("#customOutput").textContent="Ready. Run a sample prompt to see the model's raw generated response.";
  $("#inferBtn").disabled=false;$("#suiteBtn").disabled=false;$("#exportBtn").disabled=!(suiteResults.length||customResult);
  updateStorage();
 }catch(err){
  generator=null;loadedDevice=null;loadedDtype=null;loadMs=null;
  $("#readyTag").textContent="LOAD FAILED";$("#readyTag").style.color="var(--red)";$("#readyTag").style.borderColor="#633840";
  setStatus("#loadStatus","Could not initialize "+device+"/"+dtype+": "+(err&&err.message?err.message:String(err))+". Retry or choose CPU/WASM; that may download a different quantized file.","bad");
  if(device==="webgpu")$("#retryCpuBtn").classList.remove("hide");
 }finally{setBusy(false)}
}
function getText(out){
 let x=Array.isArray(out)?out[0]?.generated_text:(out&&out.generated_text!==undefined?out.generated_text:out);
 if(typeof x==="string")return x.trim();
 if(Array.isArray(x)){for(let i=x.length-1;i>=0;i--){if(x[i]&&x[i].role==="assistant"&&typeof x[i].content==="string")return x[i].content.trim()}const last=x[x.length-1];return last&&typeof last.content==="string"?last.content.trim():JSON.stringify(x)}
 if(x&&typeof x.content==="string")return x.content.trim();
 return String(x??"");
}
async function inferPrompt(prompt){
 if(!generator)throw new Error("Load the model first.");
 const started=performance.now();
 const result=await generator([{role:"system",content:SYSTEM_PROMPT},{role:"user",content:prompt}],{max_new_tokens:112,do_sample:false,repetition_penalty:1.04});
 const latency=performance.now()-started,text=getText(result);
 const pageMemoryBytes=await samplePageMemory();if(pageMemoryBytes!==null)updatePageMemoryFact(pageMemoryBytes);
 return {text,latencyMs:Number(latency.toFixed(1)),charsPerSecond:Number((text.length/(latency/1000||1)).toFixed(1)),pageMemoryBytes};
}
function normalTime(v){
 if(v===null||v===undefined)return null;
 let s=String(v).trim().replace(/^["']|["']$/g,"");
 if(!s||/^(null|none|n\/a|unknown|not applicable)$/i.test(s))return null;
 let m=s.match(/\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b/i);
 if(m){let hr=Number(m[1]);const suffix=m[3].toLowerCase().startsWith("a")?"AM":"PM";return (hr%12||12)+" "+suffix}
 m=s.match(/\b(\d{1,2}):(\d{2})\b/);
 if(m){const hour=Number(m[1]),minute=Number(m[2]);if(hour<=23&&minute<=59){const hr=hour%12||12;return hr+" "+(hour>=12?"PM":"AM")}}
 return s.toUpperCase();
}
function parseJSON(text){
 let s=String(text||"").replace(/﻿/g,"").trim();
 const fenced=s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);if(fenced)s=fenced[1];
 const start=s.indexOf("{"),end=s.lastIndexOf("}");
 if(start<0||end<=start)throw new Error("No JSON object found.");
 const obj=JSON.parse(s.slice(start,end+1));
 const valid=!!obj&&typeof obj==="object"&&!Array.isArray(obj)&&["review","consistent","insufficient"].includes(String(obj.decision||"").toLowerCase())&&typeof obj.conflict==="boolean"&&Object.prototype.hasOwnProperty.call(obj,"proposed_closing_time")&&(obj.proposed_closing_time===null||typeof obj.proposed_closing_time==="string")&&typeof obj.reason==="string";
 const out={decision:String(obj.decision||"").toLowerCase(),conflict:obj.conflict,proposed_closing_time:normalTime(obj.proposed_closing_time),reason:typeof obj.reason==="string"?obj.reason:""};
 return {valid,object:out,rawObject:obj};
}
function expectedText(x){return x.decision+" · conflict="+x.conflict+" · time="+(x.proposed_closing_time===null?"null":x.proposed_closing_time)}
function actualText(r){
 if(!r.parsed)return "Not parseable";
 const x=r.parsed.object;return expectedText(x);
}
function isExact(parsed,expected){
 if(!parsed||!parsed.valid)return false;
 const x=parsed.object;return x.decision===expected.decision&&x.conflict===expected.conflict&&x.proposed_closing_time===expected.proposed_closing_time;
}
function renderRows(){
 const body=$("#resultRows");body.innerHTML="";
 if(!suiteResults.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=6;td.textContent="No tests run.";tr.appendChild(td);body.appendChild(tr);return}
 suiteResults.forEach((r,i)=>{
  const tr=document.createElement("tr"),values=[CASES[i].label,expectedText(CASES[i].expected),actualText(r),r.parsed&&r.parsed.valid?"Valid":"Invalid",r.exact?"PASS":"FAIL",r.latencyMs.toFixed(1)+" ms"];
  values.forEach((v,j)=>{const td=document.createElement("td");td.textContent=v;if((j===3&&r.parsed&&r.parsed.valid)||(j===4&&r.exact))td.className="pass";else if((j===3&&!r.parsed?.valid)||(j===4&&!r.exact))td.className="fail";tr.appendChild(td)});
  body.appendChild(tr);
 });
 const n=suiteResults.length,exact=suiteResults.filter(r=>r.exact).length,schema=suiteResults.filter(r=>r.parsed&&r.parsed.valid).length;
 const lat=[...suiteResults].map(r=>r.latencyMs).sort((a,b)=>a-b),median=n%2?lat[(n-1)/2]:(lat[n/2-1]+lat[n/2])/2;
 const chars=suiteResults.reduce((s,r)=>s+r.text.length,0),secs=suiteResults.reduce((s,r)=>s+r.latencyMs,0)/1000;
 $("#metricExact").textContent=exact+"/"+n;$("#metricSchema").textContent=schema+"/"+n;$("#metricLatency").textContent=median.toFixed(0)+" ms";$("#metricRate").textContent=(chars/(secs||1)).toFixed(0);
 $("#exportBtn").disabled=false;
 setStatus("#suiteStatus",exact===n?"All five decisions exactly matched the synthetic labels. This small suite is not a general quality guarantee.":exact+" of "+n+" decisions matched the synthetic labels. Inspect each response and expand the labelled set before using this model for decisions.",exact===n?"good":"warn");
}
async function runCustom(){
 if(busy||!generator)return;
 const prompt=$("#customPrompt").value.trim();if(!prompt){setStatus("#customStatus","Enter a prompt first.","warn");return}
 setBusy(true);$("#customOutput").textContent="Generating locally…";$("#customParsed").textContent="Waiting for model output…";
 try{
  customResult=await inferPrompt(prompt);$("#customOutput").textContent=customResult.text;
  try{const parsed=parseJSON(customResult.text);customResult.parsed=parsed;$("#customParsed").textContent=JSON.stringify({schemaValid:parsed.valid,...parsed.object},null,2);setStatus("#customStatus","Inference "+customResult.latencyMs.toFixed(1)+" ms · "+customResult.charsPerSecond+" output chars/s. Check the facts and source records; model output is advisory.",parsed.valid?"good":"warn")}
  catch(e){customResult.parseError=e.message;$("#customParsed").textContent=e.message;setStatus("#customStatus","Generated text is not valid structured output. This is a quality failure, not a successful decision.","warn")}
  $("#exportBtn").disabled=!(suiteResults.length||customResult);
 }catch(err){setStatus("#customStatus","Inference failed: "+(err&&err.message?err.message:String(err)),"bad")}
 finally{setBusy(false)}
}
async function runSuite(){
 if(busy||!generator)return;
 suiteResults=[];renderRows();setBusy(true);setStatus("#suiteStatus","Running synthetic test 1 of "+CASES.length+" locally…","");
 $("#metricExact").textContent="—";$("#metricSchema").textContent="—";$("#metricLatency").textContent="—";$("#metricRate").textContent="—";
 try{
  for(let i=0;i<CASES.length;i++){
   setStatus("#suiteStatus","Running synthetic test "+(i+1)+" of "+CASES.length+" locally…","");
   const r={caseId:CASES[i].id,expected:CASES[i].expected,prompt:CASES[i].prompt};
   try{Object.assign(r,await inferPrompt(CASES[i].prompt));try{r.parsed=parseJSON(r.text)}catch(e){r.parseError=e.message}r.exact=isExact(r.parsed,CASES[i].expected)}
   catch(err){r.text="";r.latencyMs=0;r.charsPerSecond=0;r.error=err&&err.message?err.message:String(err);r.exact=false}
   suiteResults.push(r);renderRows();
  }
  renderRows();
 }finally{setBusy(false);$("#exportBtn").disabled=!(suiteResults.length||customResult)}
}
function exportResults(){
 const payload={format:"lkm-local-model-evaluation-v1",exportedAt:new Date().toISOString(),modelId:MODEL_ID,transformersJs:"4.3.1",runtime:loadedDevice,dtype:loadedDtype,modelLoadMs:loadMs,measurementNotes:{peakProcessRam:"not exposed reliably to this browser page",pageMemory:"optional performance.measureUserAgentSpecificMemory samples of whole-page use; may omit GPU allocations and must not be treated as peak model RSS",deviceMemory:"coarse browser-reported device RAM only",originStorage:"approximate origin storage usage, not exact model cache size",quality:"small synthetic labelled suite; not a production accuracy evaluation"},memory:{crossOriginIsolated:!!window.crossOriginIsolated,baselineBytes:basePageMemoryBytes,afterLoadBytes:afterLoadPageMemoryBytes,highestSampledPageBytes:highestPageMemoryBytes,samples:memorySamples},suite:suiteResults,customInference:customResult?{prompt:$("#customPrompt").value.trim(),...customResult}:null};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}),url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="lkm-local-model-evaluation.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$("#loadBtn").addEventListener("click",()=>loadModel());
$("#retryCpuBtn").addEventListener("click",()=>{ $("#runtimeChoice").value="wasm";loadModel("wasm") });
$("#inferBtn").addEventListener("click",runCustom);
$("#suiteBtn").addEventListener("click",runSuite);
$("#exportBtn").addEventListener("click",exportResults);
window.LKMModelLab={MODEL_ID,CASES,normalTime,parseJSON,getText};
initDeviceFacts();renderRows();
})();