(function(){
"use strict";
const $=s=>document.querySelector(s),E=window.LKMEngine;
function assert(ok,msg){if(!ok)throw new Error(msg||"Assertion failed")}
function clone(x){return JSON.parse(JSON.stringify(x))}
function fixture(){
 const nodes=[
  {id:"signal",name:"Holiday hours conflict",type:"SIGNAL",summary:"Two business schedules disagree",source:"Local discrepancy",a:.82,c:.76,u:.42},
  {id:"policy",name:"Holiday schedule policy",type:"POLICY",summary:"Effective-date policy for special hours",source:"Synthetic policy",a:.63,c:.88,u:.16},
  {id:"override",name:"Manager override",type:"EVIDENCE",summary:"Dated holiday hours submitted by a manager",source:"Synthetic override",a:.79,c:.84,u:.19},
  {id:"profile",name:"Published business profile",type:"EVIDENCE",summary:"Published opening schedule conflicts with override",source:"Synthetic profile",a:.76,c:.92,u:.12},
  {id:"review",name:"Needs human review",type:"ACTION",summary:"A reviewer confirms evidence before update",source:"Synthetic workflow",a:.71,c:.89,u:.23}
 ];
 const adjacency={signal:["policy","override","profile","review"],policy:["signal","override"],override:["signal","policy"],profile:["signal"],review:["signal"]};
 return {nodes,adjacency,byId:Object.fromEntries(nodes.map(n=>[n.id,n]))};
}
function sameJSON(a,b){return JSON.stringify(a)===JSON.stringify(b)}
function buildBenchGraph(size){
 const nodes=new Array(size),adjacency=Object.create(null);
 for(let i=0;i<size;i++){const rare=i%64===0;nodes[i]={id:"n"+i,name:rare?"Needle beacon marker "+i:"Ordinary record "+i,type:"FACT",summary:rare?"Signal for a sparse locality fixture":"Generic synthetic object for timing",source:"Synthetic benchmark",a:.2+(i%70)/100,c:.5+(i%40)/100,u:.1+(i%30)/100}}
 for(let i=0;i<size;i++){const neighbors=new Set();[i-1,i+1,i-7,i+7].forEach(j=>{if(j>=0&&j<size&&j!==i)neighbors.add("n"+j)});adjacency["n"+i]=Array.from(neighbors)}
 return {nodes,adjacency};
}
function runBenchmarks(){
 const wrap=$("#benchmarkRows");wrap.innerHTML="";
 [32,128,512,2048].forEach(size=>{
  const g=buildBenchGraph(size),q="needle beacon",index=E.buildIndex(g.nodes,g.adjacency);
  E.transition(g.nodes,g.adjacency,q);E.sparseTransition(g.nodes,g.adjacency,q,index);
  let denseTotal=0,sparseTotal=0,denseResult=null,sparseResult=null;
  const now=()=>window.performance&&performance.now?performance.now():Date.now();
  for(let i=0;i<3;i++){let t=now();denseResult=E.transition(g.nodes,g.adjacency,q);denseTotal+=now()-t;t=now();sparseResult=E.sparseTransition(g.nodes,g.adjacency,q,index);sparseTotal+=now()-t}
  const row=document.createElement("tr"),vals=[String(size),String(denseResult.metrics.nodesEvaluated),String(sparseResult.metrics.nodesEvaluated),String(sparseResult.metrics.nodesSkipped),String(sparseResult.metrics.sparseWrites),(denseTotal/3).toFixed(2)+" ms",(sparseTotal/3).toFixed(2)+" ms"];
  vals.forEach((v,i)=>{const td=document.createElement("td");td.textContent=v;if(i>0)td.className="mono";row.appendChild(td)});wrap.appendChild(row);
 });
 $("#benchmarkStamp").textContent="This browser · 3 runs after warm-up · index build excluded";
}
function runOne(name,fn){
 const row=document.createElement("div");row.className="test-row";
 try{const note=fn();row.innerHTML='<span class="test-icon pass">✓</span><div class="test-copy"><strong></strong><small>PASS</small></div>';row.querySelector("strong").textContent=name;if(note)row.querySelector("small").textContent=note;return {ok:true,row}}
 catch(e){row.innerHTML='<span class="test-icon fail">×</span><div class="test-copy"><strong></strong><small class="fail-text"></small></div>';row.querySelector("strong").textContent=name;row.querySelector("small").textContent=e&&e.message?e.message:String(e);return {ok:false,row}}
}
function runTests(){
 const tests=[
  ["Engine module is loaded and versioned",()=>assert(E&&E.VERSION==="3.0.0","Expected LKMEngine 3.0.0")],
  ["Relevance matching ignores letter case",()=>{const g=fixture();assert(E.relevance(g.nodes[0],"HOLIDAY HOURS")>0,"Relevant terms did not match")}],
  ["Same graph and signal produce deterministic dense output",()=>{const g=fixture(),a=E.transition(clone(g.nodes),g.adjacency,"holiday hours conflict"),b=E.transition(clone(g.nodes),g.adjacency,"holiday hours conflict");assert(sameJSON(a,b),"Dense runs returned different outputs")}],
  ["Dense reference does not mutate input nodes",()=>{const g=fixture(),before=clone(g.nodes);E.transition(g.nodes,g.adjacency,"holiday hours conflict");assert(sameJSON(before,g.nodes),"Input graph was mutated")}],
  ["Every proposed state is finite and bounded",()=>{const g=fixture();E.transition(g.nodes,g.adjacency,"holiday hours conflict").changes.forEach(ch=>assert(E.validState(ch.after),"Invalid next state: "+ch.nodeId))}],
  ["Sparse writes meet the configured threshold",()=>{const g=fixture();E.transition(g.nodes,g.adjacency,"holiday hours conflict").changes.forEach(ch=>assert(ch.delta>=E.SPARSE_THRESHOLD,"Sub-threshold write"))}],
  ["Each node appears at most once per transition",()=>{const g=fixture(),ids=E.transition(g.nodes,g.adjacency,"holiday hours conflict").changes.map(ch=>ch.nodeId);assert(new Set(ids).size===ids.length,"Duplicate node writes")}],
  ["Isolated node and empty signal are supported by dense reference",()=>{const n=[{id:"solo",name:"Quiet record",summary:"No edges",source:"Fixture",type:"FACT",a:.4,c:.7,u:.2}],r=E.transition(n,{solo:[]},"");assert(r.metrics.nodesEvaluated===1,"Isolated node not evaluated")}],
  ["Apply then reverse restores original state",()=>{const g=fixture(),before=clone(g.nodes),r=E.transition(g.nodes,g.adjacency,"holiday hours conflict");assert(r.changes.length>0,"No fixture changes");assert(E.applyRecordedChanges(g.byId,r.changes,"before","after",true).ok,"Apply failed");assert(E.applyRecordedChanges(g.byId,r.changes,"after","before",true).ok,"Reverse failed");assert(sameJSON(g.nodes,before),"State was not restored")}],
  ["Replay re-applies after-state after reversal",()=>{const g=fixture(),r=E.transition(g.nodes,g.adjacency,"holiday hours conflict");assert(E.applyRecordedChanges(g.byId,r.changes,"before","after",true).ok,"Apply failed");assert(E.applyRecordedChanges(g.byId,r.changes,"after","before",true).ok,"Reverse failed");assert(E.applyRecordedChanges(g.byId,r.changes,"before","after",true).ok,"Replay failed");r.changes.forEach(ch=>assert(E.sameState(g.byId[ch.nodeId],ch.after),"Replay mismatch"))}],
  ["Stale reversal is rejected atomically",()=>{const nodes=[{id:"a",a:.9,c:.8,u:.2},{id:"b",a:.3,c:.7,u:.3}],byId=Object.fromEntries(nodes.map(n=>[n.id,n])),changes=[{nodeId:"a",before:{a:.2,c:.7,u:.2},after:{a:.4,c:.8,u:.1}},{nodeId:"b",before:{a:.2,c:.6,u:.4},after:{a:.3,c:.7,u:.3}}],bBefore=clone(byId.b),out=E.applyRecordedChanges(byId,changes,"after","before",true);assert(!out.ok,"Stale batch was allowed");assert(sameJSON(byId.b,bBefore),"Partial update occurred")}],
  ["Unknown-node batch cannot partially mutate graph",()=>{const g=fixture(),before=clone(g.nodes),changes=[{nodeId:"signal",before:{a:.82,c:.76,u:.42},after:{a:.4,c:.7,u:.2}},{nodeId:"missing",before:{a:.1,c:.1,u:.1},after:{a:.2,c:.2,u:.2}}],r=E.applyRecordedChanges(g.byId,changes,"before","after",false);assert(!r.ok,"Unknown node was accepted");assert(sameJSON(g.nodes,before),"Partial write occurred")}],
  ["Duplicate node in an event batch is rejected",()=>{const g=fixture(),s={a:.82,c:.76,u:.42},ch=[{nodeId:"signal",before:s,after:{a:.83,c:.77,u:.41}},{nodeId:"signal",before:s,after:{a:.84,c:.78,u:.4}}];assert(!E.applyRecordedChanges(g.byId,ch,"before","after",false).ok,"Duplicate was accepted")}],
  ["Invalid graph state fails validation",()=>{let failed=false;try{E.transition([{id:"bad",a:2,c:.5,u:.2}],{bad:[]},"x")}catch(e){failed=String(e.message).includes("invalid")}assert(failed,"Invalid activation was not rejected")}],
  ["Index stores exact-token and prefix postings",()=>{const g=fixture(),idx=E.buildIndex(g.nodes,g.adjacency);assert(idx.byToken.holiday.includes("override"),"Exact token missing");assert(idx.byPrefix.conf.includes("profile"),"Four-character prefix posting missing")}],
  ["Sparse transition is deterministic and non-mutating",()=>{const g=buildBenchGraph(128),idx=E.buildIndex(g.nodes,g.adjacency),before=clone(g.nodes),a=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx),b=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx);assert(sameJSON(a,b),"Sparse results changed between runs");assert(sameJSON(before,g.nodes),"Sparse proposal mutated graph")}],
  ["Sparse scheduling evaluates a strict subset of a large graph",()=>{const g=buildBenchGraph(512),idx=E.buildIndex(g.nodes,g.adjacency),r=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx);assert(r.metrics.nodesEvaluated<g.nodes.length/2,"Sparse scheduler evaluated too much of the graph");assert(r.metrics.nodesSkipped===g.nodes.length-r.metrics.nodesEvaluated,"Skipped-node metric is inconsistent")}],
  ["Sparse transition skips all nodes for an unmatched signal",()=>{const g=buildBenchGraph(128),idx=E.buildIndex(g.nodes,g.adjacency),r=E.sparseTransition(g.nodes,g.adjacency,"xyzzynotfound",idx);assert(r.metrics.nodesEvaluated===0,"Unmatched signal evaluated nodes");assert(r.changes.length===0,"Unmatched signal proposed changes")}],
  ["Stale index is rejected instead of silently missing nodes",()=>{const g=fixture(),idx=E.buildIndex(g.nodes,g.adjacency);let failed=false;try{E.sparseTransition(g.nodes.concat([{id:"extra",name:"Extra",type:"FACT",summary:"",source:"",a:.2,c:.5,u:.1}]),g.adjacency,"holiday",idx)}catch(e){failed=String(e.message).includes("stale")}assert(failed,"A stale index was allowed")}],
  ["Changed-node frontier enables local multi-cycle propagation",()=>{const g=buildBenchGraph(128),idx=E.buildIndex(g.nodes,g.adjacency),first=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx),frontier=first.changes.map(ch=>ch.nodeId),next=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx,undefined,frontier);assert(next.metrics.frontierNodes===frontier.length,"Frontier size was not reported");assert(next.metrics.nodesEvaluated>=first.metrics.nodesEvaluated,"Frontier unexpectedly reduced its candidate set")}],
  ["Scaling fixture reports dense and sparse evaluation counts",()=>{const g=buildBenchGraph(24),idx=E.buildIndex(g.nodes,g.adjacency),dense=E.transition(g.nodes,g.adjacency,"needle beacon"),sparse=E.sparseTransition(g.nodes,g.adjacency,"needle beacon",idx);assert(dense.metrics.nodesEvaluated===24,"Wrong dense count");assert(sparse.metrics.nodesEvaluated>0&&sparse.metrics.nodesEvaluated<24,"Sparse count did not reduce evaluation")}]
 ];
 const results=tests.map(t=>runOne(t[0],t[1]));const list=$("#testResults");list.innerHTML="";results.forEach(r=>list.appendChild(r.row));
 const passed=results.filter(r=>r.ok).length;$("#passCount").textContent=String(passed);$("#testCount").textContent=String(results.length);
 $("#testSummary").textContent=passed===results.length?"All checks passed":"Some checks need attention";
 $("#testState").className="status "+(passed===results.length?"good":"bad");$("#testState").textContent=passed===results.length?"PASS":"REVIEW";
 $("#ranAt").textContent="Last run: "+new Date().toLocaleTimeString();runBenchmarks();
}
$("#runTests").addEventListener("click",runTests);
runTests();
})();