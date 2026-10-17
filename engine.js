(function(root){
"use strict";
const VERSION="2.0.0";
const SPARSE_THRESHOLD=0.018;
const STOP_WORDS=new Set(["the","and","for","this","that","with","from","about","into","when","what","where","does","could","would","should"]);
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function tokenise(value){return String(value||"").toLowerCase().match(/[a-z0-9]+/g)||[]}
function relevance(node,query){
 const words=tokenise(query).filter(w=>w.length>2&&!STOP_WORDS.has(w));
 if(!words.length)return 0;
 const corpus=tokenise([node.name,node.summary,node.source,node.type].join(" "));
 let hit=0;
 words.forEach(w=>{if(corpus.includes(w))hit++;else if(w.length>5&&corpus.some(t=>t.startsWith(w.slice(0,4))))hit+=.35});
 return clamp(hit/Math.max(2,words.length*.62),0,1);
}
function validState(s){return !!s&&["a","c","u"].every(k=>Number.isFinite(s[k])&&s[k]>=0&&s[k]<=1)}
function readState(node){return {a:node.a,c:node.c,u:node.u}}
function sameState(a,b,epsilon){return !!a&&!!b&&["a","c","u"].every(k=>Number.isFinite(a[k])&&Number.isFinite(b[k])&&Math.abs(a[k]-b[k])<=(epsilon===undefined?1e-6:epsilon))}
function validateGraph(nodes,adjacency){
 if(!Array.isArray(nodes))throw new TypeError("nodes must be an array");
 if(!adjacency||typeof adjacency!=="object")throw new TypeError("adjacency must be an object keyed by node id");
 const ids=new Set();
 nodes.forEach(n=>{if(!n||typeof n.id!=="string"||!n.id)throw new TypeError("every node needs a non-empty string id");if(ids.has(n.id))throw new TypeError("duplicate node id: "+n.id);ids.add(n.id);if(!validState(n))throw new TypeError("node "+n.id+" has invalid activation/confidence/uncertainty values")});
 nodes.forEach(n=>{const neighbors=adjacency[n.id]||[];if(!Array.isArray(neighbors))throw new TypeError("neighbors for "+n.id+" must be an array");neighbors.forEach(id=>{if(!ids.has(id))throw new TypeError("unknown neighbor "+id+" referenced by "+n.id)})});
 return ids;
}
function transition(nodes,adjacency,query,threshold){
 threshold=threshold===undefined?SPARSE_THRESHOLD:threshold;
 if(!Number.isFinite(threshold)||threshold<0)throw new TypeError("threshold must be a non-negative finite number");
 validateGraph(nodes,adjacency);
 const previous=Object.create(null),changes=[];
 nodes.forEach(n=>{previous[n.id]=n.a});
 let neighborReferences=0;
 nodes.forEach(n=>{
  const ids=adjacency[n.id]||[];neighborReferences+=ids.length;
  const neighbor=ids.length?ids.reduce((sum,id)=>sum+(previous[id]||0),0)/ids.length:n.a;
  const r=relevance(n,query),own=n.a,conf=n.c,unc=n.u;
  const h1=Math.tanh(.82*own+.43*neighbor+.25*conf-.39*unc+.72*r-.67);
  const h2=Math.tanh(-.24+.26*own+.77*neighbor-.16*unc+.61*r);
  const h3=Math.tanh(.61*conf+.37*neighbor-.7*unc+.41*r-.28);
  const a=clamp(.72*own+.28*(.5+.27*h1+.25*h2+.1*h3),.02,.99);
  const c=clamp(conf+.052*r+.013*(neighbor-.5)-.011*unc,.05,.99);
  const u=clamp(unc-.034*r+.008*(.5-neighbor),.03,.98);
  const delta=Math.abs(a-own)+Math.abs(c-conf)+Math.abs(u-unc);
  if(delta>=threshold)changes.push({nodeId:n.id,before:{a:own,c:conf,u:unc},after:{a,c,u},delta:Number(delta.toFixed(5)),relevance:Number(r.toFixed(3))});
 });
 return {changes,metrics:{engineVersion:VERSION,nodesEvaluated:nodes.length,neighborReferences,sparseWrites:changes.length,writeRatio:nodes.length?Number((changes.length/nodes.length).toFixed(4)):0,threshold,complexity:"O(V + E + text matching)"}};
}
function applyRecordedChanges(nodesById,changes,expectedSide,targetSide,strict){
 if(!nodesById||typeof nodesById!=="object")return {ok:false,applied:0,changes:[],reason:"A node lookup is required."};
 if(!Array.isArray(changes)||!["before","after"].includes(expectedSide)||!["before","after"].includes(targetSide)||expectedSide===targetSide)return {ok:false,applied:0,changes:[],reason:"The change batch or direction is invalid."};
 const proposals=[],seen=new Set();
 for(const ch of changes){
  if(!ch||typeof ch.nodeId!=="string"||seen.has(ch.nodeId))return {ok:false,applied:0,changes:[],reason:"The batch contains a missing or duplicate node reference."};
  seen.add(ch.nodeId);
  const node=nodesById[ch.nodeId],expected=ch[expectedSide],target=ch[targetSide];
  if(!node||!validState(expected)||!validState(target))return {ok:false,applied:0,changes:[],reason:"A node or recorded state is missing or invalid."};
  const current=readState(node);
  if(strict!==false&&!sameState(current,expected))return {ok:false,applied:0,changes:[],reason:"Current state for "+ch.nodeId+" no longer matches the event. No changes were applied."};
  proposals.push({node,change:{nodeId:ch.nodeId,before:current,after:{a:target.a,c:target.c,u:target.u},delta:Number((Math.abs(current.a-target.a)+Math.abs(current.c-target.c)+Math.abs(current.u-target.u)).toFixed(5))}});
 }
 proposals.forEach(p=>{p.node.a=p.change.after.a;p.node.c=p.change.after.c;p.node.u=p.change.after.u});
 return {ok:true,applied:proposals.length,changes:proposals.map(p=>p.change),reason:""};
}
root.LKMEngine={VERSION,SPARSE_THRESHOLD,clamp,tokenise,relevance,transition,applyRecordedChanges,validState,sameState};
})(typeof globalThis!=="undefined"?globalThis:this);
