(function(root){
"use strict";
const VERSION="3.0.0", INDEX_VERSION="3.0.0", SPARSE_THRESHOLD=0.018;
const STOP_WORDS=new Set(["the","and","for","this","that","with","from","about","into","when","what","where","does","could","would","should"]);
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function tokenise(value){return String(value||"").toLowerCase().match(/[a-z0-9]+/g)||[]}
function queryWords(value){return tokenise(value).filter(w=>w.length>2&&!STOP_WORDS.has(w))}
function relevance(node,query){
 const words=queryWords(query);if(!words.length)return 0;
 const corpus=tokenise([node.name,node.summary,node.source,node.type].join(" "));
 let hit=0;words.forEach(w=>{if(corpus.includes(w))hit++;else if(w.length>5&&corpus.some(t=>t.startsWith(w.slice(0,4))))hit+=.35});
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
function validateThreshold(threshold){threshold=threshold===undefined?SPARSE_THRESHOLD:threshold;if(!Number.isFinite(threshold)||threshold<0)throw new TypeError("threshold must be a non-negative finite number");return threshold}
function proposeNode(n,neighbor,r,threshold){
 const own=n.a,conf=n.c,unc=n.u;
 const h1=Math.tanh(.82*own+.43*neighbor+.25*conf-.39*unc+.72*r-.67);
 const h2=Math.tanh(-.24+.26*own+.77*neighbor-.16*unc+.61*r);
 const h3=Math.tanh(.61*conf+.37*neighbor-.7*unc+.41*r-.28);
 const a=clamp(.72*own+.28*(.5+.27*h1+.25*h2+.1*h3),.02,.99);
 const c=clamp(conf+.052*r+.013*(neighbor-.5)-.011*unc,.05,.99);
 const u=clamp(unc-.034*r+.008*(.5-neighbor),.03,.98);
 const delta=Math.abs(a-own)+Math.abs(c-conf)+Math.abs(u-unc);
 return delta<threshold?null:{nodeId:n.id,before:{a:own,c:conf,u:unc},after:{a,c,u},delta:Number(delta.toFixed(5)),relevance:Number(r.toFixed(3))};
}
/* Build once for a static graph. Rebuild after changing text or topology. */
function buildIndex(nodes,adjacency){
 adjacency=adjacency||Object.create(null);validateGraph(nodes,adjacency);
 const nodeById=Object.create(null),neighbors=Object.create(null),incidentSets=Object.create(null),byToken=Object.create(null),byPrefix=Object.create(null);
 nodes.forEach(n=>{nodeById[n.id]=n;neighbors[n.id]=(adjacency[n.id]||[]).slice();incidentSets[n.id]=new Set()});
 nodes.forEach(n=>{
  neighbors[n.id].forEach(id=>{incidentSets[n.id].add(id);incidentSets[id].add(n.id)});
  const terms=new Set(tokenise([n.name,n.summary,n.source,n.type].join(" ")));
  terms.forEach(word=>{(byToken[word]||(byToken[word]=[])).push(n.id);if(word.length>=4){const prefix=word.slice(0,4);(byPrefix[prefix]||(byPrefix[prefix]=[])).push(n.id)}});
 });
 const incident=Object.create(null);Object.keys(incidentSets).forEach(id=>incident[id]=Array.from(incidentSets[id]));
 return {version:INDEX_VERSION,nodeCount:nodes.length,nodesRef:nodes,adjacencyRef:adjacency,nodeIds:nodes.map(n=>n.id),nodeById,neighbors,incident,byToken,byPrefix};
}
function transition(nodes,adjacency,query,threshold){
 threshold=validateThreshold(threshold);validateGraph(nodes,adjacency);
 const changes=[],previous=Object.create(null);let neighborReferences=0;
 nodes.forEach(n=>{previous[n.id]=n.a});
 nodes.forEach(n=>{
  const ids=adjacency[n.id]||[];neighborReferences+=ids.length;
  const neighbor=ids.length?ids.reduce((sum,id)=>sum+previous[id],0)/ids.length:n.a;
  const change=proposeNode(n,neighbor,relevance(n,query),threshold);if(change)changes.push(change);
 });
 return {changes,metrics:{engineVersion:VERSION,mode:"dense-reference",totalNodes:nodes.length,candidateSeeds:nodes.length,candidateNodes:nodes.length,nodesEvaluated:nodes.length,nodesSkipped:0,frontierNodes:0,neighborReferences,sparseWrites:changes.length,writeRatio:nodes.length?Number((changes.length/nodes.length).toFixed(4)):0,graphWriteRatio:nodes.length?Number((changes.length/nodes.length).toFixed(4)):0,coverageRatio:nodes.length?1:0,threshold,complexity:"O(V + E + text matching)"}};
}
/* Indexed sparse scheduler: match text, include its neighborhood and optional prior changed-node frontier. */
function sparseTransition(nodes,adjacency,query,index,threshold,frontierIds){
 threshold=validateThreshold(threshold);
 if(!Array.isArray(nodes)||!adjacency||typeof adjacency!=="object")throw new TypeError("nodes and adjacency are required");
 if(!index||index.version!==INDEX_VERSION)throw new TypeError("A V3 graph index is required.");
 if(index.nodesRef!==nodes||index.adjacencyRef!==adjacency||index.nodeCount!==nodes.length)throw new TypeError("The graph index is stale. Rebuild it after changing nodes or topology.");
 const seeds=new Set(),words=queryWords(query);
 words.forEach(word=>{(index.byToken[word]||[]).forEach(id=>seeds.add(id));if(word.length>5)(index.byPrefix[word.slice(0,4)]||[]).forEach(id=>seeds.add(id))});
 const base=new Set(seeds),frontier=new Set();
 if(Array.isArray(frontierIds))frontierIds.forEach(id=>{if(index.nodeById[id]){base.add(id);frontier.add(id)}});
 const candidateIds=new Set(base);base.forEach(id=>(index.incident[id]||[]).forEach(neighborId=>candidateIds.add(neighborId)));
 const changes=[];let neighborReferences=0;
 candidateIds.forEach(id=>{
  const n=index.nodeById[id],ids=index.neighbors[id]||[];neighborReferences+=ids.length;
  const neighbor=ids.length?ids.reduce((sum,neighborId)=>sum+index.nodeById[neighborId].a,0)/ids.length:n.a;
  const change=proposeNode(n,neighbor,relevance(n,query),threshold);if(change)changes.push(change);
 });
 const total=nodes.length,evaluated=candidateIds.size;
 return {changes,metrics:{engineVersion:VERSION,mode:"indexed-sparse",totalNodes:total,candidateSeeds:seeds.size,candidateNodes:evaluated,nodesEvaluated:evaluated,nodesSkipped:total-evaluated,frontierNodes:frontier.size,neighborReferences,sparseWrites:changes.length,writeRatio:evaluated?Number((changes.length/evaluated).toFixed(4)):0,graphWriteRatio:total?Number((changes.length/total).toFixed(4)):0,coverageRatio:total?Number((evaluated/total).toFixed(4)):0,threshold,complexity:"O(query postings + C + E_C)",indexBuildComplexity:"O(V + E + tokens) once per graph"}};
}
function applyRecordedChanges(nodesById,changes,expectedSide,targetSide,strict){
 if(!nodesById||typeof nodesById!=="object")return {ok:false,applied:0,changes:[],reason:"A node lookup is required."};
 if(!Array.isArray(changes)||!["before","after"].includes(expectedSide)||!["before","after"].includes(targetSide)||expectedSide===targetSide)return {ok:false,applied:0,changes:[],reason:"The change batch or direction is invalid."};
 const proposals=[],seen=new Set();
 for(const ch of changes){
  if(!ch||typeof ch.nodeId!=="string"||seen.has(ch.nodeId))return {ok:false,applied:0,changes:[],reason:"The batch contains a missing or duplicate node reference."};
  seen.add(ch.nodeId);const node=nodesById[ch.nodeId],expected=ch[expectedSide],target=ch[targetSide];
  if(!node||!validState(expected)||!validState(target))return {ok:false,applied:0,changes:[],reason:"A node or recorded state is missing or invalid."};
  const current=readState(node);
  if(strict!==false&&!sameState(current,expected))return {ok:false,applied:0,changes:[],reason:"Current state for "+ch.nodeId+" no longer matches the event. No changes were applied."};
  proposals.push({node,change:{nodeId:ch.nodeId,before:current,after:{a:target.a,c:target.c,u:target.u},delta:Number((Math.abs(current.a-target.a)+Math.abs(current.c-target.c)+Math.abs(current.u-target.u)).toFixed(5))}});
 }
 proposals.forEach(p=>{p.node.a=p.change.after.a;p.node.c=p.change.after.c;p.node.u=p.change.after.u});
 return {ok:true,applied:proposals.length,changes:proposals.map(p=>p.change),reason:""};
}
root.LKMEngine={VERSION,INDEX_VERSION,SPARSE_THRESHOLD,clamp,tokenise,relevance,transition,sparseTransition,buildIndex,applyRecordedChanges,validState,sameState};
})(typeof globalThis!=="undefined"?globalThis:this);
