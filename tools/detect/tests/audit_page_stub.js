// A DOM small enough to run the audit page's script and large enough to catch
// it lying. Everything the page touches is here; anything it reaches for that
// is not becomes a TypeError, which is the point.
var els = {}, listeners = {};
function mk(id){
  return {id:id, textContent:'', innerHTML:'', className:'', title:'',
    hidden:false, disabled:false, style:{}, children:[], value:'', src:'',
    dataset:{},
    setAttribute:function(){}, getAttribute:function(){return null},
    addEventListener:function(t,f){(listeners[id]=listeners[id]||{})[t]=f},
    querySelectorAll:function(){return []}, scrollIntoView:function(){},
    classList:{toggle:function(){}, add:function(){}, remove:function(){}},
    closest:function(){return null}, select:function(){},
    setSelectionRange:function(){}};
}
function E(id){ return els[id] || (els[id] = mk(id)) }
global.document = {getElementById:E, createElement:function(){return mk('new')},
  addEventListener:function(t,f){(listeners.doc = listeners.doc || {})[t] = f},
  body:{appendChild:function(){}, removeChild:function(){}},
  execCommand:function(){return true}};
global.window = {isSecureContext:false};
global.setTimeout = function(){return 1};
global.clearTimeout = function(){};
var FETCHES = [];
// Enough of a thenable to carry the page's .then().then().catch() chains
// without pretending to be async -- the assertions run after a synchronous
// boot, so nothing has to be awaited.
global.fetch = function(u, o){
  FETCHES.push(u);
  return {then:function(f){
    var r = f({json:function(){return RESP(u)}});
    return {then:function(g){ g && g(r); return {catch:function(){return {}}} },
            catch:function(){ return {then:function(){return {}}} }};
  }};
};
function RESP(u){
  if (/audit\/page/.test(u)) return {page:PAGE, index:0, total:1};
  if (/audit\/stats/.test(u)) return STATS;
  if (/verdict/.test(u)) return {ok:true};
  if (/draw/.test(u)) return {page:PAGE, index:0, total:1};
  return {};
}
var PAGE = {index:0, dropped:0, items:[
  {key:'111#0', image_id:'111', det_idx:0, p_dog:0.0,  conf:0.31, band:0,
   seq:'s1', drive:'lynx',   cell:'c'},
  {key:'222#1', image_id:'222', det_idx:1, p_dog:0.47, conf:0.90, band:4,
   seq:'s2', drive:'bobcat', cell:'c'}]};
var STATS = {judged:12, missed:1, weighted_rate:0.0031, pool:3945390,
  covered:0.9, bands:[
  {lo:0.0, hi:0.1, judged:8, missed:0, rate:0.0,  boxes:3530147},
  {lo:0.1, hi:0.2, judged:0, missed:0, rate:0.0,  boxes:162866},
  {lo:0.2, hi:0.3, judged:0, missed:0, rate:0.0,  boxes:102513},
  {lo:0.3, hi:0.4, judged:0, missed:0, rate:0.0,  boxes:80241},
  {lo:0.4, hi:0.5, judged:4, missed:1, rate:0.25, boxes:69623}]};
