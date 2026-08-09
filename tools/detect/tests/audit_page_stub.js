// A DOM small enough to run the audit page's script and large enough to catch
// it lying. Everything the page touches is here; anything it reaches for that
// is not becomes a TypeError, which is the point.
var els = {}, listeners = {};
function mk(id){
  var el = {id:id, textContent:'', className:'', title:'',
    hidden:false, disabled:false, style:{}, children:[], value:'', src:'',
    dataset:{}, options:[],
    // a <select> the page fills at boot: without these the option list is
    // never built and every assertion below runs against a control that does
    // not exist, which is a test passing on code it never executed
    appendChild:function(c){ this.children.push(c); this.options.push(c) },
    setAttribute:function(k,v){ this.dataset[k] = v },
    removeAttribute:function(k){ delete this.dataset[k] },
    getAttribute:function(){ return null },
    addEventListener:function(t,f){(listeners[id]=listeners[id]||{})[t]=f},
    // a card really does contain its three action buttons; returning []
    // made paintCard() read b[0].classList off undefined
    querySelectorAll:function(sel){
      if (/\.act/.test(sel || '')) {
        this._acts = this._acts ||
          [mk(this.id + ':flag'), mk(this.id + ':no'), mk(this.id + ':unsure')];
        return this._acts;
      }
      return [];
    }, scrollIntoView:function(){},
    classList:{toggle:function(){}, add:function(){}, remove:function(){}},
    closest:function(){return null}, select:function(){},
    setSelectionRange:function(){}};
  // innerHTML must MAKE CHILDREN. Without this, grid.children stayed [] for
  // ever, so every check about a card being hidden, restored or counted
  // passed against a grid that had none -- a test of nothing at all.
  var _html = '';
  Object.defineProperty(el, 'innerHTML', {
    get: function(){ return _html },
    set: function(v){
      _html = String(v);
      var n = (_html.match(/<div class="card/g) || []).length;
      el.children = [];
      for (var i = 0; i < n; i++) el.children.push(mk(el.id + ':card' + i));
    }
  });
  return el;
}
function E(id){ return els[id] || (els[id] = mk(id)) }
// Elements the MARKUP starts hidden. Defaulting every node to visible made
// the keydown handler believe the lightbox was open, so it returned before
// any verdict key was read and the check passed on a branch it never entered.
['lb', 'toast', 'empty'].forEach(function(id){ E(id).hidden = true });
global.document = {getElementById:E, createElement:function(){return mk('new')},
  addEventListener:function(t,f){(listeners.doc = listeners.doc || {})[t] = f},
  body:{appendChild:function(){}, removeChild:function(){}},
  execCommand:function(){return true}};
global.window = {isSecureContext:false};
global.localStorage = {_d:{}, getItem:function(k){return this._d[k] || null},
  setItem:function(k,v){this._d[k] = String(v)}};
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
// summarise() returns lo95/hi95 on every band; a fixture without them was a
// payload the server never sends, and the page drew left:NaN% against it
var STATS = {judged:12, wrong:1, pool:4688510, covered:0.9, threshold:0.5,
  rejected:{rate:0.0031, judged:12, wrong:1, boxes:3945390, covered:0.9},
  kept:{rate:0.0, judged:0, wrong:0, boxes:743120, covered:0.0},
  bands:[
  {lo:0.0, hi:0.1, judged:8, dogs:0, wrong:0, kept:false, rate:0.0,
   lo95:0.0, hi95:0.324, boxes:3530147},
  {lo:0.1, hi:0.2, judged:0, dogs:0, wrong:0, kept:false, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:162866},
  {lo:0.2, hi:0.3, judged:0, dogs:0, wrong:0, kept:false, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:102513},
  {lo:0.3, hi:0.4, judged:0, dogs:0, wrong:0, kept:false, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:80241},
  {lo:0.4, hi:0.5, judged:4, dogs:1, wrong:1, kept:false, rate:0.25,
   lo95:0.046, hi95:0.699, boxes:69623},
  {lo:0.5, hi:0.6, judged:0, dogs:0, wrong:0, kept:true, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:120000},
  {lo:0.6, hi:0.7, judged:0, dogs:0, wrong:0, kept:true, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:110000},
  {lo:0.7, hi:0.8, judged:0, dogs:0, wrong:0, kept:true, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:120000},
  {lo:0.8, hi:0.9, judged:0, dogs:0, wrong:0, kept:true, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:150000},
  {lo:0.9, hi:1.0, judged:0, dogs:0, wrong:0, kept:true, rate:0.0,
   lo95:0.0, hi95:0.0, boxes:243120}]};
