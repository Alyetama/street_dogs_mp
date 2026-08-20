"""The strip that tells an annotator how far through their target they are.

ONE SPELLING, THREE PAGES. The review queue is rendered by dashboard.py and
the two audit sheets by audit.py, and the tab strip they share is already
duplicated between those files with a guard pinning the copies byte for byte.
This is the same contract with the duplication removed: both files import
these three constants, so there is one set of class names, one way the numbers
are formatted, and one answer to what happens when the count does not arrive.

It is self-contained on purpose. The script defines its own escape rather than
reaching for the page's ``esc()``, because a strip that works on two pages and
silently renders nothing on the third -- the one whose helper happens to be
named differently -- is exactly the failure a shared file is supposed to end.

WHAT IT IS NOT. It is not a gate. Nothing on any page is withheld from
somebody who has not reached their number, and nobody is stopped at it: the
target is a thing to see, and judging past it is not an error. A quota that
locks the work away would make the fastest thing to do the wrong thing --
stop reading crops carefully and start clicking.
"""

# The words the surfaces go by, in the reader's language rather than the
# database's. They match the tab strip: a target on 'gate' is a target on the
# tab the reader can see called Dog-bin audit.
SURFACE_WORDS = {
    'any': 'every surface',
    'review': 'the review queue',
    'gate': 'the dog-bin audit',
    'leash': 'the leash audit',
}

# One line, sitting under the tab strip: a label, then one entry per target
# that applies to this page. Two is the most anybody can have here (their
# target on this surface, and a target on every surface), and in practice it
# is one -- which is why they sit on a row rather than stacking into a panel
# that would push the crops down the screen on every visit.
STRIP_CSS = """
.asg{display:flex;align-items:center;flex-wrap:wrap;gap:10px 16px;
  margin:0 0 14px;font-size:12px;color:var(--mut)}
.asg[hidden]{display:none}
.asgl{color:var(--dim);font-size:10.5px;text-transform:uppercase;
  letter-spacing:.07em;font-weight:600}
.asg1{display:inline-flex;align-items:center;gap:9px;
  border:1px solid var(--bd);border-radius:999px;padding:5px 13px 5px 12px;
  background:rgba(130,140,150,.05)}
.asgs{color:var(--dim)}
/* The bar is for the glance -- am I nearly there -- and the numbers are for
   the answer, because a bar alone cannot tell 40 of 50 from 400 of 500. */
.asgbar{width:74px;height:4px;border-radius:3px;flex:none;
  background:rgba(130,140,150,.18);overflow:hidden}
.asgbar i{display:block;height:100%;background:var(--acc);border-radius:3px;
  transition:width .3s ease}
.asg1 b{color:var(--tx);font-weight:650;
  font-variant-numeric:tabular-nums}
.asg1 em{font-style:normal;color:var(--dim)}
.asg1.done{border-color:rgba(67,181,129,.4)}
.asg1.done .asgbar i{background:var(--green)}
.asg1.done em{color:var(--green)}
/* Overdue is a state of an OPEN target and never of a finished one: work that
   landed late is still work, and a row that turns red the day after somebody
   finished it is a scoreboard they will stop believing. */
.asg1.late{border-color:rgba(216,116,58,.42)}
.asg1.late .asgd{color:#e08a5a}
.asgd{color:var(--dim);font-size:11px}
@media(prefers-reduced-motion:reduce){.asgbar i{transition:none}}
"""

# data-surface is what the page is: the strip shows a target set on THIS
# surface and one set on every surface, and nothing else. A leash target has
# no business drawing a bar over the dog-bin sheet, where none of the work
# being done counts towards it.
STRIP_HTML = ('<div class="asg" id="asg" data-surface="%s" hidden '
              'aria-live="polite"></div>')


def strip_html(surface):
    """The empty strip for one page. It fills itself, or stays hidden."""
    return STRIP_HTML % (surface,)


# refreshWorkStrip() is the page's handle: call it after a verdict lands and
# the bar moves. It is debounced here rather than at each call site, so a
# reader holding D down does not put one request per crop on the server.
STRIP_JS = r"""
var refreshWorkStrip=(function(){
  var box=document.getElementById('asg');
  if(!box)return function(){};
  var want=box.getAttribute('data-surface')||'',timer=null,busy=false;
  var WORDS={any:'every surface',review:'the review queue',
             gate:'the dog-bin audit',leash:'the leash audit'};
  function esc(s){var d=document.createElement('div');
    d.textContent=String(s==null?'':s);return d.innerHTML}
  function num(v){
    return String(v).replace(/\B(?=(\d{3})+(?!\d))/g,',')}
  function paint(list){
    var as=(list||[]).filter(function(a){
      return a&&(a.surface===want||a.surface==='any')});
    if(!as.length){box.hidden=true;box.innerHTML='';return}
    box.innerHTML='<span class="asgl">your target</span>'+
      as.map(function(a){
        var pct=Math.max(0,Math.min(100,+a.pct||0)),done=pct>=100;
        return '<span class="asg1'+(done?' done':'')+
          (!done&&a.state==='overdue'?' late':'')+'">'+
          '<span class="asgs">'+esc(WORDS[a.surface]||a.surface)+'</span>'+
          '<span class="asgbar"><i style="width:'+pct+'%"></i></span>'+
          '<span><b>'+num(a.done)+'</b> / '+num(a.target)+
          (done?' <em>done</em>':' <em>'+pct+'%</em>')+'</span>'+
          (a.due_txt?'<span class="asgd">'+
            (a.state==='overdue'?'was due ':'due ')+esc(a.due_txt)+
            '</span>':'')+
          '</span>';
      }).join('');
    box.hidden=false;
  }
  function pull(){
    if(busy)return;
    busy=true;
    fetch('/api/assignment',{credentials:'same-origin'})
      .then(function(r){if(!r.ok)throw 0;return r.json()})
      .then(function(j){busy=false;paint(j&&j.assignments)})
      /* A count that did not arrive leaves the LAST one on screen. Blanking
         the strip on a dropped request would read as the target having been
         called off, which is a thing an admin does and a network does not. */
      .catch(function(){busy=false});
  }
  pull();
  return function(){
    if(timer)clearTimeout(timer);
    timer=setTimeout(function(){timer=null;pull()},900);
  };
})();
"""
