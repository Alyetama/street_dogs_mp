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

import json

# The words the surfaces go by, in the reader's language rather than the
# database's. They match the tab strip: a target on 'gate' is a target on the
# tab the reader can see called Dog-bin audit.
SURFACE_WORDS = {
    'any': 'every surface',
    'review': 'the review queue',
    'gate': 'the dog-bin audit',
    'leash': 'the leash audit',
}

# THE DATE PAIR, for the three judging pages. It was written in audit.py and
# copied into dashboard.py, which is two files that have to stay identical
# with nothing making them -- so it lives here beside the strip, for the same
# reason the strip does. Sized to the select it stands next to: 7px of
# padding is 33px tall, which is what a control on these pages is.
DATE_CSS = """/* JUDGED, BETWEEN TWO DATES. Native inputs, so the calendar is the
   platform's: a hand-built one is a month of edge cases -- locale order,
   which day a week starts on, every keyboard path -- to arrive somewhere
   worse than the control the browser already ships. color-scheme is the
   whole reason it looks like it belongs: without it Chrome draws a white
   calendar panel and a black-on-black picker icon on a dark field. */
.pdate{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:7px 8px;font-size:12.5px;font-family:inherit;
  cursor:pointer;color-scheme:dark;font-variant-numeric:tabular-nums}
.pdate:hover{color:var(--tx)}
.pdate:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
.pdash{color:var(--dim)}
/* Only where there is something to clear: a x standing over two empty fields
   is a control for undoing nothing. Its own display rule because an author
   `display` beats the browser's [hidden], which is how a control this page
   meant to hide has shipped visible twice. */
.pclr{background:0;border:0;color:var(--dim);font:inherit;font-size:14px;
  line-height:1;cursor:pointer;padding:3px 6px;border-radius:7px}
.pclr:hover{background:rgba(130,140,150,.12);color:var(--tx)}
.pclr[hidden]{display:none}
"""


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
  var WORDS=__WORDS__;
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
        /* THE STAMP WINS. A target met and then dented by an undo shows 497
           of 500 -- and it was still reached, on the day it was reached, so
           it does not go back to looking unfinished. */
        var pct=Math.max(0,Math.min(100,+a.pct||0)),
            done=pct>=100||a.state==='done';
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

# The script's copy of the vocabulary is GENERATED from the one above, not
# typed out beside it. A browser cannot import a Python dict, so the choice
# was between a second literal and this line -- and a second literal is how
# the admin page came to say "any surface" over a bar that said "every
# surface", which is the drift this whole file exists to end.
STRIP_JS = STRIP_JS.replace('__WORDS__', json.dumps(SURFACE_WORDS))
