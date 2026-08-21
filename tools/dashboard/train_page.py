#!/usr/bin/env python3
"""The page that builds a dataset and trains a model on it.

Its own page rather than a card on the front one. This is three steps, a
parameter form and a live log; the front page is already seven sections tall,
and the run in progress is reported there by the training tracker either way,
so a card would have been the same thing twice with less room for it.

WHAT IT DOES NOT DO. It runs nothing itself. Every button submits a JOB --
tools/dashboard/jobs.py -- which is detached from this process, so the build
survives the tab being closed and the training survives the dashboard being
restarted. This page is a way to start those and a way to watch them, and if
it is closed the work carries on.

THE COMMANDS ARE SHOWN. Every action prints the argv it submitted into the
job's log, and the page shows it, because the answer to "what did that button
just do" should not be "read the source". Everything here can be run from a
terminal with the same arguments.
"""
import html
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))

import build_dataset as bd                                    # noqa: E402
import jobs                                                   # noqa: E402
import work_strip                                             # noqa: E402


def esc(v):
    return html.escape('' if v is None else str(v), quote=True)


# The three models, in the order the pipeline runs them: the detector finds
# boxes, the gate throws away what is not a dog, the leash model reads what is
# left. Presented that way because it is the order somebody retrains them in.
ORDER = ('dogdet', 'dogbin', 'leash')


def overview(family=None):
    """Everything the page draws, in one answer.

    One request rather than four, because the four are read together and a
    page that assembles itself from four is a page with four ways to be half
    drawn.
    """
    fams = []
    for key in ORDER:
        spec = bd.FAMILIES[key]
        stores = {}
        for name, meta in bd.STORES.items():
            if key not in meta['for']:
                continue
            st = bd.store_state(name)
            stores[name] = {
                'lines': st['lines'],
                'files': (sum(len(x) for x in st['files'].values())
                          if isinstance(st['files'], dict)
                          else len(st['files'] or [])),
            }
        fams.append({'key': key, 'title': spec['title'], 'what': spec['what'],
                     'kind': spec['kind'], 'base': spec['base'],
                     'stores': stores})
    return {
        'families': fams,
        'datasets': bd.catalogue(family),
        'jobs': [_job_row(j) for j in jobs.listing(limit=25)],
        'lanes': {lane: (lambda j: j and j['id'])(jobs.lane_holder(lane))
                  for lane in jobs.LANES},
    }


def _job_row(job):
    """One job as the page needs it -- never the environment, and never more
    of the record than a page has any use for."""
    return {'id': job.get('id'), 'kind': job.get('kind'),
            'lane': job.get('lane'), 'label': job.get('label'),
            'by': job.get('by'), 'state': job.get('state'),
            'created_at': job.get('created_at'),
            'started_at': job.get('started_at'),
            'ended_at': job.get('ended_at'),
            'exit_code': job.get('exit_code'),
            'argv': job.get('argv'),
            'progress': jobs.progress(job.get('id') or ''),
            'meta': job.get('meta') or {}}


TRAIN_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Train &mdash; Street Dogs</title><style>
:root{--bg:#13151a;--panel:#1b2027;--panel2:#21262d;--bd:rgba(130,140,150,.13);
--tx:#eef1f4;--mut:#98a2ad;--dim:#69727d;--acc:#e8a645;--green:#43b581;
--red:#ef5350;
--num:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);-webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  line-height:1.5;padding:0 22px 80px}
.wrap{max-width:1180px;margin:0 auto}
a{color:inherit}
header{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap;
  padding:22px 0 16px;border-bottom:1px solid var(--bd);margin-bottom:18px}
h1{font-size:20px;font-weight:660;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:12.5px;margin-top:3px;max-width:64ch}
.hdrend{display:flex;align-items:center;gap:12px;margin-left:auto}
.back{font-size:12px;color:var(--mut);text-decoration:none;
  border:1px solid var(--bd);border-radius:8px;padding:6px 11px}
.back:hover{color:var(--tx);border-color:rgba(130,140,150,.3)}
__ACCTCSS__
/* ── the three models, as one control ──
   A segmented picker rather than a dropdown: there are exactly three, they
   never change, and which one you are working on decides everything else on
   the page -- so it is worth the width to have it always readable. */
.models{display:flex;gap:2px;padding:3px;border:1px solid var(--bd);
  border-radius:12px;margin-bottom:20px;flex-wrap:wrap}
.model{flex:1 1 200px;text-align:left;background:0;border:0;cursor:pointer;
  font-family:inherit;color:var(--dim);padding:10px 14px;border-radius:9px;
  transition:background .12s,color .12s}
.model b{display:block;font-size:13.5px;font-weight:640;color:var(--mut)}
.model span{font-size:11.5px}
.model:hover{background:rgba(130,140,150,.06)}
.model.on{background:rgba(232,166,69,.13)}
.model.on b{color:var(--acc)}
.model.on span{color:var(--mut)}
.model:focus-visible{outline:2px solid var(--acc);outline-offset:-2px}
/* ── steps ──
   Numbered because this IS a sequence: there is no dataset to choose before
   one is built, and nothing to train before one is chosen. */
.step{background:var(--panel);border:1px solid var(--bd);border-radius:13px;
  padding:16px 18px;margin-bottom:14px}
.shead{display:flex;align-items:baseline;gap:11px;margin-bottom:11px}
.snum{font-family:var(--num);font-size:11px;color:var(--dim);
  border:1px solid var(--bd);border-radius:6px;padding:1px 7px}
.stitle{font-size:13.5px;font-weight:620}
.ssub{color:var(--dim);font-size:12px;margin-left:auto;text-align:right}
.btn{background:var(--panel2);border:1px solid var(--bd);color:var(--mut);
  border-radius:9px;padding:8px 14px;font-size:12.5px;cursor:pointer;
  font-family:inherit}
.btn:hover:not(:disabled){color:var(--tx);border-color:rgba(130,140,150,.32)}
.btn:disabled{opacity:.45;cursor:default}
.btn.go{color:var(--acc);border-color:rgba(232,166,69,.4);font-weight:620}
.btn.go:hover:not(:disabled){background:rgba(232,166,69,.1)}
.btn.warn:hover:not(:disabled){color:var(--red);
  border-color:rgba(239,83,80,.4)}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
select,input{background:var(--panel2);border:1px solid var(--bd);
  color:var(--mut);border-radius:9px;padding:7px 9px;font-size:12.5px;
  font-family:inherit}
select:hover,input:hover{color:var(--tx)}
select:focus-visible,input:focus-visible{outline:2px solid var(--acc);
  outline-offset:1px}
.wide{width:100%;max-width:620px}
/* what a build would read, so the button is not a leap of faith */
.stores{display:flex;gap:8px 18px;flex-wrap:wrap;font-size:11.5px;
  color:var(--dim);margin-bottom:12px}
.stores b{color:var(--mut);font-family:var(--num);font-weight:620}
/* ── the parameters ──
   Two columns of label and field, dense on purpose: thirty of them stacked
   one per row is a page nobody scrolls to the bottom of. Anything inherited
   from the last run is marked, because "why is this 1280" has an answer. */
.params{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:9px 16px;margin-bottom:14px}
.p{display:flex;align-items:center;gap:8px;font-size:12px}
.p label{flex:1;color:var(--dim);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.p input,.p select{width:96px;flex:none;font-family:var(--num);
  font-size:12px;padding:5px 7px}
.p.inherited label{color:var(--mut)}
.p.inherited label::after{content:' \00b7';color:var(--acc)}
.p.changed input,.p.changed select{border-color:rgba(232,166,69,.5);
  color:var(--acc)}
.pmore{font-size:12px;color:var(--dim);background:0;border:0;cursor:pointer;
  font-family:inherit;padding:0}
.pmore:hover{color:var(--tx)}
.note{font-size:12px;color:var(--dim);margin-top:10px;max-width:76ch}
.note b{color:var(--mut);font-weight:600}
/* ── jobs ── */
.job{border:1px solid var(--bd);border-radius:11px;padding:11px 14px;
  margin-bottom:9px;background:var(--panel)}
.jtop{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.jlabel{font-weight:620;font-size:13px}
.jid{font-family:var(--num);font-size:11px;color:var(--dim)}
.tag{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  border:1px solid var(--bd);border-radius:6px;padding:1px 6px;
  color:var(--dim)}
.tag.running{color:var(--acc);border-color:rgba(232,166,69,.4)}
.tag.done{color:var(--green);border-color:rgba(67,181,129,.35)}
.tag.failed,.tag.lost{color:var(--red);border-color:rgba(239,83,80,.35)}
.jspace{flex:1}
.bar{height:4px;border-radius:3px;background:rgba(130,140,150,.16);
  overflow:hidden;margin:9px 0 6px}
.bar i{display:block;height:100%;background:var(--acc);border-radius:3px;
  transition:width .3s ease}
/* THE BAR MEANS WHAT THE STATE MEANS. Left amber, a job that failed a quarter
   of the way through reads as one still working, and a finished one reads as
   one that stopped. Amber is only for work in progress. */
.job.done .bar i{background:var(--green)}
.job.failed .bar i,.job.lost .bar i,.job.cancelled .bar i{
  background:rgba(130,140,150,.3)}
.cmd{font-family:var(--num);font-size:11px;color:var(--dim);
  overflow-x:auto;white-space:pre;padding-bottom:2px}
.log{font-family:var(--num);font-size:11.5px;line-height:1.45;color:var(--mut);
  background:#0e1014;border:1px solid var(--bd);border-radius:8px;
  padding:10px 12px;margin-top:8px;max-height:300px;overflow:auto;
  white-space:pre-wrap;word-break:break-word}
.log[hidden]{display:none}
.empty{color:var(--dim);font-size:12.5px;padding:8px 0}
/* One line in the dataset's own description, not a banner: it is a fact
   about that dataset, and it belongs where the counts are. */
.warnnote{color:#e8a645}
.msg{border-radius:10px;padding:9px 13px;font-size:12.5px;margin-bottom:12px}
.msg.bad{background:rgba(239,83,80,.1);border:1px solid rgba(239,83,80,.3);
  color:#ffb4b2}
.msg.ok{background:rgba(67,181,129,.1);border:1px solid rgba(67,181,129,.3);
  color:#9fe3c0}
.msg[hidden]{display:none}
@media(prefers-reduced-motion:reduce){.bar i{transition:none}}
</style></head><body><div class="wrap">
<header>
  <div><h1>Train</h1>
    <div class="sub">Build a dataset out of every annotation on record, then
      train on it. Both run detached &mdash; close this page and they carry
      on.</div></div>
  <div class="hdrend"><a class="back" href="/">&larr; dashboard</a>__ACCOUNT__</div>
</header>

<div class="msg bad" id="err" hidden></div>
<div class="msg ok" id="say" hidden></div>

<div class="models" id="models" role="tablist"></div>

<div class="step">
  <div class="shead"><span class="snum">1</span>
    <span class="stitle">Build a dataset</span>
    <span class="ssub" id="basesub"></span></div>
  <div class="stores" id="stores"></div>
  <div class="row">
    <button class="btn go" id="build" type="button">Build dataset</button>
    <span class="ssub" id="buildsub"></span>
  </div>
  <div class="note">Every build starts from the base set and re-applies every
    annotation on record, so building twice from the same ledgers gives the
    same dataset. It lands in a new directory with its own
    <b>bundle/</b> &mdash; every image it used, with a digest, plus the
    ledgers it read and the command that ran.</div>
</div>

<div class="step">
  <div class="shead"><span class="snum">2</span>
    <span class="stitle">Choose one to train on</span>
    <span class="ssub" id="dssub"></span></div>
  <select class="wide" id="dataset"></select>
  <div class="note" id="dsnote"></div>
</div>

<div class="step">
  <div class="shead"><span class="snum">3</span>
    <span class="stitle">Parameters</span>
    <span class="ssub" id="psub"></span></div>
  <div class="params" id="params"></div>
  <button class="pmore" id="pmore" type="button">show every parameter</button>
  <div class="row" style="margin-top:13px">
    <button class="btn go" id="train" type="button">Start training</button>
    <button class="btn" id="preset" type="button">reset to the last run</button>
    <span class="ssub" id="trainsub"></span>
  </div>
</div>

<div class="step">
  <div class="shead"><span class="snum">&#9679;</span>
    <span class="stitle">Work</span>
    <span class="ssub" id="jobsub"></span></div>
  <div id="jobs"><div class="empty">Nothing has been run yet.</div></div>
</div>

<script>
var FAM=null, STATE=null, FIELDS=[], OPEN={}, ALL=false, POLL=null;
function $(id){return document.getElementById(id)}
function esc(s){var d=document.createElement('div');
  d.textContent=(s==null?'':String(s));return d.innerHTML}
function n(v){return (v==null?'—':(+v).toLocaleString('en-US'))}
function say(el,text,ms){var e=$(el);e.textContent=text||'';e.hidden=!text;
  if(text&&ms)setTimeout(function(){e.hidden=true},ms)}
function when(iso){ return iso ? String(iso).replace('T',' ').slice(0,16) : '—' }
function ago(ts){
  if(!ts)return '';
  var s=Math.max(0,Math.floor(Date.now()/1000-ts));
  if(s<60)return s+'s'; if(s<3600)return Math.floor(s/60)+'m';
  if(s<86400)return Math.floor(s/3600)+'h'; return Math.floor(s/86400)+'d';
}
/* Every request goes through here so a failure is a sentence on the page
   rather than a promise nobody caught. */
function api(path,body){
  var opt={credentials:'same-origin'};
  if(body){opt.method='POST';opt.headers={'Content-Type':'application/json'};
           opt.body=JSON.stringify(body)}
  return fetch(path,opt).then(function(r){
    return r.json().catch(function(){throw new Error('the server did not answer with JSON')})
      .then(function(j){
        if(!r.ok||j.error)throw new Error(j.error||('the server answered '+r.status));
        return j})});
}
function fail(e){say('err',(e&&e.message)||String(e))}

/* ── the models ── */
function paintModels(){
  $('models').innerHTML=STATE.families.map(function(f){
    return '<button type="button" class="model'+(f.key===FAM?' on':'')+
      '" data-f="'+esc(f.key)+'" role="tab" aria-selected="'+(f.key===FAM)+'">'+
      '<b>'+esc(f.title)+'</b><span>'+esc(f.what)+'</span></button>';
  }).join('');
}
function famOf(){for(var i=0;i<STATE.families.length;i++)
  if(STATE.families[i].key===FAM)return STATE.families[i]; return null}

/* ── step 1 ── */
function paintBuild(){
  var f=famOf(); if(!f)return;
  $('basesub').textContent='derived from '+f.base;
  var st=f.stores, keys=Object.keys(st).sort();
  $('stores').innerHTML=keys.length?keys.map(function(k){
    var v=st[k];
    return '<span>'+esc(k.replace(/_/g,' '))+' <b>'+
      (v.lines!=null?n(v.lines)+' verdicts':n(v.files)+' crops')+'</b></span>';
  }).join(''):'<span>no annotation stores feed this model</span>';
  var busy=STATE.lanes.build;
  $('build').disabled=!!busy;
  $('buildsub').textContent=busy?'a build is already running':'';
}

/* ── step 2 ── */
function paintDatasets(){
  var sel=$('dataset'), rows=STATE.datasets.filter(function(d){
    return d.family===FAM});
  var keep=sel.value;
  sel.innerHTML=rows.length?rows.map(function(d){
    var c=d.counts, size=c?(n(c.total)+' images'):'no record';
    return '<option value="'+esc(d.id)+'">'+esc(d.id)+' — '+when(d.built_at_iso)+
      ' · '+esc(size)+(d.bundle?(d.label_studio?'':' · no hand-drawn boxes')
                              :' · built before bundles')+'</option>';
  }).join(''):'<option value="">nothing built for this model yet</option>';
  if(keep&&rows.some(function(d){return d.id===keep}))sel.value=keep;
  $('dssub').textContent=rows.length?(rows.length+' available'):'';
  paintDsNote();
}
function paintDsNote(){
  var id=$('dataset').value, d=null;
  STATE.datasets.forEach(function(x){if(x.id===id)d=x});
  if(!d){$('dsnote').textContent='';return}
  if(!d.counts){
    $('dsnote').innerHTML='Built before this page existed, so it carries no '+
      'bundle: it can be trained on, but there is no record of which '+
      'annotations went into it.';
    return;
  }
  var c=d.counts, parts=[];
  Object.keys(c.splits).sort().forEach(function(s){
    var p=c.splits[s], per=Object.keys(p.classes).sort().map(function(k){
      return esc(k)+' '+n(p.classes[k])}).join(', ');
    parts.push('<b>'+esc(s)+'</b> '+n(p.total)+' ('+Math.round(p.share*100)+
      '%) — '+per);
  });
  var ls=d.label_studio
    ? n(d.label_studio)+' hand-drawn frames from Label&nbsp;Studio'
    : '<span class="warnnote">no Label&nbsp;Studio export — built before '+
      'every build made one, so the hand-drawn boxes are not in it</span>';
  $('dsnote').innerHTML=n(c.total)+' images · '+parts.join(' &nbsp;·&nbsp; ')+
    (d.built_by?' &nbsp;·&nbsp; built by '+esc(d.built_by):'')+
    '<br>'+ls;
}

/* ── step 3 ── */
var HEADLINE=['epochs','batch','imgsz','optimizer','patience','lr0'];
function loadParams(){
  $('params').innerHTML='<span class="empty">reading what the last run used…</span>';
  api('/api/train/params?family='+encodeURIComponent(FAM)).then(function(j){
    FIELDS=j.fields||[]; OPEN={};
    $('psub').textContent=j.inherited_from
      ? 'inherited from '+j.inherited_from.split('/').slice(-2)[0]+
        ' · ultralytics '+j.ultralytics
      : 'ultralytics '+j.ultralytics+' defaults';
    paintParams();
  }).catch(function(e){
    FIELDS=[];
    $('params').innerHTML='<span class="empty">could not read the '+
      'parameters: '+esc(e.message)+'</span>';
  });
}
function paintParams(){
  var show=FIELDS.filter(function(f){
    return ALL||HEADLINE.indexOf(f.key)>=0||f.from==='the last run'});
  $('params').innerHTML=show.map(function(f){
    var inh=f.from==='the last run';
    var ctl;
    if(f.type==='bool'){
      ctl='<select data-k="'+esc(f.key)+'">'+
        ['true','false'].map(function(v){
          return '<option value="'+v+'"'+
            ((String(f.value)==='true')===(v==='true')?' selected':'')+
            '>'+v+'</option>'}).join('')+'</select>';
    }else{
      ctl='<input data-k="'+esc(f.key)+'" value="'+esc(f.value)+'">';
    }
    return '<span class="p'+(inh?' inherited':'')+'" title="'+esc(f.why)+
      ' — '+esc(f.from)+'">'+
      '<label for="">'+esc(f.key)+'</label>'+ctl+'</span>';
  }).join('');
  $('pmore').textContent=ALL?'show fewer':'show every parameter ('+
    FIELDS.length+')';
}
function overrides(){
  var out={};
  FIELDS.forEach(function(f){
    var el=$('params').querySelector('[data-k="'+f.key+'"]');
    if(!el)return;
    var v=el.value.trim();
    if(v===''||v===String(f.value))return;
    out[f.key]=v;
  });
  return out;
}

/* ── the work ── */
function paintJobs(){
  var rows=STATE.jobs;
  $('jobsub').textContent=STATE.lanes.build||STATE.lanes.train
    ? 'something is running' : '';
  if(!rows.length){$('jobs').innerHTML=
    '<div class="empty">Nothing has been run yet.</div>';return}
  $('jobs').innerHTML=rows.map(function(j){
    var p=j.progress, pct=p&&p.total?Math.round(p.done/p.total*100):null;
    var run=j.state==='running';
    return '<div class="job '+esc(j.state)+'" data-j="'+esc(j.id)+'">'+
      '<div class="jtop">'+
        '<span class="jlabel">'+esc(j.label||j.kind)+'</span>'+
        '<span class="tag '+esc(j.state)+'">'+esc(j.state)+'</span>'+
        (j.by?'<span class="jid">'+esc(j.by)+'</span>':'')+
        '<span class="jid">'+esc(j.id)+'</span>'+
        '<span class="jspace"></span>'+
        (j.exit_code!=null&&j.state!=='done'
          ? '<span class="jid">exit '+esc(j.exit_code)+'</span>':'')+
        '<span class="jid">'+(run?ago(j.started_at)+' so far'
          :(j.ended_at?ago(j.ended_at)+' ago':''))+'</span>'+
        '<button class="btn" data-log="'+esc(j.id)+'">'+
          (OPEN[j.id]?'hide log':'log')+'</button>'+
        (run?'<button class="btn warn" data-stop="'+esc(j.id)+
             '">stop</button>'
            :'<button class="btn" data-forget="'+esc(j.id)+
             '" title="remove this record">clear</button>')+
      '</div>'+
      (pct!=null?'<div class="bar"><i style="width:'+pct+'%"></i></div>'+
        '<div class="jid">'+esc(p.what)+' — '+p.done+' of '+p.total+'</div>':'')+
      '<div class="cmd">'+esc((j.argv||[]).join(' '))+'</div>'+
      '<div class="log" id="log-'+esc(j.id)+'"'+(OPEN[j.id]?'':' hidden')+
        '>'+(OPEN[j.id]===true?'loading…':esc(OPEN[j.id]||''))+'</div>'+
    '</div>';
  }).join('');
}

/* ── wiring ── */
function refresh(){
  return api('/api/train/overview'+(FAM?'?family='+encodeURIComponent(FAM):''))
    .then(function(j){
      STATE=j;
      if(!FAM)FAM=(j.families[0]||{}).key;
      paintModels();paintBuild();paintDatasets();paintJobs();
      var live=j.jobs.some(function(x){return x.state==='running'});
      clearTimeout(POLL);
      POLL=setTimeout(refresh, live?3000:20000);
      Object.keys(OPEN).forEach(function(id){
        if(OPEN[id]!==undefined&&OPEN[id]!==false)pullLog(id)});
    }).catch(fail);
}
function pullLog(id){
  return api('/api/train/log?job='+encodeURIComponent(id)).then(function(j){
    OPEN[id]=j.tail||'(nothing yet)';
    var el=$('log-'+id);
    if(el){var stick=el.scrollTop+el.clientHeight>=el.scrollHeight-24;
      el.textContent=OPEN[id]; el.hidden=false;
      if(stick)el.scrollTop=el.scrollHeight;}
  }).catch(function(){});
}
document.addEventListener('click',function(e){
  var t=e.target;
  if(!t||!t.getAttribute)return;
  var f=t.closest&&t.closest('.model');
  if(f&&f.getAttribute('data-f')){
    FAM=f.getAttribute('data-f');
    paintModels();paintBuild();paintDatasets();loadParams();
    return;
  }
  var lg=t.getAttribute('data-log');
  if(lg){ if(OPEN[lg]){delete OPEN[lg];var el=$('log-'+lg);if(el)el.hidden=true;
            t.textContent='log';}
          else {OPEN[lg]=true;t.textContent='hide log';pullLog(lg);} return }
  var fg=t.getAttribute('data-forget');
  if(fg){
    if(!window.confirm('Clear this record? The log goes with it. Whatever it '+
       'built stays on disk.'))return;
    t.disabled=true;
    delete OPEN[fg];
    api('/api/train/forget',{job:fg}).then(refresh).catch(fail);
    return;
  }
  var stop=t.getAttribute('data-stop');
  if(stop){
    if(!window.confirm('Stop this? Whatever it has written stays where it is.'))
      return;
    t.disabled=true;
    api('/api/train/cancel',{job:stop}).then(refresh).catch(fail);
    return;
  }
});
$('build').addEventListener('click',function(){
  var f=famOf(); if(!f)return;
  if(!window.confirm('Build a '+f.title+' dataset from every annotation on '+
     'record? It runs in the background and lands in a new directory.'))return;
  $('build').disabled=true;
  api('/api/train/build',{family:FAM}).then(function(j){
    say('say','building — '+j.job.id,8000); return refresh();
  }).catch(function(e){fail(e);$('build').disabled=false});
});
$('train').addEventListener('click',function(){
  var ds=$('dataset').value;
  if(!ds){fail(new Error('there is no dataset to train on yet'));return}
  var over=overrides();
  var lines=Object.keys(over).sort().map(function(k){return k+'='+over[k]});
  if(!window.confirm('Train '+(famOf()||{}).title+' on '+ds+
     (lines.length?'\nwith '+lines.join(', '):'\nwith the inherited parameters')+
     '?\n\nIt runs in the background on the GPU.'))return;
  $('train').disabled=true;
  api('/api/train/start',{family:FAM,dataset:ds,params:over})
    .then(function(j){say('say','training — '+j.job.id,8000);return refresh()})
    .catch(fail).then(function(){$('train').disabled=false});
});
$('pmore').addEventListener('click',function(){ALL=!ALL;paintParams()});
$('preset').addEventListener('click',loadParams);
$('dataset').addEventListener('change',paintDsNote);
document.addEventListener('visibilitychange',function(){
  if(!document.hidden)refresh()});
refresh().then(loadParams);
</script></body></html>"""


def page_html(account=('', '')):
    """The page, with the identity strip spliced in like every other."""
    out = TRAIN_HTML
    for key, value in (('__ACCTCSS__', account[0]),
                       ('__ACCOUNT__', account[1])):
        out = out.replace(key, value)
    return out
