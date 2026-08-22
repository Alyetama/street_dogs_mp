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
        # THE HAND-DRAWN BOXES COUNT TOO. They are not a ledger in data/ --
        # every build fetches a fresh export from Label Studio and keeps it in
        # the bundle -- so the line above, which lists what a build reads,
        # was silently missing the only labels in this project a person drew
        # rather than corrected. How many came in last time is the honest
        # number to show before a build: the next export is fetched when the
        # button is pressed and will be that size or larger.
        # ONE PROJECT SERVES ALL THREE MODELS, so a build of any of them
        # says how big the last export was. Without the fallback the two
        # models that have not been built since the export chain landed
        # showed nothing at all, which reads as 'no hand-drawn boxes here'
        # -- the opposite of the truth.
        seen = None
        for rows in (bd.catalogue(key), bd.catalogue()):
            for row in rows:
                if row.get('label_studio'):
                    # WHAT THIS MODEL TAKES, not what the export holds. One
                    # export, three readings: the leash model is asked
                    # leashed or unleashed about a dog, so the goats and cows
                    # somebody boxed are not harder examples for it -- they
                    # are not examples at all, and counting them here would
                    # promise the build twice the data it will get.
                    per = row.get('label_studio_classes') or {}
                    if not isinstance(per, dict):
                        per = {}          # a hand-edited or truncated record
                    want = bd.ls_wanted(key)
                    mine = sum(v for k, v in per.items() if k in want)
                    seen = {'tasks': row['label_studio'], 'from': row['id'],
                            'at': row.get('built_at_iso'),
                            'mine': row.get('family') == key,
                            # a number, including zero: a model that takes
                            # nothing from an export it HAS read is not the
                            # same as a model with no export behind it, and
                            # both read as 'fetched at build' when this was
                            # None -- so a class added in Label Studio and
                            # dropped by every model looked like silence
                            'takes': mine,
                            'unit': 'boxes' if key == 'dogdet' else 'crops',
                            'skipped': sum(v for k, v in per.items()
                                           if k not in want) or None}
                    break
            if seen:
                break
        fams.append({'key': key, 'title': spec['title'], 'what': spec['what'],
                     'kind': spec['kind'], 'base': spec['base'],
                     'stores': stores, 'label_studio': seen})
    return {
        'families': fams,
        'datasets': bd.catalogue(family),
        'jobs': [_job_row(j) for j in jobs.listing(limit=25)],
        'lanes': {lane: (lambda j: j and j['id'])(jobs.lane_holder(lane))
                  for lane in jobs.LANES},
    }


def _run_state(family, name):
    """What became of the run a job produced: the score, and the weights.

    Read from the run's own bundle, which is written even when the run falls
    over, with results.csv behind it for a run still going. Without this a
    training job that somebody waited hours for reported 'done, exit 0' and
    nothing else -- the score was in ultralytics' last screenful of log, and
    the weights were wherever the reader could work out they had gone.
    """
    if not family or not name:
        return None
    try:
        import train_model as tm
        root = tm.runs_root(family)
    except Exception:                     # noqa: BLE001 - a page, not a tool
        return None
    path = os.path.join(root, str(name))
    if not os.path.isdir(path):
        return None
    out = {'name': str(name), 'path': path, 'metrics': None, 'error': None,
           'epochs': None, 'weights': None, 'resumed': False}
    # The run's own record, and then whatever a resume wrote after it: the
    # first manifest still holds the error that interrupted the run, and a
    # resume that then succeeded left its outcome in resume.json. Reading only
    # the first one reported a finished run as still broken.
    for who in ('manifest.json', 'resume.json'):
        man = os.path.join(path, 'bundle', who)
        if not os.path.isfile(man):
            continue
        try:
            with open(man) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            continue
        out['metrics'] = doc.get('metrics') or out['metrics']
        out['error'] = doc.get('error')
        if who == 'resume.json':
            out['resumed'] = True
    csv = os.path.join(path, 'results.csv')
    if os.path.isfile(csv):
        try:
            with open(csv) as fh:
                head = fh.readline().strip().split(',')
                rows = [ln for ln in fh.read().splitlines() if ln.strip()]
            out['epochs'] = len(rows)
            if rows and not out['metrics']:
                # a run still going has no bundle metrics yet, and its last
                # written epoch is the only honest answer to "how is it doing"
                last = rows[-1].split(',')
                got = {}
                for key, value in zip(head, last):
                    try:
                        got[key.strip()] = float(value)
                    except ValueError:
                        pass
                out['metrics'] = got or None
        except OSError:
            pass
    try:
        import train_model as tm2
        out['resumable'] = bool(tm2.resumable(family, name))
    except Exception:                     # noqa: BLE001
        out['resumable'] = False
    best = os.path.join(path, 'weights', 'best.pt')
    last = os.path.join(path, 'weights', 'last.pt')
    for cand in (best, last):
        if os.path.isfile(cand):
            out['weights'] = cand
            break
    return out


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
            'run': _run_state((job.get('meta') or {}).get('family'),
                              (job.get('meta') or {}).get('run')),
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
/* it comes off a server rather than out of data/, and the dotted rule under
   it is the only thing that says so */
.stores .fetched{border-bottom:1px dotted var(--bd);cursor:help}
/* ── the parameters ──
   Two columns of label and field, dense on purpose: thirty of them stacked
   one per row is a page nobody scrolls to the bottom of. Anything inherited
   from the last run is marked, because "why is this 1280" has an answer. */
.params{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
  gap:9px 16px;margin-bottom:14px}
/* A group header spans the grid. Only the expanded view draws them: six
   fields need no table of contents. */
.pgrp{grid-column:1/-1;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.08em;color:var(--dim);margin:6px 0 -2px;
  border-bottom:1px solid var(--bd);padding-bottom:3px}
.pgrp:first-child{margin-top:0}
.p{display:flex;align-items:center;gap:8px;font-size:12px}
.p label{flex:1;color:var(--dim);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.p input,.p select{width:96px;flex:none;font-family:var(--num);
  font-size:12px;padding:5px 7px}
.p.inherited label{color:var(--mut)}
.p.inherited label::after{content:' \00b7';color:var(--acc)}
.p.changed input,.p.changed select{border-color:rgba(232,166,69,.5);
  color:var(--acc)}
.wsel{margin-left:auto;display:flex;align-items:center;gap:6px;
  font-size:11px;color:var(--dim)}
.wsel select{width:auto;font-family:var(--num);font-size:11.5px;
  padding:3px 7px}
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
/* pre, not pre-wrap: a progress line is ~100 columns of aligned figures,
   and wrapping it stacks the columns into hash. The box scrolls sideways
   for the rare longer line; the server has already collapsed the redraw
   storms, so a line here is what a terminal would have shown. */
.log{font-family:var(--num);font-size:11.5px;line-height:1.45;color:var(--mut);
  background:#0e1014;border:1px solid var(--bd);border-radius:8px;
  padding:10px 12px;margin-top:8px;max-height:300px;overflow:auto;
  white-space:pre}
.log[hidden]{display:none}
.empty{color:var(--dim);font-size:12.5px;padding:8px 0}
/* One line in the dataset's own description, not a banner: it is a fact
   about that dataset, and it belongs where the counts are. */
.warnnote{color:#e8a645}
option.dead{color:var(--dim)}
.pextra{margin-top:9px;font-size:12px;color:var(--dim)}
.pextra summary{cursor:pointer;padding:3px 0}
.pextra a{color:var(--mut)}
.pextra textarea{display:block;width:100%;margin-top:7px;padding:8px 10px;
  font-family:var(--num);font-size:12px;color:var(--tx);background:#0e1014;
  border:1px solid var(--bd);border-radius:8px;resize:vertical}
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
  <div class="row" style="margin-top:9px">
    <button class="btn warn" id="dsdel" type="button">Delete this dataset</button>
    <span class="ssub" id="dsdelsub"></span>
  </div>
</div>

<div class="step">
  <div class="shead"><span class="snum">3</span>
    <span class="stitle">Parameters</span>
    <span class="ssub" id="psub"></span>
    <!-- The one choice that is not an ultralytics cfg key: which
         architecture to start from. Sizes of the family's own base model,
         because a run's weights decide its speed and its ceiling, and
         "make the next one an m" should not require knowing the file
         naming scheme. "inherited" keeps whatever the recipe used. -->
    <span class="wsel"><label for="wsel">weights</label>
      <select id="wsel"></select></span></div>
  <div class="params" id="params"></div>
  <button class="pmore" id="pmore" type="button">show every parameter</button>
  <!-- ultralytics settles more keys than anyone wants as a form, and the
       curated list is the ones that get changed. The rest are reachable
       rather than absent: one per line, checked against ultralytics itself
       before the run starts, the same as every other parameter. -->
  <details class="pextra"><summary>anything else from the ultralytics
    <a href="https://docs.ultralytics.com/usage/cfg/" target="_blank"
       rel="noopener">cfg docs</a></summary>
    <textarea id="pextra" rows="3" spellcheck="false"
      placeholder="one per line&#10;device=0&#10;save_period=10"></textarea>
  </details>
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
/* What has been typed, kept apart from what was inherited. The form is
   rebuilt whenever the list is expanded, and rebuilding it from FIELDS
   alone threw away every edit made before the click -- silently, so the
   run then started with the inherited value instead of the typed one. */
var EDITS={};
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
  var chips=keys.map(function(k){
    var v=st[k];
    return '<span>'+esc(k.replace(/_/g,' '))+' <b>'+
      (v.lines!=null?n(v.lines)+' verdicts':n(v.files)+' crops')+'</b></span>';
  });
  /* the hand-drawn boxes, which are fetched rather than read off disk */
  var ls=f.label_studio;
  chips.push('<span class="fetched" title="'+
    (ls?'One Label Studio project feeds all three models and each takes a '+
        'different part of it'+
        (ls.skipped?' — '+n(ls.skipped)+' boxes in the last export carry '+
                     'labels this model has no use for':'')+
        '. Counted from the export of '+esc(ls.from)+', which held '+
        n(ls.tasks)+' frames, and a fresh export is taken when you press '+
        'Build. '+
        (ls.unit==='crops'
          ? 'Boxes offered, not crops cut: one whose frame cannot be '+
            'fetched is skipped.'
          : 'The next build sees at least this many.')
       :'Every build fetches the whole Label Studio project. Each model '+
        'takes the labels it can use.')+
    '">label studio <b>'+
    (ls&&ls.tasks!=null?n(ls.takes)+' '+esc(ls.unit):'fetched at build')+
    '</b></span>');
  $('stores').innerHTML=chips.join('');
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
    if(d.unfinished)
      return '<option value="'+esc(d.id)+'" class="dead">'+esc(d.id)+
        ' — unfinished build, safe to delete</option>';
    return '<option value="'+esc(d.id)+'">'+esc(d.id)+' — '+when(d.built_at_iso)+
      ' · '+esc(size)+(d.bundle?(d.label_studio?'':' · no hand-drawn boxes')
                              :' · built before bundles')+'</option>';
  }).join(''):'<option value="">nothing built for this model yet</option>';
  if(keep&&rows.some(function(d){return d.id===keep}))sel.value=keep;
  /* Nothing is auto-selected onto an unfinished build -- step 4 would post a
     dataset that was never finished. But a pick the person made themselves
     stands, including an unfinished one they are about to delete: snapping it
     away on the next poll put the delete button out of reach of the only
     thing the page labels safe to delete. */
  var pick=null; rows.forEach(function(d){if(!pick&&!d.unfinished)pick=d.id});
  if(!sel.value||!rows.some(function(d){return d.id===sel.value}))
    sel.value=(keep&&rows.some(function(d){return d.id===keep}))?keep:(pick||'');
  $('dssub').textContent=rows.length?(rows.length+' available'):'';
  paintDsNote();
}
function paintDsNote(){
  var id=$('dataset').value, d=null;
  STATE.datasets.forEach(function(x){if(x.id===id)d=x});
  if(!d){$('dsnote').textContent='';return}
  if(d.unfinished){
    $('dsnote').innerHTML='<span class="warnnote">An unfinished build: the '+
      'bundle is the last thing a build writes and this has none, so what is '+
      'here is however much was copied before the build stopped. It cannot '+
      'be trained on.</span>';
    return;
  }
  if(!d.counts){
    $('dsnote').innerHTML='Built before this page existed, so it carries no '+
      'bundle: it can be trained on, but there is no record of which '+
      'annotations went into it.'+
      (d.damaged?' Its bundle is there but unreadable.':'');
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
/* The expanded view in three named rows, in the order somebody tunes them:
   when to stop, what goes on the card, what jitters the pictures. */
var GROUPS=[
  ['schedule',['epochs','patience','optimizer','lr0','lrf','momentum',
               'weight_decay','warmup_epochs','cos_lr']],
  ['data & batch',['batch','imgsz','rect','cache','workers','seed',
                   'fraction','freeze','dropout','single_cls',
                   'close_mosaic']],
  ['augmentation',['hsv_h','hsv_s','hsv_v','degrees','translate','scale',
                   'fliplr','flipud','mosaic','mixup','erasing']]];
/* Sizes of the family's own base architecture. The names are the whole
   contract -- the server refuses anything shaped differently. */
function sizeNames(){
  return FAM==='dogdet'
    ? ['yolo26n.pt','yolo26s.pt','yolo26m.pt','yolo26l.pt','yolo26x.pt']
    : ['yolo11n-cls.pt','yolo11s-cls.pt','yolo11m-cls.pt','yolo11l-cls.pt',
       'yolo11x-cls.pt'];
}
function paintSizes(inherited){
  var el=$('wsel');
  el.innerHTML='<option value="">inherited — '+esc(inherited||'?')+
    '</option>'+sizeNames().map(function(w){
      return '<option value="'+w+'">'+w.replace('-cls','').replace('.pt','')
        .replace(/^yolo\d+/,'size ')+' — '+w+'</option>'}).join('');
}
function loadParams(){
  $('params').innerHTML='<span class="empty">reading what the last run used…</span>';
  api('/api/train/params?family='+encodeURIComponent(FAM)).then(function(j){
    FIELDS=j.fields||[]; OPEN={}; EDITS={};
    $('psub').textContent=j.inherited_from
      ? 'inherited from '+j.inherited_from.split('/').slice(-2)[0]+
        ' · ultralytics '+j.ultralytics
      : 'ultralytics '+j.ultralytics+' defaults';
    paintSizes(j.weights);
    paintParams();
  }).catch(function(e){
    FIELDS=[];
    $('params').innerHTML='<span class="empty">could not read the '+
      'parameters: '+esc(e.message)+'</span>';
  });
}
function harvest(){
  FIELDS.forEach(function(f){
    var el=$('params').querySelector('[data-k="'+f.key+'"]');
    if(!el)return;                       /* not on screen: leave the edit be */
    var v=el.value.trim();
    if(v===''||v===String(f.value))delete EDITS[f.key]; else EDITS[f.key]=v;
  });
}
function fieldHtml(f){
  var inh=f.from!=='the ultralytics default';
  var edited=(f.key in EDITS);
  var shown=edited?EDITS[f.key]:f.value;
  var ctl;
  if(f.type==='bool'){
    ctl='<select data-k="'+esc(f.key)+'">'+
      ['true','false'].map(function(v){
        return '<option value="'+v+'"'+
          ((String(shown)==='true')===(v==='true')?' selected':'')+
          '>'+v+'</option>'}).join('')+'</select>';
  }else{
    /* inputmode, never type=number: the numeric keyboard without the
       spinner arrows that crowd a 96px field. The DEFAULT sits in the
       placeholder, so clearing a field shows what letting go returns to. */
    var mode=f.type==='int'?' inputmode="numeric"'
      :(f.type==='float'||f.type==='fraction')?' inputmode="decimal"':'';
    ctl='<input data-k="'+esc(f.key)+'" value="'+esc(shown)+'"'+mode+
      ' placeholder="'+esc(f.default)+'">';
  }
  return '<span class="p'+(inh?' inherited':'')+(edited?' changed':'')+
    '" title="'+esc(f.why)+' — '+esc(f.from)+
    ' (ultralytics default '+esc(f.default)+')">'+
    '<label for="">'+esc(f.key)+'</label>'+ctl+'</span>';
}
function paintParams(){
  harvest();
  if(!ALL){
    var show=FIELDS.filter(function(f){
      return HEADLINE.indexOf(f.key)>=0||f.from!=='the ultralytics default'
        ||(f.key in EDITS)});
    $('params').innerHTML=show.map(fieldHtml).join('');
    $('pmore').textContent='show the other '+
      Math.max(0,FIELDS.length-show.length)+' common parameters';
    return;
  }
  var byKey={}; FIELDS.forEach(function(f){byKey[f.key]=f});
  var out=[], seen={};
  GROUPS.forEach(function(g){
    var fs=g[1].filter(function(k){return k in byKey});
    if(!fs.length)return;
    out.push('<span class="pgrp">'+esc(g[0])+'</span>');
    fs.forEach(function(k){seen[k]=1;out.push(fieldHtml(byKey[k]))});
  });
  var rest=FIELDS.filter(function(f){return !(f.key in seen)});
  if(rest.length){
    out.push('<span class="pgrp">other</span>');
    rest.forEach(function(f){out.push(fieldHtml(f))});
  }
  $('params').innerHTML=out.join('');
  $('pmore').textContent='show fewer';
}
function overrides(){
  harvest();
  var out={};
  Object.keys(EDITS).forEach(function(k){out[k]=EDITS[k]});
  /* the escape hatch, last so it wins: somebody who typed a key meant it.
     Nothing is validated here -- train_model checks every key against
     ultralytics' own table, in the environment that has ultralytics, and a
     key that is not real fails the run in a second saying which one. */
  ($('pextra').value||'').split('\n').forEach(function(line){
    var at=line.indexOf('='); if(at<1)at=line.indexOf(':');
    if(at<1)return;
    var k=line.slice(0,at).trim(), v=line.slice(at+1).trim();
    if(k&&v!=='')out[k]=v;
  });
  return out;
}

/* ── the work ── */
/* The headline number for each task. A detector is judged on mAP, a
   classifier on top-1, and reporting six ultralytics keys instead would be
   the same as reporting none. */
var SCORE={detect:['metrics/mAP50-95(B)','metrics/mAP50(B)'],
           classify:['metrics/accuracy_top1']};
function score(run,family){
  var m=run&&run.metrics; if(!m)return null;
  var keys=SCORE[family==='dogdet'?'detect':'classify'];
  for(var i=0;i<keys.length;i++)
    if(typeof m[keys[i]]==='number')
      return keys[i].replace('metrics/','').replace('(B)','')+' '+
             m[keys[i]].toFixed(3);
  return null;
}
/* What became of the run: the thing somebody waited hours for. */
function outcome(j){
  var r=j.run; if(!r)return '';
  var fam=(j.meta||{}).family, bits=[];
  var s=score(r,fam); if(s)bits.push('<b>'+esc(s)+'</b>');
  if(r.epochs)bits.push(r.epochs+(j.state==='running'?' epochs so far'
                                                     :' epochs'));
  if(r.error)bits.push('<span class="warnnote">'+esc(r.error)+'</span>');
  if(!bits.length&&j.state!=='running')bits.push('no result recorded');
  return '<div class="jid">'+bits.join(' &nbsp;·&nbsp; ')+'</div>'+
    (r.weights?'<div class="cmd">'+esc(r.weights)+'</div>':'');
}
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
        /* a run that finished has nothing to continue: ultralytics answers
           resume with "nothing to resume", after a job record was made */
        (!run&&j.run&&j.run.resumable&&j.state!=='done'
          ? '<button class="btn" data-resume="'+esc(j.id)+
            '" title="continue from weights/last.pt, with the arguments this '+
            'run recorded">resume</button>':'')+
        (run?'<button class="btn warn" data-stop="'+esc(j.id)+
             '">stop</button>'
            :'<button class="btn" data-forget="'+esc(j.id)+
             '" title="remove this record">clear</button>')+
      '</div>'+
      (pct!=null?'<div class="bar"><i style="width:'+pct+'%"></i></div>'+
        '<div class="jid">'+esc(p.what)+' — '+p.done+' of '+p.total+'</div>':'')+
      outcome(j)+
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
  var rs=t.getAttribute('data-resume');
  if(rs){
    var job=null; STATE.jobs.forEach(function(x){if(x.id===rs)job=x});
    if(!job||!job.run){fail(new Error('there is no run to resume'));return}
    if(!window.confirm('Resume '+job.run.name+' from epoch '+
       (job.run.epochs||0)+'? It continues with the arguments that run '+
       'recorded, not with anything set above.'))return;
    t.disabled=true;
    api('/api/train/resume',{family:(job.meta||{}).family,run:job.run.name})
      .then(function(r){say('say','resuming — '+r.job.id,8000);return refresh()})
      .catch(fail).then(function(){t.disabled=false});
    return;
  }
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
/* Gigabytes, one button, no undo -- so it says what it is about to remove
   and what that costs, and the server refuses a base or anything a running
   job is reading whatever this asks for. */
$('dsdel').addEventListener('click',function(){
  var sel=$('dataset'), id=sel.value||
    (sel.options[sel.selectedIndex]||{}).value;
  if(!id){fail(new Error('there is no dataset selected'));return}
  var d=null; STATE.datasets.forEach(function(x){if(x.id===id)d=x});
  var size=d&&d.counts?n(d.counts.total)+' images':'its images';
  if(!window.confirm('Delete '+id+'? That removes '+size+' and the bundle '+
     'that records how it was built. Runs already trained on it keep their '+
     'own copy of that record.'))return;
  $('dsdel').disabled=true;
  api('/api/train/dataset-delete',{dataset:id}).then(function(j){
    say('say','deleted '+id+(j.freed?' — '+Math.round(j.freed/1e9*10)/10+
        ' GB back':''),7000);
    $('dataset').value='';
    return refresh();
  }).catch(fail).then(function(){$('dsdel').disabled=false});
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
  /* selectable so it can be deleted, never trainable: the launcher refuses
     it too, this is only so the refusal arrives before the confirm box */
  var pick=null; STATE.datasets.forEach(function(x){if(x.id===ds)pick=x});
  if(pick&&pick.unfinished){
    fail(new Error(ds+' is an unfinished build — delete it and build again'));
    return;
  }
  var over=overrides();
  var lines=Object.keys(over).sort().map(function(k){return k+'='+over[k]});
  if(!window.confirm('Train '+(famOf()||{}).title+' on '+ds+
     ($('wsel').value?'\nstarting from '+$('wsel').value:'')+
     (lines.length?'\nwith '+lines.join(', '):'\nwith the inherited parameters')+
     '?\n\nIt runs in the background on the GPU.'))return;
  $('train').disabled=true;
  var body={family:FAM,dataset:ds,params:over};
  if($('wsel').value)body.weights=$('wsel').value;
  api('/api/train/start',body)
    .then(function(j){say('say','training — '+j.job.id,8000);return refresh()})
    .catch(fail).then(function(){$('train').disabled=false});
});
$('pmore').addEventListener('click',function(){ALL=!ALL;paintParams()});
$('preset').addEventListener('click',function(){EDITS={};loadParams()});
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
