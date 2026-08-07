#!/usr/bin/env python3
"""
Adversarial test for the /review bulk-flagging page's client JS.

``node --check`` only proves the script parses. This EXECUTES the script
extracted from dashboard.REVIEW_HTML under node against a stub DOM and drives
the real user path: load, flag, backfill, undo, paginate, keyboard, lightbox.

The stub DOM is deliberately small but honest about the three things this page
actually leans on:

  * ``innerHTML`` assignment creates queryable children AND registers their
    ids, because tile()/showUndo()/openLb() all build markup that way and then
    immediately look the pieces back up.
  * ``querySelector('.card[data-name="..."]')`` -- the page addresses tiles by
    crop name, not index, so a stale index can never flag the wrong image.
  * ``getComputedStyle(grid).gridTemplateColumns`` -- arrow-key up/down needs
    the live column count.

Cases cover the normal payload, a flag that the server REFUSES (must roll
back, not silently drop the tile), undo restoring position, a fetch that
fails, the empty queue, quote/tag injection in image_id, and every keyboard
binding. A ReferenceError, a TypeError or any other throw fails the test.

Requires node on PATH; skips (exit 0, loud message) if absent.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DASH = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')


def load_dashboard():
    spec = importlib.util.spec_from_file_location('dash_under_test', DASH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def crop(i, conf=0.5, full=True, iid=None):
    return {'name': '%d_%s_%03d.jpg' % (1_700_000_000_000 + i,
                                        iid or ('img%d' % i),
                                        int(round(conf * 100))),
            'image_id': iid or ('img%d' % i),
            'ts': 1_700_000_000_000 + i,
            'conf': conf,
            'has_full': full}


HARNESS = r"""
// ── stub DOM ────────────────────────────────────────────────────────────────
const failures = [];
let COLS = 5;

function parseKids(html) {
  // shallow: every tag with a class= and/or id= becomes one child node
  const out = [];
  const re = /<(\w+)([^>]*)>/g;
  let m;
  while ((m = re.exec(html))) {
    const attrs = m[2];
    const cm = /class="([^"]*)"/.exec(attrs);
    const im = /id="([^"]*)"/.exec(attrs);
    const sm = /src="([^"]*)"/.exec(attrs);
    if (!cm && !im) continue;
    const el = new El(m[1]);
    if (cm) el.className = cm[1];
    if (im) { el.id = im[1]; byId[im[1]] = el; }
    if (sm) el.src = sm[1];
    // every data-* attribute, not one hardcoded name: delegated handlers use
    // them to work out which control a click came from
    let dm; const dre = /data-([\w-]+)="([^"]*)"/g;
    while ((dm = dre.exec(attrs))) el.dataset[dm[1]] = dm[2];
    const am = /aria-expanded="([^"]*)"/.exec(attrs);
    if (am) el._attrs['aria-expanded'] = am[1];
    out.push(el);
  }
  return out;
}

const byId = {};
const allEls = [];

// what a browser's textContent -> innerHTML round trip actually escapes.
// NOT quotes -- which is exactly why the page needs att() for title="".
function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;');
}

class El {
  constructor(tag) {
    this.tagName = (tag || 'div').toUpperCase();
    this.className = ''; this._id = ''; this.dataset = {}; this.style = {};
    this.children = []; this.parentNode = null; this._text = '';
    this.hidden = false; this.disabled = false; this.src = '';
    this._html = ''; this.onclick = null; this.onchange = null;
    this._attrs = {};
    this.onmousedown = null; this.value = ''; this._listeners = {};
    this.onload = null; this.naturalWidth = 0; this.naturalHeight = 0;
    this.clientWidth = 0; this.clientHeight = 0;
    this.offsetLeft = 0; this.offsetTop = 0;
    this.scrollLeft = 0; this.scrollTop = 0;
    allEls.push(this);
  }
  // assigning .id must make the node findable, exactly as in a real document
  // (showUndo() builds the toast with `t.id='tbox'` and later looks it up)
  set id(v) { this._id = v; if (v) byId[v] = this; }
  get id() { return this._id; }
  set innerHTML(v) {
    this._html = v; this._text = '';
    for (const c of this.children) {
      const k = allEls.indexOf(c); if (k >= 0) allEls.splice(k, 1);
    }
    this.children = parseKids(v);
    for (const c of this.children) c.parentNode = this;
  }
  get innerHTML() { return this._html; }
  // A <select> is not just a box with a value: four painters on this page
  // BUILD its options and a fifth reads the chosen one's text back out to
  // label a chip. Without options/selectedIndex the read threw, load()'s
  // promise swallowed it, and every later assertion failed for the wrong
  // reason.
  get options() {
    const out = [];
    const re = /<option([^>]*)>([\s\S]*?)<\/option>/g;
    let m;
    while ((m = re.exec(this._html))) {
      const v = /value="([^"]*)"/.exec(m[1]);
      out.push({ value: v ? v[1].replace(/&quot;/g, '"') : m[2],
                 // a browser hands back option.text already decoded, so
                 // the stub must too or a label reads as raw entities
                 text: m[2].replace(/&middot;/g, '·')
                           .replace(/&mdash;/g, '—')
                           .replace(/&ldquo;/g, '\u201c')
                           .replace(/&rdquo;/g, '\u201d')
                           .replace(/&amp;/g, '&').replace(/&lt;/g, '<')
                           .replace(/&gt;/g, '>') });
    }
    return out;
  }
  get selectedIndex() {
    const o = this.options;
    for (let i = 0; i < o.length; i++) if (o[i].value === this.value) return i;
    return o.length ? 0 : -1;
  }
  // esc() in the page is `d.textContent = t; return d.innerHTML` -- model it
  set textContent(v) { this._text = String(v); this._html = escHtml(v);
                       this.children = []; }
  get textContent() {
    if (this._text) return this._text;
    // set via innerHTML: a browser still reports the text inside it
    return this._html.replace(/<[^>]*>/g, '')
      .replace(/&middot;/g, '·').replace(/&mdash;/g, '—')
      .replace(/&times;/g, '×').replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>');
  }
  get classList() {
    const self = this;
    return {
      add(c) { if (!self.className.split(' ').includes(c))
                 self.className = (self.className + ' ' + c).trim(); },
      remove(c) { self.className = self.className.split(' ')
                    .filter(x => x && x !== c).join(' '); },
      contains(c) { return self.className.split(' ').includes(c); },
      toggle(c, on) { on ? this.add(c) : this.remove(c); },
    };
  }
  appendChild(n) {
    if (n && n.__frag) { for (const c of n.children) this.appendChild(c); return n; }
    n.parentNode = this; this.children.push(n); return n;
  }
  insertBefore(n, ref) {
    n.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(n); else this.children.splice(i, 0, n);
    return n;
  }
  removeChild(n) {
    const i = this.children.indexOf(n);
    if (i >= 0) this.children.splice(i, 1);
    n.parentNode = null; return n;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  addEventListener(t, f) { (this._listeners[t] = this._listeners[t] || []).push(f); }
  focus() {}
  scrollIntoView() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth,
                                     height: this.clientHeight }; }
  matches(sel) { return matchSel(this, sel); }
  // delegated handlers walk up from the event target; the chip row and
  // the grid both rely on it
  closest(sel) {
    for (let n = this; n; n = n.parentNode) if (matchSel(n, sel)) return n;
    return null;
  }
  querySelector(sel) { return descendants(this).find(e => matchSel(e, sel)) || null; }
  querySelectorAll(sel) { return descendants(this).filter(e => matchSel(e, sel)); }
  getAttribute(k) {
    if (k.startsWith('data-')) {
      const v = this.dataset[k.slice(5)];
      return v === undefined ? null : v;
    }
    return this._attrs[k] === undefined ? null : this._attrs[k];
  }
  setAttribute(k, v) { this._attrs[k] = String(v); }
}

function descendants(root) {
  const out = [];
  (function walk(n) { for (const c of n.children) { out.push(c); walk(c); } })(root);
  return out;
}
function matchSel(el, sel) {
  // supports ".cls", ".cls[attr="v"]", "#id"
  const am = /^\.([\w-]+)\[([\w-]+)="(.*)"\]$/.exec(sel);
  if (am) {
    if (!el.classList.contains(am[1])) return false;
    const key = am[2].replace(/^data-/, '');
    return String(el.dataset[key]) === am[3].replace(/\\(.)/g, '$1');
  }
  // compound classes: '.fbtn.no' must not be read as one class named 'fbtn.no'
  if (sel[0] === '.')
    return sel.slice(1).split('.').every(c => el.classList.contains(c));
  if (sel[0] === '#') return el.id === sel.slice(1);
  return el.tagName === sel.toUpperCase();
}

const root = new El('body');
// getElementById only sees attached nodes -- otherwise a removed toast stays
// "findable" and the page silently reuses a detached element forever
function attached(el) {
  for (let n = el; n; n = n.parentNode) if (n === root) return true;
  return false;
}
const document = {
  body: root,
  createElement: t => new El(t),
  createDocumentFragment: () => { const f = new El('frag'); f.__frag = true; return f; },
  getElementById: id => {
    const e = byId[id];
    return e && (attached(e) || e.__page) ? e : null;
  },
  querySelector: s => descendants(root).find(e => matchSel(e, s)) || null,
  querySelectorAll: s => descendants(root).filter(e => matchSel(e, s)),
  addEventListener: (t, f) => (docL[t] = docL[t] || []).push(f),
};
const docL = {};
const CSS = { escape: s => String(s).replace(/([^\w-])/g, '\\$1') };
const beacons = [];
let scrolls = [];
const window = {
  matchMedia: () => ({ matches: false }),
  addEventListener: (t, f) => (winL[t] = winL[t] || []).push(f),
  scrollTo: (a) => scrolls.push(a),
};
const winL = {};
const navigator = { sendBeacon: (u, b) => { beacons.push(u); return true; } };
const Blob = function (parts, opts) { this.parts = parts; this.type = opts && opts.type; };
function getComputedStyle() {
  return { gridTemplateColumns: new Array(COLS).fill('100px').join(' ') };
}
function requestAnimationFrame(f) { f(); }
function setTimeout(f, ms) { timers.push({ f, ms }); return timers.length; }
// setInterval stubbed to a no-op handle: a real one keeps node's event
// loop alive forever, hanging this test the moment the page starts a poll
function setInterval() { return 0; }
function clearInterval() {}
function clearTimeout(h) { if (h) timers[h - 1] = null; }
const timers = [];
function runTimers() { const t = timers.slice(); timers.length = 0;
                       for (const x of t) if (x) x.f(); }

// ── controllable fetch ──────────────────────────────────────────────────────
let RESP = {};           // url-substring -> () => value | 'reject'
const CALLS = [];
function fetch(url, opts) {
  CALLS.push({ url, body: opts && opts.body ? JSON.parse(opts.body) : null });
  // longest key first: '/api/review' is a prefix of '/api/review/seen', so
  // insertion order would silently answer the seen POST with a page payload
  for (const k of Object.keys(RESP).sort((a, b) => b.length - a.length)) {
    if (String(url).includes(k)) {
      const v = RESP[k](url, opts);
      if (v === 'reject') return Promise.reject(new Error('boom'));
      return Promise.resolve({ ok: true, json: () => Promise.resolve(v) });
    }
  }
  return Promise.reject(new Error('unstubbed ' + url));
}

// ── the page's own element graph (built from the real markup ids) ───────────
for (const id of ['left','done','seen','dups','unkeep','bal','balFill','balPend','balMain','balSub','balLg','pg','pg2','next','next2','mode','verdict',
                  'foot','grid','state','sort','size','reload','country','leftlab',
                  // the model-suggestion filter; without it paintSuggest and
                  // its onchange bind against null and kill the whole script
                  'suggest','balNum','balNumU','balLeft',
                  // crop-suggestion progress strip, moved here from the dashboard
                  'trg','trgState','trgSub','trgPct','trgFill','trgDot','trgRun','leashN',
                  // findmsg is what says the search cannot work; leaving it
                  // out of the stub makes paintFind's guard skip the whole
                  // branch, so every state would 'pass' untested
                  // the guesser toggle and its caveat line; without them
                  // paintBackends' guard skips and t24 would pass untested
                  'leashf','find','findterms','findmsg','trgModel','trgNote',
                  // the folded legend that explains the dropdown's percentage
                  'trgNoteSum','trgNoteBasis','trgNoteWhich','trgNoteCaveat',
                  // the gate's own filter axis
                  'gatef',
                  // the redesigned block: caption, applied-filter chips
                  // and the disclosure holding the controls
                  'cap','chips','narrow','npanel','ngrpLooks','ngrpWho']) {
  const e = new El(id === 'grid' || id === 'state' || id === 'foot' ? 'div' : 'span');
  e.id = id; e.__page = true; root.appendChild(e);
}
byId['sort'].value = 'conf';
byId['size'].value = '50';
byId['country'].value = '';

// ── run the page script ─────────────────────────────────────────────────────
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
let API;
try {
  API = new Function('document','window','CSS','fetch','getComputedStyle',
    'requestAnimationFrame','setTimeout','clearTimeout','setInterval','clearInterval','docL','navigator','Blob',
    src + '\nreturn {load,render,flag,undo,openLb,closeLb,stepLb,tile,score,'
        + 'idx,mark,cols,hideToast,showUndo,'
        + 'markSeen,imgScale,saveBox,paintBox,fitBox,fitImage,zoomBy,'
        + 'flushSave,dirty,'
        // the guesser strip repaints on a 5s timer, and setInterval is a
        // no-op here; without a way to drive one poll by hand nothing in
        // that closure is ever exercised
        + 'trgPoll:()=>window.__trgPoll&&window.__trgPoll(),'
        + 'setBackend:(b)=>{const s=document.getElementById("trgModel");'
        + 's.value=b;(s._listeners.change||[]).forEach(f=>f.call(s));},'
        + 'st:()=>({page,size,sort,items,reserve,pages,sel,todoN,flaggedN,'
        + 'seenN,session,lastUndo,lb})};')(
    document, window, CSS, fetch, getComputedStyle, requestAnimationFrame,
    setTimeout, clearTimeout, setInterval, clearInterval, docL, navigator, Blob);
} catch (e) {
  console.log('FAIL: could not evaluate the review script: ' + e);
  process.exit(1);
}

const FIX = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const CROPS = FIX.crops;
// Which page elements the real markup ships hidden. Taken from the
// markup rather than assumed, because a stub that starts everything
// visible lets a panel 'pass' a test for being shut.
for (const id of (FIX.hidden || [])) if (byId[id]) byId[id].hidden = true;
// ...and the options a <select> ships in the markup. Four painters BUILD
// their options at runtime, but #mode, #verdict, #sort and #size carry
// theirs in the HTML -- so a stub that starts them empty makes every
// read of a chosen option's text return '' and any test of one vacuous.
for (const [id, html] of Object.entries(FIX.options || {}))
  if (byId[id]) { byId[id].innerHTML = html; byId[id].value =
    (byId[id].options[0] || {}).value || ''; }
// The panel's shape: which controls sit in which group, read off the
// markup. trimGroups() walks that tree to decide whether a group still
// offers anything, and a flat stub gave it nothing to walk -- so the
// heading-over-nothing it exists to prevent could not be tested at all.
for (const [gid, ids] of Object.entries(FIX.groups || {})) {
  const g = byId[gid]; if (!g) continue;
  g.className = ((g.className || '') + ' ngrp').trim();
  const row = new El('div'); row.className = 'nrow'; g.appendChild(row);
  for (const id of ids) if (byId[id]) row.appendChild(byId[id]);
}
for (const gid of (FIX.owned || [])) if (byId[gid]) byId[gid].dataset.own = '1';
function payload(items, reserve, extra) {
  return Object.assign({ items, reserve: reserve || [], page: 0, pages: 2,
                         size: 50, sort: 'conf',
                         total_unflagged: 120, flagged_total: 30 }, extra || {});
}
function key(k) {
  for (const f of (docL['keydown'] || []))
    f({ key: k, preventDefault(){}, target: { tagName: 'BODY' } });
}
function ck(cond, msg) { if (!cond) failures.push(msg); }
// keydown handlers fire flag()/undo() without awaiting; drain the microtask
// queue deep enough that their fetch chains have settled
async function flush(n) { for (let i = 0; i < (n || 12); i++) await Promise.resolve(); }
function toastUp() { return root.children.some(c => c.id === 'tbox'); }

// ── 1. load + render ────────────────────────────────────────────────────────
async function t1() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)) };
  await API.load(); await flush();
  const cards = document.querySelectorAll('.card');
  ck(cards.length === 6, 't1: rendered ' + cards.length + ' tiles, want 6');
  ck(byId['left'].textContent === '120', 't1: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '30', 't1: done=' + byId['done'].textContent);
  /* The queue is consumed from the head, not paged through: nav() banks the
     screen before loading, so an offset on top of that skipped a screenful
     every turn. The label counts what is LEFT rather than naming an offset
     that no longer moves, and Prev is gone -- Restore kept is the way back. */
  ck(/^6 shown \u00b7 \d+ left$/.test(byId['pg'].textContent),
     't1: pg=' + byId['pg'].textContent);
  ck(byId['next'].disabled === false, 't1: next disabled with crops left');
  ck(byId['prev'] === undefined || byId['prev'] === null ||
     !byId['prev'].onclick, 't1: Prev still wired');
  ck(byId['next'].disabled === false, 't1: next disabled with 2 pages');
  // nothing may be pre-selected: a highlighted first tile reads as a choice
  // the user did not make
  ck(API.st().sel === -1, 't1: something was pre-selected, sel=' + API.st().sel);
  ck(!document.querySelectorAll('.card').some(c => c.classList.contains('sel')),
     't1: a tile is marked selected on load');
  // the confidence rail must reflect conf, not be a constant
  const rails = document.querySelectorAll('.rail');
  ck(rails.length === 6, 't1: ' + rails.length + ' rails for 6 tiles');
}

// ── 2. flag: surgical removal + backfill from reserve ───────────────────────
async function t2() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  const before = API.st().items.map(c => c.name);
  await API.flag(2); await flush();
  const after = API.st().items.map(c => c.name);
  ck(!after.includes(before[2]), 't2: flagged crop still in items');
  ck(after.length === 6, 't2: grid shrank to ' + after.length + ', want backfill to 6');
  ck(after[5] === CROPS.normal[6].name, 't2: backfilled with the wrong crop');
  ck(API.st().reserve.length === 2, 't2: reserve not consumed');
  ck(document.querySelectorAll('.card').length === 6,
     't2: DOM has ' + document.querySelectorAll('.card').length + ' tiles');
  ck(byId['left'].textContent === '119', 't2: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '31', 't2: done=' + byId['done'].textContent);
  const post = CALLS[CALLS.length - 1];
  ck(post.body && post.body.name === before[2] && post.body.label === 'false_positive',
     't2: wrong flag POST body: ' + JSON.stringify(post.body));
  ck(!!byId['undoB'], 't2: no undo control in the toast');
  // DOM order must still track items order, or arrow keys select the wrong tile
  const dom = document.querySelectorAll('.card').map(e => e.dataset.name);
  ck(JSON.stringify(dom) === JSON.stringify(after),
     't2: DOM order diverged from items order');
}

// ── 3. undo restores the crop AT ITS OLD INDEX ─────────────────────────────
async function t3() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6),
                                        CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  const before = API.st().items.map(c => c.name);
  const s0 = API.st().session;
  await API.flag(2); await flush();
  await API.undo(); await flush();
  const after = API.st().items.map(c => c.name);
  ck(after[2] === before[2], 't3: undo put the crop at index ' +
     after.indexOf(before[2]) + ', want 2');
  ck(byId['left'].textContent === '120', 't3: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '30', 't3: done=' + byId['done'].textContent);
  ck(API.st().session === s0, 't3: session counter not decremented');
  const dom = document.querySelectorAll('.card').map(e => e.dataset.name);
  ck(JSON.stringify(dom) === JSON.stringify(after), 't3: DOM order wrong after undo');
  ck(!toastUp(), 't3: toast still present after undo');
  // a flag pulls one crop out of `reserve`; undo must hand it back, or
  // repeated flag/undo cycles grow the page without bound
  ck(after.length === before.length,
     't3: page length drifted ' + before.length + ' -> ' + after.length);
  ck(API.st().reserve.length === 3, 't3: reserve not restored, has ' +
     API.st().reserve.length + ' want 3');
}

// ── 4. a REFUSED flag must roll back, never drop the tile ──────────────────
async function t4() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/detect/flag': () => ({ ok: false, error: 'nope' }) };
  await API.load(); await flush();
  const n = API.st().items.length;
  const name = API.st().items[1].name;
  await API.flag(1); await flush();
  ck(API.st().items.length === n, 't4: refused flag still removed the crop');
  ck(API.st().items[1].name === name, 't4: refused flag reordered items');
  const card = document.querySelector('.card[data-name="' + name + '"]');
  ck(card && !card.classList.contains('go'),
     't4: tile left in the exiting state after a refusal');
  ck(byId['left'].textContent === '120', 't4: counters moved on a refusal');
}

// ── 5. fetch failure -> error state with a retry, not a blank page ─────────
async function t5() {
  RESP = { '/api/review': () => 'reject' };
  await API.load(); await flush(); await Promise.resolve();
  ck(/Could not reach/.test(byId['state'].innerHTML), 't5: no error state shown');
  ck(!!byId['retry'], 't5: no retry control');
  ck(byId['foot'].hidden === true, 't5: pager left visible over an error');
  ck(document.querySelectorAll('.card').length === 0, 't5: stale tiles kept');
}

// ── 6. empty queue -> an invitation, not a void ────────────────────────────
async function t6() {
  RESP = { '/api/review': () => payload([], [], { total_unflagged: 0,
                                                  flagged_total: 500, pages: 1 }) };
  await API.load(); await flush();
  ck(/Queue is clear/.test(byId['state'].innerHTML), 't6: no empty state');
  ck(!!byId['rl2'], 't6: no way to re-check from the empty state');
  ck(byId['left'].textContent === '0', 't6: left=' + byId['left'].textContent);
  ck(byId['done'].textContent === '500', 't6: done=' + byId['done'].textContent);
  ck(byId['foot'].hidden === true, 't6: pager shown for a single page');
}

// ── 7. injection in image_id must not reach markup unescaped ───────────────
async function t7() {
  RESP = { '/api/review': () => payload(CROPS.hostile, []) };
  await API.load(); await flush();
  const h = byId['grid'].children.map(c => c.innerHTML).join('');
  // no raw tag may appear that we did not write ourselves
  ck(!/<script/i.test(h), 't7: <script survived into tile markup');
  ck(!/<img\s+src=x/i.test(h), 't7: <img src=x survived into tile markup');
  ck(h.includes('&lt;'), 't7: nothing was escaped at all');
  // The id also lands in a title="". esc() does NOT touch quotes, so a bare
  // esc() there lets `"><script>` close the attribute AND the tag. Assert the
  // exact fully-escaped attribute value rather than hunting for fragments.
  ck(h.includes('title="&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"'),
     't7: title="" not fully escaped -- got ' +
     String((/title="([^"]*)"/.exec(h) || [])[1]));
  const src = byId['grid'].children[0].querySelector('.thumb').src;
  ck(!src.includes('"') && !src.includes('<'), 't7: thumb src not URL-encoded: ' + src);
}

// ── 8. keyboard: arrows honour the column count, F/U/Enter/Esc all bound ───
async function t8() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 12), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  COLS = 5;
  const last = API.st().items.length - 1;
  ck(API.cols() === 5, 't8: cols()=' + API.cols());
  // first arrow from "nothing selected" lands on tile 0, not tile 1
  ck(API.st().sel === -1, 't8: page did not open unselected');
  key('ArrowRight'); ck(API.st().sel === 0, 't8: first arrow -> ' + API.st().sel);
  key('ArrowRight'); ck(API.st().sel === 1, 't8: right -> ' + API.st().sel);
  key('ArrowDown');  ck(API.st().sel === 6, 't8: down -> ' + API.st().sel + ', want 6');
  key('ArrowUp');    ck(API.st().sel === 1, 't8: up -> ' + API.st().sel);
  key('ArrowLeft');  ck(API.st().sel === 0, 't8: left -> ' + API.st().sel);
  key('ArrowLeft');  ck(API.st().sel === 0, 't8: left ran past 0');
  for (let i = 0; i < 40; i++) key('ArrowDown');
  ck(API.st().sel === last, 't8: down ran past the end -> ' + API.st().sel +
     ', last is ' + last);
  for (let i = 0; i < 40; i++) key('ArrowUp');
  ck(API.st().sel === 0, 't8: up ran past 0 -> ' + API.st().sel);
  // Enter opens the lightbox on a crop that has a full frame
  key('Enter');
  ck(!!API.st().lb, 't8: Enter did not open the lightbox');
  key('Escape'); ck(!API.st().lb, 't8: Escape did not close the lightbox');
  // F flags the selection
  const n0 = API.st().items.length;
  key('f'); await flush();
  ck(API.st().items.length === n0 - 1, 't8: F did not flag the selection');
  // U undoes it
  key('u'); await flush();
  ck(API.st().items.length === n0, 't8: U did not undo');
  // typing in a control must not steal the key
  const sel0 = API.st().sel, n1 = API.st().items.length;
  for (const f of (docL['keydown'] || []))
    f({ key: 'f', preventDefault(){}, target: { tagName: 'SELECT' } });
  await flush();
  ck(API.st().items.length === n1 && API.st().sel === sel0,
     't8: F fired while focus was in a <select>');
}

// ── 9. lightbox: only steps to crops that HAVE a full frame ───────────────
async function t9() {
  RESP = { '/api/review': () => payload(CROPS.mixed, []) };
  await API.load(); await flush();
  // has_full only says whether a burned-in PREVIEW was saved; the editor
  // reads the original jpg, so every crop opens
  API.openLb(0);
  ck(!!API.st().lb, 't9: refused a crop with no preview frame');
  ck(String(byId['lbi'].src).startsWith('/orig?name='),
     't9: lightbox did not load the ORIGINAL (needed to edit): ' + byId['lbi'].src);
  ck(!String(byId['lbi'].src).includes(' '),
     't9: lightbox src not URL-encoded: ' + byId['lbi'].src);
  API.openLb(1);
  ck(API.st().sel === 1, 't9: opening did not move the selection');
  const first = byId['lbi'].src;
  await API.stepLb(1); await flush();
  ck(byId['lbi'].src !== first, 't9: step(1) did not advance');
  await API.stepLb(1); await API.stepLb(1); await API.stepLb(1); await flush();
  ck(!!API.st().lb, 't9: stepping past the end closed the lightbox');
  ck(API.st().sel >= 0 && API.st().sel < CROPS.mixed.length,
     't9: stepped outside the page, sel=' + API.st().sel);
  API.closeLb(); ck(!API.st().lb, 't9: closeLb left the overlay up');
  ck(document.body.style.overflow === '', 't9: page scroll not restored');
}

// ── 10. flagging the LAST crop falls into the empty state ─────────────────
async function t10() {
  RESP = { '/api/review': () => payload([CROPS.normal[0]], []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  await API.flag(0); await flush();
  ck(/Queue is clear/.test(byId['state'].innerHTML),
     't10: no empty state after the last crop was flagged');
  ck(API.st().sel === -1, 't10: sel should be -1 on an empty page, got ' + API.st().sel);
  // and undo must climb back out of the empty state
  await API.undo(); await flush();
  ck(API.st().items.length === 1, 't10: undo did not restore the last crop');
  ck(document.querySelectorAll('.card').length === 1,
     't10: undo restored state but not the tile');
}

// ── 11. double-flag of the same crop must not double-post ─────────────────
async function t11() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  CALLS.length = 0;
  const p1 = API.flag(1), p2 = API.flag(1);
  await p1; await p2; await Promise.resolve(); await Promise.resolve();
  const posts = CALLS.filter(c => String(c.url).includes('/api/detect/flag'));
  ck(posts.length === 1, 't11: ' + posts.length + ' POSTs for one crop');
  ck(API.st().items.length === 3, 't11: items=' + API.st().items.length);
}

// ── 12. the 5 s undo window really expires ────────────────────────────────
async function t12() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  await API.flag(0); await flush();
  ck(!!API.st().lastUndo, 't12: nothing staged for undo right after a flag');
  runTimers();
  ck(!API.st().lastUndo, 't12: undo still live after the timer fired');
  const n = API.st().items.length;
  await API.undo(); await Promise.resolve();
  ck(API.st().items.length === n, 't12: undo worked after the window closed');
}

// ── 13. flagging must not advance the selection on its own ───────────────
async function t13() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), CROPS.normal.slice(6, 9)),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  // MOUSE path: pressing the flag button must not select the tile, and after
  // removal nothing may be selected -- otherwise the highlight lands on
  // whatever slid into that index, which reads as an auto-advance
  const card = document.querySelectorAll('.card')[1];
  card.onmousedown({ target: { closest: sel => sel === '.acts' ? {} : null } });
  ck(API.st().sel === -1, 't13: pressing a verdict button selected the tile');
  await API.flag(1); await flush();
  ck(API.st().sel === -1,
     't13: mouse flag left a selection (auto-advance), sel=' + API.st().sel);
  ck(!document.querySelectorAll('.card').some(c => c.classList.contains('sel')),
     't13: a tile is highlighted after a mouse flag');
  // pressing elsewhere on the tile still selects it (needed for the lightbox)
  const c2 = document.querySelectorAll('.card')[2];
  c2.onmousedown({ target: { closest: () => null } });
  ck(API.st().sel === 2, 't13: clicking the tile body no longer selects');
  // KEYBOARD path: F keeps the position so the next crop can be judged.
  // items.length does NOT drop -- the reserve backfills -- so assert on
  // identity: the crop that was selected must be gone.
  const gone = API.st().items[2].name;
  key('f'); await flush();
  ck(!API.st().items.some(c => c.name === gone),
     't13: F did not flag the selected crop');
  ck(API.st().sel === 2,
     't13: F flow lost its position, sel=' + API.st().sel + ' want 2');
  // D marks a low-confidence detection as a REAL dog -> the other ledger
  const lbl = [];
  RESP['/api/detect/flag'] = (u, o) => { lbl.push(JSON.parse(o.body).label);
                                         return { ok: true }; };
  const dogCrop = API.st().items[2].name;
  key('d'); await flush();
  ck(lbl[lbl.length - 1] === 'true_positive',
     't13: D sent label ' + lbl[lbl.length - 1]);
  ck(!API.st().items.some(c => c.name === dogCrop),
     't13: D did not remove the crop from the queue');
  // undo must return it under the SAME label, or it is undone in the wrong
  // ledger and stays flagged forever in the other one
  await API.undo(); await flush();
  const last = CALLS[CALLS.length - 1];
  ck(last.body && last.body.undo === true && last.body.label === 'true_positive',
     't13: undo sent ' + JSON.stringify(last.body));
}

// ── 14. paging banks the screen as reviewed, so it never comes back ──────
async function t14() {
  let posted = null;
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/review/seen': (u, o) => {
             posted = JSON.parse(o.body).names; return { ok: true, seen_total: 99 };
           } };
  await API.load(); await flush();
  const onScreen = API.st().items.map(c => c.name);
  CALLS.length = 0;
  byId['next'].onclick(); await flush();
  ck(posted !== null, 't14: paging did not record the screen as reviewed');
  ck(JSON.stringify(posted) === JSON.stringify(onScreen),
     't14: banked the wrong crops');
  ck(API.st().seenN === 99, 't14: reviewed total not tracked, ' + API.st().seenN);
  // the seen POST must land BEFORE the next page is fetched, or the server
  // computes the next page from a pool that still contains what we just kept
  const order = CALLS.map(c => String(c.url).includes('/seen') ? 'seen' : 'page');
  ck(order.indexOf('seen') >= 0 && order.indexOf('seen') < order.indexOf('page'),
     't14: fetched the next page before banking this one: ' + order.join(','));
  // an empty grid must not POST an empty list
  RESP['/api/review'] = () => payload([], []);
  posted = null;
  await API.load(); await flush();
  await API.markSeen(); await flush();
  ck(posted === null, 't14: posted an empty screen as reviewed');
}

// ── 15. Restore kept: confirmed, scoped, and never touches the flags ─────
async function t15() {
  let body = null, alerted = null, kept = 7;   // server-side kept count
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), [], {seen_total: kept}),
           '/api/review/seen': (u, o) => { body = JSON.parse(o.body);
                                           if (body.reset) kept = 0;
                                           return { ok: true, restored: 7, seen_total: kept }; },
           '/api/dataset': () => ({ dog: 10, not_dog: 2, new_flags: 0,
                                    yield_per_flag: 0.822, dataset: 'x' }) };
  await API.load(); await flush();
  ck(API.st().seenN === 7, 't15: kept total not read from the payload');

  // declining the confirm must do nothing at all
  window.confirm = () => false;
  body = null;
  byId['unkeep'].onclick(); await flush();
  ck(body === null, 't15: acted despite the confirm being declined');
  ck(API.st().seenN === 7, 't15: cleared the counter on a declined confirm');

  // accepting sends reset:true -- never a names list, which would BANK them
  window.confirm = () => true;
  byId['unkeep'].onclick(); await flush();
  ck(body && body.reset === true, 't15: did not send reset:true, sent ' +
     JSON.stringify(body));
  ck(!body.names, 't15: sent a names list on reset -- that would re-bank them');
  ck(API.st().seenN === 0, 't15: kept total not cleared after restore');

  // with nothing kept it must warn instead of prompting to restore nothing
  window.alert = m => { alerted = m; };
  window.confirm = () => { throw new Error('should not prompt with 0 kept'); };
  byId['unkeep'].onclick(); await flush();
  ck(alerted && /Nothing to restore/.test(alerted),
     't15: no guard when there is nothing to restore');
}

// ── 16. a new page starts at the top ─────────────────────────────────────
async function t16() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 6), []),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  scrolls = [];
  byId['next'].onclick(); await flush();
  ck(scrolls.length >= 1, 't16: paging did not scroll to the top');
  ck(scrolls[scrolls.length - 1] &&
     (scrolls[scrolls.length - 1].top === 0 || scrolls[scrolls.length - 1] === 0),
     't16: scrolled somewhere other than the top: ' +
     JSON.stringify(scrolls[scrolls.length - 1]));
  // flagging must NOT jump the page -- the user is mid-grid judging crops
  scrolls = [];
  await API.flag(0); await flush();
  ck(scrolls.length === 0, 't16: a flag scrolled the page');
  // undo must not either
  scrolls = [];
  await API.undo(); await flush();
  ck(scrolls.length === 0, 't16: an undo scrolled the page');
}

// ── 17. box editing keeps ORIGINAL pixels, whatever the render scale ─────
async function t17() {
  let posted = null;
  const BOX = { ok: true, image_id: 'img1', w: 4000, h: 3000, has_file: true,
                boxes: [{det_idx: 0, x1: 1000, y1: 800, x2: 1400, y2: 1200,
                         conf: 0.5}], saved: null };
  RESP = { '/api/review/box': (u, o) => { if (o) { posted = JSON.parse(o.body);
                                                   return { ok: true }; }
                                          return BOX; },
           '/api/review': () => payload([crop0()], []),
           '/api/detect/flag': () => ({ ok: true }) };
  await API.load(); await flush();
  // openLb rebuilds the overlay, so size the <img> AFTER it exists:
  // a 4000px image rendered at 800px is scale 0.2
  async function open0(){
    API.openLb(0); await flush();
    const im = byId['lbi'];
    im.naturalWidth = 4000; im.naturalHeight = 3000;
    byId['lbw'].clientWidth = 800; byId['lbw'].clientHeight = 600;
    API.fitImage();            // 800/4000 == 600/3000 == 0.2
  }
  await open0();
  ck(byId['lbbox'].hidden === false, 't17: box overlay never shown');
  ck(Math.abs(API.imgScale() - 0.2) < 1e-9, 't17: scale=' + API.imgScale());
  // overlay is placed in DISPLAY px
  ck(byId['lbbox'].style.left === '200px',
     't17: left=' + byId['lbbox'].style.left + ' want 200px (1000*0.2)');
  ck(byId['lbbox'].style.width === '80px',
     't17: width=' + byId['lbbox'].style.width + ' want 80px (400*0.2)');

  // drag the whole box 100 display px right = 500 ORIGINAL px
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => null },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: 100, clientY: 0 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(posted && Math.round(posted.box[0]) === 1500,
     't17: moved box x1=' + (posted && posted.box[0]) + ' want 1500 ORIGINAL px');
  ck(Math.round(posted.box[2]) === 1900, 't17: x2=' + posted.box[2]);
  ck(Math.round(posted.box[1]) === 800 && Math.round(posted.box[3]) === 1200,
     't17: vertical drifted on a horizontal drag');
  ck(posted.det_idx === 0, 't17: wrong det_idx ' + posted.det_idx);

  // resizing by a corner must move only that corner
  await open0();
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => 'se' },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: 20, clientY: 20 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(Math.round(posted.box[0]) === 1000 && Math.round(posted.box[1]) === 800,
     't17: SE handle moved the NW corner');
  ck(Math.round(posted.box[2]) === 1500 && Math.round(posted.box[3]) === 1300,
     't17: SE corner went to ' + posted.box[2] + ',' + posted.box[3]);

  // dragging far off-image must clamp inside the picture, not save negatives
  await open0();
  byId['lbbox']._listeners.mousedown[0]({ target: { getAttribute: () => null },
    clientX: 0, clientY: 0, preventDefault(){}, stopPropagation(){} });
  for (const f of (docL['mousemove'] || [])) f({ clientX: -99999, clientY: -99999 });
  for (const f of (docL['mouseup'] || [])) f({});
  await API.saveBox(); await flush();
  ck(posted.box.every(v => v >= 0), 't17: saved a negative coordinate: ' +
     JSON.stringify(posted.box));
  ck(posted.box[2] <= 4000 && posted.box[3] <= 3000,
     't17: saved past the image bounds');

  // a SMALL object must open zoomed in, not fitted to the whole frame --
  // a 30px box on a 4000px image is 6 screen px at fit, which is the
  // complaint this whole zoom model exists to answer
  BOX.boxes = [{det_idx: 0, x1: 2000, y1: 1500, x2: 2030, y2: 1530, conf: 0.5}];
  await open0();
  API.fitBox();
  const zBox = API.imgScale(), zFit = 0.2;
  ck(zBox > zFit * 5, 't17: fitBox barely zoomed: ' + zBox + ' vs fit ' + zFit);
  const px = 30 * zBox;
  ck(px > 150, 't17: a 30px object renders at only ' + Math.round(px) +
     ' screen px after Fit box');
  // and the handles must NOT have grown with it -- they are plain px in CSS,
  // so assert the box itself is what scaled
  ck(byId['lbbox'].style.width === px + 'px',
     't17: box width ' + byId['lbbox'].style.width + ' want ' + px + 'px');
  // one-pixel nudge stays one ORIGINAL pixel however deep the zoom
  const x0 = 2000;
  for (const f of (docL['keydown'] || []))
    f({ key: 'ArrowRight', shiftKey: true, preventDefault(){},
        target: { tagName: 'BODY' } });
  await API.saveBox(); await flush();
  ck(Math.round(posted.box[0]) === x0 + 1,
     't17: Shift+Arrow moved ' + (posted.box[0] - x0) + 'px, want exactly 1');
}
function crop0(){ return CROPS.normal[0]; }

// ── 18. box edits save themselves, and always before the verdict ─────────
async function t18() {
  const order = [];
  const BOX = { ok: true, image_id: 'img1', w: 4000, h: 3000, has_file: true,
                boxes: [{det_idx: 0, x1: 100, y1: 100, x2: 500, y2: 500,
                         conf: 0.5}], saved: null };
  RESP = { '/api/review/box': (u, o) => { if (o) { order.push('box');
                                                   return { ok: true }; }
                                          return BOX; },
           // three crops: stepping away must have somewhere to go
           '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           '/api/detect/flag': () => { order.push('flag'); return { ok: true }; } };
  await API.load(); await flush();
  API.openLb(0); await flush();
  byId['lbi'].naturalWidth = 4000; byId['lbi'].naturalHeight = 3000;
  byId['lbw'].clientWidth = 800; byId['lbw'].clientHeight = 600;
  API.fitImage();

  // there is no Save button any more
  ck(!byId['lbsave'], 't18: a Save box button still exists');

  // an edit schedules a save on its own -- no click
  order.length = 0;
  API.dirty(true);
  ck(order.length === 0, 't18: saved instantly, losing the debounce');
  runTimers(); await flush();
  ck(order[0] === 'box', 't18: an edit did not autosave, order=' + order);

  // a verdict must not reach the server before the pending box does
  order.length = 0;
  API.dirty(true);                     // dirty again, still debouncing
  byId['lbf'].onclick(); await flush(); await flush();
  ck(order.join(',') === 'box,flag',
     't18: verdict raced the box save, order=' + order.join(','));

  // stepping away also flushes first
  API.openLb(0); await flush();
  order.length = 0;
  API.dirty(true);
  await API.stepLb(1); await flush();
  ck(order[0] === 'box', 't18: stepping away dropped the pending edit');
}

// ── 19. the collapse count is surfaced, not silently swallowed ──────────
async function t19() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 4), [],
                                        {collapsed: 669, total_unflagged: 1881}) };
  await API.load(); await flush();
  ck(byId['dups'].textContent === '669',
     't19: hidden-repeat count not shown, got ' + byId['dups'].textContent);
  ck(byId['left'].textContent === '1,881', 't19: left=' + byId['left'].textContent);
  // a payload without the field must not print undefined/NaN
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 4), []);
  await API.load(); await flush();
  ck(/^[0-9,]+$/.test(byId['dups'].textContent),
     't19: non-numeric when the server omits collapsed: ' + byId['dups'].textContent);
}

// ── 20. the country filter reaches the server and repaints its options ──
async function t20() {
  const LIST = [{iso:'DEU',name:'Germany',n:904},{iso:'JPN',name:'Japan',n:838}];
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
                                        {countries: LIST, country: ''}),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  const opts = byId['country'].innerHTML;
  ck(/All countries/.test(opts), 't20: no "All countries" option');
  ck(/Germany \(904\)/.test(opts), 't20: option text lacks name+count: ' + opts);
  ck(/value="DEU"/.test(opts), 't20: option value is not the ISO code');

  // choosing one must send ?country= and reset to page 0
  CALLS.length = 0;
  byId['country'].value = 'DEU';
  await byId['country'].onchange.call(byId['country']);
  await flush(); await flush();
  const req = CALLS.map(c => c.url).filter(u => /\/api\/review\?/.test(u)).pop();
  ck(/country=DEU/.test(req || ''), 't20: filter not sent, url=' + req);
  ck(/page=0/.test(req || ''), 't20: filter did not reset to page 1: ' + req);

  // an unchanged option set must NOT be rewritten -- doing so on every page
  // turn drops an open dropdown mid-click
  const before = byId['country'].innerHTML;
  byId['country'].innerHTML = before + '<!--sentinel-->';
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
                                      {countries: LIST, country: 'DEU'});
  await API.load(); await flush();
  ck(/sentinel/.test(byId['country'].innerHTML),
     't20: options rebuilt although the list was identical');

  // a payload with no countries key must not blow up or wipe the control
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), []);
  await API.load(); await flush();
  ck(byId['country'].innerHTML.length > 0, 't20: control emptied when the '
     + 'server omitted countries');
}

// ── 21. a filtered count must not read as a global one ──────────────────
// 'left to review' is scoped to the country filter while 'flagged' and 'kept'
// stay all-time. Side by side with no marker, 198 next to 1,166 reads as
// "198 left in total".
async function t21() {
  const LIST = [{iso:'DEU',name:'Germany',n:904},{iso:'JPN',name:'Japan',n:838}];
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             {countries: LIST, country: '', total_unflagged: 2100,
              flagged_total: 1166}),
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  ck(byId['leftlab'].textContent === 'left to review',
     't21: unfiltered label changed: ' + byId['leftlab'].textContent);

  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
        {countries: LIST, country: 'DEU', total_unflagged: 198,
         flagged_total: 1166});
  await API.load(); await flush();
  ck(byId['left'].textContent === '198', 't21: left=' + byId['left'].textContent);
  ck(/Germany/.test(byId['leftlab'].textContent),
     't21: filtered count not scoped to the country, label=' +
     byId['leftlab'].textContent);
  ck(byId['done'].textContent === '1,166',
     't21: global flagged count changed under a filter: ' + byId['done'].textContent);

  // clearing the filter restores the global wording
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
        {countries: LIST, country: '', total_unflagged: 2100});
  await API.load(); await flush();
  ck(byId['leftlab'].textContent === 'left to review',
     't21: label stuck on a country after clearing: ' + byId['leftlab'].textContent);
}

// ── 22. an option's count must equal what selecting it returns ──────────
// The first cut tallied the dropdown from the country INDEX, which spans the
// rolling pool plus both flag ledgers, while the queue excludes everything
// judged/kept/collapsed. Measured on the live server: 60 of 60 options were
// dead, promising 4,090 crops that did not exist.
async function t22() {
  const LIST = [{iso:'BRA',name:'Brazil',n:1073},{iso:'JPN',name:'Japan',n:312}];
  RESP = { '/api/review': (url) => {
             const iso = /country=([A-Z]*)/.exec(url);
             const sel = iso && iso[1];
             const hit = LIST.filter(c => c.iso === sel)[0];
             return payload(CROPS.normal.slice(0, 3), [],
               {countries: LIST, country: sel || '',
                total_unflagged: hit ? hit.n : 1385});
           },
           '/api/review/seen': () => ({ ok: true, seen_total: 1 }) };
  await API.load(); await flush();
  // pick each option and check the queue size matches what it advertised
  for (const c of LIST) {
    byId['country'].value = c.iso;
    await byId['country'].onchange.call(byId['country']);
    await flush(); await flush();
    const shown = byId['left'].textContent.replace(/,/g, '');
    ck(shown === String(c.n),
       't22: ' + c.iso + ' advertised ' + c.n + ' but the queue shows ' +
       byId['left'].textContent);
  }
  // and an option that would return nothing must not be offered at all
  RESP['/api/review'] = () => payload([], [], {countries: [], country: ''});
  await API.load(); await flush();
  ck(!/value="[A-Z]{3}"/.test(byId['country'].innerHTML),
     't22: a country was still offered with an empty queue');
}

// ── 23. a search that cannot work has to say so ─────────────────────────
// The vectors belong to crop FILES and the pool rotates hourly, so coverage
// decays to nothing whenever the guesser is stopped. Measured on the live
// box: 4,513 vectors, 3,010 crops in the pool, zero in both -- and the page
// reported the search as working while the queue did not move, which reads
// as the model returning nonsense. Every state that fails to reorder the
// queue must put a sentence on screen.
async function t23() {
  const FIND = {find: 'a cat', find_terms: ['a cat'], find_cover: [0, 3010]};
  for (const [state, want] of [['cold', /embedded/],
                               // mismatch now clears itself: the words are
                               // re-encoded under whichever model embedded the
                               // crops, so the message must promise that and
                               // not send the reader off to run a tool
                               ['mismatch', /re-encoding the search words/],
                               ['learning', /moment/], ['unknown', /encoded/],
                               ['failed', /crop_search\.log/],
                               ['novectors', /guesser/]]) {
    RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
               Object.assign({}, FIND, {find_state: state})) };
    await API.load(); await flush();
    ck(!byId['findmsg'].hidden,
       't23: ' + state + ' said nothing on screen');
    ck(want.test(byId['findmsg'].textContent),
       't23: ' + state + ' message unhelpful: ' + byId['findmsg'].textContent);
    ck(/\bwarn\b/.test(byId['find'].className || ''),
       't23: ' + state + ' left the box looking healthy');
  }
  // and a search that DID order the queue must not nag
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             Object.assign({}, FIND, {find_state: 'on', find_hits: 2663,
                                      find_cover: [2663, 3010]})) };
  await API.load(); await flush();
  ck(byId['findmsg'].hidden, 't23: a working search still warned');
  ck(!/\bwarn\b/.test(byId['find'].className || ''),
     't23: a working search kept the warning border');

  // 'cold' with most of the pool embedded is a FILTER problem, not a stopped
  // guesser -- telling the reviewer to start one that is already running and
  // has covered 4,014 of 5,018 crops sends them after the wrong thing.
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             Object.assign({}, FIND, {find_state: 'cold',
                                      find_cover: [4014, 5018]})) };
  await API.load(); await flush();
  ck(/4,014/.test(byId['findmsg'].textContent),
     't23: cold ignored how much IS embedded: ' + byId['findmsg'].textContent);
  ck(!/start the guesser/.test(byId['findmsg'].textContent),
     't23: cold blamed the guesser with the pool mostly embedded');

  // the term is written with textContent, so it must not arrive pre-escaped
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             {find: 'cats & dogs', find_terms: ['a cat'],
              find_state: 'learning', find_cover: [0, 10]}) };
  await API.load(); await flush();
  ck(/cats & dogs/.test(byId['findmsg'].textContent),
     't23: term double-escaped in the message: ' + byId['findmsg'].textContent);
  ck(!/&amp;/.test(byId['findmsg'].textContent),
     't23: entities shown literally: ' + byId['findmsg'].textContent);
}

// ── 24. the guesser toggle names the weaker guesser ─────────────────────
// Two backends with very different accuracy on this data -- SigLIP calls 98%
// of confirmed dogs 'dog', RF-DETR 56% -- so a bare "SigLIP / RF-DETR"
// dropdown is a trap. The number the server measured has to reach the option
// text, and the caveat has to be on screen, not in a title attribute. The
// control also must not exist at all when there is only one guesser.
async function t24() {
  const TRG = {ever: true, can_run: true, running: false, pool: 100,
               guessed: 100, coverage: 1};
  // the real measured values, so a stale fixture cannot make a stale claim
  // in the UI look correct
  const TWO = [{key: 'siglip', label: 'SigLIP 2', recall: 0.977, clears: 0.943,
                buckets: ['dog', 'animal', 'object'],
                note: 'leaves behind the vectors the search box needs'},
               {key: 'rfdetr', label: 'RF-DETR', recall: 0.678, clears: 0.957,
                buckets: ['dog', 'animal', 'object'],
                note: 'writes no search vectors'}];
  const BASIS = '120 crops a reviewer confirmed are dogs, and the share ' +
                'each guesser files under "dog".';
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           '/api/triage': () => Object.assign({}, TRG,
                                              {backend: 'siglip',
                                               recall_basis: BASIS,
                                               backends: TWO}) };
  await API.load(); API.trgPoll(); await flush(); await flush();
  const sel = byId['trgModel'];
  const WANT_OPT = Math.round(TWO[0].recall * 100) + '%';
  ck(!sel.hidden, 't24: two guessers offered but the control stayed hidden');
  const WANT_OPT2 = Math.round(TWO[1].recall * 100) + '%';
  ck(sel.innerHTML.includes(WANT_OPT) && sel.innerHTML.includes(WANT_OPT2),
     't24: accuracy missing from the options: ' + sel.innerHTML);
  ck(/SigLIP 2/.test(sel.innerHTML) && /RF-DETR/.test(sel.innerHTML),
     't24: a guesser is missing from the options: ' + sel.innerHTML);
  ck(!byId['trgNote'].hidden,
     't24: no legend for the guesser percentages');
  // "75% of known dogs" is unreadable on its own -- WHICH dogs, and what did
  // the guesser have to do to count? The summary has to answer that without
  // being unfolded, and the body has to say what the test set is.
  const WANT = Math.round(TWO[0].recall * 100) + '%';
  ck(byId['trgNoteSum'].textContent.includes(WANT) &&
     /test set/.test(byId['trgNoteSum'].textContent),
     't24: the legend summary does not say what the number counts: ' +
     byId['trgNoteSum'].textContent);
  ck(/confirmed are dogs/.test(byId['trgNoteBasis'].textContent),
     't24: the legend does not say where the test set comes from: ' +
     byId['trgNoteBasis'].textContent);
  ck(/vectors/.test(byId['trgNoteWhich'].textContent),
     't24: the caveat for the chosen guesser is not on screen: ' +
     byId['trgNoteWhich'].textContent);

  // one guesser is not a choice
  RESP['/api/triage'] = () => Object.assign({}, TRG,
        {backend: 'siglip', backends: [TWO[0]]});
  API.trgPoll(); await flush(); await flush();
  ck(byId['trgModel'].hidden,
     't24: a dropdown with one option was still drawn');
  ck(byId['trgNote'].hidden, 't24: a caveat with no choice to make');

  // and the queue must be asked for the SELECTED guesser's guesses
  RESP['/api/triage'] = () => Object.assign({}, TRG,
        {backend: 'siglip', backends: TWO});
  API.trgPoll(); await flush(); await flush();
  API.setBackend('rfdetr'); await flush(); await flush();
  const asked = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  ck(/backend=/.test(asked.url),
     't24: the queue request does not say whose guesses it wants: ' +
     asked.url);
}

// ── 25. one guesser running must not be reported as the other ───────────
// The two share ONE status file. Before this was fixed, starting RF-DETR and
// then moving the dropdown to SigLIP showed SigLIP as running, with its
// progress bar and a Pause button that would have killed the RF-DETR run.
async function t25() {
  const BASE = {ever: true, can_run: true, pool: 100, guessed: 40,
                coverage: 0.4,
                backends: [{key: 'siglip', label: 'SigLIP 2', recall: 0.75},
                           {key: 'rfdetr', label: 'RF-DETR', recall: 0.56}]};
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
           // asked about SigLIP while RF-DETR holds the card
           '/api/triage': () => Object.assign({}, BASE, {backend: 'siglip',
                                running: false, busy_with: 'RF-DETR'}) };
  // t24 leaves the page on another guesser and a repaint of its own still in
  // flight; settle both before asserting, or this reads t24's last payload
  API.setBackend('siglip');
  await flush(); await flush(); await flush();
  API.trgPoll(); await flush(); await flush();
  ck(/RF-DETR/.test(byId['trgState'].textContent + byId['trgSub'].textContent),
     't25: the strip does not say which guesser is busy: ' +
     byId['trgState'].textContent);
  const btn = byId['trgRun'];
  ck(btn.disabled, 't25: Run was offered while the other guesser had the card');
  ck(btn.textContent !== 'Pause',
     't25: offered to Pause a run belonging to the other guesser');

  // once it frees up, the button has to come back -- the disabled flag must
  // not latch
  RESP['/api/triage'] = () => Object.assign({}, BASE, {backend: 'siglip',
                              running: false, busy_with: null});
  API.trgPoll(); await flush(); await flush();
  ck(!btn.disabled, 't25: Run stayed disabled after the card freed up');
  ck(btn.textContent === 'Run guesses',
     't25: button label stuck: ' + btn.textContent);
}

// ── 26. the dog-bin gate is its own axis, not a rival to the guess filter ──
// It answers the question the reviewer is answering -- is this a dog -- where
// the guess filter answers what KIND of thing it is. So it gets its own
// control, usable at the same time, and it must not move when the guesser
// toggle does.
async function t26() {
  const GATE = {gate_ready: true, gate: 'all', gate_label: 'Dog-bin gate',
                gate_counts: {all: 2157, dog: 887, not_dog: 796, none: 474}};
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
             Object.assign({backend: 'siglip', suggest_ready: true,
                            buckets: [{key: 'dog', label: 'Looks like a dog'},
                                      {key: 'animal', label: 'Other animal'},
                                      {key: 'object', label: 'Not an animal'}]},
                           GATE)) };
  await API.load(); await flush();
  const g = byId['gatef'];
  ck(!g.hidden, 't26: the gate has no control of its own');
  ck(/887/.test(g.innerHTML) && /796/.test(g.innerHTML) &&
     /474/.test(g.innerHTML),
     't26: the gate options carry no counts: ' + g.innerHTML);
  ck(/Gate says dog/.test(g.innerHTML) && /Gate says not a dog/.test(g.innerHTML),
     't26: the gate verdicts are not offered: ' + g.innerHTML);
  // both axes at once: the guess filter is still there beside it
  ck(!byId['suggest'].hidden,
     't26: the guess filter vanished when the gate appeared');

  // choosing a gate verdict has to reach the server
  g.value = 'not_dog';
  (g._listeners.change || []).forEach(f => f.call(g));
  if (g.onchange) g.onchange.call(g);
  await flush(); await flush();
  const asked = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  ck(/gate=not_dog/.test(asked.url),
     't26: the gate verdict never reached the queue request: ' + asked.url);

  // and it must not be offered before the gate has judged anything
  RESP['/api/review'] = () => payload(CROPS.normal.slice(0, 3), [],
        {backend: 'siglip', suggest_ready: true, gate_ready: false});
  await API.load(); await flush();
  ck(byId['gatef'].hidden,
     't26: an empty gate filter was still offered');
}

// ── 27. the caption, the chips, and the one disclosure ──────────────────
// Nine controls sat in one row holding four different kinds of thing. The
// block is now a caption over a fold: it says what the queue is, shows only
// the filters actually applied, and keeps the rest behind one button. Each of
// those three claims is checked, because each replaced something visible.
async function t27() {
  const FULL = {total_unflagged: 2157, pool_unfiltered: 2157,
                suggest_ready: true, gate_ready: true,
                gate_label: 'Dog-bin gate', gate: 'all',
                gate_counts: {all: 2157, dog: 887, not_dog: 796, none: 474},
                countries: [{iso: 'JPN', name: 'Japan', n: 838}], country: ''};
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [], FULL) };
  await API.load(); await flush();
  ck(/2,157/.test(byId['cap'].textContent),
     't27: the caption does not say what the queue holds: ' +
     byId['cap'].textContent);
  ck(!/narrowed from/.test(byId['cap'].textContent),
     't27: claimed a narrowing with no filter applied: ' +
     byId['cap'].textContent);
  ck(byId['chips'].hidden, 't27: an empty chip row still took a line');
  ck(byId['npanel'].hidden, 't27: the panel is open before it is asked for');

  // apply one: the chip appears, and the caption says what it narrowed from
  RESP['/api/review'] = () => Object.assign({}, FULL,
        {total_unflagged: 887, gate: 'dog', pool_unfiltered: 2157});
  byId['gatef'].value = 'dog';
  await API.load(); await flush();
  ck(!byId['chips'].hidden && /Gate says dog/.test(byId['chips'].innerHTML),
     't27: the applied filter is not shown as a chip: ' +
     byId['chips'].innerHTML);
  ck(/narrowed from/.test(byId['cap'].textContent) &&
     /2,157/.test(byId['cap'].textContent) &&
     /887/.test(byId['cap'].textContent),
     't27: the caption does not report the narrowing: ' +
     byId['cap'].textContent);
  ck(/1/.test(byId['narrow'].textContent),
     't27: the shut panel does not say how many filters are on: ' +
     byId['narrow'].textContent);

  // clearing from the chip resets the control it came from
  RESP['/api/review'] = () => Object.assign({}, FULL, {gate: 'all'});
  const x = byId['chips'].querySelector('.chipx');
  ck(!!x, 't27: the chip cannot be cleared where it is read');
  (byId['chips']._listeners.click || []).forEach(f => f.call(byId['chips'],
      {target: x}));
  await flush(); await flush();
  // Asserted on the REQUEST, not on the control's value afterwards: the next
  // payload echoes `gate` back and paintGate writes it into the select, so a
  // chip that cleared nothing would still look cleared a moment later. The
  // URL is the only observable the echo cannot fake.
  const after = CALLS.filter(c => /\/api\/review\?/.test(c.url)).pop();
  ck(/gate=all/.test(after.url),
     't27: clearing the chip did not clear the filter it names: ' + after.url);
  ck(byId['chips'].hidden, 't27: the chip outlived the filter');

  // the disclosure holds the rest, and says so
  const nb = byId['narrow'];
  (nb._listeners.click || []).forEach(f => f.call(nb));
  ck(!byId['npanel'].hidden, 't27: Narrow does not open the panel');
  ck(nb.getAttribute && nb.getAttribute('aria-expanded') === 'true',
     't27: the disclosure does not report its state to a screen reader');
  (nb._listeners.click || []).forEach(f => f.call(nb));
  ck(byId['npanel'].hidden, 't27: Narrow does not shut the panel again');
}

// ── 28. every state of the two lines is still a line ────────────────────
// Both rows set className wholesale to add a state class, and 'line' is what
// gives them their flex, their gap and their track. Two of those writes
// predated the redesign and dropped it, so the commonest guesser state --
// "Not running" -- shipped as run-together text with no bar, while the suite
// stayed green because no test drove that branch.
async function t28() {
  const BASE = {ever: true, can_run: true, pool: 5018, guessed: 1633,
                coverage: 0.325,
                backends: [{key: 'siglip', label: 'SigLIP 2', recall: 0.977,
                            clears: 0.943}]};
  const STATES = [
    ['not running',  {running: false}],
    ['guessing',     {running: true, done: 176, total: 2864, rate: 39.2}],
    ['stalled',      {running: false, stalled: true, age_s: 6840}],
    ['stopped, why', {running: false, why: 'the GPU was full'}],
    ['up to date',   {running: false, guessed: 5018, coverage: 1}],
    ['waiting',      {running: false, busy_with: 'RF-DETR'}],
  ];
  for (const [what, extra] of STATES) {
    RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), []),
             '/api/triage': () => Object.assign({}, BASE,
                                    {backend: 'siglip'}, extra) };
    await API.load(); API.trgPoll(); await flush(); await flush();
    const cls = (byId['trg'].className || '').split(' ');
    ck(cls.indexOf('line') >= 0,
       't28: the guesser row lost its layout class while ' + what +
       ': class="' + byId['trg'].className + '"');
    ck(cls.indexOf('trg') >= 0,
       't28: the guesser row lost its own class while ' + what +
       ': class="' + byId['trg'].className + '"');
  }

  // ...and the balance row, whose painter has three exits of its own
  for (const [what, bal] of [
      ['no dataset', {ok: false, error: 'nope'}],
      ['short',      {ok: true, have: 1549, want: 1652, pending: 0,
                      judged: 900, n_pos: 100, yield_per_flag: 0.5,
                      dataset: 'dogbin_v5'}],
      ['balanced',   {ok: true, have: 1700, want: 1652, pending: 0,
                      judged: 900, n_pos: 100, yield_per_flag: 0.5,
                      dataset: 'dogbin_v5'}]]) {
    RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 3), [],
                                          {balance: bal}) };
    await API.load(); await flush();
    ck((byId['bal'].className || '').split(' ').indexOf('line') >= 0,
       't28: the balance row lost its layout class while ' + what +
       ': class="' + byId['bal'].className + '"');
  }
}

// ── 29. the chips describe the request that was actually sent ───────────
// The two views fetch different things. The audit list is fetched with label=
// and leash= and nothing else, so a guess or a country left set from the queue
// narrows nothing there. The chip row advertised both anyway and hid the
// verdict filter -- the one that does apply -- so it explained an empty list
// with a cause that was not the cause, and offered no way to undo the real one.
async function t29() {
  const Q = () => ({items: [], reserve: [], page: 0, size: 50, pages: 1,
      total_unflagged: 100, pool_unfiltered: 100, suggest_ready: true,
      countries: [{iso: 'JPN', name: 'Japan', n: 9}], country: 'JPN',
      suggest: 'dog'});
  const A = () => ({items: [], page: 0, pages: 1, total: 5,
      pool_unfiltered: 7, n_false_positive: 5, n_true_positive: 2});
  RESP = {'/api/review': Q, '/api/review/annotated': A};
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };

  fire('suggest', 'dog'); await flush(); await flush();
  ck(/Looks like a dog/.test(byId['chips'].innerHTML),
     't29: a queue filter produced no chip: ' + byId['chips'].innerHTML);

  fire('mode', 'audit'); await flush(); await flush();
  const sent = (CALLS.filter(c => /annotated/.test(c.url)).pop() || {}).url;
  ck(!/Looks like a dog/.test(byId['chips'].innerHTML),
     't29: the audit view kept a chip for a filter its request does not ' +
     'carry (' + sent + '): ' + byId['chips'].innerHTML);

  fire('verdict', 'false_positive'); await flush(); await flush();
  ck(/not a dog/.test(byId['chips'].textContent),
     't29: the audit view shows no chip for the one filter it applies: ' +
     byId['chips'].textContent);
  ck(!byId['npanel'].hidden || !byId['verdict'].hidden,
     't29: the audit view\'s only filter is behind a fold');

  // and clearing it has to reach the audit request, not the queue's
  const x = byId['chips'].querySelector('.chipx');
  (byId['chips']._listeners.click || []).forEach(f =>
    f.call(byId['chips'], {target: x}));
  await flush(); await flush();
  const after = (CALLS.filter(c => /annotated/.test(c.url)).pop() || {}).url;
  ck(/label=all/.test(after),
     't29: clearing the verdict chip did not clear the verdict: ' + after);

  fire('mode', 'queue'); await flush(); await flush();
}

// ── 30. nothing narrows the queue without a way to see and undo it ──────
// The whole point of the chip row. Hiding a control does not unset it: the
// page hides the guess filter when the gate's own axis covers it, and the
// value stayed in the request and kept being honoured — so choosing the gate
// could empty the queue with no chip, no cross, no control on screen and no
// "narrowed from". The server decides what it applied and echoes it; the page
// has to adopt that rather than re-send the dropped value.
async function t30() {
  let asked = '';
  RESP = { '/api/review': (url) => {
    asked = url;
    // the server drops a filter the page is not offering, and says so
    const sg = /suggest=(\w*)/.exec(url);
    const applied = (/backend=dogbin/.test(url)) ? '' : (sg ? sg[1] : '');
    return payload(CROPS.normal.slice(0, 2), [], {
      suggest: applied, suggest_ready: true, backend: 'siglip',
      total_unflagged: applied ? 300 : 2206, pool_unfiltered: 2206});
  }};
  const fire = (id, v) => { const e = byId[id]; e.value = v;
    if (e.onchange) e.onchange.call(e);
    (e._listeners.change || []).forEach(f => f.call(e)); };

  fire('suggest', 'animal'); await flush(); await flush();
  ck(/suggest=animal/.test(asked) && /Other animal/.test(byId['chips'].innerHTML),
     't30: a filter the server applied has no chip: ' + byId['chips'].innerHTML);

  // now the server says it did NOT apply it. The page must stop sending it.
  RESP['/api/review'] = (url) => { asked = url; return payload(
      CROPS.normal.slice(0, 2), [], {suggest: '', suggest_ready: true,
      backend: 'dogbin', gate_ready: true, total_unflagged: 2206,
      pool_unfiltered: 2206}); };
  await API.load(); await flush(); await flush();
  await API.load(); await flush(); await flush();
  ck(/suggest=(&|$)/.test(asked),
     't30: the page kept sending a filter the server refused: ' + asked);
  ck(byId['chips'].hidden || !/Other animal/.test(byId['chips'].innerHTML),
     't30: a chip for a filter that was not applied: ' + byId['chips'].innerHTML);
}

// ── 31. the panel offers no control that does nothing ───────────────────
// Two ways it did. A group whose every control is hidden rendered as an
// uppercase heading over an empty row. And the Run button, which used to be
// hidden by living inside the progress strip, moved into the panel and stayed
// on screen on a checkout with no guesser at all — showing the markup's raw
// placeholder, enabled, and clickable, since it reads its own label to decide
// what to do.
async function t31() {
  RESP = { '/api/review': () => payload(CROPS.normal.slice(0, 2), [],
             {suggest_ready: false, gate_ready: false}),
           '/api/triage': () => ({ever: false, can_run: false}) };
  await API.load(); API.trgPoll(); await flush(); await flush();
  const looks = document.getElementById('ngrpLooks');
  ck(looks && looks.hidden,
     't31: a heading with nothing under it');
  const who = document.getElementById('ngrpWho');
  ck(who && who.hidden,
     't31: the guesser controls are offered with no guesser to run');

  // and they come back when there is one
  RESP['/api/triage'] = () => ({ever: true, can_run: true, pool: 10,
      guessed: 10, coverage: 1, running: false,
      backends: [{key: 'siglip', label: 'SigLIP 2', recall: .977, clears: .943}]});
  API.trgPoll(); await flush(); await flush();
  ck(!who.hidden, 't31: the guesser group never came back');
  ck(byId['trgRun'].textContent !== '—',
     't31: the Run button is still showing its raw placeholder: ' +
     byId['trgRun'].textContent);

  // a reload must not undo that: two painters own this group
  await API.load(); await flush(); await flush();
  ck(!who.hidden, 't31: a queue reload hid the guesser group again');
}

(async () => {
  const tests = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31];
  for (const t of tests) {
    try { await t(); console.log('ok   ' + t.name); }
    catch (e) {
      failures.push(t.name + ': THREW ' + (e && e.stack || e));
      console.log('FAIL ' + t.name + ' — ' + e);
    }
  }
  if (failures.length) {
    console.log('FAILURES: ' + failures.join(' | '));
    process.exit(1);
  }
  console.log('all review cases passed');
})();
"""


def main():
    if not shutil.which('node'):
        print('SKIP: node not on PATH — cannot execute the review page JS')
        return 0
    mod = load_dashboard()
    html = mod.REVIEW_HTML
    script = html[html.rindex('<script>') + 8:html.rindex('</script>')]

    # parse the whole block first: one syntax error kills every handler, and
    # driving the functions below would report a confusing cascade instead
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(script)
        probe = f.name
    try:
        p = subprocess.run(['node', '--check', probe],
                           capture_output=True, text=True)
    finally:
        os.unlink(probe)
    if p.returncode:
        print('FAIL: /review script does not parse:\n' + p.stderr.strip()[:900])
        return 1
    print('ok   whole review script parses (%d chars)' % len(script))

    fixtures = {
        'normal': [crop(i, conf=round(0.95 - i * 0.05, 2)) for i in range(9)],
        'mixed': [crop(0, full=False), crop(1, full=True), crop(2, full=False),
                  crop(3, full=True), crop(4, full=True)],
        'hostile': [{
            'name': '1700000000000_x_090.jpg',
            'image_id': '"><script>alert(1)</script>',
            'ts': 1_700_000_000_000, 'conf': 0.9, 'has_full': True,
        }, {
            'name': '1700000000001_y_080.jpg',
            'image_id': '<img src=x onerror=alert(1)>',
            'ts': 1_700_000_000_001, 'conf': 0.8, 'has_full': False,
        }],
    }

    with tempfile.TemporaryDirectory() as tmp:
        js = os.path.join(tmp, 'review.js')
        fx = os.path.join(tmp, 'crops.json')
        run = os.path.join(tmp, 'run.js')
        with open(js, 'w') as f:
            f.write(script)
        # Which ids the real markup ships hidden, read off the markup so the
        # stub starts where the page does. Asserting a panel is shut against a
        # stub that starts everything visible proves nothing.
        hidden = re.findall(r'<[a-z]+[^>]*\bid="(\w+)"[^>]*\bhidden\b',
                            html)
        hidden += re.findall(r'<[a-z]+[^>]*\bhidden\b[^>]*\bid="(\w+)"',
                             html)
        # The options each <select> ships in the markup, so the stub starts
        # with the same choices the page does.
        opts = dict(re.findall(
            r'<select id="(\w+)"[^>]*>(.*?)</select>', html, re.S))
        opts = {k: v.strip() for k, v in opts.items() if '<option' in v}
        # Which controls each panel group holds, so the stub has the tree
        # trimGroups() walks.
        # Split the panel at each group start rather than trying to match
        # balanced divs: the nesting differs per group (one carries a
        # <details>), and a regex that assumed otherwise silently dropped the
        # Run button from its group and made a test fail for the wrong reason.
        groups, owned = {}, []
        panel = html.split('<div class="npanel"', 1)[-1]
        parts = re.split(r'<div class="ngrp"', panel)[1:]
        for part in parts:
            # the id must be in the group's OWN opening tag. Searching the
            # whole block found the first control's id instead for a group
            # that has none, which wired that control as its own parent --
            # a cycle the stub's descendant walk never returned from.
            gid = re.match(r'[^>]*id="(\w+)"', part)
            if not gid:
                continue
            groups[gid.group(1)] = re.findall(
                r'<(?:select|button|input)[^>]*\bid="(\w+)"', part)
            # a group that owns its own visibility, so trimGroups leaves it be
            if re.match(r'[^>]*data-own=', part):
                owned.append(gid.group(1))
        with open(fx, 'w') as f:
            json.dump({'crops': fixtures, 'hidden': sorted(set(hidden)),
                       'options': opts, 'groups': groups,
                       'owned': owned}, f)
        with open(run, 'w') as f:
            f.write(HARNESS)
        p = subprocess.run(['node', run, js, fx],
                           capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.stderr.strip():
        sys.stderr.write(p.stderr)
    return p.returncode


if __name__ == '__main__':
    sys.exit(main())
