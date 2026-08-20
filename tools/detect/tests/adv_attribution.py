#!/usr/bin/env python3
"""Every annotation says who made it, and the ones that predate accounts say
so by being read as the admin's rather than by being rewritten.

The dashboard grew real accounts -- an admin plus invite-only members -- over
stores that had recorded 3,247 human judgements without ever recording a
person. This grades the answer to that:

  * ONE word for the annotator of a row that has none. `admin` is spelled in
    exactly one module and imported by the rest, because a second spelling is
    the day one reader says admin and another says None over the same lines;
  * and NOBODY ELSE CAN BE CALLED IT. The word is also an ordinary username,
    and on a deployment whose admin was renamed an invited member could take
    it from the signup form and inherit every judgement made before they
    existed;
  * a NEW record in any of the human stores carries the signed-in username --
    the two flag ledgers, both audit verdict ledgers, the box corrections, the
    leash database, the wrong-label flags, and the reviewed-and-kept ledger,
    which is the largest of them and records the affirmative half of the same
    decision the flag button records;
  * an OLD record without one reads as the admin at every reader -- the audit
    statistics, the dataset export, the review page's audit view, the "my
    annotations" list, the box corrections and both databases;
  * each database column arrives by a migration that can be run twice AND by
    several threads at once, which is the shape a ThreadingHTTPServer actually
    produces, and the rows already in it keep their NULL;
  * a write that cannot name its annotator is REFUSED and appends nothing.
    The login gate makes that unreachable for a real request, so it is a
    caller that forgot the session, not a reviewer -- and recording it as the
    admin would forge attribution, which is the one thing this field exists
    to prevent;
  * the two surfaces that read work back say who judged a crop, quietly, and
    neither grew a filter-by-user nobody asked for.

EVERY PATH IS REDIRECTED. This file writes fixture verdicts, fixture flags and
fixture boxes; a past session redirected one path and not another and put
seventeen invented verdicts into the live gate audit, which no command can
take back out. So the live ledgers are fingerprinted before anything runs and
compared afterwards, byte for byte and line for line, and the fixtures carry a
mark this file greps the real stores for.
"""

import ast
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'detect'))

# A string only this file invents. Nothing that reaches a real store can
# carry it, so finding it in one names the redirect that did not take.
MARK = 'zzattrguard'
WHO = MARK + '_alice'

# The stores this change touches, as (path, what it is). Fingerprinted at
# import and checked again at the end.
LIVE = (
    (os.path.join(REPO, 'data', 'hard_positives', 'labels.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'hard_negatives', 'labels.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'fn_audit', 'verdicts.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'leash_audit', 'verdicts.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'box_corrections', 'boxes.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'hard_negatives', 'reviewed.jsonl'), 'jsonl'),
    (os.path.join(REPO, 'data', 'leash_labels', 'leash.db'), 'db'),
    (os.path.join(REPO, 'data', 'label_flags', 'label_flags.db'), 'db'),
    # not an annotation, and never opened by this file: a test that writes
    # here mints accounts on the live dashboard
    (os.path.join(REPO, 'data', 'dashboard', 'accounts.db'), 'db'),
)

# A crop name of the shape the review page mints: <ts>_<image_id>_<conf>.
CROP = '1785663300000_1606751523958968_073.jpg'
OLD_CROP = '1785663300001_1606751523958969_064.jpg'

# leash.db as it stood before the annotator column -- the fixture for the
# migration. Kept verbatim rather than derived from SCHEMA, because a
# migration test that builds its "before" out of the "after" is testing
# nothing.
LEASH_SCHEMA_V0 = """
CREATE TABLE IF NOT EXISTS leash (
    crop        TEXT PRIMARY KEY,
    image_id    TEXT NOT NULL,
    label       TEXT NOT NULL CHECK (label IN ('leashed', 'unleashed')),
    conf        REAL,
    ts          INTEGER,
    labelled_at INTEGER NOT NULL,
    source      TEXT NOT NULL DEFAULT 'review_page',
    note        TEXT
);
CREATE INDEX IF NOT EXISTS leash_label ON leash(label);
CREATE INDEX IF NOT EXISTS leash_when  ON leash(labelled_at);
"""

# label_flags.db as it stood before the annotator column, for the same reason.
FLAGS_SCHEMA_V0 = """
CREATE TABLE IF NOT EXISTS flags (
    file       TEXT PRIMARY KEY,
    image_id   TEXT,
    dataset    TEXT,
    class_was  TEXT,
    should_be  TEXT,
    run        TEXT,
    note       TEXT,
    flagged_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS flags_image ON flags(image_id);
"""

# A crop name of the shape a dataset build mints, which is what a wrong-label
# flag is raised against.
DS_CROP = 'no_1606751523958968_0.jpg'


def fingerprint():
    """(md5, mtime, size, lines) for every live store, as it is right now.

    Lines as well as bytes: a file can be rewritten to the same length, and
    the number of human judgements in it is the fact worth stating in the
    failure message.
    """
    out = {}
    for path, kind in LIVE:
        try:
            with open(path, 'rb') as fh:
                blob = fh.read()
        except OSError:
            out[path] = None
            continue
        lines = (len([b for b in blob.split(b'\n') if b.strip()])
                 if kind == 'jsonl' else None)
        out[path] = (hashlib.md5(blob).hexdigest(),
                     int(os.path.getmtime(path)), len(blob), lines)
    return out


BEFORE = fingerprint()


def live_untouched(bad):
    """Nothing this file did reached a store a person's work lives in."""
    after = fingerprint()
    for path, _kind in LIVE:
        rel = os.path.relpath(path, REPO)
        if BEFORE[path] != after[path]:
            bad.append(f'{rel} CHANGED during this run: {BEFORE[path]} -> '
                       f'{after[path]} — a redirect did not take, and the '
                       f'fixtures went into human data')
        try:
            with open(path, 'rb') as fh:
                if MARK.encode() in fh.read():
                    bad.append(f'{rel} holds this test\'s own mark')
        except OSError:
            pass


# ── one spelling ────────────────────────────────────────────────────────────

def one_spelling_checks(bad):
    """The legacy author is defined once and imported everywhere else.

    The failure this stops is not a crash. It is two modules that each know
    what an unattributed row means, disagreeing about it a year from now --
    the audit page reporting `admin` while a dataset export writes null, off
    the same line of the same file.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import accounts
    import audit
    import dashboard as dash

    owner = os.path.join('tools', 'detect', 'fn_audit.py')
    assigned = []
    for rel in (owner, os.path.join('tools', 'detect', 'leash_store.py'),
                os.path.join('tools', 'detect', 'label_flags.py'),
                os.path.join('tools', 'dashboard', 'accounts.py'),
                os.path.join('tools', 'dashboard', 'audit.py'),
                os.path.join('tools', 'dashboard', 'dashboard.py')):
        tree = ast.parse(open(os.path.join(REPO, rel)).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'LEGACY_AUTHOR':
                    assigned.append((rel, node.lineno,
                                     isinstance(node.value, ast.Constant)))
    literal = [f'{r}:{n}' for r, n, lit in assigned if lit]
    mine = [f'{r}:{n}' for r, n, lit in assigned if lit and r == owner]
    if len(literal) != 1 or literal != mine:
        bad.append(f'LEGACY_AUTHOR is written out as a literal in {literal}; '
                   f'exactly one module may spell it -- {owner} -- and every '
                   f'other one imports it')
    # and the value really is the same object everywhere it is used
    for mod in (audit, dash, accounts):
        if mod.LEGACY_AUTHOR is not fa.LEGACY_AUTHOR:
            bad.append(f'{mod.__name__}.LEGACY_AUTHOR is a copy, not '
                       f'fn_audit\'s — two spellings, one of which will move')
    for mod in (ls, lf):
        if mod.author_of is not fa.author_of:
            bad.append(f'{mod.__name__} resolves the legacy author with its '
                       f'own function rather than the shared one')
    # the constant says what an absent author means and why the ledgers were
    # left alone -- a bare `LEGACY_AUTHOR = 'admin'` is the version of this
    # that the next reader has to guess at
    src = open(os.path.join(REPO, owner)).read()
    doc = src[src.index('LEGACY_AUTHOR ='):src.index('def author_of')]
    for want in ('rewrit', 'append-only', 'accounts'):
        if want not in doc.lower():
            bad.append(f'the LEGACY_AUTHOR docstring does not mention '
                       f'{want!r} — it has to say what an absent author '
                       f'means and why 3,247 lines were not rewritten')
    if fa.author_of(None) != fa.LEGACY_AUTHOR or fa.author_of('') != \
            fa.LEGACY_AUTHOR or fa.author_of('bo') != 'bo':
        bad.append('author_of does not resolve absent -> legacy, present -> '
                   'itself')


# ── writing ─────────────────────────────────────────────────────────────────

def _flag_layout(dash, tmp):
    """Point every flag path at a throwaway tree. Returns what to restore.

    The reviewed-and-kept ledger goes with them. It lives inside HN_DIR, so
    redirecting that alone moves the file -- but SEEN_FILE was resolved at
    import and still names the real one, and `_seen` caches the real ids on
    top. Both are reset here: this is the largest human ledger in the repo and
    the one a half-done redirect appends to.
    """
    keep = (dash.CROPS, dash.HN_DIR, dash.HN_CROPS, dash.HN_FULL,
            dash.HN_LABELS, dash.HP_DIR, dash.BOX_DIR, dash.BOX_LABELS,
            dash._flagged, dash.SEEN_FILE, dash._seen)
    crops = os.path.join(tmp, 'recent_crops')
    os.makedirs(os.path.join(crops, 'full'), exist_ok=True)
    for name in (CROP, OLD_CROP):
        with open(os.path.join(crops, name), 'wb') as fh:
            fh.write(b'\xff\xd8crop')
    dash.CROPS = crops
    dash.HN_DIR = os.path.join(tmp, 'hard_negatives')
    dash.HN_CROPS = os.path.join(dash.HN_DIR, 'crops')
    dash.HN_FULL = os.path.join(dash.HN_DIR, 'full')
    dash.HN_LABELS = os.path.join(dash.HN_DIR, 'labels.jsonl')
    dash.HP_DIR = os.path.join(tmp, 'hard_positives')
    dash.BOX_DIR = os.path.join(tmp, 'box_corrections')
    dash.BOX_LABELS = os.path.join(dash.BOX_DIR, 'boxes.jsonl')
    dash.SEEN_FILE = os.path.join(dash.HN_DIR, 'reviewed.jsonl')
    dash._flagged = None
    dash._seen = None
    return keep


def _restore_flags(dash, keep):
    (dash.CROPS, dash.HN_DIR, dash.HN_CROPS, dash.HN_FULL, dash.HN_LABELS,
     dash.HP_DIR, dash.BOX_DIR, dash.BOX_LABELS, dash._flagged,
     dash.SEEN_FILE, dash._seen) = keep


def _lines(path):
    try:
        with open(path) as fh:
            return [json.loads(x) for x in fh if x.strip()]
    except OSError:
        return []


def _audit_layout(fa, tmp, stage='gate'):
    """A whole audit stage in a temp directory, the way adv_fn_audit does."""
    lay = dict(fa.paths(stage))
    lay.update(out=tmp, pages=os.path.join(tmp, 'pages'),
               verdicts=os.path.join(tmp, 'v.jsonl'),
               drawn=os.path.join(tmp, 'drawn.jsonl'),
               full=os.path.join(tmp, 'full'),
               crops=os.path.join(tmp, 'crops'),
               dataset=os.path.join(tmp, 'ds'),
               pool=os.path.join(tmp, 'pool.parquet'))
    os.makedirs(lay['pages'], exist_ok=True)
    return lay


def new_record_checks(bad):
    """A verdict recorded today names the person who recorded it.

    Every store, because "the author is threaded through" is a claim about
    every write path and not about the ones that were easiest to change.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    import audit
    import dashboard as dash

    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        real_paths = fa.paths
        real_box = audit.BOX_FILE
        try:
            # 1. the two flag ledgers
            for label, store in ((dash.FLAG_LABEL, 'hard_negatives'),
                                 (dash.POS_LABEL, 'hard_positives')):
                body, code = dash.flag_crop(CROP, label, by=WHO)
                path = dash._store_for(label)['labels']
                rows = _lines(path)
                if code != 200 or not body.get('ok') or not rows:
                    bad.append(f'flagging into {store} with an annotator '
                               f'failed: {body}')
                elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                    bad.append(f'{store} row does not name its annotator: '
                               f'{rows[-1]}')

            # 2. the box corrections, from the review page
            got = dash.save_box(CROP, 0, [1, 2, 30, 40], by=WHO)
            rows = _lines(dash.BOX_LABELS)
            if not got.get('ok') or not rows:
                bad.append(f'saving a box with an annotator failed: {got}')
            elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                bad.append(f'the box correction does not name who drew it: '
                           f'{rows[-1]}')

            # 3. both audit verdict ledgers
            for stage in fa.STAGES:
                lay = _audit_layout(fa, os.path.join(tmp, stage), stage)
                fa.paths = lambda s=stage, _l=lay: _l
                sp = fa.spec(stage)
                out = audit.record(MARK + '#0', sp['positive'],
                                   {'band': 3, 'p_dog': 0.31}, stage=stage,
                                   by=WHO)
                rows = _lines(lay['verdicts'])
                if not out.get('ok') or not rows:
                    bad.append(f'{stage}: recording a verdict with an '
                               f'annotator failed: {out}')
                elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                    bad.append(f'{stage} verdict does not name its annotator: '
                               f'{rows[-1]}')

            # 4. a box redrawn on the audit page, which writes the same
            #    corrections ledger from the other side
            lay = _audit_layout(fa, os.path.join(tmp, 'corr'))
            fa.paths = lambda s='gate', _l=lay: _l
            audit.BOX_FILE = os.path.join(tmp, 'audit_boxes.jsonl')
            real_row, real_cut = audit._pool_row, audit._cut_one
            audit._pool_row = lambda key, stage='gate': {
                'key': MARK + '#0', 'image_id': '1606751523958968',
                'det_idx': 0, 'x1': 1.0, 'y1': 2.0, 'x2': 30.0, 'y2': 40.0}
            audit._cut_one = lambda *a, **k: False
            try:
                out = audit.save_correction(MARK + '#0', [1, 2, 30, 40],
                                            by=WHO)
                rows = _lines(audit.BOX_FILE)
                if not out.get('ok') or not rows:
                    bad.append(f'saving an audit correction with an annotator '
                               f'failed: {out}')
                elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                    bad.append(f'the audit correction does not name who drew '
                               f'it: {rows[-1]}')
            finally:
                audit._pool_row, audit._cut_one = real_row, real_cut

            # 5. the leash database
            db = os.path.join(tmp, 'leash.db')
            body, code = ls.record(CROP, 'leashed', path=db, by=WHO)
            if code != 200 or not body.get('ok'):
                bad.append(f'recording a leash verdict with an annotator '
                           f'failed: {body}')
            con = ls.connect(db)
            try:
                row = con.execute('SELECT * FROM leash WHERE crop = ?',
                                  (CROP,)).fetchone()
                if row is None or row[ls.AUTHOR_FIELD] != WHO:
                    bad.append(f'the leash row does not name its annotator: '
                               f'{dict(row) if row else None}')
                # re-deciding is one row, and the person whose call it now is
                ls.record(CROP, 'unleashed', path=db, by=WHO + '2')
                row = con.execute('SELECT * FROM leash WHERE crop = ?',
                                  (CROP,)).fetchone()
                if row[ls.AUTHOR_FIELD] != WHO + '2' or \
                        row['label'] != 'unleashed':
                    bad.append(f're-deciding a leash verdict kept the old '
                               f'annotator: {dict(row)}')
            finally:
                con.close()

            # 6. the wrong-label flags -- a judgement about somebody else's
            #    judgement, which the next dataset build acts on
            fdb = os.path.join(tmp, 'label_flags.db')
            body, code = lf.add(DS_CROP, dataset='dogbin_v5',
                                class_was='not_dog', should_be='dog',
                                run='r1', path=fdb, by=WHO)
            if code != 200 or not body.get('ok'):
                bad.append(f'raising a label flag with an annotator failed: '
                           f'{body}')
            rows = lf.flagged_files(path=fdb)
            if DS_CROP not in rows or rows[DS_CROP].get(fa.AUTHOR_FIELD) != WHO:
                bad.append(f'the label flag does not name who raised it: '
                           f'{rows}')

            # 7. the reviewed-and-kept ledger: paging past a screen is the
            #    affirmative half of the same decision the flag button makes
            got = dash.mark_seen([CROP], by=WHO)
            rows = _lines(dash.SEEN_FILE)
            if not got.get('ok') or not rows:
                bad.append(f'marking a crop reviewed-and-kept with an '
                           f'annotator failed: {got}')
            elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                bad.append(f'the reviewed-and-kept row does not name who '
                           f'kept it: {rows[-1]}')
        finally:
            fa.paths = real_paths
            audit.BOX_FILE = real_box
            _restore_flags(dash, keep)


# ── reading ─────────────────────────────────────────────────────────────────

def legacy_read_checks(bad):
    """A row with no author on it reads as the admin, at every reader.

    Written as fixtures in the OLD shape -- exactly the lines on disk, with no
    `by` key at all -- because the point of the default is that those lines
    never had to change.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    import audit
    import dashboard as dash

    A = fa.AUTHOR_FIELD
    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        real_paths = fa.paths
        real_box = audit.BOX_FILE
        try:
            lay = _audit_layout(fa, tmp)
            fa.paths = lambda s='gate', _l=lay: _l
            # before anything reads a correction: corrections() is called from
            # inside judged(), and the live ledger is not this test's to read
            # a fixture out of
            audit.BOX_FILE = os.path.join(tmp, 'audit_boxes.jsonl')
            # one legacy verdict and one written since accounts landed
            with open(lay['verdicts'], 'w') as fh:
                fh.write(json.dumps({'key': MARK + 'old#0', 'verdict': 'dog',
                                     'band': 0, 'p_dog': 0.04,
                                     'ts': 1_700_000_000}) + '\n')
                fh.write(json.dumps({'key': MARK + 'new#0', 'verdict': 'dog',
                                     'band': 0, 'p_dog': 0.05,
                                     'ts': 1_800_000_000, A: WHO}) + '\n')
            got = {v['key']: v.get(A) for v in fa.read_verdicts(stage='gate')}
            if got != {MARK + 'old#0': fa.LEGACY_AUTHOR,
                       MARK + 'new#0': WHO}:
                bad.append(f'read_verdicts hands out {got}; the ledger\'s own '
                           f'past has to read as the admin and a new line as '
                           f'itself')

            # the "my annotations" list
            rows = {it['key']: it.get(A)
                    for it in audit.judged(which='all')['items']}
            if rows != {MARK + 'old#0': fa.LEGACY_AUTHOR,
                        MARK + 'new#0': WHO}:
                bad.append(f'the annotations list shows {rows}')

            # the sheet, read back
            doc = audit.with_verdicts(
                {'items': [{'key': MARK + 'old#0'}, {'key': MARK + 'new#0'},
                           {'key': MARK + 'none#0'}]})
            seen = [(it['key'], it.get(A)) for it in doc['items']]
            if seen != [(MARK + 'old#0', fa.LEGACY_AUTHOR),
                        (MARK + 'new#0', WHO), (MARK + 'none#0', None)]:
                bad.append(f'a page read back says {seen}; a judged box names '
                           f'its judge and an unjudged one names nobody')

            # the dataset export's manifest
            import argparse
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                fa.export(argparse.Namespace(stage='gate', model='m'))
            man = _lines(os.path.join(lay['dataset'], 'manifest.jsonl'))
            byline = {r['image_id']: r.get('judged_by') for r in man}
            if byline and set(byline.values()) - {fa.LEGACY_AUTHOR, WHO}:
                bad.append(f'the exported manifest attributes rows to '
                           f'{byline} — a dataset row with no judge is the '
                           f'reason this field exists')

            # the box corrections, both readers
            for path in (audit.BOX_FILE, dash.BOX_LABELS):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w') as fh:
                    fh.write(json.dumps({
                        'crop': OLD_CROP, 'image_id': '1606751523958969',
                        'det_idx': 0, 'x1': 1, 'y1': 2, 'x2': 3, 'y2': 4,
                        'saved_at': 1_700_000_000}) + '\n')
            fixed = audit.corrections().get(('1606751523958969', 0))
            if not fixed or fixed[5] != fa.LEGACY_AUTHOR:
                bad.append(f'a box drawn before accounts reads as {fixed}; '
                           f'the audit page has to be able to say who drew it')
            saved = dash._saved_box(OLD_CROP)
            if not saved or saved.get(A) != fa.LEGACY_AUTHOR:
                bad.append(f'the review editor reads a legacy box as {saved}')

            # the review page's audit view, over one legacy flag and one new
            os.makedirs(dash._store_for(dash.FLAG_LABEL)['dir'], exist_ok=True)
            with open(dash._store_for(dash.FLAG_LABEL)['labels'], 'w') as fh:
                fh.write(json.dumps({
                    'image_id': '1606751523958969', 'conf': 0.64,
                    'ts': 1785663300001, 'crop': OLD_CROP,
                    'label': dash.FLAG_LABEL, 'copied': True,
                    'flagged_at': 1_700_000_000}) + '\n')
            dash._flagged = None
            dash.flag_crop(CROP, dash.FLAG_LABEL, by=WHO)
            ls_mod = dash.leash_store()
            real_db = ls_mod.DB_PATH if ls_mod else None
            if ls_mod:
                # the leash lookups inside the payload open this path, and
                # opening the real one would migrate a store this test has no
                # business touching
                ls_mod.DB_PATH = os.path.join(tmp, 'leash.db')
            try:
                shown = {it['name']: it.get(A)
                         for it in dash.annotated_payload()['items']}
            finally:
                if ls_mod:
                    ls_mod.DB_PATH = real_db
            if shown != {OLD_CROP: fa.LEGACY_AUTHOR, CROP: WHO}:
                bad.append(f'the audit view attributes crops to {shown}')

            # the leash store, whose legacy rows are NULLs rather than absent
            # keys -- same meaning, same answer
            db = os.path.join(tmp, 'legacy_leash.db')
            con = sqlite3.connect(db)
            con.executescript(LEASH_SCHEMA_V0)
            con.execute('INSERT INTO leash (crop, image_id, label, conf, ts, '
                        'labelled_at, source) VALUES (?,?,?,?,?,?,?)',
                        (OLD_CROP, '1606751523958969', 'leashed', 0.64,
                         1785663300001, 1_700_000_000, 'review_page'))
            con.commit()
            con.close()
            ls.record(CROP, 'unleashed', path=db, by=WHO)
            con = ls.connect(db)
            try:
                seen = {r['crop']: ls.row_dict(r)[A]
                        for r in con.execute('SELECT * FROM leash')}
            finally:
                con.close()
            if seen != {OLD_CROP: fa.LEGACY_AUTHOR, CROP: WHO}:
                bad.append(f'the leash store reads back as {seen}; a NULL '
                           f'column means what an absent key means')

            # and the wrong-label flags, whose one live row predates the
            # column exactly the same way
            fdb = os.path.join(tmp, 'legacy_flags.db')
            con = sqlite3.connect(fdb)
            con.executescript(FLAGS_SCHEMA_V0)
            con.execute('INSERT INTO flags (file, image_id, dataset, '
                        'class_was, should_be, run, note, flagged_at) '
                        'VALUES (?,?,?,?,?,?,?,?)',
                        ('no_1606751523958969_0.jpg', '1606751523958969',
                         'dogbin_v5', 'not_dog', 'dog', 'r0', '',
                         1_700_000_000))
            con.commit()
            con.close()
            lf.add(DS_CROP, dataset='dogbin_v5', class_was='dog',
                   should_be='not_dog', run='r1', path=fdb, by=WHO)
            seen = {f: r.get(A) for f, r in lf.flagged_files(path=fdb).items()}
            if seen != {'no_1606751523958969_0.jpg': fa.LEGACY_AUTHOR,
                        DS_CROP: WHO}:
                bad.append(f'the label-flag store reads back as {seen}; the '
                           f'flag raised before the column existed is the '
                           f'admin\'s and is not rewritten to say so')
        finally:
            fa.paths = real_paths
            audit.BOX_FILE = real_box
            _restore_flags(dash, keep)


# ── the migration ───────────────────────────────────────────────────────────

def _shape(db, table):
    """(columns, rows) straight off disk, by a connection that cannot migrate."""
    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    try:
        cols = [r['name'] for r in c.execute(f'PRAGMA table_info({table})')]
        rows = [dict(r) for r in c.execute(f'SELECT * FROM {table}')]
        return cols, rows
    finally:
        c.close()


def _race_migrate(mod, db, n=8):
    """Migrate one store from n threads at once; return what raised.

    The barrier sits around _migrate() rather than around connect(). A sqlite3
    connection belongs to the thread that opened it, so each thread has to
    open its own first -- and the jitter of doing that is enough to hide the
    window on most runs, which is exactly why three sequential connects saw
    nothing. What is graded here is the two-step inside _migrate: read the
    table's columns, then ALTER it. Several requests arriving together run
    that concurrently, one wins the ALTER, and every other one used to take
    "duplicate column name" back to the reviewer as a failed click.
    """
    errs = []
    gate = threading.Barrier(n)

    def go():
        con = sqlite3.connect(db, timeout=10)
        con.row_factory = sqlite3.Row
        gate.wait()
        try:
            mod._migrate(con)
        except Exception as e:            # noqa: BLE001 - the thing measured
            errs.append(f'{type(e).__name__}: {e}')
        finally:
            con.close()

    ts = [threading.Thread(target=go) for _ in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return errs


def migration_checks(bad):
    """The column arrives without touching a row, arrives once, and survives
    arriving from several threads at the same moment.

    A migration that is not idempotent is a dashboard that starts once. These
    run on every connect(), which is every request that reads a verdict or
    lights a flagged tile, so "twice" is the normal case rather than the edge
    one -- and the dashboard is a ThreadingHTTPServer, so "twice at once" is
    the shape the first click on an unmigrated store actually presents.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls

    # (module, table, schema-before, seed, what the seeded row must still say)
    cases = (
        (ls, 'leash', LEASH_SCHEMA_V0,
         ('INSERT INTO leash (crop, image_id, label, conf, ts, labelled_at, '
          ' source, note) VALUES (?,?,?,?,?,?,?,?)',
          (OLD_CROP, '1606751523958969', 'leashed', 0.64, 1785663300001,
           1_700_000_000, 'review_page', 'kept')),
         {'crop': OLD_CROP, 'label': 'leashed', 'note': 'kept'}),
        (lf, 'flags', FLAGS_SCHEMA_V0,
         ('INSERT INTO flags (file, image_id, dataset, class_was, should_be, '
          ' run, note, flagged_at) VALUES (?,?,?,?,?,?,?,?)',
          (DS_CROP, '1606751523958968', 'dogbin_v5', 'not_dog', 'dog', 'r0',
           'kept', 1_700_000_000)),
         {'file': DS_CROP, 'should_be': 'dog', 'note': 'kept'}),
    )
    for mod, table, v0, (sql, args), keeper in cases:
        name = mod.__name__
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, 'store.db')
            con = sqlite3.connect(db)
            con.executescript(v0)
            con.execute(sql, args)
            con.commit()
            con.close()

            cols, _ = _shape(db, table)
            if mod.AUTHOR_FIELD in cols:
                bad.append(f'{name}: the fixture already has the column; this '
                           f'test cannot see the migration run at all')
            for run in (1, 2, 3):
                con = mod.connect(db)
                try:
                    mode = con.execute('PRAGMA journal_mode').fetchone()[0]
                    sync = con.execute('PRAGMA synchronous').fetchone()[0]
                finally:
                    con.close()
                cols, rows = _shape(db, table)
                if cols.count(mod.AUTHOR_FIELD) != 1:
                    bad.append(
                        f'{name}: after {run} connect(s) the annotator column '
                        f'appears {cols.count(mod.AUTHOR_FIELD)} times in '
                        f'{cols} — ALTER TABLE ran again and raised, or the '
                        f'schema grew a duplicate')
                if len(rows) != 1 or any(rows[0].get(k) != v
                                         for k, v in keeper.items()):
                    bad.append(f'{name}: the judgement already on record did '
                               f'not survive the migration: {rows}')
                if rows and rows[0].get(mod.AUTHOR_FIELD) is not None:
                    bad.append(
                        f'{name}: the migration filled in an author '
                        f'({rows[0][mod.AUTHOR_FIELD]!r}) — an existing row '
                        f'keeps its NULL and is read as the admin, it is not '
                        f'rewritten to say so')
                if rows and mod.row_dict(rows[0])[mod.AUTHOR_FIELD] != \
                        fa.LEGACY_AUTHOR:
                    bad.append(f'{name}: a migrated row does not read as the '
                               f'admin')
                if str(mode).lower() != 'wal' or str(sync) not in ('2', 'FULL'):
                    bad.append(f'{name}: connect() left journal_mode={mode} '
                               f'synchronous={sync}; the migration must not '
                               f'cost the store the durability it was built '
                               f'with')

            # And a store created fresh today needs no migration to have it.
            # SCHEMA on its own, without _migrate() behind it: going through
            # connect() cannot tell "the schema declares the column" from
            # "the migration added it a moment ago", and a schema that has
            # quietly lost it still passes while every new store is built one
            # ALTER short of the shape this file says it has.
            fresh = os.path.join(tmp, 'fresh.db')
            con = sqlite3.connect(fresh)
            try:
                con.executescript(mod.SCHEMA)
                cols = [r[1]
                        for r in con.execute(f'PRAGMA table_info({table})')]
            finally:
                con.close()
            if mod.AUTHOR_FIELD not in cols:
                bad.append(f'{name}: SCHEMA creates the table without the '
                           f'annotator column ({cols}) — a new store would '
                           f'only ever get one by migration, which is a '
                           f'schema that exists in no single place')

            # THE FIRST CLICK, racing the page reads beside it. One thread
            # wins the ALTER; the losers must carry on rather than hand the
            # reviewer a failed write.
            race = os.path.join(tmp, 'race.db')
            con = sqlite3.connect(race)
            con.executescript(v0)
            con.execute(sql, args)
            con.commit()
            con.close()
            errs = _race_migrate(mod, race)
            if errs:
                bad.append(
                    f'{name}: {len(errs)} of 8 concurrent first-opens of an '
                    f'unmigrated store raised {errs[0]} — one thread wins the '
                    f'ALTER and the rest have to carry on; a loser raising is '
                    f'a verdict the reviewer clicked and never got recorded')
            cols, rows = _shape(race, table)
            if cols.count(mod.AUTHOR_FIELD) != 1:
                bad.append(f'{name}: the raced migration left {cols}')
            if len(rows) != 1 or any(rows[0].get(k) != v
                                     for k, v in keeper.items()):
                bad.append(f'{name}: the raced migration did not leave the '
                           f'existing judgement alone: {rows}')


# ── the unauthenticated write ───────────────────────────────────────────────

def refusal_checks(bad):
    """No session, no line. Not a line signed `admin`.

    The gate answers every request before a route is matched, so a write that
    arrives without a session is a caller that forgot to pass one. Refusing is
    a visible failure on the first click; recording it as the admin would put
    somebody's name on a judgement they never made, in a file with no undo.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    import audit
    import dashboard as dash

    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        real_paths = fa.paths
        real_box = audit.BOX_FILE
        real_row, real_cut = audit._pool_row, audit._cut_one
        try:
            lay = _audit_layout(fa, tmp)
            fa.paths = lambda s='gate', _l=lay: _l
            audit.BOX_FILE = os.path.join(tmp, 'audit_boxes.jsonl')
            audit._pool_row = lambda key, stage='gate': {
                'key': MARK + '#0', 'image_id': '1606751523958968',
                'det_idx': 0, 'x1': 1.0, 'y1': 2.0, 'x2': 30.0, 'y2': 40.0}
            audit._cut_one = lambda *a, **k: False
            db = os.path.join(tmp, 'leash.db')
            fdb = os.path.join(tmp, 'label_flags.db')

            # every write entry point, with no annotator and with an empty
            # one -- '' is what a session-less handler actually produces
            for nobody in (None, ''):
                tries = (
                    ('flag_crop',
                     lambda: dash.flag_crop(CROP, dash.FLAG_LABEL,
                                            by=nobody)[0],
                     dash.HN_LABELS),
                    ('flag_crop undo',
                     lambda: dash.flag_crop(CROP, dash.FLAG_LABEL, undo=True,
                                            by=nobody)[0],
                     dash.HN_LABELS),
                    ('save_box',
                     lambda: dash.save_box(CROP, 0, [1, 2, 30, 40],
                                           by=nobody),
                     dash.BOX_LABELS),
                    ('audit.record',
                     lambda: audit.record(MARK + '#0', 'dog', by=nobody),
                     lay['verdicts']),
                    ('audit.save_correction',
                     lambda: audit.save_correction(MARK + '#0',
                                                   [1, 2, 30, 40], by=nobody),
                     audit.BOX_FILE),
                    ('leash_store.record',
                     lambda: ls.record(CROP, 'leashed', path=db,
                                       by=nobody)[0],
                     db),
                    ('label_flags.add',
                     lambda: lf.add(DS_CROP, dataset='dogbin_v5',
                                    should_be='dog', path=fdb, by=nobody)[0],
                     fdb),
                    ('mark_seen',
                     lambda: dash.mark_seen([CROP], by=nobody),
                     dash.SEEN_FILE),
                )
                for what, call, path in tries:
                    was = _lines(path) if path not in (db, fdb) else None
                    body = call()
                    if body.get('ok'):
                        bad.append(f'{what} accepted a write with no '
                                   f'annotator ({nobody!r}): {body}')
                    msg = str(body.get('error') or body.get('msg') or '')
                    if 'annotator' not in msg:
                        bad.append(f'{what} refused with {msg!r}, which does '
                                   f'not say what was missing')
                    if path in (db, fdb):
                        if os.path.exists(path):
                            bad.append(f'a refused {what} created the store '
                                       f'anyway')
                    elif _lines(path) != was:
                        bad.append(f'{what} appended a line while refusing '
                                   f'the write')
                    for rec in (_lines(path) if path not in (db, fdb)
                                else []):
                        if rec.get(fa.AUTHOR_FIELD) == fa.LEGACY_AUTHOR:
                            bad.append(f'{what} wrote a row attributed to the '
                                       f'admin for a caller with no session '
                                       f'— that is forged attribution, and it '
                                       f'is exactly what this field exists to '
                                       f'prevent')

            # the refusal is a REFUSAL, not a validation quirk: the same call
            # with an annotator goes through
            body, code = dash.flag_crop(CROP, dash.FLAG_LABEL, by=WHO)
            if not body.get('ok') or code != 200:
                bad.append(f'the same flag with an annotator was also '
                           f'refused ({body}) — the checks above prove '
                           f'nothing')
        finally:
            fa.paths = real_paths
            audit.BOX_FILE = real_box
            audit._pool_row, audit._cut_one = real_row, real_cut
            _restore_flags(dash, keep)


# ── the callers ─────────────────────────────────────────────────────────────

# Annotation writes reachable from do_POST, in the two shapes they take: a
# function defined in dashboard.py itself, and a method on one of the store
# modules the handler imports lazily (`mod`, or `a` for the audit module).
# Matched this narrowly on purpose -- `add` alone matches twenty-one ordinary
# set inserts in this file, and a rule that cries wolf at those is a rule the
# next person deletes.
WRITE_FUNCS = ('flag_crop', 'save_box', 'mark_seen')
WRITE_METHODS = ('record', 'add', 'save_correction')
STORE_NAMES = ('mod', 'a')
# `remove` is deliberately absent from both. Taking a verdict back is not an
# annotation and none of the three stores records who did it -- see the report
# beside this change; it is a known gap, not an oversight here.


def _annotation_writes(tree):
    """Every annotation write inside do_POST, as (node, printable name)."""
    posts = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == 'do_POST']
    out = []
    for post in posts:
        for node in ast.walk(post):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id in WRITE_FUNCS:
                out.append((node, f.id))
            elif (isinstance(f, ast.Attribute) and f.attr in WRITE_METHODS
                    and isinstance(f.value, ast.Name)
                    and f.value.id in STORE_NAMES):
                out.append((node, ast.unparse(f)))
    return out


def caller_checks(bad):
    """Every write route hands over the SESSION's username, and only that.

    Read out of the source rather than driven, because the failure is a route
    added next month that forgets -- and the thing that catches that is a rule
    over every call, not a test of the ones that exist today.
    """
    path = os.path.join(REPO, 'tools', 'dashboard', 'dashboard.py')
    src = open(path).read()
    tree = ast.parse(src)
    seen = 0
    for node, name in _annotation_writes(tree):
        f = node.func
        seen += 1
        kw = {k.arg: k.value for k in node.keywords}
        got = kw.get('by')
        ok = (isinstance(got, ast.Call)
              and isinstance(got.func, ast.Attribute)
              and got.func.attr == '_annotator'
              and isinstance(got.func.value, ast.Name)
              and got.func.value.id == 'self')
        if not ok:
            bad.append(f'dashboard.py:{node.lineno} calls '
                       f'{ast.unparse(f)} without by=self._annotator() '
                       f'({ast.unparse(node.keywords[-1]) if node.keywords else "no keywords"})'
                       f' — a write route that does not name its annotator '
                       f'either refuses every click or records nobody')
    if seen < 7:
        bad.append(f'only {seen} annotation writes found in do_POST; this '
                   f'check has stopped looking at the routes it is supposed '
                   f'to cover (flag, box, audit verdict, audit box, leash, '
                   f'label flag, reviewed-and-kept)')
    # and the username comes off the session, never off the wire
    body = src[src.index('def _annotator'):src.index('def _gate_broken')]
    if "'session'" not in body and 'self.session' not in body:
        bad.append('_annotator does not read the session the gate resolved')
    for wire in ('data.get', 'json.loads', 'rfile', 'headers'):
        if wire in body:
            bad.append(f'_annotator reads {wire} — a client that can name its '
                       f'own annotator can sign somebody else\'s name to a '
                       f'verdict')
    post = src[src.index('def do_POST'):]
    if "get('by'" in post or 'get("by"' in post:
        bad.append('a POST handler takes the annotator out of the request '
                   'body; the session is the only authority for who is '
                   'writing')


def route_checks(bad):
    """The routes themselves, driven: session in, username on the line.

    The check above reads the source; this one runs it. The two fail
    differently -- a rename of the session's `username` key passes an AST
    check and records nobody -- and the second is the one that catches a
    handler whose plumbing is right and whose vocabulary is not.

    No socket and no live server: the handler is built directly, the gate is
    the one thing stubbed out (it is what would otherwise answer the request),
    and every path it can write to points at a temp directory. POSTing at the
    running dashboard would record verdicts in the ledgers this file exists to
    protect.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    import audit
    import dashboard as dash

    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        real_paths = fa.paths
        real_box = audit.BOX_FILE
        real_leash = (ls.DB_PATH, ls.CROPS_OUT, ls.FULL_OUT)
        real_flags = lf.DB_PATH
        try:
            lay = _audit_layout(fa, tmp)
            fa.paths = lambda s='gate', _l=lay: _l
            audit.BOX_FILE = os.path.join(tmp, 'audit_boxes.jsonl')
            open(lay['pool'], 'wb').close()   # pool_ready() only asks it is there
            ls.DB_PATH = os.path.join(tmp, 'leash.db')
            ls.CROPS_OUT = os.path.join(tmp, 'leash_crops')
            ls.FULL_OUT = os.path.join(tmp, 'leash_full')
            lf.DB_PATH = os.path.join(tmp, 'label_flags.db')
            real_row = audit._pool_row
            audit._pool_row = lambda key, stage='gate': {
                'key': MARK + '#0', 'image_id': '1606751523958968',
                'det_idx': 0, 'x1': 1.0, 'y1': 2.0, 'x2': 30.0, 'y2': 40.0}

            def post(path, payload, session):
                """One request through the real dispatch. (reply, status)."""
                import io
                h = dash.BoardHandler.__new__(dash.BoardHandler)
                body = json.dumps(payload).encode()
                h.path = path
                h.command = 'POST'
                h.headers = {'Content-Length': str(len(body))}
                h.rfile = io.BytesIO(body)
                h.session = session
                got = []
                h._json = lambda o, c=200: got.append((o, c))
                h._gate = lambda: False      # the gate has already said yes
                h.do_POST()
                return got[0] if got else (None, None)

            routes = (
                ('/api/detect/flag', {'name': CROP,
                                      'label': dash.FLAG_LABEL},
                 lambda: _lines(dash.HN_LABELS)),
                ('/api/review/box', {'name': CROP, 'det_idx': 0,
                                     'box': [1, 2, 30, 40]},
                 lambda: _lines(dash.BOX_LABELS)),
                ('/api/audit/verdict?stage=gate', {'key': MARK + '#0',
                                                   'verdict': 'dog',
                                                   'band': 3, 'p_dog': 0.31},
                 lambda: _lines(lay['verdicts'])),
                ('/api/audit/box?stage=gate', {'key': MARK + '#0',
                                               'box': [1, 2, 30, 40]},
                 lambda: _lines(audit.BOX_FILE)),
                ('/api/review/leash', {'name': CROP, 'label': 'leashed'},
                 lambda: [dict(r) for r in
                          ls.connect(ls.DB_PATH).execute('SELECT * FROM leash')]),
                ('/api/training/relabel', {'file': DS_CROP,
                                           'dataset': 'dogbin_v5',
                                           'was': 'not_dog', 'should': 'dog',
                                           'run': 'r1'},
                 lambda: list(lf.flagged_files(path=lf.DB_PATH).values())),
                ('/api/review/seen', {'names': [CROP]},
                 lambda: _lines(dash.SEEN_FILE)),
            )
            # signed in: the row names the account the session names
            for path, payload, read in routes:
                reply, _code = post(path, payload,
                                    {'id': 1, 'username': WHO,
                                     'role': 'member'})
                rows = read()
                if not rows:
                    bad.append(f'POST {path} with a session wrote nothing: '
                               f'{reply}')
                elif rows[-1].get(fa.AUTHOR_FIELD) != WHO:
                    bad.append(f'POST {path} recorded {rows[-1]} — the row '
                               f'does not name the account the session named')
            # signed out: nothing is written and nothing is signed
            for path, payload, read in routes:
                was = read()
                reply, _code = post(path, payload, None)
                if read() != was:
                    bad.append(f'POST {path} with no session still wrote a '
                               f'row: {read()[-1]}')
                if isinstance(reply, dict) and reply.get('ok'):
                    bad.append(f'POST {path} with no session answered ok: '
                               f'{reply}')
        finally:
            fa.paths = real_paths
            audit.BOX_FILE = real_box
            audit._pool_row = real_row
            ls.DB_PATH, ls.CROPS_OUT, ls.FULL_OUT = real_leash
            lf.DB_PATH = real_flags
            _restore_flags(dash, keep)


# ── the surfaces ────────────────────────────────────────────────────────────

def surface_checks(bad):
    """The two views that read work back can say who judged a crop.

    Quietly, and only there: a byline on the tile, no column, and no
    filter-by-user -- which is a separate decision nobody has made.
    """
    import fn_audit as fa
    import audit
    A = fa.AUTHOR_FIELD

    page = audit.page_html('gate')
    if 'it.by' not in page:
        bad.append('the audit page never renders an annotation\'s author — '
                   'the "my annotations" view cannot say whose they are')
    if '.by{' not in page:
        bad.append('the audit page has no style for the byline, so it would '
                   'paint as unstyled text over the crop')
    import dashboard as dash
    review = dash.REVIEW_HTML
    if 'c.by' not in review:
        bad.append('the review page\'s audit view never renders the author')
    if '.meta .by{' not in review:
        bad.append('the review page has no style for the byline')
    # a byline, not a control: neither page may send the annotator back as a
    # filter, which is a feature nobody asked for and a way to rank people
    for name, doc in (('audit', page), ('review', review)):
        for wire in ('&by=', '?by=', "'by='"):
            if wire in doc:
                bad.append(f'the {name} page sends {wire!r} to the server — '
                           f'this change surfaces who judged a crop, it does '
                           f'not add a filter by person')
    # the markup reads it.by / c.by, and both payloads are keyed on the
    # store's own field name -- one of the two moving alone paints nothing at
    # all, silently (legacy_read_checks is what proves the payloads carry it)
    if A != 'by':
        bad.append(f'the stored field is {A!r} but both pages read .by')


# ── the name itself ─────────────────────────────────────────────────────────

def reserved_name_checks(bad):
    """Nobody but the .env admin may be called what the old rows are read as.

    LEGACY_AUTHOR is a perfectly ordinary username: the charset allows it and
    the only other gate is "is it taken already". On a deployment whose admin
    has been renamed -- DASHBOARD_USER changed, or data/ restored beside a
    fresh accounts.db -- the name is free, and an invited member taking it
    from the signup form would be credited with all 3,247 judgements made
    before they existed, while their own rows became indistinguishable from
    the founder's. No crafted request is needed for that, which is why it is
    graded here and not left to a comment.

    A THROWAWAY DATABASE AND A THROWAWAY .env. The live accounts.db is never
    opened: a test that mints accounts on the real dashboard hands somebody a
    login.
    """
    import fn_audit as fa
    sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
    import accounts as A

    founder = MARK + 'founder'
    pw = MARK + '-a-long-enough-passphrase'
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, 'accounts.db')
        envf = os.path.join(tmp, '.env')
        with open(envf, 'w') as fh:
            fh.write(f'DASHBOARD_USER={founder}\nDASHBOARD_PASSWORD={pw}\n')
        real_path = A.ENV_PATH
        had = os.environ.get(A.ENV_USER)
        A.ENV_PATH = envf
        # a process that never called load_env(): the reservation has to read
        # the file rather than fall back to the default, which IS the name
        os.environ.pop(A.ENV_USER, None)
        try:
            env = {A.ENV_USER: founder, A.ENV_PASSWORD: pw}
            got = A.ensure_admin(path=db, env=env)
            if not got.get('ok') or got.get('username') != founder:
                bad.append(f'the renamed admin could not be created: {got}')
            inv = A.create_invite(founder, path=db)
            try:
                A.redeem_invite(inv['token'], fa.LEGACY_AUTHOR, pw, path=db)
                bad.append(
                    f'a member signed up as {fa.LEGACY_AUTHOR!r} on a '
                    f'deployment whose admin is {founder!r} — every '
                    f'annotation made before there were accounts now reads as '
                    f'theirs, and their own rows are indistinguishable from '
                    f'the founder\'s')
            except A.AccountError as e:
                if e.code != 'username_reserved':
                    bad.append(f'signing up as {fa.LEGACY_AUTHOR!r} was '
                               f'refused for the wrong reason: {e.code}')
            holders = [u['username'] for u in A.list_users(path=db)
                       if A.normalise_username(u['username'])
                       == A.normalise_username(fa.LEGACY_AUTHOR)
                       and u['role'] != 'admin']
            if holders:
                bad.append(f'{holders} hold the legacy author name without '
                           f'being the admin')
            # The refusal is a refusal, not a signup form nobody can use. Its
            # own invite: if the check above has regressed, the member took
            # the first one with it, and reusing it here would report a spent
            # link instead of the thing that actually broke.
            spare = A.create_invite(founder, path=db)
            who = A.redeem_invite(spare['token'], MARK + 'mia', pw, path=db)
            if who['username'] != MARK + 'mia':
                bad.append(f'an ordinary member could not sign up: {who}')
        finally:
            A.ENV_PATH = real_path
            if had is None:
                os.environ.pop(A.ENV_USER, None)
            else:
                os.environ[A.ENV_USER] = had

    # and where the admin really is called that -- this deployment, and every
    # one that never renamed it -- the name is still theirs to hold
    try:
        A.check_username(fa.LEGACY_AUTHOR,
                         env={A.ENV_USER: fa.LEGACY_AUTHOR})
    except A.AccountError as e:
        bad.append(f'the .env admin cannot be called {fa.LEGACY_AUTHOR!r} '
                   f'({e.code}) — the reservation has locked out the one '
                   f'account it is reserving the name for')


# ── the ledger an undo rewrites ─────────────────────────────────────────────

def torn_line_checks(bad):
    """An undo takes out one crop and nothing else -- not a line it cannot read.

    _rewrite_labels is the one path in the repo that takes a line back out of
    a flag ledger, and it used to keep only what it could parse: a torn final
    line (a crash mid-append, which every reader is written to survive) was
    dropped by the next person's undo, silently, out of a file holding
    everybody's work. Undo is something any signed-in member can now do to any
    crop, so this is one member spending another's damaged line.
    """
    import dashboard as dash

    torn = '{"image_id": "1606751523958777", "conf":'
    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        try:
            dash.flag_crop(CROP, dash.FLAG_LABEL, by=WHO)
            dash.flag_crop(OLD_CROP, dash.FLAG_LABEL, by=WHO)
            path = dash._store_for(dash.FLAG_LABEL)['labels']
            with open(path, 'a') as fh:
                fh.write(torn + '\n')
            dash._flagged = None
            dash.flag_crop(CROP, dash.FLAG_LABEL, undo=True, by=WHO)
            with open(path) as fh:
                now = [x.strip() for x in fh if x.strip()]
            crops = [json.loads(x)['crop'] for x in now if x != torn]
            if torn not in now:
                bad.append('an undo dropped a line the rewriter could not '
                           'parse; a line it cannot read is not the crop '
                           'being withdrawn, and no command puts it back')
            if crops != [OLD_CROP]:
                bad.append(f'the undo left {crops}; it must remove exactly '
                           f'the one crop and leave every other line alone')
        finally:
            _restore_flags(dash, keep)


# ── re-deciding ─────────────────────────────────────────────────────────────

def redecision_checks(bad):
    """Pressing a verdict a second time: three stores, three shapes, each
    answering for the shape it is.

    An append-only ledger keeps the first line and the person who wrote it -- a
    second reviewer pressing the verdict a crop already carries is agreeing,
    not re-deciding. A table holding one row per crop keeps the call standing
    now, and the person whose call it now is. Both are defensible; what is not
    is nobody being able to say which a given store does, so each is asserted
    here and stated in the source beside the branch that does it.
    """
    import fn_audit as fa
    import label_flags as lf
    import leash_store as ls
    import audit
    import dashboard as dash

    A = fa.AUTHOR_FIELD
    alice, bob = MARK + 'alice', MARK + 'bob'
    with tempfile.TemporaryDirectory() as tmp:
        keep = _flag_layout(dash, tmp)
        real_paths = fa.paths
        try:
            # append-only, one line per crop: the first flag stands
            dash.flag_crop(CROP, dash.FLAG_LABEL, by=alice)
            again, _ = dash.flag_crop(CROP, dash.FLAG_LABEL, by=bob)
            rows = _lines(dash._store_for(dash.FLAG_LABEL)['labels'])
            if not again.get('duplicate'):
                bad.append(f'a second flag of the same crop was not treated '
                           f'as a duplicate: {again}')
            if [r.get(A) for r in rows] != [alice]:
                bad.append(f'the flag ledger reads {[r.get(A) for r in rows]}; '
                           f'agreeing with a verdict already on record does '
                           f'not take it off the person who made it')

            # one row per crop: the annotator moves with the verdict
            db = os.path.join(tmp, 'leash.db')
            ls.record(CROP, 'leashed', path=db, by=alice)
            ls.record(CROP, 'unleashed', path=db, by=bob)
            con = ls.connect(db)
            try:
                got = [(r['label'], ls.row_dict(r)[A])
                       for r in con.execute('SELECT * FROM leash')]
            finally:
                con.close()
            if got != [('unleashed', bob)]:
                bad.append(f'the leash store reads {got}; one row holds the '
                           f'verdict standing now and the person whose call '
                           f'it now is')

            # same shape, same answer
            fdb = os.path.join(tmp, 'label_flags.db')
            lf.add(DS_CROP, should_be='dog', path=fdb, by=alice)
            lf.add(DS_CROP, should_be='not_dog', path=fdb, by=bob)
            rows = lf.flagged_files(path=fdb)
            if [(r['should_be'], r[A]) for r in rows.values()] != \
                    [('not_dog', bob)]:
                bad.append(f'the label-flag store reads {rows}; re-flagging '
                           f'replaces the overrule and its author together')

            # append-only, last line wins: the reader keeps bob's answer
            lay = _audit_layout(fa, os.path.join(tmp, 'audit'))
            fa.paths = lambda s='gate', _l=lay: _l
            audit.record(MARK + '#0', 'dog', by=alice)
            audit.record(MARK + '#0', 'not_dog', by=bob)
            seen = [(v['key'], v.get(A))
                    for v in fa.read_verdicts(stage='gate')]
            if seen != [(MARK + '#0', bob)]:
                bad.append(f'the audit ledger reads back {seen}; a mind '
                           f'changed later is another line and the reader '
                           f'keeps the last one, annotator included')
        finally:
            fa.paths = real_paths
            _restore_flags(dash, keep)


def main():
    bad = []
    for fn in (one_spelling_checks, new_record_checks, legacy_read_checks,
               migration_checks, refusal_checks, caller_checks, route_checks,
               surface_checks, reserved_name_checks, torn_line_checks,
               redecision_checks):
        try:
            fn(bad)
        except Exception as e:                 # noqa: BLE001 - report, not die
            bad.append(f'{fn.__name__} threw {type(e).__name__}: {e}')
    # last, and unconditionally: everything above wrote fixtures, and the
    # only acceptable number of them in a human store is zero
    live_untouched(bad)
    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print('every annotation names its annotator, the ledgers that predate '
          'accounts read as the admin, and a write with no session is '
          'refused rather than signed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
