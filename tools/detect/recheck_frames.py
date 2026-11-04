#!/usr/bin/env python3
"""A label batch of frames somebody already labelled, served again to fix.

    python tools/detect/recheck_frames.py make --name recheck_001 \\
        --list <frames.json>

The Annotate page serves a frame until it has a label, so a frame labelled
wrong -- a dog left out, a box round three dogs, "no dog" on a frame with a
dog in it -- can never come back through the batch it was labelled in. A
recheck batch is the way back: the page serves its frames until each is
SAVED AGAIN after the batch was made (annotate_page.py, kind `recheck`),
shows the boxes already drawn, says what the check found, and marks where a
detector is sure of a dog that no box covers.

THE LIST is JSON: [{"image_id", "issues": [short sentences], "suspects":
[[x1, y1, x2, y2], ...]}, ...], boxes in the frame's own pixels. Whatever
produced it -- a model, a person looking at a contact sheet -- it is a reason
to look again, never a label: nothing here writes the label ledger, and the
suspect boxes are drawn as hints only.

Every frame must already be labelled and must come from an existing batch,
whose manifest row (the jpg, the sequence, the size) the recheck row copies.
A batch is never rebuilt over.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCHES = os.path.join(REPO, 'data', 'label_batches')
# \Z, not $: `$` also matches before a trailing newline
NAME_RE = re.compile(r'^[a-z0-9][a-z0-9_]{2,60}\Z')
# What a recheck row copies from the frame's own batch. Not `role`: a row
# carrying a hidden-dog role would make a broken recheck batch read as a
# hidden-dog batch.
KEEP = ('image_id', 'jpg', 'seq', 'w', 'h', 'cell', 'drive')
# how far past the frame's edge a suspect box may reach and be clamped
EDGE = 2.0


def _manifests(batches):
    """{image_id: [(batch, manifest row), ...]} over every batch that is not
    itself a recheck, in name order."""
    out = {}
    for name in sorted(os.listdir(batches)):
        man = os.path.join(batches, name, 'manifest.jsonl')
        if not os.path.isfile(man):
            continue
        try:
            with open(os.path.join(batches, name, 'summary.json')) as fh:
                if (json.load(fh) or {}).get('kind') == 'recheck':
                    continue
        except (OSError, ValueError, AttributeError):
            pass
        with open(man) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r.get('from_batch'):
                        continue  # a recheck row, whatever its summary says
                    out.setdefault(str(r['image_id']), []).append((name, r))
    return out


def _pick(sources, prefer):
    """The (batch, row) to copy: the first of `prefer` that lists the frame
    and whose jpg is on disk, else any whose jpg is. None if no jpg is."""
    order = sorted(sources,
                   key=lambda x: (prefer.index(x[0])
                                  if x[0] in prefer else len(prefer)))
    for name, r in order:
        if os.path.isfile(str(r.get('jpg') or '')):
            return name, r
    return None


def _size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def plan(items, batches, labelled, size_of=_size):
    """(rows, problems): the recheck manifest rows, and why any item was
    refused. Nothing is written."""
    found = _manifests(batches)
    rows, problems, seen = [], [], set()
    for it in items:
        iid = str(it.get('image_id') or '')
        if not iid or iid in seen:
            problems.append(f'{iid or "?"}: missing or listed twice')
            continue
        seen.add(iid)
        if iid not in found:
            problems.append(f'{iid}: in no label batch')
            continue
        if iid not in labelled:
            problems.append(f'{iid}: has no label to recheck')
            continue
        raw = it.get('issues')
        if not isinstance(raw, list) or not all(
                isinstance(x, str) for x in raw):
            problems.append(f'{iid}: issues must be a list of sentences')
            continue
        issues = [x.strip() for x in raw if x.strip()]
        if not issues:
            problems.append(f'{iid}: no reason given to look again')
            continue
        # the batch the frame was labelled in, then the one the list names
        prefer = [
            str(labelled[iid].get('batch') or ''),
            str(it.get('from') or '')
        ]
        got = _pick(found[iid], prefer)
        if got is None:
            problems.append(f'{iid}: no batch it is in has its jpg on disk')
            continue
        src, r = got
        try:
            w, h = (float(v) for v in size_of(r['jpg']))
        except Exception:  # noqa: BLE001 - an unreadable frame is refused
            problems.append(f'{iid}: its jpg cannot be read')
            continue
        row = {k: r[k] for k in KEEP if k in r}
        row['image_id'] = iid
        row['from_batch'] = src
        row['issues'] = issues
        sus, bad = [], ''
        raw_sus = it.get('suspects') or []
        if not isinstance(raw_sus, list):
            bad = 'suspects must be a list of boxes'
        for b in raw_sus if not bad else []:
            if not isinstance(b, (list, tuple)) or len(b) != 4:
                bad = 'a suspect box is not four numbers'
                break
            try:
                x1, y1, x2, y2 = (float(v) for v in b)
            except (TypeError, ValueError):
                bad = 'a suspect box is not four numbers'
                break
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                bad = 'a suspect box is not four numbers'
                break
            if not (x1 < x2 and y1 < y2 and x1 >= -EDGE and y1 >= -EDGE
                    and x2 <= w + EDGE and y2 <= h + EDGE):
                bad = (f'a suspect box {[x1, y1, x2, y2]} does not fit the '
                       f'{int(w)}x{int(h)} frame')
                break
            sus.append([
                round(max(0.0, x1), 1),
                round(max(0.0, y1), 1),
                round(min(w, x2), 1),
                round(min(h, y2), 1)
            ])
        if bad:
            problems.append(f'{iid}: {bad}')
            continue
        row['suspects'] = sus
        rows.append(row)
    return rows, problems


def make(name,
         items,
         batches=BATCHES,
         labelled=None,
         source='',
         now=None,
         size_of=_size):
    """Write the batch. Refuses to write anything if any item is refused, or
    if the batch already exists. Written into a temporary directory and
    renamed into place, so a batch is either whole or not there at all."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise SystemExit('--name: lowercase letters, digits and _ only')
    out = os.path.join(batches, name)
    if os.path.exists(out):
        raise SystemExit(f'{out} already exists -- a batch is never rebuilt '
                         'over; pick a new name')
    if labelled is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import frame_labels
        labelled = frame_labels.read()
    rows, problems = plan(items, batches, labelled, size_of=size_of)
    if problems:
        raise SystemExit('nothing written -- %d item(s) refused:\n  %s' %
                         (len(problems), '\n  '.join(problems[:20])))
    if not rows:
        raise SystemExit('nothing to recheck')
    froms = {}
    for r in rows:
        froms[r['from_batch']] = froms.get(r['from_batch'], 0) + 1
    tmp = tempfile.mkdtemp(prefix='.' + name + '.', dir=batches)
    try:
        with open(os.path.join(tmp, 'summary.json'), 'w') as fh:
            json.dump(
                {
                    'kind': 'recheck',
                    'batch': name,
                    'planned_at': int(time.time() if now is None else now),
                    'picked': len(rows),
                    'from_batches': froms,
                    'source': source,
                },
                fh,
                indent=1)
        with open(os.path.join(tmp, 'manifest.jsonl'), 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        os.chmod(tmp, 0o775)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out, rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make', help='write a recheck batch from a list')
    mk.add_argument('--name', required=True)
    mk.add_argument('--list', required=True, help='JSON list of frames')
    mk.add_argument('--source',
                    default='',
                    help='what found the problems, for the summary')
    a = ap.parse_args(argv)
    with open(a.list) as fh:
        items = json.load(fh)
    out, rows = make(a.name, items, source=a.source)
    print(f'wrote {out}: {len(rows)} frame(s) to recheck')
    return 0


if __name__ == '__main__':
    sys.exit(main())
