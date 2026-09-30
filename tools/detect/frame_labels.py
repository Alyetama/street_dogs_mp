"""Whole-frame labels from the dashboard's Annotate page, as detector rows.

The Annotate page (tools/dashboard/annotate_page.py) is where a person boxes
EVERY dog in a frame, or says there is none, and each save is one line of
data/frame_labels/labels.jsonl. This reads that ledger for the detector build
(build_dogdet_v3.py), and is the only place the build learns what it says.

WHAT MAKES A FRAME LABEL DIFFERENT from every other addition the build makes.
A confirmed positive is one box a person agreed with, and the silence around
it is only as true as the sweep could make it -- which is why those additions
are single-detection frames only. A frame label is a person saying "these are
all the dogs in this frame": the silence is theirs. So a frame label may carry
any number of boxes, and a frame declared empty is a real background, written
as an empty label file the way YOLO takes one.

THE SPLIT RULE IS THE BUILD'S. A frame whose sequence is in val is left out,
never moved: a labelled neighbour of a val frame is the val scene handed to
the model in advance. A frame with no known sequence goes to train, as the
build pins every unresolved frame. A frame already in the dataset keeps the
label it came in with -- the frames were chosen to be new, so this happening
means the source changed, and the build is not the place to settle which
human was right.

A label is the ledger's LATEST line for its image, so a correction on the page
is a second save and the build sees only the second.
"""

import json
import math
import os

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER = os.path.join(REPO, 'data', 'frame_labels', 'labels.jsonl')
SOURCE = 'annotate_page'


def read(path=LEDGER):
    """{image_id: the latest usable record}. A missing ledger is no labels.

    Only a record the page wrote and a signed-in person signed is a label: a
    line without an author, or from anything other than the page, is not
    somebody saying what is in the frame.
    """
    out = {}
    try:
        fh = open(path)
    except OSError:
        return out
    with fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a torn last line is not a label
            if not isinstance(r, dict) or not r.get('image_id'):
                continue
            if r.get('source') != SOURCE or not r.get('by'):
                continue
            out[str(r['image_id'])] = r
    return out


def yolo(rec):
    """The record as YOLO label text: one `0 cx cy w h` line per box, or ''
    for a frame declared empty. None when the record cannot be trusted --
    a size that is not a size, a box that is not a box, or boxes on a frame
    that says it has none."""
    try:
        w, h = float(rec['w']), float(rec['h'])
    except (KeyError, TypeError, ValueError):
        return None
    if not (w > 0 and h > 0 and math.isfinite(w) and math.isfinite(h)):
        return None
    boxes = rec.get('boxes') or []
    if not isinstance(boxes, list):
        return None
    if bool(rec.get('empty')) != (not boxes):
        return None
    lines = []
    for b in boxes:
        try:
            x1, y1, x2, y2 = (float(v) for v in b)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        lines.append('0 %.6f %.6f %.6f %.6f\n' %
                     ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w,
                      (y2 - y1) / h))
    return ''.join(lines)


def plan(recs, seq, val_seqs, have, where, size_of=None):
    """Which frame labels go into train, and why the rest do not.

    recs      {image_id: record} from read()
    seq       {image_id: sequence}, resolved by the build
    val_seqs  the sequences the build put in val
    have      image ids already in the dataset
    where     {image_id: path of the frame's jpg on disk}
    size_of   path -> (w, h), to check the file is the one that was labelled

    Returns ([(image_id, jpg, label_text)], {reason: count}).
    """
    added = []
    skipped = {
        'already_in_dataset': 0,
        'val_sequence': 0,
        'no_file': 0,
        'bad_record': 0,
        'size_changed': 0
    }
    for iid in sorted(recs):
        rec = recs[iid]
        if iid in have:
            skipped['already_in_dataset'] += 1
            continue
        if seq.get(iid) is not None and seq.get(iid) in val_seqs:
            skipped['val_sequence'] += 1
            continue
        text = yolo(rec)
        if text is None:
            skipped['bad_record'] += 1
            continue
        src = where.get(iid) or ''
        if not src or not os.path.isfile(src):
            skipped['no_file'] += 1
            continue
        if size_of is not None:
            try:
                got = tuple(int(v) for v in size_of(src))
            except Exception:  # noqa: BLE001 - unreadable is not the frame
                got = None
            if got != (int(rec['w']), int(rec['h'])):
                skipped['size_changed'] += 1
                continue
        added.append((iid, src, text))
    return added, skipped


def image_size(path):
    """(w, h) off the file's header."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size
