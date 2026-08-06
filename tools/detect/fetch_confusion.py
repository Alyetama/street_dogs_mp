#!/usr/bin/env python3
"""
Cache each training run's confusion matrix from Comet into data/confusion.json.

    python tools/detect/fetch_confusion.py            # show what is cached
    python tools/detect/fetch_confusion.py --update   # pull anything missing
    python tools/detect/fetch_confusion.py --update --projects dog-bin
    python tools/detect/fetch_confusion.py --update --all   # refetch everything

Ultralytics writes a confusion_matrix.png next to every finished run, but a PNG
is a picture of the numbers, not the numbers -- it cannot be restyled, summed,
or compared. The same matrix goes to Comet as ``confusion-matrix.json``, which
is where the actual counts are, so that is what this pulls.

ORIENTATION -- the one thing worth getting right here. Comet's own envelope
says ``rowLabel: "Actual Category"``, and for an ultralytics matrix that is
WRONG. ``ConfusionMatrix.process_cls_preds`` does ``self.matrix[p][t] += 1``
(ultralytics/utils/metrics.py), so rows are the PREDICTION and columns are the
TRUTH, which is also what its own plot says: xlabel "True", ylabel "Predicted".
Verified against a held split rather than taken on faith -- dogbin_006's val
set is 330 dog / 169 not_dog, and those are the COLUMN sums of its matrix, not
the row sums. Believing Comet's label would transpose the matrix and swap every
false positive with a false negative.

So the cache records the orientation explicitly, and readers use that rather
than re-deriving it.

The API key is read from a .env (never printed, never written to the JSON).
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(REPO, 'data', 'confusion.json')
ENV_FILES = tuple(p for p in (os.environ.get('COMET_ENV_FILE'),
                              os.path.join(REPO, '.env')) if p)
ASSET = 'confusion-matrix.json'
# rows are the prediction, columns the truth -- see the module docstring
ORIENTATION = 'rows=predicted, cols=true'


def load_key():
    """COMET_API_KEY from the environment or a .env. Never logged."""
    import re
    if os.environ.get('COMET_API_KEY'):
        return True
    for p in ENV_FILES:
        try:
            for ln in open(p):
                m = re.match(r'^COMET_API_KEY=(.+)$', ln.strip())
                if m:
                    os.environ['COMET_API_KEY'] = \
                        m.group(1).strip().strip('"').strip("'")
                    return True
        except OSError:
            continue
    return False


def proj_name(p):
    """The project's NAME, given whatever was passed as ``project=``.

    ultralytics takes project= as a directory, so it is equally happy with a
    bare name and an absolute path, and logs back whatever it was given. Both
    spellings mean one project, so both must reduce to one key -- the same rule
    the dashboard's tracker uses, or the cache would miss every run launched
    with a full path.
    """
    p = str(p or '').strip()
    return (os.path.basename(os.path.normpath(p)) or p) if p else p


def load_state():
    try:
        with open(STATE) as fh:
            got = json.load(fh)
        if isinstance(got, dict) and isinstance(got.get('runs'), dict):
            return got
    except (OSError, ValueError):
        pass
    return {'_comment': [
        'Confusion matrices pulled from Comet by',
        '  tools/detect/fetch_confusion.py --update',
        'orientation is rows=predicted, cols=true -- Comet labels its rows',
        '"Actual Category", which is wrong for an ultralytics matrix. See the',
        'script docstring; it was checked against a val split, not assumed.'],
        'workspace': None, 'updated_at': None, 'runs': {}}


def save_state(st):
    tmp = STATE + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(st, fh, indent=2, sort_keys=False)
        fh.write('\n')
    os.replace(tmp, STATE)


def trim(labels, matrix):
    """Drop trailing classes that never occur, in either direction.

    A classify run always carries ultralytics' 'background' row and column and
    they are always zero -- it is a detection concept. Showing an empty third
    row and column in a two-class matrix invites the reader to wonder what fell
    into it. A class with any count anywhere is kept.
    """
    n = len(labels)
    keep = [i for i in range(n)
            if any(matrix[i][j] for j in range(n))
            or any(matrix[j][i] for j in range(n))]
    if not keep or len(keep) == n:
        return labels, matrix
    return ([labels[i] for i in keep],
            [[matrix[i][j] for j in keep] for i in keep])


def fetch(update=False, refetch=False, workspace=None, only=None):
    if not load_key():
        print('no COMET_API_KEY (set $COMET_ENV_FILE to the .env holding it)',
              file=sys.stderr)
        return 1
    import comet_ml
    api = comet_ml.API()
    st = load_state()
    ws = workspace or st.get('workspace') or (api.get_workspaces() or [None])[0]
    if not ws:
        print('no Comet workspace visible', file=sys.stderr)
        return 1
    st['workspace'] = ws
    runs = st['runs']
    added = skipped = 0
    want = {x.strip().lower() for x in (only or []) if x.strip()}
    for proj in api.get_projects(ws):
        pname = proj if isinstance(proj, str) else proj.get('projectName')
        # Sweeping every project in the workspace costs a parameter fetch and
        # an asset listing per experiment, which is ten minutes across a
        # workspace holding unrelated work.
        if want and str(pname).lower() not in want:
            continue
        for exp in api.get(ws, pname):
            params = {p['name']: p['valueCurrent']
                      for p in (exp.get_parameters_summary() or [])}
            name = params.get('name')
            if not name:
                continue
            key = f'{proj_name(params.get("project") or pname)}/{name}'
            if key in runs and not refetch:
                skipped += 1
                continue
            hit = [a for a in exp.get_asset_list()
                   if a.get('fileName') == ASSET]
            if not hit:
                continue
            try:
                doc = json.loads(exp.get_asset(hit[0]['assetId'],
                                               return_type='text'))
                labels = [str(x) for x in doc['labels']]
                matrix = [[int(round(float(v))) for v in row]
                          for row in doc['matrix']]
            except (ValueError, KeyError, TypeError) as e:
                print(f'  {key}: unreadable asset ({type(e).__name__})')
                continue
            labels, matrix = trim(labels, matrix)
            runs[key] = {'labels': labels, 'matrix': matrix,
                         'orientation': ORIENTATION,
                         'experiment': exp.name, 'comet_project': pname}
            added += 1
            n = len(labels)
            print(f'  + {key}  {n}x{n}  from {exp.name}')
    if added:
        import datetime as _dt
        st['updated_at'] = _dt.date.today().isoformat()
    if update:
        save_state(st)
    print(f'{added} fetched, {skipped} already cached, '
          f'{len(runs)} total in {os.path.relpath(STATE, REPO)}')
    return 0


def show():
    st = load_state()
    runs = st.get('runs') or {}
    if not runs:
        print('nothing cached yet -- run with --update')
        return 0
    print(f'{len(runs)} runs cached, updated {st.get("updated_at")}, '
          f'orientation {ORIENTATION}')
    for key in sorted(runs):
        r = runs[key]
        tot = sum(sum(row) for row in r['matrix'])
        diag = sum(r['matrix'][i][i] for i in range(len(r['labels'])))
        acc = f'{diag / tot:.4f}' if tot else '-'
        print(f'  {key:34s} {len(r["labels"])} classes  n={tot:<6d} acc={acc}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--update', action='store_true',
                    help='pull matrices missing from the cache and save')
    ap.add_argument('--all', action='store_true',
                    help='with --update, refetch runs already cached')
    ap.add_argument('--workspace', default=None)
    ap.add_argument('--projects', default=None,
                    help='comma-separated Comet projects to sweep '
                         '(default: every project in the workspace)')
    a = ap.parse_args(argv)
    if a.update:
        return fetch(update=True, refetch=a.all, workspace=a.workspace,
                     only=(a.projects or '').split(','))
    return show()


if __name__ == '__main__':
    sys.exit(main())
