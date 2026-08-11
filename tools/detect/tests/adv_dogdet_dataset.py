#!/usr/bin/env python3
"""The detector dataset must be a measurement instrument, not just a pile.

dogdet_v2 shipped with 63.1% of val in sequences that also appear in train --
the same leakage measured at 70.8% in leash_binary_v1 -- so every recall
number read off it was part memorisation. These checks run against whatever
dogdet_v3 build is on disk and fail if the split has rotted.

Nothing here trusts the builder's summary alone: splits, labels and files are
re-read from the dataset itself; the manifest supplies only the sequence map,
and any frame it cannot vouch for must be sitting in train.

INTERPRETER. Set $TRAINING_ROOT and this runs anywhere. Without it the
training repo is resolved by importing tools/dashboard/dashboard.py, which is
a 3.12+ file -- so under an older interpreter (mp is 3.11, and it is the env
MEMORY.md names for most tools/detect scripts) the import raises and there is
no other way to find the dataset. That case now exits 1 and says so; it used
to be swallowed into 'SKIP: no dogdet_v3 on disk' beside a dogdet_v3 that was
on disk, which is this suite's only check on the sequence split reporting
success without reading a single file.
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
def _training_root():
    """(root, where it came from) -- env first, then the dashboard's config.

    NO FALLBACK. There used to be one: dirname(dirname(REPO)), the mount the
    repo sits on, which is not a training repo on any machine and cannot
    become one. Its only effect was to turn "I could not work out where the
    datasets are" into "there are no datasets", and main() reported that as a
    clean skip with exit 0. An unresolvable root is a broken invocation and
    has to read as one -- an answer this file cannot give is not a pass.
    """
    got = os.environ.get('TRAINING_ROOT')
    if got:
        return got, '$TRAINING_ROOT'
    try:
        sys.path.insert(0, os.path.join(REPO, 'tools', 'dashboard'))
        import dashboard as dash
        got = dash.training_root()
    except Exception as exc:
        return '', (f'importing tools/dashboard/dashboard.py raised '
                    f'{type(exc).__name__}: {exc}')
    if not got:
        return '', 'the dashboard config names no training_root'
    return got, 'the dashboard config'


TRAINING_ROOT, ROOT_FROM = _training_root()
V3 = os.path.join(TRAINING_ROOT, 'dogdet_v3') if TRAINING_ROOT else ''


def main():
    if not TRAINING_ROOT:
        print(f'SKIP-AS-FAILURE: could not resolve the training root — '
              f'{ROOT_FROM}')
        print('  Nothing was checked, and the sequence split is the one thing '
              'this file exists to check.')
        print('  Set TRAINING_ROOT=<training repo>, or run it under the '
              'interpreter the dashboard runs on (dashboard.py is 3.12+).')
        return 1
    if not os.path.isdir(V3):
        print(f'SKIP: no dogdet_v3 under {TRAINING_ROOT} ({ROOT_FROM}) — '
              f'nothing to check yet')
        return 0
    bad = []
    man = json.load(open(os.path.join(V3, 'manifest.json')))
    seq = man.get('sequences') or {}
    splits = {}
    for sp in ('train', 'val'):
        ids = {f[:-4] for f in os.listdir(os.path.join(V3, 'images', sp))}
        lbl = {f[:-4] for f in os.listdir(os.path.join(V3, 'labels', sp))}
        if ids != lbl:
            bad.append(f'{sp}: {len(ids ^ lbl)} frames where image and label '
                       f'do not pair up')
        splits[sp] = ids

    both = splits['train'] & splits['val']
    if both:
        bad.append(f'{len(both)} frames sit in BOTH splits: '
                   f'{sorted(both)[:3]}')

    # ── the reason this file exists ─────────────────────────────────────────
    tr_seq = {seq[i] for i in splits['train'] if seq.get(i)}
    leak = [i for i in splits['val'] if seq.get(i) and seq[i] in tr_seq]
    if leak:
        bad.append(f'{len(leak)} val frames share a sequence with train — '
                   f'recall read off them is memorisation, e.g. '
                   f'{sorted(leak)[:3]}')
    unvouched = [i for i in splits['val'] if not seq.get(i)]
    if unvouched:
        bad.append(f'{len(unvouched)} val frames have no sequence on record; '
                   f'a frame that cannot prove it is unrelated to train '
                   f'belongs in train')

    # the train-30-blind holdout survives inside val
    hold = set(man.get('holdout') or [])
    if not hold:
        bad.append('the manifest names no holdout — there is no ground left '
                   'on which the old and new model can be compared fairly')
    elif not hold <= splits['val']:
        bad.append(f'{len(hold - splits["val"])} holdout frames are not in '
                   f'val — the train-30-blind benchmark has been eaten')

    # ── labels are labels ───────────────────────────────────────────────────
    n_bg = {'train': 0, 'val': 0}
    for sp in ('train', 'val'):
        for f in os.listdir(os.path.join(V3, 'labels', sp)):
            p = os.path.join(V3, 'labels', sp, f)
            txt = open(p).read().strip()
            if not txt:
                n_bg[sp] += 1
                continue
            for ln in txt.splitlines():
                parts = ln.split()
                ok = (len(parts) == 5 and parts[0] == '0')
                if ok:
                    try:
                        vals = [float(x) for x in parts[1:]]
                        ok = (all(0.0 <= v <= 1.0 for v in vals)
                              and vals[2] > 0 and vals[3] > 0)
                    except ValueError:
                        ok = False
                if not ok:
                    bad.append(f'{sp}/{f}: malformed label line {ln!r}')
                    break

    # backgrounds teach "no dog here"; past a point they buy precision with
    # recall, and this dataset exists for recall
    share = n_bg['train'] / max(1, len(splits['train']))
    if share > 0.25:
        bad.append(f'{share:.0%} of train is background — that trades away '
                   f'the recall this retrain is for')
    for sp in ('train', 'val'):
        if n_bg[sp] != man.get(f'backgrounds_{sp}'):
            bad.append(f'manifest says {man.get(f"backgrounds_{sp}")} {sp} '
                       f'backgrounds, the directory holds {n_bg[sp]}')

    # every confirmed near-miss the builder claims is really there, labelled
    added = man.get('added_ids') or []
    for i in added[:50] + added[-50:]:
        p = os.path.join(V3, 'labels', 'train', f'{i}.txt')
        if not os.path.exists(p):
            bad.append(f'added frame {i} has no label in train')
        elif len(open(p).read().strip().splitlines()) != 1:
            bad.append(f'added frame {i} should carry exactly the one '
                       f'confirmed box')
    if added and set(added) & splits['val']:
        bad.append('confirmed positives were added to val — additions may '
                   'only ever teach, never grade')

    if bad:
        for b in bad:
            print(f'FAIL {b}')
        return 1
    print(f"dogdet_v3 is sequence-clean: {len(splits['train'])} train / "
          f"{len(splits['val'])} val, 0 leaked, holdout {len(hold)} intact, "
          f"{len(added)} confirmed positives in train")
    return 0


if __name__ == '__main__':
    sys.exit(main())
