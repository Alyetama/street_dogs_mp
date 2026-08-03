#!/usr/bin/env python3
"""
Track the best model per Comet project in data/best_models.json.

    python tools/detect/best_models.py            # show what is recorded
    python tools/detect/best_models.py --update   # refresh candidate metrics

``--update`` re-reads every experiment in the tracked projects and rewrites the
``candidates`` lists with current metrics and URLs. It deliberately does NOT
touch ``best``.

That separation is the whole point. Promoting a model is a claim that it is fit
for its job, and the evidence for that does not live in the leaderboard:
``dogbin_001`` has the highest top-1 in its project and is still not deployable,
because 70.8% of its validation images shared a Mapillary sequence with training
and it rejects only 16% of real sweep false positives. An argmax over
``accuracy_top1`` would have promoted it. So ``best`` is edited by hand, with a
``why``, and a project with nothing worthy keeps ``best: null`` rather than
being handed its least-bad run.

The API key is read from a .env (never printed, never written to the JSON).
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(REPO, 'data', 'best_models.json')
ENV_FILES = (os.path.join(REPO, '.env'),
             '<home>/dogs_detection/.env')
# metrics worth carrying per model type; first match wins for the headline
METRIC_KEYS = ('metrics/accuracy_top1', 'metrics/mAP50(B)',
               'metrics/mAP50-95(B)', 'metrics/precision(B)',
               'metrics/recall(B)')


def load_key():
    """COMET_API_KEY from the environment or a .env. Never logged."""
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


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fetch(workspace, project):
    """[{run, key, url, date, metrics}] for every experiment in a project."""
    from comet_ml import API
    import datetime
    api = API()
    out = []
    for e in api.get_experiments(workspace, project_name=project):
        try:
            params = {p['name']: p['valueCurrent']
                      for p in (e.get_parameters_summary() or [])}
        except Exception:
            params = {}
        try:
            summ = {m['name']: m for m in e.get_metrics_summary()}
        except Exception:
            summ = {}
        met = {}
        for k in METRIC_KEYS:
            if k in summ:
                v = _num(summ[k].get('valueMax'))
                if v is None:
                    v = _num(summ[k].get('valueCurrent'))
                if v is not None:
                    met[k.split('/')[-1].replace('(B)', '')] = round(v, 4)
        t = (e.get_metadata() or {}).get('startTimeMillis')
        out.append({
            'run': params.get('name'),
            'key': e.id,
            'url': f'https://www.comet.com/{workspace}/{project}/{e.id}',
            'date': (datetime.datetime.fromtimestamp(t / 1000)
                     .strftime('%Y-%m-%d') if t else None),
            'metrics': met,
        })
    return sorted(out, key=lambda r: r['date'] or '')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--update', action='store_true',
                    help='refresh candidate metrics from Comet (never '
                         'changes which model is marked best)')
    ap.add_argument('--state', default=STATE)
    args = ap.parse_args()

    with open(args.state) as f:
        state = json.load(f)

    if not args.update:
        for proj, d in state['projects'].items():
            b = d.get('best')
            print(f'\n{proj}  -- {d.get("role", "")}')
            if b:
                m = ' '.join(f'{k}={v}' for k, v in (b.get('metrics') or {}).items())
                print(f'  BEST: {b["run"]}  {m}')
                print(f'        {b["url"]}')
            else:
                print('  BEST: (none yet)')
                print(f'        {d.get("why_blank", "")[:200]}')
            for c in d.get('candidates', []):
                m = ' '.join(f'{k}={v}' for k, v in (c.get('metrics') or {}).items())
                print(f'    - {c["run"]:<20} {m}')
        return 0

    if not load_key():
        print('COMET_API_KEY not found in the environment or ' +
              ', '.join(ENV_FILES), file=sys.stderr)
        return 2
    ws = state['workspace']
    for proj, d in state['projects'].items():
        try:
            rows = fetch(ws, proj)
        except Exception as e:
            print(f'{proj}: could not refresh ({e})', file=sys.stderr)
            continue
        keep = {c['run']: c for c in d.get('candidates', [])}
        fresh = []
        for r in rows:
            old = keep.get(r['run'], {})
            # notes and dataset provenance are hand-written -- carry them over
            for k in ('note', 'dataset'):
                if old.get(k):
                    r[k] = old[k]
            fresh.append(r)
        # keep only what was already tracked, plus anything new since
        d['candidates'] = [r for r in fresh
                           if r['run'] in keep or r['run'] not in keep]
        best = d.get('best')
        if best:
            cur = next((r for r in rows if r['key'] == best.get('key')), None)
            if cur and cur['metrics']:
                # refresh the numbers on the pinned model, not the pin itself
                best['metrics'] = {**best.get('metrics', {}), **cur['metrics']}
        print(f'{proj}: {len(rows)} experiments refreshed'
              f'{" (best unchanged: " + best["run"] + ")" if best else " (best still unset)"}')
    import datetime
    state['updated_at'] = datetime.date.today().isoformat()
    tmp = args.state + '.part'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, args.state)
    print(f'\nwrote {args.state}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
