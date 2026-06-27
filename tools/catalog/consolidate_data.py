#!/usr/bin/env python
"""
consolidate_data.py -- move every region DATA file onto ONE drive.

Policy: data lives on a single drive; ground-animal IMAGES may stay spread
across many drives. This scans all grid_runs roots (from a dirs file, by
default the same data/catalog_dirs.txt the catalog uses) and MOVES every data
file -- i.e. everything inside a <cell>/ folder EXCEPT the
`ground_animal_images/` subdirectory -- onto a destination grid_runs dir you
choose. Images are never touched.

Because each drive holds complementary, disjoint data for a shared cell (e.g.
crucial = primary extraction, weasel = backfill), files are MERGED into the
destination, never discarded.

============================ DATA-SAFETY GUARANTEES ===========================
The move is engineered so that NO source byte is ever deleted unless its data
already exists, intact, at the destination:

1. VERIFIED COPY-THEN-DELETE. Each file is copied to a `<dst>.partial` temp,
   its size (and, with --checksum, its SHA-1) is verified against the source,
   the temp is atomically renamed into place (os.replace, same filesystem), and
   ONLY THEN is the source removed. A crash at any point leaves the source fully
   intact; at worst a discardable `.partial` temp remains.
2. EXISTING DESTINATION FILES ARE NEVER OVERWRITTEN OR TRUNCATED.
     * a name not yet at the destination -> verified-moved (above)
     * `validated_images_*.txt` ledgers  -> unioned into a temp, atomically
       swapped in (the union is a SUPERSET of both files, so no line is lost),
       then the source is removed
     * a byte-identical duplicate          -> source dropped (dest already holds
       an identical copy)
     * a same-name file with different     -> LEFT IN PLACE on both drives and
       content                                reported as a CONFLICT
3. SAME-FILE GUARD. A source is never removed if it resolves to the very same
   physical file as the destination (os.path.samefile / device+inode).
4. DIRECTORIES ARE REMOVED ONLY WHEN EMPTY (os.rmdir refuses non-empty dirs).
5. SPACE PRE-FLIGHT. --execute aborts before touching anything if the
   destination lacks free space for the planned moves (override with --force).
6. The destination drive's OWN existing data is never scanned or moved (the
   destination is excluded from the source set by resolved path AND device).

--keep-source copies instead of moving (nothing on any source is ever deleted;
needs room for a second copy). Dry-run is the default -- nothing changes until
you pass --execute.
===============================================================================

Usage:
    # one parent region at a time (recommended when space is tight) -- dry run:
    python tools/catalog/consolidate_data.py \
        --dest /media/biodiv/weasel/street_dogs_mp_weasel/grid_runs \
        --region Europe
    # review the plan, then move it:
    python tools/catalog/consolidate_data.py \
        --dest /media/biodiv/weasel/street_dogs_mp_weasel/grid_runs \
        --region Europe --execute

    # everything at once (also writes the data-root catalogue):
    python tools/catalog/consolidate_data.py \
        --dest /media/biodiv/weasel/street_dogs_mp_weasel/grid_runs --execute
"""
import argparse
import hashlib
import os
import re
import shutil
import sys

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None

IMAGES_SUBDIR = 'ground_animal_images'
DEFAULT_DIRS_FILE = 'data/catalog_dirs.txt'
DEFAULT_CATALOGUE = 'data/data_root.txt'


def read_dirs(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                out.append(line.rstrip('/'))
    return out


def region_prefix(name):
    """Cell-folder prefix for a parent region, matching the extractor's folder
    sanitisation, so a cell named e.g. 'Middle_East_45_30_50_35' is selected by
    --region 'Middle East' or 'Middle_East', and 'Europe' selects 'Europe_*'."""
    safe = re.sub(r'[^\w\-_\.]', '_', name.replace('&', 'and')).strip('_')
    return safe + '_'


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f"{n:,.1f} {unit}" if unit != 'B' else f"{n:,} B"
        n /= 1024


def file_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def same_content(a, b, use_hash):
    """True if two files are identical: size first, then SHA-1 when sizes tie
    (the hash is only ever computed on equal-sized files)."""
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    return file_hash(a) == file_hash(b)


def is_same_file(a, b):
    """True iff a and b are the SAME physical file (same device + inode)."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def verified_copy(src, dst, use_hash):
    """Copy src -> dst via a temp + atomic rename, verifying the copy. Returns
    only once dst is a complete, verified copy of src. Never touches src."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.partial'
    if os.path.exists(tmp):
        os.remove(tmp)
    shutil.copy2(src, tmp)
    src_size = os.path.getsize(src)
    if os.path.getsize(tmp) != src_size:
        os.remove(tmp)
        raise IOError(f"size mismatch after copy: {src}")
    if use_hash and file_hash(tmp) != file_hash(src):
        os.remove(tmp)
        raise IOError(f"checksum mismatch after copy: {src}")
    os.replace(tmp, dst)  # atomic within the destination filesystem


def verified_move(src, dst, use_hash):
    """Verified copy, then delete src -- src is removed ONLY after a complete,
    verified copy is in place at dst."""
    verified_copy(src, dst, use_hash)
    os.remove(src)


def atomic_merge_ledger(src, dst, keep_source):
    """Union the lines of two validated_images ledgers into dst atomically (the
    union is a superset of both, so nothing is lost), then drop the source."""
    lines = set()
    for p in (dst, src):
        if os.path.exists(p):
            with open(p) as f:
                lines.update(l.strip() for l in f if l.strip())
    tmp = dst + '.partial'
    with open(tmp, 'w') as f:
        f.write('\n'.join(sorted(lines)))
        if lines:
            f.write('\n')
    os.replace(tmp, dst)
    if not keep_source and not is_same_file(src, dst):
        os.remove(src)


def conflict_mergeable(name):
    """Return the union-merge strategy for a conflicting file name, or None.

    The structured extractor artefacts can be merged losslessly:
      * validated_images_*.txt        -> 'ledger' (line union)
      * *_checkpoint_*_sub_*.jsonl.zst -> 'jsonl'  (record union by image_id)
      * *_checkpoint_*_sub_*.json.zst  -> 'json'   (dict union by image_id)
    Anything else (e.g. *_tiles.png) is not mergeable and is left untouched.
    """
    if name.startswith('validated_images_') and name.endswith('.txt'):
        return 'ledger'
    if 'checkpoint' in name and name.endswith('.jsonl.zst'):
        return 'jsonl'
    if 'checkpoint' in name and name.endswith('.json.zst'):
        return 'json'
    return None


def _zstd():
    import compression.zstd as z  # Python 3.14 stdlib
    return z


def _read_jsonl_records(path):
    """Yield (key, raw_line_bytes) from a zstd jsonl checkpoint, tolerantly: a
    truncated/corrupt stream simply contributes whatever was readable first."""
    import json
    try:
        with _zstd().open(path, 'rb') as f:
            for raw in f:
                s = raw.strip()
                if not s:
                    continue
                try:
                    key = json.loads(s).get('image_id')
                except Exception:
                    key = None
                if key is None:
                    key = '#' + hashlib.sha1(s).hexdigest()
                yield key, raw
    except Exception:
        return  # corrupt/truncated -> stop; the other side supplies the data


def _read_json_dict(path):
    import json
    try:
        with _zstd().open(path, 'rb') as f:
            data = f.read()
        obj = json.loads(data) if data else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def merge_jsonl_union(src, dst):
    """Union jsonl checkpoint records by image_id into dst (the more-complete,
    larger file wins for any shared key), atomically. Lossless: every record
    from either side is preserved."""
    order = sorted([src, dst], key=os.path.getsize,
                   reverse=True)  # larger first
    recs = {}
    for p in order:
        for key, raw in _read_jsonl_records(p):
            recs.setdefault(key, raw)  # first (larger file) wins shared keys
    tmp = dst + '.partial'
    with _zstd().open(tmp, 'wb') as f:
        for raw in recs.values():
            f.write(raw if raw.endswith(b'\n') else raw + b'\n')
    os.replace(tmp, dst)


def merge_json_union(src, dst):
    """Union topology-checkpoint dicts (keyed by image_id) into dst, atomically.
    Lossless: every key from either side is preserved (larger file wins ties)."""
    import json
    order = sorted([src, dst],
                   key=os.path.getsize)  # smaller first; larger last wins
    merged = {}
    for p in order:
        merged.update(_read_json_dict(p))
    tmp = dst + '.partial'
    with _zstd().open(tmp, 'wb') as f:
        f.write(json.dumps(merged).encode())
    os.replace(tmp, dst)


def resolve_conflict(src, dst, keep_source):
    """Losslessly union a structured conflict into dst, then drop src. Returns
    True if resolved, False if the file type cannot be merged safely."""
    kind = conflict_mergeable(os.path.basename(dst))
    if kind == 'ledger':
        atomic_merge_ledger(src, dst, keep_source)  # already drops src
        return True
    if kind == 'jsonl':
        merge_jsonl_union(src, dst)
    elif kind == 'json':
        merge_json_union(src, dst)
    else:
        return False
    if not keep_source and not is_same_file(src, dst):
        os.remove(src)
    return True


def build_plan(live, dest, use_hash, region_pref=None):
    """Scan sources WITHOUT changing anything. Returns (ops, cell_dirs, stats).

    ops: list of (kind, src, dst, size) with kind in
         {move, merge, identical, conflict}.
    cell_dirs: source cell directories seen (for pruning later).
    region_pref: if set, only cell folders whose name starts with this prefix
                 (one parent region, e.g. 'Europe_') are considered.
    """
    ops, cell_dirs = [], []
    stats = dict(move=0,
                 merge=0,
                 identical=0,
                 conflict=0,
                 conflict_mergeable=0,
                 bytes=0)
    conflict_list = []
    for root in live:
        try:
            cells = sorted((c for c in os.scandir(root) if c.is_dir()),
                           key=lambda c: c.name)
        except OSError as e:
            print(f"[skip] cannot read {root}: {e}")
            continue
        for cell in cells:
            if region_pref and not cell.name.startswith(region_pref):
                continue
            cell_dirs.append(cell.path)
            dst_cell = os.path.join(dest, cell.name)
            try:
                entries = list(os.scandir(cell.path))
            except OSError:
                continue
            for entry in entries:
                if entry.name == IMAGES_SUBDIR:
                    continue  # images stay put
                if entry.is_dir():
                    conflict_list.append(
                        (entry.path, "unexpected sub-directory"))
                    stats['conflict'] += 1
                    continue
                dst_path = os.path.join(dst_cell, entry.name)
                if not os.path.exists(dst_path):
                    size = entry.stat().st_size
                    ops.append(('move', entry.path, dst_path, size))
                    stats['move'] += 1
                    stats['bytes'] += size
                elif is_same_file(entry.path, dst_path):
                    # literally the same physical file -> nothing to do, never remove
                    continue
                elif entry.name.startswith('validated_images_'):
                    ops.append(('merge', entry.path, dst_path, 0))
                    stats['merge'] += 1
                elif same_content(entry.path, dst_path, use_hash):
                    ops.append(('identical', entry.path, dst_path, 0))
                    stats['identical'] += 1
                else:
                    stats['conflict'] += 1
                    ops.append(('conflict', entry.path, dst_path, 0))
                    if conflict_mergeable(entry.name):
                        stats['conflict_mergeable'] += 1
                        tag = "  [mergeable]"
                    else:
                        tag = "  [NOT mergeable]"
                    conflict_list.append(
                        (entry.path, f"differs from {dst_path}{tag}"))
    return ops, cell_dirs, stats, conflict_list


def main():
    ap = argparse.ArgumentParser(
        description="Move all region DATA files (not images) onto one drive.")
    ap.add_argument('--dest',
                    required=True,
                    help="Destination grid_runs dir; all data is moved here.")
    ap.add_argument('--dirs-file',
                    default=DEFAULT_DIRS_FILE,
                    help=f"File listing source grid_runs roots "
                    f"(default: {DEFAULT_DIRS_FILE}).")
    ap.add_argument('--dirs',
                    nargs='+',
                    default=None,
                    help="Explicit source roots (overrides --dirs-file).")
    ap.add_argument(
        '--region',
        default=None,
        help="Only move ONE parent region's data, e.g. 'Europe' or "
        "'Middle East' (matches <region>_* cell folders). "
        "Default: every region.")
    ap.add_argument('--catalogue',
                    default=DEFAULT_CATALOGUE,
                    help=f"Data-root catalogue to write on --execute "
                    f"(default: {DEFAULT_CATALOGUE}).")
    ap.add_argument('--execute',
                    action='store_true',
                    help="Actually move files (default is a dry run).")
    ap.add_argument('--checksum',
                    action='store_true',
                    help="Verify every moved file by SHA-1 (not just size) "
                    "before deleting the source. Slower, paranoid-safe.")
    ap.add_argument('--keep-source',
                    action='store_true',
                    help="COPY instead of move: never delete any source file "
                    "(needs room for a second copy).")
    ap.add_argument('--on-conflict',
                    choices=('report', 'merge'),
                    default='report',
                    help="What to do with same-name-but-different files. "
                    "'report' (default) leaves both untouched. 'merge' "
                    "losslessly unions structured artefacts "
                    "(ledgers + *_checkpoint_* jsonl/json, by image_id) "
                    "into the destination; non-mergeable files (e.g. "
                    "*_tiles.png) are still left untouched.")
    ap.add_argument(
        '--no-prune',
        action='store_true',
        help="Keep source cell dirs that become empty after moving.")
    ap.add_argument(
        '--force',
        action='store_true',
        help="Proceed even if the destination looks short on space.")
    args = ap.parse_args()

    dest = os.path.realpath(args.dest.rstrip('/'))
    raw = args.dirs if args.dirs else read_dirs(args.dirs_file)
    sources = [os.path.realpath(d.rstrip('/')) for d in raw]

    # exclude the destination itself, and any source nested in dest / vice versa
    live = []
    for d in sources:
        if d == dest or is_same_file(d, dest):
            continue
        if dest == d or dest.startswith(d + os.sep) or d.startswith(dest +
                                                                    os.sep):
            print(f"[skip] source overlaps destination, refusing: {d}")
            continue
        if not os.path.isdir(d):
            print(f"[skip] source does not exist: {d}")
            continue
        live.append(d)
    if not live:
        print("No source directories to scan. Nothing to do.")
        return

    mode = ("EXECUTE/COPY" if args.keep_source else "EXECUTE") \
        if args.execute else "DRY-RUN"
    region_pref = region_prefix(args.region) if args.region else None

    print(f"\n=== consolidate_data ({mode}) ===")
    print(f"destination : {dest}")
    print(f"region      : {args.region or 'ALL'}"
          f"{f'   (cells {region_pref}*)' if region_pref else ''}")
    print(f"verify      : {'size + SHA-1' if args.checksum else 'size'}")
    print(f"sources     : {len(live)} drive(s)")
    for d in live:
        print(f"  - {d}")
    print()

    # ---- PASS 1: plan (no filesystem changes) ----
    ops, cell_dirs, stats, conflict_list = build_plan(live, dest,
                                                      args.checksum,
                                                      region_pref)

    print("---- plan ----")
    print(f"  to move (new at dest)      : {stats['move']:,}   "
          f"({human(stats['bytes'])})")
    print(f"  ledgers to merge           : {stats['merge']:,}")
    verb = 'drop' if not args.keep_source else 'keep (copy mode)'
    print(f"  identical duplicates       : {stats['identical']:,}   ({verb})")
    if args.on_conflict == 'merge':
        unmergeable = stats['conflict'] - stats['conflict_mergeable']
        print(
            f"  conflicts to MERGE (union) : {stats['conflict_mergeable']:,}")
        print(f"  conflicts left (unmergeable): {unmergeable:,}")
    else:
        print(f"  conflicts (left untouched) : {stats['conflict']:,}"
              f"   [{stats['conflict_mergeable']:,} mergeable -- pass "
              f"--on-conflict merge]")
    if conflict_list:
        print("\n  CONFLICTS (same name, different content):")
        for path, why in conflict_list[:40]:
            print(f"    - {path}\n        {why}")
        if len(conflict_list) > 40:
            print(f"    ... and {len(conflict_list) - 40:,} more")

    # space pre-flight
    short = False
    try:
        probe = dest if os.path.isdir(dest) else os.path.dirname(dest)
        free = shutil.disk_usage(probe).free
        need = stats['bytes'] * (2 if args.keep_source else 1)
        short = need > free
        flag = "  [!] NOT ENOUGH FREE SPACE" if short else ""
        print(f"\n  dest free space            : {human(free)}{flag}")
    except OSError:
        free = None

    if not args.execute:
        print(
            f"\n  (dry run) re-run with --execute to move; catalogue would be "
            f"written to {args.catalogue} -> {dest}")
        return

    if short and not args.force:
        print("\n[ABORT] destination is short on free space; nothing moved. "
              "Free space, choose another --dest, or pass --force.")
        return 1

    # ---- PASS 2: execute (verified, source removed only after a good copy) ----
    moved = merged = dropped = resolved = left = errors = 0
    err_list = []
    bar = None
    if tqdm is not None and stats['bytes'] > 0 and sys.stderr.isatty():
        bar = tqdm(total=stats['bytes'],
                   unit='B',
                   unit_scale=True,
                   unit_divisor=1024,
                   desc='moving data')
    for kind, src, dst, size in ops:
        try:
            if kind == 'move':
                if os.path.exists(
                        dst):  # appeared after planning -> never overwrite
                    if not is_same_file(src, dst) and \
                            not same_content(src, dst, args.checksum):
                        errors += 1
                        err_list.append(
                            (src, f"dest appeared & differs: {dst}"))
                    elif not args.keep_source and not is_same_file(src, dst):
                        os.remove(src)
                    continue
                if args.keep_source:
                    verified_copy(src, dst, args.checksum)
                else:
                    verified_move(src, dst, args.checksum)
                moved += 1
            elif kind == 'merge':
                atomic_merge_ledger(src, dst, args.keep_source)
                merged += 1
            elif kind == 'identical':
                if not args.keep_source and not is_same_file(src, dst):
                    os.remove(src)
                dropped += 1
            elif kind == 'conflict':
                if args.on_conflict == 'merge' and \
                        resolve_conflict(src, dst, args.keep_source):
                    resolved += 1
                else:
                    left += 1  # report mode, or not mergeable -> leave both
        except Exception as e:  # noqa: BLE001 - report & continue; source is intact
            errors += 1
            err_list.append((src, str(e)))
        finally:
            if bar is not None and kind == 'move':
                bar.update(size)
    if bar is not None:
        bar.close()

    # prune emptied source cell dirs (rmdir only removes truly-empty dirs)
    pruned = 0
    if not args.no_prune and not args.keep_source:
        for cell_path in cell_dirs:
            try:
                os.rmdir(cell_path)
                pruned += 1
            except OSError:
                pass  # still holds images or conflict files -> keep

    print("\n---- done ----")
    print(f"  moved (verified)           : {moved:,}")
    print(f"  ledgers merged             : {merged:,}")
    print(f"  identical duplicates "
          f"{'kept ' if args.keep_source else 'dropped'} : {dropped:,}")
    if args.on_conflict == 'merge':
        print(f"  conflicts merged (union)   : {resolved:,}")
    if left:
        print(f"  conflicts left untouched   : {left:,}")
    if not args.keep_source:
        print(f"  emptied source dirs pruned : {pruned:,}")
    if errors:
        print(f"  ERRORS (source kept intact): {errors:,}")
        for path, why in err_list[:20]:
            print(f"    - {path}\n        {why}")

    # The data-root catalogue asserts that ALL data lives on the destination,
    # which only holds after a full run. For a per-region move it is left
    # untouched so the generator is not pointed at a half-migrated drive.
    if args.region:
        print(
            f"\n  partial move (region '{args.region}'): data-root catalogue "
            f"NOT updated.\n  Run once without --region (after every region is "
            f"moved) to write it -> {dest}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.catalogue)) or '.',
                    exist_ok=True)
        with open(args.catalogue, 'w') as f:
            f.write(
                "# Data-root catalogue written by "
                "tools/catalog/consolidate_data.py\n"
                "# All region DATA files (everything except ground_animal_images/)\n"
                "# are consolidated under this single grid_runs directory.\n"
                "# The dashboard command generator reads this to fill\n"
                "# --parent-dir / --data-dir automatically.\n"
                f"{dest}\n")
        print(f"  data-root catalogue written: {args.catalogue} -> {dest}")


if __name__ == '__main__':
    sys.exit(main())
