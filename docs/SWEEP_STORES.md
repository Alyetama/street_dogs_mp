# Sweep Store Map

Where the sweep writes its results, for each of the three models in the chain, and what
each model decides. Paths are relative to the repository root unless marked otherwise.

Counts measured against the live stores on 2026-08-21. They are the number of rows on
disk, not a claim from a status file — where a store reports its own progress, the shards
are the record.

---

## The chain

Three models in series, each reading the one before it. The pipeline is a funnel, and the
narrowing is the point: a loose detector is cheap to run over 32 million frames, and the
expensive judgements only ever see what survived the previous stage.

| # | model | decides | in | out |
|---|-------|---------|----|-----|
| 1 | Detector `train-30` | where the dogs are — one class, tuned for recall | 32,542,334 frames | 4,785,890 boxes |
| 2 | Dog-bin gate `dogbin_008` | which of those boxes hold a dog | 4,688,510 boxes judged | 743,120 dog · 3,945,390 not_dog |
| 3 | Leash classifier `leash_v2_001` | which of those dogs are on a leash | 743,120 dogs | 469,150 leashed · 273,970 unleashed |

About one box in five off the detector is a dog: the rest are the animals and
objects it fires on anyway, and they are **false positives** — catching them is stage 2's
whole job. 15.9% of what it finds survives the gate, and stage 3 has judged every one of those.

Keep two subjects apart when writing anything here. **The harvest is ground animals** — the
Mapillary query, the `ground_animals` parquets, the "Ground animals" figure on the dashboard.
**The detector is dogs.** Every detector metric — recall, precision, mAP — is a statement
about dogs, because the model is `single_cls` on a set whose only class is `target` = dog.

**Every row in every store below is stamped `unverified`.** A verdict written by a model is
its opinion, for filtering and measurement, and nothing that assembles a training set reads
one. Ground truth is a human verdict in a reviewer ledger, written only by a person
clicking. `tools/detect/tests/adv_triage_isolation.py` enforces that separation by reading
the source rather than trusting the convention.

---

## Stage 1 — Detector

**`dogdetection` · run `train-30` · yolo26x · deployed**

When anyone here says "the sweep", this is it: the only model that reads the image drives.

A single-class YOLO: `single_cls: true`, and the dataset's only class name is `target`,
which means dog. So this model looks for dogs, and every metric it reports is about dogs.

It fires on other things anyway — goats, cows, the occasional parked motorbike — and each of
those is a **false positive**, which is what stage 2 exists to catch. That is a different
statement from "it detects animals", and the difference matters when you write UI copy or a
commit message: the harvest collected ground animals, the model was trained on the dogs in
that harvest.

Promoted on recall, deliberately: a missed dog is unrecoverable, and a false positive costs
one click in the review page.

### Results — authoritative

```
data/detect/shards/gen=NNNN/region=<region>/cell=<cell>/drive=<drive>/sNNNNN.pXXXXXX_YYYYYY.img.parquet
                                                                     sNNNNN.pXXXXXX_YYYYYY.det.parquet
```

Hive-partitioned, **18,010 files, 434 MB**. Everything else about the sweep is derived from
this.

| file kind | rows | columns |
|-----------|------|---------|
| `*.det.parquet` | 4,785,890 | `image_id, det_idx, gen, region, cell, drive, run_id, model_sha8, conf, x1, y1, x2, y2, leash_class, leash_conf, shard_idx` |
| `*.img.parquet` | 32,542,334 | `image_id, gen, region, cell, drive, run_id, model_sha8, status, n_det, max_conf, orig_w, orig_h, reduce, guards, ts_off, shard_idx` |

`img` holds one row per frame read, including frames with no detection.

`leash_class` / `leash_conf` on the detection rows are legacy columns from an earlier
in-sweep experiment — **the leash verdicts you want are stage 3's**, in `data/leash/`.

### Queryable mirror — stale by design

`data/detect/sweep.duckdb` — 1.9 GB. Tables: `images`, `detections`, `image_events`,
`detection_events`, `runs`, `_files`, `_meta`.

Built 2026-08-04 02:02 from 6,944 shards, so it currently holds **11,798,660 images /
1,790,772 detections** against a store that has since grown to 18,010 shards. The parquet
store is authoritative; this is a materialised copy that is stale the moment the sweep
writes again. `_meta` records what it was built from and when — read it before trusting a
count out of this file. Refresh with `tools/detect/build_sqldb.py`, which is incremental
and safe to run while the sweep is running.

### Everything else about stage 1

| | |
|---|---|
| runner | `tools/detect/sweep.py` |
| log | `data/sweep_resume.log` |
| live state | `data/detect/status.json` |
| remaining work | `data/detect/worklist/gen=NNNN/<cell>` |
| output root | pointer file `data/detect_root.txt` (gitignored — no drive paths in tracked files) |
| weights | `data/engines/yolo26x_train30.engine` (TensorRT) and `…_train30.pt`; hashes in `data/engines/SHA256SUMS` |
| metrics | mAP50 0.8708 · mAP50-95 0.4973 · precision 0.8977 · recall 0.8764 |

The sweep stamps `model_sha8` on every row it writes, so a shard always says which engine
produced it.

---

## Stage 2 — Dog-bin gate

**`dog-bin` · run `dogbin_008` · trained on `dogbin_v5`**

The binary classifier trained on this project's own reviewers' verdicts. It puts a
dog / not-a-dog call and a probability on each of the 4.7 million boxes the detector
committed. It reads those boxes, not images — it crops them out of the frames.

### Results

```
data/gate/
├─ gate-00000.parquet … gate-00164.parquet   165 shards · 188 MB · 4,688,510 rows
├─ work.parquet                              125 MB — the planned work list
├─ plan.json                                 the plan, including the drive roots
├─ shard_size                                rows per shard
└─ progress.json                             heartbeat, only while a run is live
```

Columns: `image_id, det_idx, label, p_dog, unverified, model`

- `label` — `dog` or `not_dog`
- `p_dog` — the probability, and the thing to threshold on
- `model` — `dogbin_008` on every row written so far

**Verdicts:** 743,120 `dog` · 3,945,390 `not_dog`.

**Coverage: 4,688,510 of 4,785,890 boxes — 98.0%.** The ~97K difference is boxes the sweep
committed after this plan was built; re-plan to pick them up.

### Everything else about stage 2

| | |
|---|---|
| runner | `tools/detect/gate_store.py --stage gate` — `plan` / `run` / `status` |
| log | `data/gate_run.log` |
| weights | `<training_root>/dog-bin/dogbin_008/weights/best.pt` |
| metrics | top-1 0.9393 · ROC AUC 0.9794 · acceptance ROC AUC 0.9756 |

`training_root` is a key in `tools/dashboard/dashboard.config.json` and points outside the
repository. The weights are never hardcoded: `gate_store.py` reads `data/best_models.json`
and follows whatever is promoted there.

Shards are the progress record. Each is written under a dot-name and `os.replace()`d into
place, so what is on disk is what is done, and a resumed run skips the indices it already
has.

---

## Stage 3 — Leash classifier

**`leash-models` · run `leash_v2_001` · trained on `leash_v2`**

The same code path as the gate with a different stage table — not a second file, and not a
second head on the gate. It judges only the crops the gate called `dog`, which is why it is
a separate pass: 4.7M boxes become 743K, and most of the saving is in not spending a leash
decision on a goat.

### Results

```
data/leash/
├─ leash-00000.parquet … leash-00028.parquet   29 shards · 34 MB · 743,120 rows
├─ work.parquet · plan.json · shard_size
└─ progress.json                               heartbeat, only while a run is live
```

Columns: `image_id, det_idx, label, p_leashed, unverified, model`

- `label` — `leashed` or `unleashed`
- `model` — `leash_v2_001`

**Verdicts:** 469,150 `leashed` · 273,970 `unleashed` — 743,120 rows, exactly the gate's dog
count, so this stage is complete over its input.

### Everything else about stage 3

| | |
|---|---|
| runner | `tools/detect/gate_store.py --stage leash` |
| log | `data/leash_run.log` |
| weights | `<training_root>/leash-models/leash_v2_001/weights/best.pt` |

`feeds_on: gate` in the stage table, so it refuses to plan work the gate has not judged yet.

Stages 2 and 3 are declared once, in `STAGES` at the top of `tools/detect/gate_store.py`,
which is the authority. The dashboard keeps its own copy for the panel (`GATE_STAGES`) and a
guard checks the two against each other, because they are read by different interpreters and
cannot import one another.

---

## Freshness at a glance

Measured 2026-08-21.

| store | rows | size | state |
|-------|------|------|-------|
| `data/detect/shards/` | 4,785,890 det / 32,542,334 img | 434 MB | current — **authoritative** |
| `data/detect/sweep.duckdb` | 1,790,772 det / 11,798,660 img | 1.9 GB | **stale** — built 2026-08-04 from 6,944 of 18,010 shards |
| `data/gate/` | 4,688,510 | 188 MB | 98.0% of boxes |
| `data/leash/` | 743,120 | 34 MB | 100% of the gate's dogs |

The two things most likely to mislead: the duckdb mirror's row counts, and the gate's 98%.

---

## Re-running any of it

Each stage plans before it runs, and each is resumable: stop it, start it, and it picks up
from what is on disk.

```bash
# stage 1 — the sweep
python tools/detect/sweep.py

# stages 2 and 3 — same file, different stage table
python tools/detect/gate_store.py plan   --stage gate
python tools/detect/gate_store.py run    --stage gate
python tools/detect/gate_store.py status --stage leash

# refresh the queryable mirror of stage 1 (safe while the sweep runs)
python tools/detect/build_sqldb.py
```

The sweep runs on its own interpreter and the dashboard on another — both are named in
`tools/dashboard/dashboard.config.json` (`sweep_python`, and a dashboard interpreter new
enough for its f-strings) rather than assumed. Promotion decisions, the weights each stage
loads, and the reasoning behind them live in `data/best_models.json`.
