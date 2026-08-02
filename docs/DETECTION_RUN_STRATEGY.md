# yolo26x @1280 over 32.5M images — definitive run strategy

**Status:** resolved design. **Read Addendum A first** -- the pipeline is
two-stage (detect -> crop -> classify leashed/unleashed/not_a_dog); where it
and §1-§10 disagree, Addendum A wins.

**Status detail:** resolved design. Every number below is labelled `[M]` measured on this machine, `[D]` derived arithmetic from measured inputs, or `[E]` estimated. Where an earlier design and its critique disagreed, only the resolved answer appears.

---

## 1. Bottom line

**Do this:** export a TensorRT FP16 static-batch-8 engine at **square 1280×1280**, bypass `ultralytics.predict()` entirely, feed it from 11 reader threads (lynx 1 / capybara 1 / jackal 1 / bobcat 8) → 4 byte-bounded raw-JPEG queues → 4 decode threads that letterbox directly into a pinned batch ring → one GPU consumer. Store **one row per detection with full float32 conf and original-pixel xyxy at conf ≥ 0.05, NMS iou = 0.90**, plus one row per image, as per-shard immutable Parquet on crucial with JSON progress sidecars. Run it as a systemd user unit with per-lane drive quarantine and closed-loop pacing so all four drives finish at the same moment.

**Expected wall clock:**

| Bound | Arithmetic | Result |
|---|---|---|
| Pure TRT forward `[M 126.97 img/s]` | 32,542,334 / 126.97 = 256,299 s | 2.97 d |
| Full pipeline, GPU-bound `[E 110–117 img/s]` | 32,542,334 / 117 = 278,140 s<br>32,542,334 / 110 = 295,839 s | 3.22–3.42 d |
| Disk floor under simultaneous-finish pacing `[D]` | max over drives of imgs/capacity = 10,043,204 / 32.5 = 309,022 s | 3.58 d |
| **Plan** | + ~10% for restarts, quarantine backoff, Phase-0 residue | **3.5–4.0 d wall clock** |

Against today's baseline: 32,542,334 / 7.9 `[M naive predict()]` = 4,119,283 s = **47.7 d**. Against PyTorch fp16 bs8: 32,542,334 / 54.82 `[M]` = 593,622 s = **6.87 d**. TensorRT plus a decoupled reader/decoder is a **13.5× end-to-end speedup**, and the GPU stops being the bottleneck.

**Which bound applies is decided by one number:** jackal's sustained read rate under the real pipeline load. At 32.5 img/s `[M, 4-way concurrent]` jackal alone floors the run at 3.58 d and the GPU idles ~7%. At ≥ 36.1 img/s the GPU binds at 3.22 d. Phase-0 experiment **P8** settles it in 90 minutes and also tests the fix (jackal shares PCIe device `0000:11:00` with capybara — they are functions `.3` and `.4` behind root port `0000:00:08.1` `[M]`, so the drive to pace down is **capybara, not lynx/bobcat**).

**Single biggest risk: the run finishes, reports 100% complete, and a large fraction of the output is silently wrong.** Nothing downstream detects it and the only remedy is a 3.5-day re-run. Four concrete instances were found and all four are fixed in the design below:

1. **Decoder ring-slot publish race.** The prototype published a batch on a row-*fill* count, not a row-*completion* count, so the GPU consumed stale pixels from the previous occupant of the slot while `meta` asserted a different `image_id`. Reproduced with the real 4.9 MB pinned memcpy at 4 decoders: **206 of 8,000 images (2.58%) fed as stale pixels** `[M]` → extrapolated ~840,000 of 32.5M images with another image's detections under their id. Fix: §4.3.
2. **`non_max_suppression`'s wall-clock `time_limit`** (`2.0 + max_time_img * bs` = 2.4 s at bs 8) `break`s out of the loop and returns `torch.zeros((0,6))` for every remaining image in the batch — byte-identical to a genuine negative, with only a stderr warning nobody reads on day 3. Fix: pin `max_time_img=1e9` and hard-fail on the log string.
3. **Coordinate space.** ~97% of images take an `IMREAD_REDUCED_COLOR_r` branch; a missing `×r` on the inverse transform halves every box. Detection *counts*, positive rates, mAP and the whole threshold table are unchanged — nothing in a naive Phase-0 catches it — but every crop, every area filter and the pano prior are wrong forever. Fix: §4.5 + gate P6.
4. **Drive unmount → ENOENT storm.** If an image drive unmounts mid-run the mountpoint becomes an empty directory on the root fs; every read returns ENOENT with no I/O wait, so a lane commits its entire remaining backlog as MISSING at enormous speed and marks it done. Jackal's 10.04M images would be "processed" in minutes, and the obvious completeness audit (`count(*) == 32,542,334`) still passes because MISSING rows count. Fix: §6.4.

---

## 2. Reality check — measured constraints and scale arithmetic

### 2.1 Corpus

```
raw (cell,drive) rows in catalog.images   1,192 rows / 32,582,319 images   [M]
distinct cells                              665                             [M]
unique image_ids                         32,542,334                         [M]
cross-drive duplicate ids                    39,985 (0.12%), in 16 cells    [M]
```

`1,192` is the (cell, drive) **pair** count, not a cell count — planning against it double-schedules 527 units. The 39,985 duplicates decompose exactly: 13 South_Asia cells on bobcat+capybara+jackal = 37,972 and 3 Europe cells on capybara+lynx = 2,013; 37,972 + 2,013 = 39,985 `[M]`. The other 484 multi-drive cells have `raw_sum == n_unique` exactly — complementary splits, zero overlap. **Dedup 16 cells by filename union at worklist-build time; never build a global 32.5M-element set.** Contested ids go to bobcat (South_Asia) and lynx (Europe) — the two lanes with the most idle capacity.

Per-drive, raw: capybara 14,841,591 (45.6%) / jackal 10,043,204 (30.8%) / bobcat 4,043,431 (12.4%) / lynx 3,654,093 (11.2%) `[M]`. Post-dedup ≈ capybara 14.82M, jackal 10.02M, bobcat and lynx unchanged `[E]`; the authoritative figure is written by `enumerate` into `worklist/gen=NNNN/_meta.json` and asserted against 32,542,334.

**Cross-CELL duplication is a second, larger effect that the above does not cover** `[M, measured on the live sweep at ~2.2%]`. The 39,985 figure counts ids shared by two *drives* within one cell. Separately, the harvest wrote some images under **several different cells**, and the worklist never dedups across cells — so the detector processes those jpgs once per cell. At 735,080 image rows the store held 720,387 distinct ids: **14,693 redundant rows, 2.0%**, from 10,254 images appearing in 2–6 cells each. Three properties were checked, and they are why this is tolerated rather than fixed mid-run:

- the repeats are **bit-identical** — same `n_det`, same `orig_w/orig_h`, identical box geometry for every image that had detections `[M]`;
- every duplicated image stays inside a **single region**, so region attribution is unaffected `[M]`;
- the twins are **not adjacent-cell neighbours** — observed spanning 20° of longitude within one region (`Africa_-5_10_0_15` + `Africa_10_10_15_15` + `Africa_0_10_5_15` for one id) `[M]`. This is harvest-side cell attribution, almost certainly the `/images?bbox` endpoint returning out-of-bbox results; **do not try to fix it by nudging cell bounds.**

Cost is ~2% of GPU time (≈ 3.7 h of a 7.7-day run). Rebuilding the worklist with a global cross-cell dedup would change shard boundaries and so invalidate the tiling resume, costing far more than it saves. **Decision: leave the sweep alone; collapse at read time via `store.unique_src()`, which is lossless by the three properties above.** Use `_sql_src()` only for per-(image, cell) work such as drive/cell throughput.

Weighted mean JPEG 1.26 MB `[M, 2,000-file sample]` → **~41 TB to read, not 52 TB**.

### 2.2 Storage hardware — the shape of the whole schedule

Three of four drives are Bulk-Only-Transport USB with **`queue_depth=1`, `nr_requests=2`** `[M]`. The WD Elements 25A3 enclosure exposes no UAS alt-interface (`bNumInterfaces=1`, `bInterfaceProtocol=80`), so this is unfixable in software. Only bobcat (SABRENT) binds `uas` at qd 30.

Consequence, measured on paired disjoint stripes of 300 inode-sorted files:

| drive | 1 thread | 8 threads | use |
|---|---|---|---|
| lynx | 60.6 | 48.2 | **1** |
| capybara | 84.1 | 76.5 | **1** |
| jackal | 53.1 | 47.3 | **1** |
| bobcat | 15.8 (readdir order) / 37.9 (inode order) | 43.6 | **8** |

`[M all]`. **Adding threads to the BOT drives makes them slower.** 11 reader threads total, not 32–64.

ext4 htree makes readdir order ~50% uncorrelated with inode order on all four `[M]`. Inode-sorting each directory before opening is free (`e.inode()` rides in the getdents dirent) and worth +8% on the BOT drives and **+140% on bobcat**. Never call `e.stat()` in the enumerator: 3.8–4.1M entries/s with name+inode vs 423–1,439 stat/s `[M]` — capybara's largest single directory (1,527,654 files) would be **1.0 h of `stat()` alone**.

All-four-concurrent, inode-sorted, at each drive's optimal thread count: lynx 71.0 / capybara 71.1 / jackal 32.5 / bobcat 22.9 = 197.6 img/s instantaneous `[M]`.

**The floor is not 197.6, and it is not the 2.40 d "aggregate ceiling" figure.** Under proportional pacing (all four finish together, which is the only schedule with no single-drive tail) the floor is `max_d (images_d / capacity_d)`:

```
capybara 14,820,000 / 71.1 = 208,439 s
jackal   10,043,204 / 32.5 = 309,022 s   <-- binding
bobcat    4,043,431 / 22.9 = 176,569 s
lynx      3,654,093 / 71.0 =  51,466 s
```
`[D]` → **disk floor 309,022 s = 3.576 d, set by jackal alone.**

Topology: lynx (2-2) + bobcat (2-3.2) + **weasel** (the parquet SSD, 2-4) share xHCI `0000:0e:00.0`; capybara (6-1) is on `0000:11:00.4` and jackal (4-2.2) on `0000:11:00.3` — **two functions of the same PCIe device behind root port `0000:00:08.1`** `[M]`. This explains jackal's otherwise unexplained 38–45 solo → 32.5 concurrent drop, and it means throttling lynx/bobcat (a different controller entirely) cannot help jackal. The lever is capybara.

### 2.3 GPU

| config | img/s | 32.54M / rate | note |
|---|---|---|---|
| naive `predict(txt source)`, cold | 7.9 `[M]` | 4,119,283 s = 47.7 d | single-threaded decode on the main thread |
| PyTorch fp16 bs16, square 1280 | 51.9 `[M]` | 627,020 s = 7.26 d | |
| PyTorch fp16 bs8, square 1280 | 54.82 `[M]` | 593,622 s = 6.87 d | |
| **TRT fp16 static bs8, square 1280** | **126.97 `[M]`** | **256,299 s = 2.97 d** | 2.31× over PyTorch bs8 |
| full pipeline prototype | 116.98 `[M]` | 278,187 s = 3.22 d | no readers/writer/fadvise attached |
| production pipeline | 105–117 `[E]` | 278,140–309,927 s = 3.22–3.59 d | re-measured in P11 |

Batch size does **not** buy throughput: bs4 56.93 / bs8 54.82 / bs16 52.47 / bs32 51.18 img/s `[M, PyTorch fp16]` — monotonically decreasing. VRAM is a total non-constraint: engine + context 1,832 MiB, whole pipeline 4,067 MiB of 15,850 MiB `[M]`, 1,234 MiB of which belongs to three unrelated `prediction_api.py` workers.

**INT8 is rejected outright** — not on accuracy grounds but on arithmetic: the run is drive-bound the moment the GPU passes ~117 img/s, so any GPU rate above that is dead weight. Same reasoning kills `torch.compile`. **Aspect-ratio bucketing is rejected on accuracy**: per-bucket engines run 1.54× faster (960×1280 172.72, 736×1280 226.20, 640×1280 267.36 img/s `[M]`) but lose **10.8% of boxes and 13.6% of positive images** on 700 real images vs square 1280, concentrated in the 4:3 bucket `[M]`; even a ≥1120-only variant loses 2.4% of positives for 1.14×. The user already rejected imgsz 640 for losing detections; this fails the same test, and the payoff is capped at ~0.3 d because jackal floors the run anyway.

### 2.4 Decode

Adaptive `IMREAD_REDUCED_COLOR_r` at 4 threads: 268.4 img/s; fixed `/2` 229.7; full decode 105.6 `[M]`. Reduction histogram over 700 real images: r=1 1.1%, r=2 67.6%, r=4 25.3%, r=8 6.0% `[M]`. Mean decoded 2.29 MPix vs 15.59 MPix full = **6.8× less decode work** `[M]`.

Full decode at 4 threads (105.6 img/s) sits *below* the GPU's 110–117 appetite, so adaptive is not a nicety — it is what gives the decode stage 2.3× headroom without stealing cores from the 11 readers.

**Two corrections to earlier decode analysis, both load-bearing:**

- The `phase0_decode.json` experiment claiming REDUCED_2 costs 15 points of recall is an **artifact and is discarded**. All 459 images in the shipped val set are exactly 1280 px on the long side `[M]`, so REDUCED_2 there halves to 640 and then *upscales* back to 1280 — a condition that cannot occur on a corpus where 97%+ of images exceed 2560 px. **The shipped val set structurally cannot validate the decode path.**
- The width < 2560 share does **not** reproduce at 11.87%. Deduped over 1,469,710 unique image_ids from 120 random cells: **2.62% overall, 4.50% non-pano, 0.78% pano** `[M]`. The 11.87% figure was computed over raw parquet rows, which carry **9.57% duplicate image_ids that are 88.15% pano** `[M]` — backfill parquets re-fetch preferentially pano rows. Every composition weight must be recomputed deduped and inner-joined to the swept worklist (P10). The adaptive rule is unaffected and stays.

### 2.5 Yield

Two disagreeing samples: 0.129 boxes/img and 11.2% positive at conf 0.05 `[M, 847 images, rect 736×1280, capybara+bobcat]` vs 0.2514 boxes/img and 17.9% positive `[M, 700 images, square 1280]`. The square/rect difference is the likely cause (square is 1.74× the pixels on 16:9 and 2:1 sources) but neither sample is a uniform draw. **Size the store for the pessimistic case**: 32,542,334 × 0.2514 = 8,181,143 detection rows `[D]`. Both figures are storage-irrelevant (§5.4).

### 2.6 Machine

16 threads / 8 physical cores (Ryzen 7 9800X3D), 60 GB RAM with **12 GB of swap already in use and kswapd/kcompactd active** `[M]`, RTX 5080 16,303 MiB, driver 570.211.01, CUDA 12.8. crucial: 3.6 T total, 158 G free, 96% full `[M]`.

Streaming 41 TB through the page cache on this box for 3.5 days would churn reclaim continuously. Page cache has essentially zero value here — `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` after every read is three lines and the single highest-leverage memory action in the whole design.

> **Correction.** This paragraph originally justified the above with "dedup guarantees **no image is ever read twice**". That is false, and measurement says so: cross-cell twins are re-read (§2.1, 2.0% of rows `[M]`). The conclusion survives intact — the twins are 253–1204 s apart (median 713 s `[M]`), so 41 TB of unrelated traffic evicts them long before the second read, and a 2% re-read rate could not pay for the cache anyway. The action stays; only its stated premise was wrong.

---

## 3. Phase 0 — validation before the run

Total **≈ 6 h machine + a 6.5 h soak**, plus ~5 h of human labelling that runs *in parallel with the sweep*. Against a 3.5-day run whose output is unrecoverable, this is cheap. **Precondition: an idle GPU** — the three `prediction_api.py` workers must be stopped (open question Q2) or every timing number is noise.

| id | experiment | duration | pass / fail gate |
|---|---|---|---|
| **P0** | Env prep: `pip install onnx>=1.12,<1.18 onnxslim>=0.1.59 pyarrow==21.0.0 orjson==3.11.9 'rich>=13,<15'` into `dnd`; `pip freeze > tools/detect/dnd.lock.txt`. Clean bobcat's stale mount. Extend the keep-alive timer to capybara `<drive-serial>` and jackal `<drive-serial>`. Restart the dashboard with refresh disabled. `vm.swappiness 60→10`. | 40 min | `import onnx, onnxslim, pyarrow, orjson` succeeds in `dnd`; `findmnt -n -o SOURCE <mounts>/bobcat \| wc -l` == 1; 3 keep-alive timers `active`; dashboard PID's cmdline shows `--images-every 999999`. **Blocker: ultralytics AutoUpdate installs onnx into the BASE env via uv and prints "AutoUpdate success" anyway** `[M]` — install explicitly or the export dies with `ModuleNotFoundError: onnx`. |
| **P1** | Export the engine: `yolo export model=<pt> format=engine half=True imgsz=1280 batch=8 device=0 workspace=8 dynamic=False simplify=True` | 10 min `[M 9.3 min]` | 116 MiB `.engine` produced; sha256 recorded. **`dynamic=True` is broken on this stack** — TRT 10.12.0.36 aborts with `Assertion interval.max() >= 0 failed ... interval [-1374389534720,-4]` because ultralytics marks both batch and the anchor dim dynamic `[M]`. Static is the only option. |
| **P2** | Throughput verify: 30 timed iters of `AutoBackend(engine)(torch.rand(8,3,1280,1280, dtype=float32))` | 5 min | ≥ 120 img/s (measured 126.97). **Do not construct with `fp16=True` and do not pass a half tensor** — the engine's I/O bindings are FP32 even though compute is FP16; a half input causes an unrecoverable `CUDA error: an illegal memory access` `[M]`. `ab.fp16` must assert `False`. |
| **P3** | Preprocessing equivalence: production `decode → letterbox → CHW` vs `ultralytics.data.augment.LetterBox(auto=False, scaleup=False)` + `BasePredictor.preprocess`, on 100 real images spanning all 4 aspect buckets | 20 min | 0 tensor mismatches. Already 0/40 for the letterbox arithmetic alone `[M]`; this extends it through the *whole* production path. **Pin `INTER_LINEAR`, not `INTER_AREA`** — ultralytics uses `INTER_LINEAR` at `augment.py:1708` for training, val and predict, so the checkpoint has never seen an `INTER_AREA` downscale, and at 3.2× the two filters diverge exactly on small distant dogs. |
| **P4** | Numerics chain on `valset_fullres_457` through the production path: PyTorch FP32 → PyTorch FP16 → TRT FP16. Also sweep iou ∈ {0.70, 0.90, 0.95}. | 40 min | Gates in **box counts**, per stratum {pano n=98/108 boxes, 16:9 n=85/92, 4:3 n=256/302, other n=18/18}: no stratum loses > 2 boxes at conf ≥ 0.25; ≤ 4 lost across all strata; every lost box eyeballed. Pooled: ≥ 99% matched at IoU ≥ 0.7, median \|Δconf\| ≤ 0.01, identical counts at conf 0.10/0.25/0.50. **Fractional recall gates are unusable per stratum** — one pano box is 1/108 = 0.926 recall points, so a −0.005 gate demands strictly zero loss and would reject every variant by construction. iou decision: ship 0.90 if boxes/img ≤ 3× the 0.70 value, else 0.70. |
| **P5** | Adaptive-decode detection gate: 2,000 uniformly random real image_ids drawn from `catalog.duckdb` across all four drives. Arm A = full `IMREAD_COLOR` + INTER_LINEAR + PyTorch fp16. Arm B = adaptive reduce + TRT fp16. Report r=8 and r=4 subsets separately. | 2 h (mostly disk) | Same box-count gates as P4, **per reduction stratum**. If r=8 fails, clamp `decode_reduce_max` to 4 — it costs almost nothing (r=8 is 6.0% of images and decode has 2.3× headroom). **This is the gate that did not exist before**: every prior box-comparison script used full-resolution `cv2.imread`, so adaptive reduction's only evidence was a post-letterbox mean-abs pixel diff (0.30% / 0.39% / 0.76% for r=2/4/8 `[M]`), which is not a detection metric — and 0.76% acts precisely on the 0.05–0.15 confidence tail the whole store-at-0.05 design exists to preserve. |
| **P6** | Coordinate-space gate: 500 images across all 4 aspect buckets, run twice (forced full decode vs adaptive), recover original-pixel boxes both ways | 20 min | Recovered boxes agree within 2 px; every box satisfies `0 ≤ x1 < x2 ≤ orig_w` and `0 ≤ y1 < y2 ≤ orig_h`. A halved-coordinate bug passes the inequality but fails the 2-px agreement — both checks are needed. |
| **P7** | Guard harness **at production decode concurrency**: 2,000 interleaved clean / forged-EOI-truncated / bit-rotted buffers through 4 decode threads | 30 min | 0 false alarms on clean images; process stderr still writable at the end; the injected corruption band is visible in the drive-level warning rate. **`os.dup2(w, 2)` is process-global, not thread-local** — the naive per-image version measured **72.5% of real corruptions missed, 43.75% of clean images falsely flagged, and then `OSError [Errno 32] Broken pipe` on the next stderr write** `[M]`, which would kill every log line for the rest of the run and trip the auto-pause on all four drives inside the first window. Fix in §8.3. |
| **P8** | **Controller frontier.** 3×20 min arms: (a) jackal solo flat out; (b) jackal + capybara both flat out; (c) capybara token-bucketed to 53 img/s, jackal flat out. | 90 min | No gate — this is the measurement that picks the schedule. If arm (c) puts jackal ≥ 36.1 img/s, the run is GPU-bound at 3.22 d and capybara's pacing target is pinned at 53.3. If jackal stays at 32.5 regardless, accept the 3.58 d floor and stop optimizing. |
| **P9** | Determinism: 3 repeat TRT runs on `valset_fullres_457` | 15 min | Identical box sets at conf ≥ 0.05 across all three. A built TRT engine is *more* deterministic than PyTorch+cuDNN autotune (tactics frozen at build) but is invalid across a driver change — record `570.211.01` and re-run P2/P4/P9 if it moves. |
| **P10** | Composition recount (SQL only): `SELECT DISTINCT ON (image_id)` over the ground_animals parquets, inner-joined to the deduped worklist. Recompute is_pano / aspect / width<2560 shares per drive and per region. Pull `region` from `catalog.images`. | 30 min | `sum(drives.planned) == sum(regions.planned) == 32,542,334`. Replaces the raw-row weights (9.57% duplicates, 88% pano-biased `[M]`). **Never parse region from the cell name** — 8 of 17 regions contain underscores, so `cell.split('_')[0]` is wrong for 124 of 665 cells (`Central_America_and_Caribbean_-85_10_-80_15` → `Central`) `[M]`. |
| **P11** | **6-hour soak.** Full pipeline, all 4 lanes, ~2.4M images at 110 img/s. Deliberate `Ctrl+C` at t≈2 h, simulated bobcat unmount at t≈3 h, `systemctl --user restart` at t≈4 h. RSS, queue depths and per-lane img/s sampled every 60 s. | 6.5 h | `count(*) - count(DISTINCT image_id) == 0`; `positive + negative + errored == scanned`; `sum(img.n_det) == count(*) FROM det` per cell; resume introduces neither gaps nor re-processing; peak RSS < 30 GB; achieved ≥ 100 img/s; stderr intact; the bobcat unmount quarantines exactly one lane and the other three keep the GPU ≥ 70 img/s; the dashboard sidecar rehydrates correct cumulative totals after the restart. **A 50k-image dry run cannot test any of this** — it is 8 GPU-minutes and exercises no spin-up stall, no RSS drift, no restart path. |
| **P12** | Negative-density calibration: 5,000 real images drawn as whole inode-sorted directory slices (not uniformly at random — that defeats the locality every read rate depends on) | 10 min machine + ~5 h human | Runs **in parallel with the sweep**, not before it. Its output is the precision number; re-filtering is free, so the sweep need not wait. |

`valset_fullres_457` is already built and verified at `<home>/dogs_detection/archived_datasets/valset_fullres_457` — 457 images, 520 boxes, 21 negatives, 758 MB, yaml included `[M]`. 457 of 459 val image_ids were located on the drives and verified pixel-identical to the shipped copies (mean \|abs\| diff 1.888/255 = 0.74%, pure JPEG re-encode noise) `[M]`. YOLO labels are normalised, so they transfer with zero edits. **This is the only defensible baseline** — the shipped 1280-px val set cannot validate decode, and by the same argument cannot validate TensorRT against real inputs either.

---

## 4. Inference engine

### 4.1 Shape

```
11 reader threads ──> 4 byte-bounded raw-JPEG queues (4 GiB each)
   lynx 1 / capybara 1 / jackal 1 / bobcat 8
        │
        └─> 4 decode threads: SOF parse → IMREAD_REDUCED_COLOR_r → letterbox
                              → write DIRECTLY into a pinned batch-ring slot
        │
        └─> 6-slot pinned ring (6 × 8 × 1280×1280×3 B = 225 MiB page-locked)
        │
        └─> 1 GPU consumer: dst[slot].copy_(pin[slot], non_blocking) on a copy
                            stream → permute/float/÷255 → TRT → NMS → one D2H
```

The consumer does exactly **one** `queue.get()` per batch and zero per-image host work. Having decoders assemble whole batches into pinned slots (rather than the consumer draining 8 items and memcpy'ing) removes 8 queue ops and 38 MiB of memcpy per batch from the critical path: **116.98 vs 113.22 img/s and 2,951 vs 3,818 MiB RSS** `[M]`.

Thread counts are measured, not guessed. Decode-thread sweep on the full pipeline: 3T 117.56, 4T 116.98, 6T 115.23 img/s `[M]` — **more decode threads hurt** (GIL). Use 4. Standalone decode capacity at 4 threads is 268.4 img/s = 2.3× headroom `[M]`. Set `cv2.setNumThreads(1)` in every decode thread; the default here is 16 and would oversubscribe 4×.

Back-pressure needs no token bucket for memory: the free-list of ring slots *is* the mechanism. When the GPU falls behind, decoders block on `freeq.get()`, decode queues fill, readers block on the byte semaphore, drives idle. Pacing is a **separate** per-drive token bucket (§6.5) — conflating the two is what serialized bobcat in the earlier draft.

### 4.2 Bypassing `predict()`

`predict()`'s image loading is single-threaded on the main thread (`imread` inside `LoadImagesAndVideos.__next__`, `loaders.py:405-462`) — that is the entire 9–11 img/s story. It also silently forces `rect=True` (`engine/model.py:542`), producing **736×1280 on 16:9 real data**, which measurably loses detections. And `predict(list_of_paths)` runs the whole list as one batch via `autocast_list` (`data/build.py:251-253`) — reproduced as a **33.62 GiB CUDA OOM on 459 images** `[M]`.

Bypassing costs **exactly zero correctness**: the hand-rolled letterbox is byte-identical to `ultralytics.data.augment.LetterBox(auto=False, scaleup=False)` on 40 real images (0 mismatches, square and rect) `[M]`, and the full bypass reproduced `predict()`'s val output exactly — 549 boxes / 429 positive images at conf 0.05.

Corollary worth stating plainly: **switching from today's rect 736×1280 to square 1280 is an accuracy improvement**, and TensorRT's 2.31× more than pays for the 1.74× extra pixels.

### 4.3 The ring-slot fix (critical)

```python
# WRONG (the prototype):  publish when the 8th row is CLAIMED
with lock:
    s, j = claim_slot_and_row()
    if j == BS - 1: full = True
pin[s][j].copy_(frame)          # <-- happens AFTER the lock
if full: batq.put((s, BS, meta))  # rows 0..6 may still be unwritten

# RIGHT: publish when the 8th row is WRITTEN
with lock:
    s, j = claim_slot_and_row()
    meta[s][j] = m
pin[s][j].copy_(frame)
with lock:
    written[s] += 1
    if written[s] == expected[s]:
        batq.put((s, expected[s], meta[s]))
```

Cost: one extra uncontended lock acquire per image (~1 µs against a 65.9 ms batch) = 0.012% `[D]`. The graceful-shutdown partial-batch flush needs the identical treatment — it must wait for `written[s] == cur_n` before publishing.

**Second hazard, same failure mode:** allocating the H2D destination inside `with torch.cuda.stream(copy_stream)` and consuming it on the default stream orders the *kernels* via `wait_stream` but tells the caching allocator nothing — the 37.5 MiB block returns to `copy_stream`'s pool the instant `g` is rebound, and the next iteration's `.to('cuda')` can be handed the same block while the previous permute is still reading it. Fix: **preallocate one persistent device tensor per ring slot and `dst[slot].copy_(pin[slot], non_blocking=True)`**, removing the allocator from the picture entirely. H2D is 1.31 ms of a 65.9 ms batch = 2.0% `[M]`, so this is free.

### 4.4 Decode

Parse the JPEG `SOFn` marker directly from the raw bytes the reader already holds (microseconds, no extra I/O, immune to the parquet metadata drift measured at 1.53% height mismatch `[M]`). Pick the largest `r ∈ {8,4,2,1}` (capped by `decode_reduce_max`, default 8, clamped to 4 if P5's r=8 stratum fails) such that `long_side / r ≥ 1280`. Re-decode at full resolution if the result still comes back under 1280.

Fidelity after letterbox vs full decode: r=2 0.77/255 (0.30%), r=4 0.99/255 (0.39%), r=8 1.93/255 (0.76%) `[M]`. Note r=8 on a 10560×5280 pano is a DCT-domain box decimation followed by a 0.97 linear resize, whereas full decode is a single 0.121 `INTER_LINEAR` — the reduced path is arguably the *better* low-pass. P5 measures it rather than arguing it.

Set `IMREAD_IGNORE_ORIENTATION` explicitly: `cv2.imread` applies EXIF rotation, `cv2.imdecode` (which we use, on prefetched bytes) does not. Pinning it removes a silent per-image inconsistency. Measured baseline on 400 real images: **0 greyscale, 0 CMYK, 0 EXIF rotation, 0 truncation** `[M]` — Mapillary normalises on ingest, so these guards are cheap insurance and any *rate spike* is high-signal.

### 4.5 Coordinate transform — pinned explicitly

```
(W0, H0)  = SOF dims from the raw bytes           # true original
r_dec     = 1 | 2 | 4 | 8
(Wd, Hd)  = decoded array shape                   # == ceil(W0/r), ceil(H0/r)
s         = min(1280/Wd, 1280/Hd, 1.0)            # scaleup=False
nw, nh    = round(Wd*s), round(Hd*s)
dw, dh    = (1280-nw)/2, (1280-nh)/2
left      = int(round(dw-0.1));  top = int(round(dh-0.1))
right     = int(round(dw+0.1));  bot = int(round(dh+0.1))
pad value = 114, BORDER_CONSTANT, INTER_LINEAR resize

box_net (x,y) -> decoded: ((x-left)/s, (y-top)/s)
             -> ORIGINAL: * (W0/Wd, H0/Hd)        # NOT * r_dec — handles ceil()
             -> clip to [0,W0] x [0,H0]
```

Using `W0/Wd` rather than `r_dec` is the correct handling of odd dimensions. `box_space: "original_px"` goes in the manifest, and the writer asserts `x1 < x2 ≤ orig_w`, `y1 < y2 ≤ orig_h` on every row.

### 4.6 Pinned config block (gitignored `tools/detect/detect.config.json`)

```jsonc
{
  "engine":     "data/engines/yolo26x_train30.engine",
  "fallback_pt":"data/engines/yolo26x_train30.pt",
  "imgsz": 1280,
  "letterbox": "square",          // NOT rect. bucketing rejected: -13.6% positives
  "letterbox_pad": 114, "letterbox_scaleup": false, "letterbox_center": true,
  "interp": "INTER_LINEAR",       // matches ultralytics LetterBox exactly
  "batch": 8,                     // static engine; pad partial batches, drop extras
  "conf": 0.05,
  "iou": 0.90,                    // SETTLED BY P4. 0.7 is irreversible; 0.9 is a superset.
  "max_det": 300,
  "max_time_img": 1e9,            // MANDATORY: defeats the NMS wall-clock break
  "agnostic_nms": false, "classes": null,
  "class_name": "target",         // hard-coded; checkpoint says {0:'item'} (single_cls)
  "device": 0,

  "decode_threads": 4,
  "cv2_threads_per_worker": 1,
  "decode_reduce": "adaptive_sof",
  "decode_reduce_max": 8,         // clamp to 4 if P5's r=8 stratum fails
  "decode_reduce_min_long": 1280,
  "imread_flags": ["IMREAD_IGNORE_ORIENTATION"],

  "pinned_ring_slots": 6,
  "raw_queue_bytes_per_drive": 4294967296,   // 4 GiB x 4 = 16 GiB
  "fadvise_dontneed": true,

  "readers": { "lynx": 1, "capybara": 1, "jackal": 1, "bobcat": 8 },
  "read_burst_files_per_thread": { "lynx": 512, "capybara": 512, "jackal": 512, "bobcat": 256 },
  "drive_priority": ["jackal", "capybara", "bobcat", "lynx"],
  "pacing": "closed_loop_waterfill",  // recomputed every 60 s; NO fixed table
  "mount_probe_seconds": 60
}
```

Drive roots live in a gitignored `data/detect_root.txt` + `detect.config.json`, matching the `data/data_root.txt` / `data/catalog_dirs.txt` precedent. No env-specific path appears in any tracked file.

`read_burst_files_per_thread` is derived, not chosen: `budget / (threads × mean_file_bytes × 2)`. bobcat = 4 GiB / (8 × 0.90 MB × 2) = 284 → round to 256. A flat 512-file reservation across 8 bobcat threads would want 5.2 GB against a 4 GiB budget and **collapse bobcat to single-threaded** — a 2.8× regression on the drive holding 4.04M images with no margin.

### 4.7 Resource budget

```
VRAM   engine+context 1,832 MiB + torch ctx ~120 + NMS scratch ~200 + tensors ~320
       = ~2.5 GB of 15.85 GB  (MEASURED 4,067 MiB in the prototype, 1,234 of which
       belongs to three unrelated label-studio processes)        -> NON-CONSTRAINT

RAM    process (torch + CUDA + TRT + pyarrow)        ~2.5 GB  [E, M base 1.9 GB]
       raw byte queues 4 GiB x 4                     16.0 GB
       pinned ring 6 x 37.5 MiB                       0.22 GB
       resume image_id array (startup only)           0.26 GB + 0.19 transient
       writer buffers + parquet arenas               ~0.10 GB
       ------------------------------------------------------
       steady state                                  ~19 GB, peak ~21 GB at startup
       page cache: driven to ~0 by fadvise DONTNEED
```

`MemoryMax=40G`, `MemorySwapMax=0`, **no `MemoryHigh`**. In cgroup v2 page cache is charged to the same cgroup, and `memory.high` is a *throttle* — reclaim runs in the allocating task's context with a proportional sleep penalty. A job streaming 41 TB would sit pinned against `memory.high` for the entire run. Under `memory.max`, clean page cache simply reclaims. Raw queues are a config knob: raise to 8 GiB/drive (≈ 36 GB total) if P11 shows spin-up stalls draining them.

Stall coverage at 4 GiB/drive: 4 GiB / 1.26 MB = 3,404 images; at capybara's 53.3 img/s target that is 64 s, at jackal's 36.1 it is 94 s `[D]` — comfortably over the documented ~50 s spin-up.

**Any one drive can stall indefinitely and the remaining three still out-supply the GPU** `[D]`: lose lynx → 71.1+32.5+22.9 = 126.5 ≥ 117 ✓; lose capybara → 126.4 ✓; lose jackal → 165.0 ✓; lose bobcat → 174.6 ✓. The exposure is the **tail**, and proportional pacing is what removes it.

---

## 5. Predictions database

### 5.1 The one architectural decision

**Parquet written directly from the inference process, one immutable file pair per shard-part, queried with `duckdb.connect()` (no path). No hot binary log, no compactor, no `.duckdb` file anywhere in the pipeline.**

This resolves a real conflict. A numpy-only binary WAL plus an out-of-env compactor was proposed to avoid mutating `dnd` before a multi-day run. But it drags in: a versioned dtype contract across two envs, a compactor process, a single-file manifest that four concurrent compactors would clobber last-writer-wins, cold-tier staleness of 1.5–6.25 h per worker, two contradicting resume algorithms, and a durability bug where `os.fsync` without a preceding `f.flush()` leaves CPython's userspace buffer unwritten while `f.tell()` reports success — `recover()` then calls `os.truncate` to a *larger* offset, which **extends the file with zeros that parse as `image_id=0, status=ok, n_det=0`: fabricated "scanned negative" rows, with the lost unit marked DONE** `[M, reproduced under SIGKILL]`.

Installing `pyarrow` into `dnd` **before** the run and freezing the env to a tracked lockfile does not change letterbox geometry — the reproducibility concern is about `ultralytics`/`torch`, and `pip --dry-run` resolves cleanly with no torch/numpy conflict, wheels already cached `[M]`. One pip install eliminates all of the above.

The lock argument is settled by measurement and stands: with a foreign process holding a `.duckdb` write lock, `connect(read_only=True)` **fails**, `connect()` fails, and **`ATTACH '<db>' AS x (READ_ONLY)` also fails**, all with `IOException` `[M, duckdb 1.5.4]`. So a predictions `.duckdb` would block dashboard *reads* for 3.5 days. In the same run, `duckdb.connect()` over a parquet tree answered `count(*), sum(n_det)` across 32.5M rows in **0.04 s** `[M]` while that lock was held. **Never materialise a `predictions.duckdb`**, not even read-only after the run.

### 5.2 Layout

```
$DETECT_ROOT/                        # <repo>/data/detect, from gitignored data/detect_root.txt
  worklist/gen=0001/
    _meta.json                       # sha256, built_at, n_images, n_shards, dedup stats, per-pair dir_mtime
    _dirs.json                       # 1,192 x {cell, drive, region, root, dir_mtime}
    <cell>/<drive>.ids.npy           # frozen uint64 image_ids, INODE ORDER, mmap_mode='r'
  runs/gen=0001/
    _manifest.json                   # full provenance (5.5)
    .lock  .lane-<drive>.lock        # fcntl.flock, released automatically on SIGKILL
  shards/gen=0001/region=<R>/cell=<C>/drive=<D>/
    _state.json
    s00007.p000000_004000.img.parquet
    s00007.p000000_004000.det.parquet
    s00008.p000000_001536.img.parquet    # partial commit from a graceful stop
    s00008.p001536_004000.img.parquet    # its continuation
  errors/gen=0001/region=.../cell=.../drive=.../e00007.parquet
  _bootstrap/region=_bootstrap/cell=_bootstrap/drive=_bootstrap/{img,det}.parquet   # 0-row, full schema
```

**Every file is written once and never mutated.** Partial commits get their own `p<start>_<end>` filename; a shard is DONE iff its committed parts **tile `[0, shard_len)` exactly, no gaps, no overlaps**. This is what fixes the earlier scheme, where a resumed partial shard had nowhere to write but the same filename and silently overwrote its own committed prefix — and where the reconciliation rule "both files exist and read cleanly → adopt as done" would adopt a 1,536-row file as a complete 4,000-image shard, permanently losing 2,464 images while the ledger reported success.

The `_bootstrap` 0-row partition is not cosmetic: `read_parquet('.../images/**/*.parquet')` against an existing-but-empty directory raises `IOException: No files found that match the pattern` `[M]`, so without it every query, the dashboard rollup and every smoke test fail until the first shard commits.

### 5.3 Schemas

**detections** — one row per box:

| col | type | note |
|---|---|---|
| `image_id` | UINT64 | 15–17 digits, max 9.99e16 vs UBIGINT 1.8e19; 0 cast failures in 674,398 sampled `all_data` rows `[M]` |
| `det_idx` | UINT8 | 0..n−1, descending conf |
| `conf` | FLOAT32 | **full precision, never a 3-decimal string** |
| `x1,y1,x2,y2` | FLOAT32 | **original full-res pixels**, transform per §4.5. Float, not uint16 — the float→unsigned cast is formally undefined and the 66 MB it saves is noise |
| `run_id` | UINT16 | |
| `shard_idx` | UINT32 | ties the row to `_state.json` |

**images** — one row per image, 32,542,334 rows:

| col | type | note |
|---|---|---|
| `image_id` | UINT64 | UNIQUE across the whole store — asserted |
| `drive` | UINT8 | which copy was actually read |
| `status` | UINT8 | 0=ok 1=read_error 2=decode_error 3=missing(ENOENT) 4=infer_error 5=mount_lost. **Never NULL** |
| `n_det` | **UINT16** | 0 for a negative, **never SQL NULL**. UINT8 cannot hold `max_det=300` (300 → 44 silently) |
| `max_conf` | FLOAT32 | **SQL NULL when n_det=0, never NaN** |
| `orig_w, orig_h` | UINT16 | true SOF dims |
| `reduce` | UINT8 | 1\|2\|4\|8 — makes the decode path auditable after the fact |
| `guards` | UINT16 | fixed-width bitmask of fired guards (§8.3), not a comma string |
| `ts_off` | UINT32 | seconds since run epoch, DELTA-encoded |
| `run_id, shard_idx` | UINT16, UINT32 | |

Hive partition columns `gen / region / cell / drive` come from the path. **`region` is an explicit column carried from `catalog.images`, never parsed from the cell name.**

**errors** — narrow, `[image_id, status, drive, path, exc_type, msg, ts_off, run_id]`, keeps fat strings out of the 32.5M-row table.

Encoding: `zstd:3` (level 9 measured **0.7% larger** on this data — pure CPU waste `[M]`), `use_dictionary=False`, `DELTA_BINARY_PACKED` on `image_id` and `ts_off` (11.63 → 7.56 B/row, a 35% saving `[M]`), `row_group_size=131072`, rows sorted by `image_id` within each part.

### 5.4 The NaN landmine and the accounting invariant

Verified in DuckDB 1.5.4 `[M]`:

```sql
SELECT 'nan'::FLOAT >= 0.25;                              -- TRUE  (!!)
SELECT max(x) FROM (VALUES ('nan'::FLOAT),(0.9::FLOAT)) t(x);  -- nan
SELECT NULL::FLOAT >= 0.25;                               -- NULL  (correctly filtered)
```

With NaN as the negative sentinel, `count(*) WHERE max_conf >= 0.25` returned **29,995,876** where the correct NULL-based answer was **1,393,911** `[M]`. numpy has no NULL, so the writer must map NaN → NULL explicitly. Unit test: `count(*) WHERE max_conf IS NULL AND status = 0` == `count(*) WHERE n_det = 0 AND status = 0`.

The canonical progress query, with the double-count fixed:

```sql
SELECT count(*)                                AS scanned,
       sum(status=0 AND n_det>0)::BIGINT       AS positive,
       sum(status=0 AND n_det=0)::BIGINT       AS negative,   -- NOT sum(n_det=0)
       sum(status<>0)::BIGINT                  AS errored,
       sum(n_det)::BIGINT                      AS boxes
FROM img;
-- INVARIANT, asserted every commit: positive + negative + errored == scanned
```

Error rows have `n_det=0`, so the naive `sum(n_det=0) AS negative` counts every error as a negative — exactly the class of silent miscount this schema exists to eliminate.

**Invariants asserted per shard commit and hourly globally:**
```sql
SELECT count(*) - count(DISTINCT image_id) FROM img;                     -- must be 0
SELECT count(*) FROM det d ANTI JOIN img i USING (image_id);             -- must be 0
SELECT count(*) FROM img JOIN (SELECT image_id, count(*) n FROM det GROUP BY 1)
  USING (image_id) WHERE img.n_det <> n;                                 -- must be 0
```
Costs ~0.5 s over the whole store `[M]` and is the only detector for double-processing — the single biggest silent-waste risk in the project.

### 5.5 Sizing

```
detections  32,542,334 x 0.2514 boxes/img = 8,181,143 rows  [D, pessimistic sample]
            x ~26 B/row zstd  = 213 MB
images      32,542,334 rows x ~10 B/row   = 325 MB
errors      ~16,000 rows                  =  <1 MB
                                            -------
                                            ~540 MB = 0.34% of crucial's 158 GB free
design envelope 25M detection rows        = ~650 MB = 0.41%
```

At iou 0.90 the detection row count may rise up to 3× (gated in P4) → ~24.5M rows ≈ 640 MB. Still 0.4%.

**Guards that can actually fire** (the earlier "20 GB hot-log abort" guarded a rolling 24 h window against a monotonically growing quantity and could never trigger below ~300 boxes/img):
- Soft alarm: trailing-100k **global** boxes/img > 0.60 (2.4× the pessimistic 0.2514). Global, not per-worker — at lynx's paced 13.1 img/s a per-worker 100k window takes 2.1 h to fill.
- Hard abort: cumulative detection rows > 200M, **or** `shutil.disk_usage(DETECT_ROOT).free < 20 GB`, checked every shard commit. The absolute NMS ceiling (`max_det=300 × 32.54M × 26 B` = 254 GB) genuinely exceeds free space, so "it can't get that big" is false.
- Flag every image where `n_det == max_det`.

### 5.6 Durable commit

```python
f.write(...); f.flush(); os.fsync(f.fileno()); f.close()
os.replace(tmp, final)
dirfd = os.open(os.path.dirname(final), os.O_DIRECTORY)
os.fsync(dirfd); os.close(dirfd)
```

`rename(2)` guarantees atomicity of the **directory entry**, not durability of the file's data. Without the fsyncs a power loss or hard reset can land the metadata while the data blocks do not — leaving a truncated `.parquet` at the live path. That matters here more than usual: truncating a zstd parquet by **200 bytes** makes it unreadable *and* **poisons the entire `read_parquet('**/*.parquet')` glob** `[M]`. A sibling `*.parquet.tmp` of identical corrupt content is correctly ignored by the glob `[M]`, which is why `.tmp` + rename is the only safe commit primitive. `kill -9` tests leave the page cache intact and cannot surface this; on USB enclosures that have already dropped a bus once, power/reset is the realistic crash mode.

Add a `sweep.py verify` subcommand that reads every parquet footer (~30 s over 18k files `[E]`) so a poisoned glob is found by a scan, not by a failing dashboard query on day 3.

### 5.7 Post-run compaction

18k part files is fine for the run but not ideal for analytics. `sweep.py compact` rewrites each finished cell into one `img` + one `det` file (665 cells → 1,330 files) after the run completes, via the same `.tmp` + fsync + rename, leaving the parts in place until the compacted file verifies. Reference query latency at 1,437 files / 32.5M rows: total counts 0.06 s, per-region rollup 0.03 s, confidence histogram over 4.2M detections 0.06 s, re-filter with the det↔img join 0.23 s, full resume scan into a 260 MB numpy array 0.51 s `[M]`.

---

## 6. Orchestration, resumability, unattended operation

### 6.1 Unit of work

**Shard = a contiguous, inode-ordered slice of 4,000 image_ids from one (cell, drive) pair** in a frozen generation. Shard identity is *positional*: `ids[i*4000:(i+1)*4000]` of that pair's immutable `ids.npy`.

Pairs cannot be the unit — 2 exceed 1M images and the largest (`North_America_-95_30_-90_35` / capybara, 1,527,654 `[M]`) is 8.0 GPU-hours. Shard count pre-dedup: ceil-sum over all 1,192 pairs = **8,979** (capybara 3,892 / jackal 2,746 / bobcat 1,266 / lynx 1,075) `[M]`; post-dedup ≈ 8,975 `[E]`, authoritative value written by `enumerate`. Do not assert 8,979 — the frozen ids are post-dedup and several `ceil()` values change.

Shard wall clock at the paced targets: capybara 4000/53.3 = 75 s, jackal 111 s, bobcat 276 s, lynx 305 s `[D]`. Hard-crash loss with 4 lanes in flight = 16,000 images / 117 img/s = **137 s = 0.049% of a 3.22 d run** `[D]`.

### 6.2 Frozen generation

`enumerate` opens `catalog.duckdb` **read_only for < 1 s** to get the 1,192 (cell, drive, region) rows with images, then closes it. It never holds a connection. `scandir` those directories reading only `e.name` and `e.inode()`, inode-sort, dedup the 16 overlapping cells, freeze to `<cell>/<drive>.ids.npy` (uint64, 260 MB total), hash it, and record the hash in the run manifest. **The runner refuses to start if the on-disk worklist hash differs.**

Enumeration cost `[D from M rates]`: capybara 14.84M / 2.26M ent/s = 6.6 s, jackal 17.1 s, bobcat 14.2 s, lynx 2.3 s → max 17 s of getdents plus 1,192 cold directory opens, so 30–120 s `[E]` once per generation. The grid_runs roots hold ~2,900 *more* cell directories than actually contain images (jackal: 2,364 dirs vs 313 with images `[M]`) — a naive `os.walk` opens thousands of empty dirs across three qd=1 drives.

Catalog fallback if it is write-locked: 5 retries with backoff, then depth-1 scandir of the roots + `isdir` on `<cell>/ground_animal_images` (5,396 calls, 6–10 s `[E]`). After gen 1, `_dirs.json` removes the catalog dependency entirely.

Lanes open `ids.npy` with `mmap_mode='r'` and touch only the active shard's 32 KB slice — effectively zero worklist RSS. Paths are reconstructed as `<root>/<cell>/ground_animal_images/<image_id>.jpg`; a `path_suffix` column would be pure duplication and multiple GB of Python strings per worker.

### 6.3 Commit and resume

Per shard-part: (1) det `.tmp` → flush → fsync → replace; (2) img `.tmp` → flush → fsync → replace; (3) `_state.json` read-modify-write `.tmp` → fsync → replace; (4) fsync the directory fd.

Because one GPU batch of 8 mixes images from up to 4 lanes, **shard completion is not a lane-local event**. Explicit bookkeeping: a lane registers `pending[shard_idx] = shard_len` when it starts feeding; the GPU result handler decrements it and buffers rows keyed by `shard_idx`; the shard part commits when the counter hits 0 (or at shutdown, with the exact `[start, end)` it actually covered).

Startup reconciliation, per pair:
- parts present and each part's **parquet footer `num_rows` equals its declared `[start,end)` length** → adopt
- parts tile `[0, shard_len)` with no gaps/overlaps → shard DONE
- any short, unreadable or non-tiling part → delete it and redo that range (≤ 4,000 images)
- sidecar records a shard whose files are gone → drop the entry, redo

**One resume algorithm, stated once:** committed parts on disk are the truth; `_state.json` is a fast index over them and is always reconstructible by a `listdir`. There is no second resume path. `SELECT image_id FROM img` over the whole store (0.51 s `[M]`) is a **completeness audit**, explicitly not the resume path. Guard `if done.size == 0: todo = plan` — the `searchsorted` diff raises `IndexError` on an empty array `[M]`.

Decode/read failures get an explicit row (`status=1..4`, `n_det=0`) and count toward completion. **Completion is `n_ok + n_err == n_total`, never `n_ok == n_total`** — a single corrupt JPEG under the old rule makes a shard permanently incomplete, re-read and re-failed on every restart forever. Also: do **not** carry `Image.MAX_IMAGE_PIXELS = None` forward; the decompression-bomb guard is worth keeping on 32.5M externally-sourced files.

### 6.4 Drive faults

A `.detect_drive_id.json` sentinel at each grid_runs root (4 tiny dotfiles, written once by `install_service.sh`, invisible to the `.jpg` enumeration) is the **authority**; `os.stat(root).st_dev` is only a cheap fast path. Captured identities `[M]`: lynx 2097, bobcat 2112, capybara 2065, jackal 2257; `<home>/capybara/...` and `<mounts>/capybara/...` both give 2065 (a symlink, not a second mount).

Probe cadence: preflight, **every shard boundary**, and **every 60 s** — not only from the quarantine backoff path.

Error taxonomy:
- **ENOENT → re-probe the sentinel BEFORE recording anything.** If the sentinel is absent or its `drive` field mismatches, this is a drive fault, not a missing image. This is the fix for the unmount-storm: without it, an unmounted drive returns ENOENT with zero I/O wait and a lane commits its entire backlog as MISSING in minutes while the row-count audit still passes.
- Refuse to commit any shard whose MISSING fraction exceeds 5% without a passing probe.
- `errno ∈ {EIO 5, ENODEV 19, ESTALE 116, ENOTCONN 107, EREMOTEIO 121}` → drive fault.
- `imdecode` returns None → `status=2`, no retry, not a drive fault.
- Transient OSError → 3 attempts with backoff and file reopen before it becomes terminal.

3 consecutive drive faults in a shard → lane DEGRADED → shard abandoned uncommitted → QUARANTINE, backoff 30/60/120/240/480/900 s, each attempt re-probing the sentinel plus 8 random image reads with a 60 s timeout (generous enough to absorb a 50 s spin-up without a false negative). **Auto-resume after two consecutive passing probes; never auto-resume on a sentinel drive-name mismatch.** All 4 lanes quarantined > 30 min → exit 75.

Add `SELECT status, count(*) FROM img GROUP BY 1` with a hard threshold on `status=3` to the completeness audit — a status *distribution* check catches what a row-count check cannot.

**The live hazard:** `/proc/mounts` currently contains **both** `/dev/sdm <mounts>/bobcat ext4 rw,relatime,shutdown` and `/dev/sde <mounts>/bobcat ext4 rw,relatime`, and `/dev/sdm` no longer exists `[M]`. If `sde` re-enumerates, the mountpoint silently reverts to a shutdown-state filesystem that EIOs every read. bobcat is also the only one of the four mounted without `nosuid,nodev,errors=remount-ro` — it was hand-mounted after a bus drop.

### 6.5 Pacing

**Closed-loop water-fill, recomputed every 60 s**, not a fixed table:

```
target_d = clamp(remaining_d / T_remaining, 0, capacity_d)
T solved by bisection so sum(target) = min(GPU_rate, sum capacity)
capacity_d = EWMA of achieved img/s, measured only while lane d's queue was NOT full
```

A quarantined lane's weight goes to 0 and redistributes automatically, so the fault story and the pacing story share one mechanism. **Derive targets from the binding rate, not the GPU rate** — a fixed table computed as `images_d / (32.54M / GPU_rate)` manufactures exactly the single-drive tail that pacing exists to prevent: at 53.4 img/s capybara would finish 8.6 h *before* jackal, and during those 8.6 h the GPU runs at 28% fed by one drive whose 4 GiB buffer covers 94 s against a 50 s spin-up on a drive that today has **no keep-alive timer**.

Targets at the GPU-bound operating point (T = 278,140 s) `[D]`: capybara 53.3 (75% of measured concurrent capacity), jackal 36.1 (111% — the binding constraint, hence P8), bobcat 14.5 (63%), lynx 13.1 (18%).

Readers acquire byte budget for a whole burst and read back-to-back, so idle gaps land **between** bursts and stay under ~20 s. **Extend the proven lynx keep-alive systemd timer to capybara `<drive-serial>` and jackal `<drive-serial>` BEFORE enabling pacing** — same 1058:25a3 bridge, which `fix_lynx_spindown.sh` documents as ignoring `hdparm -S/-B`. Pacing is what creates the gaps.

### 6.6 systemd

`~/.config/systemd/user/dogdetect.service`, rendered from a tracked `.in` template with `@REPO@`/`@PYTHON@`/`@CFG@` placeholders (the rendered unit is gitignored). Every enabling fact verified: `Linger=yes` already set, systemd 255.4-1ubuntu8.16, `user@1000.service` has `Delegate=yes` with `DelegateControllers=cpu memory pids` and `memory` in `user-1000.slice/cgroup.subtree_control`, `/var/log/journal` exists, all five `media-biodiv-*.mount` units active `[M]`.

```ini
[Unit]
Description=Street-dogs mass inference sweep (yolo26x train-30, imgsz 1280, conf 0.05)
RequiresMountsFor=@OUT_MOUNT@
Wants=media-biodiv-lynx.mount media-biodiv-bobcat.mount media-biodiv-capybara.mount media-biodiv-jackal.mount
After=media-biodiv-lynx.mount media-biodiv-bobcat.mount media-biodiv-capybara.mount media-biodiv-jackal.mount
StartLimitIntervalSec=0
OnFailure=dogdetect-failed.service

[Service]
Type=simple
WorkingDirectory=@REPO@
Environment=PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENCV_NUM_THREADS=1 CUDA_MODULE_LOADING=LAZY
ExecStartPre=@REPO@/tools/detect/preflight.sh @CFG@
ExecStart=@PYTHON@ -u tools/detect/sweep.py run --config @CFG@
KillSignal=SIGINT
KillMode=mixed
TimeoutStopSec=600
Restart=on-failure
RestartSec=120
RestartPreventExitStatus=70
MemoryMax=40G
MemorySwapMax=0
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dogdetect
```

**`RequiresMountsFor` covers the OUTPUT drive only.** `RequiresMountsFor` adds `Requires=` + `After=`, and `Requires=` propagates *stop*: `systemctl show media-biodiv-jackal.mount -p StopPropagatedFrom` returns `dev-sdn1.device` `[M]`, so a USB re-enumeration of `/dev/sdn1` stops the mount, which stops the service — and because systemd treats that as a *clean* stop, `Restart=on-failure` does **not** restart it and nothing re-triggers when the mount returns. One line in a unit file would bypass the entire per-lane quarantine design. Image drives get `Wants=` + `After=` (boot ordering, no stop propagation); the sentinel probe is the sole authority on image-drive health. Automatic recovery on drive return is a separate `.path` unit calling `sweep.py retry --drive <name>`.

`IOSchedulingClass`/`IOSchedulingPriority` are **deleted** — all four drives use `mq-deadline` `[M]`, which ignores `ioprio` entirely (only BFQ honours it). They would be decoration that misleads anyone assuming the sweep is deprioritised.

**Exit codes split by whether a retry could ever succeed:**
- **0** — complete or clean stop.
- **70** (no restart) — weights sha256 / ultralytics / torch pin mismatch, worklist hash mismatch, sentinel *drive-name* mismatch, bobcat double-mount. Permanent by nature.
- **75** (restart after 120 s) — st_dev mismatch, unreadable drive, insufficient VRAM, insufficient free space, all 4 lanes quarantined > 30 min. Transient by nature.

Lumping all preflight failures under 70 with `RestartPreventExitStatus=70` routes the exit-75 recovery path straight into the no-restart trap: quarantine → 75 → restart → drives still absent → 70 → permanently failed. `StartLimitIntervalSec=0` prevents a long outage from exhausting a 10-restarts-per-hour budget in 20 minutes.

### 6.7 Single instance

`fcntl.flock(LOCK_EX|LOCK_NB)` on `runs/gen=NNNN/.lock` for the whole run and on `shards/gen=NNNN/.lane-<drive>.lock` per lane, so `--drives jackal` and a full run are mutually exclusive **per lane** rather than globally. `flock` releases on process death including SIGKILL, so no stale-lock cleanup is needed — unlike the dead-PID DuckDB lock this project already hit. `ExecStartPre` fails on a held lock so restarts cannot stack while the old process is still draining inside `TimeoutStopSec`.

The entire "one writer per sidecar by construction" argument depends on exactly one process existing, and `--drives jackal` is a documented invitation to violate it.

### 6.8 Graceful stop, logging, top-up

`SIGINT`/`SIGTERM` take the identical two-stage path (`KillSignal=SIGINT` makes `systemctl --user stop` literally the same code as Ctrl+C): first signal sets a module-level `SHUTDOWN` event every loop polls and reinstates `SIG_DFL`; drain in-flight (≈ 3,400 images/lane at 4 GiB budget → bounded by `TimeoutStopSec=600`), commit every in-flight shard as an exact partial part, flush sidecars, exit 0. **Zero work lost.** Message follows the repo's wording: *"Ctrl+C — finishing the current shard and checkpointing; press Ctrl+C again to force-quit."*

Logging: disable rich's live `Progress` when stdout is not a TTY (ported verbatim it would emit ANSI redraws into journald and be rate-limited into garbage). Structured JSONL to a `RotatingFileHandler` (64 MB × 5); journald sees ~1 line/min. Expected total volume `[D]`: 8,975 shard records × 260 B + 4,635 heartbeats × 200 B + 10,000 error slots × 300 B ≈ 5.3 MB — rotation is a runaway guard, not a routine mechanism.

Top-up: `stat` the 1,192 `ground_animal_images` dirs (~1 s; capybara is slowest at 423 stat/s `[M]` → 0.71 s), re-scandir only changed dirs, per-cell numpy anti-join bounded by the largest cell (1,527,654 ids = 12 MB), emit `gen=0002` with its own shard numbering. **Old generations are never mutated** — that is what keeps shard-index → image-id mapping stable and idempotency provable. This reuses `catalog.py:154-190`'s already-trusted mtime keying.

---

## 7. Live dashboard integration

### 7.1 Before anything else: stop the dashboard from competing for the drives

The live server (PID 3645894) runs with `--no-initial-refresh`, which affects the **first** build only. `serve()`'s timer at `dashboard.py:687-694` calls `_do_build(no_refresh=False, images=(cyc % images_every == 0))` with `--interval` defaulting to 3600 and `--images-every` to 1 — so **every hour** it spawns `catalog.py refresh` + `catalog.py images`. `cmd_images`' cross-drive dedup block (`catalog.py:365-385`) is **not** incremental: for every cell on more than one online drive it unconditionally `os.scandir()`s that cell's image directory on each involved drive and builds a Python set of filenames. `serve.log` shows 692 such passes, latest *"deduped 404/404 cross-drive cells"* `[M]` — roughly 400 cells and tens of millions of dirents across the exact four qd=1 drives the sweep feeds from, while holding the catalog **writer** lock. Measured build wall time 33–72 s on idle drives with a warm dentry cache; under 41 TB of concurrent reads the dentry cache is gone and it lands partly on the `0e:00.0` controller shared with lynx and bobcat, where contention already halved bobcat (45.4 → 22.9 img/s `[M]`). ~84 occurrences over 3.5 days.

**Restart the dashboard for the duration with `--interval 86400 --images-every 999999`.** Do not use `--images-every 0` — `cyc % 0` raises `ZeroDivisionError` and silently kills the timer thread. Restore afterwards and note the change in the run manifest so the schedule stays attributable. The detection panel needs no build at all.

### 7.2 Transport

The pipeline writes `data/dashboard/detect_status.json` every 5 s via `.tmp` + `os.replace` (atomic, lock-free, zero DuckDB contact). Because `serve()` already exposes `data/dashboard/` as the static document root (`functools.partial(BoardHandler, directory=OUT)`, `dashboard.py:698`), that file **is** an endpoint the instant it exists.

A thin `GET /api/detect` route goes in between `dashboard.py:595` and `:596`, matched with `self.path.split('?', 1)[0] == '/api/detect'` — this is mandatory, since `/api/board` and `/api/refresh` use `==` and `GET /api/board?t=1` returns **404** `[M]`. Body reads one file behind a 2 s memo under a lock, so N tabs collapse to ≤ 0.5 reads/s; `Cache-Control: no-store` added in `_json`. The client falls back to the static file on 404 **or 5xx** (the fallback's real value is transient-500 resilience — it cannot help against an unpatched server, because the client JS lives in that server's `TEMPLATE`), and switches mode with `schedule(0)` rather than waiting 30 s.

**Never** open `catalog.duckdb` or `history.duckdb` from a handler thread. `query_metrics` (`dashboard.py:79`) has no guard and 500s during every refresh window; opening `history.duckdb` read_only while `record_history`'s read-write handle is open **in the same process** raises `duckdb.ConnectionException` immediately `[M]`.

### 7.3 Correctness of the payload

- **`allow_nan=False` on `json.dumps`.** Python's stdlib emits bare `Infinity`/`NaN`, which is invalid JSON: `JSON.parse` throws, `r.json()` rejects, and the section degrades to "detect API unavailable" **at exactly the moment a drive parks** — the one moment it must work. Unknown ETA is JSON `null` (the client's `dur()` already renders null as "—"). Guard every division with `if rate > 1e-9`; `remaining / ema_slow` with `ema_slow == 0` raises `ZeroDivisionError` inside the daemon thread during the first minutes and the blanket `except` swallows it, so no status file is written at all at the *start* of the run.
- Serialize to a string first, write only on success, so a serialization error never leaves a truncated `.tmp`. Surface `publish_errors` in the payload — a permanently broken publisher currently looks identical to a dead pipeline, and the operator's response to "presumed dead" would be to kill a healthy 3-day run.
- `round()` every float (1 dp for rates/pct/MB-s, 0 dp for seconds). Unrounded, one EMA quotient serializes to 17–18 chars and `spark.img_s` alone goes from ~0.9 KB to ~3.1 KB `[M]`. Assert `len(blob) < 12000` in `--demo`.
- **Rehydrate at startup.** All counters are process-local accumulators; a 3.5-day run *will* be restarted. Without rehydration, day-3 restart shows ~2% complete, a fresh 3-day ETA and an empty confidence histogram — all three charts break simultaneously and it looks like catastrophic failure. At `StatusPublisher` startup, sum the committed shard sidecars for totals/regions/drives and rebuild the two 19-bin histograms with a single in-memory `duckdb.connect()` over the already-committed det parquet (~8M rows, seconds). Emit `resumed_at` and `session_done` so a resume is visible, and label `rate.mean`/`elapsed_s` explicitly as per-session.
- **All denominators from the deduped worklist.** `sum(drives.planned) == sum(regions.planned) == totals.planned == 32,542,334`, asserted at plan-build time. Mixing raw per-drive counts (32,582,319) with deduped regions makes the ETA stick at a nonzero value at the end of the run and `bound_by` permanently read `disk:<drive>` on a finished sweep.
- `bound_by` from **evidence**, not `argmax`: `gpu` only when `gpu.util ≥ 90` AND the decoded ring is near full; `feed` when `gpu.util < 70` AND the ring is near empty; `disk:<drive>` when a lane's duty is ~1.0; `paused`; else `unknown`. Deriving it from which term wins a `max()` makes the KPI chip print "GPU-bound" for *any* slowdown — contradicting the chart directly beside it.
- GPU sampling: **one long-lived `nvidia-smi ... -l 10` child**, read one line per tick. `subprocess.run` per sample forks a large-RSS CUDA-holding 16-thread process ~30,000 times over the run, copying page tables and doubling commit charge six times a minute on a box already 12 GB into swap. `pynvml` is absent from `dnd` and installing it is not worth it.
- Re-resolve mountpoint → device on **every** sample by parsing `/proc/self/mountinfo` and taking the **last** matching mount (so bobcat's stacked shutdown mount is not picked), cross-checked against `st_dev`; emit `dev_mb_s: null` (rendered "—") on failure. Kernel names are exactly what changes under the failure mode the metric exists to detect.
- `--demo` refuses the default path; it requires an explicit `--status-file`. Add `pid` and `run_id` to the client's freshness check so a changed pid is a distinct amber state, not silent continuity.

### 7.4 What the user sees

A `<details class="fold panel" id="f-det" open>` inserted after the command-generator `</details>` (`dashboard.py:881`). Fold persistence is free — the IIFE at `:1072-1085` keys purely on `details.fold` + id, and `:1076` only force-closes on `saved[d.id]===false`, so a never-seen id keeps its authored `open` state. **The one mandatory JS edit is adding the three chart ids to the resize selector at `:1081`**; index.html is regenerated hourly and the page hard-reloads via `<meta http-equiv=refresh content=3600>` (`:744`), so a user who collapsed the section otherwise gets 0×0 canvases on every reload.

**KPI chips:** % complete · done/total · remaining · live img/s (sub: 30-min EMA) · ETA + finish time + `bound_by` · positives · positive rate · boxes · GPU util/temp · VRAM · errors/corrupt · cells done · **out_free_gb** (red below 50 GB — crucial is 96% full).

**Chart 1 — throughput + GPU util, 30 min, dual axis.** Dashed markLine at **126.97 img/s** (measured TRT pure forward) labelled as the GPU ceiling. Both lines pinned high = GPU-bound, expected, nothing to do. Both dipping together = feed starvation → read the drive table. img/s low with GPU util low and one lane's `last_read_ago_s > 20` = spin-down stall on a 1058:25a3 bridge. This is the single most actionable panel; without the GPU trace overlaid you cannot distinguish the two cases, and they call for opposite responses.

**Chart 2 — per-region stacked completion + positive-rate scatter.** The baseline is drawn as a **shaded ±2.1 pp band** labelled *"measured capybara+bobcat, n=847"*, not a hard line: at n=847 the reference's own 95% CI is ±2.1 pp, so a fixed ±2 pp green/amber rule is dominated by noise in the *reference*. A region's scatter point is suppressed until it has ≥ 50k scanned images across ≥ 3 cells, because cells are consumed in scheduler order and an early reading is a spatially clustered subsample, not a random one. Tooltip carries n and cells_done/cells_total.

**Chart 3 — confidence histogram with a click-to-set threshold cursor**, over two 19-bin histograms: one of all boxes, one of each positive image's max conf. The second is not redundant — "how many *images* survive threshold t" cannot be derived from a box histogram, but `sum(img_max[i:])` gives it exactly. Readout: `≥0.25 → 3.12M boxes · 2.64M images (8.1% of scanned)`. This is what makes the store-at-0.05 decision actionable from full-corpus data while the run is still going.

**Drive table**, 4 rows: name+dev · pipeline img/s · pipeline MB/s · **device MB/s** (the delta between the last two is the only way to see USB-controller contention) · current cell + progress bar · state chip + per-drive ETA. `last_read_ago_s > 20` flips the chip amber with tooltip *"spin-down risk (1058:25a3)"*.

Not built, deliberately: GPU temperature gauge, per-drive donut, animated progress ring. Temp/VRAM/power/RAM/swap/queue depths are numbers you glance at, not charts.

Freshness bands from the HTTP `Date` header minus payload `ts` (immune to browser clock skew over Tailscale; `Date` verified present on both static and JSON responses `[M]`): ≤ 15 s live · 15–60 s amber "lagging" · > 60 s red "stalled", rates forced to 0 and counters frozen · > 30 min "presumed dead (pid N)". Poll 5 s running / 30 s otherwise / 0 when `document.hidden` or the section is collapsed — every fetch consumes a `ThreadingHTTPServer` thread.

**Security note:** `data/dashboard/` is published on `<private-host>:8050` with no auth. The sidecar emits drive **labels** (lynx/bobcat/…) and never absolute roots or image paths.

---

## 8. Correctness, auditing, provenance

### 8.1 The baseline, and what it can and cannot measure

`valset_fullres_457` (457 images / 520 boxes / 21 negatives, built and verified `[M]`) replaces the shipped set for **every** decode, engine and letterbox decision. All 459 shipped val images are 1280 px on the long side `[M]`; production images are 4080×3072 and 6503×3252. Any experiment involving a resolution change on the shipped set measures a downscale-then-upscale that cannot occur on the real corpus.

**Post-stratify every reported metric** to corpus composition along is_pano × aspect, recomputed **deduped and joined to the swept worklist** in P10. Do not use the raw-row weights: 9.57% of parquet rows are duplicate image_ids and they are 88.15% pano `[M]`, which alone moved the pano share from 53.97% to 50.36% and the 16:9 reweight factor from 1.66 to ~1.96. Report per-region too — pano share swings 5.5× by region (South_Asia 73.6% … Australia 13.4% `[M]`), so a global number hides that the highest-value region is the most under-supported condition.

**Gates in box counts, not recall deltas** (per-stratum n is 92–302 boxes; one box is 0.93–1.09 recall points, so a −0.005 fractional gate demands strictly zero loss and rejects everything by construction). Fractional gates apply only to the pooled 520-box set, where 0.005 ≈ 2.6 boxes.

Numerics chain FP32-PyTorch → FP16-PyTorch → FP32-TRT → FP16-TRT so each effect is isolable. Comparing an FP32 TRT engine against an FP16 PyTorch reference — as an earlier plan did — mixes quantisation and export error and isolates neither.

### 8.2 Recall transfers weakly; precision does not transfer at all

**Precision.** Recomputed at the threshold it is actually compared against (not at the max-F1 index of a conf=0.001 sweep): val predicted 497 boxes at conf 0.25 with TP ≈ 448 `[M]` → FP ≈ 49 → **0.107 FP boxes/image**. The real-corpus **total** box density at conf 0.25 is 0.091 boxes/image `[M]`. Val's false positives alone exceed the corpus's entire detection output. Val precision cannot transfer in either direction. With only 21 negatives (≈3 FP boxes), the Poisson 95% CI on the FP rate spans a factor of 14 `[D]`.

**Recall.** Prevalence-invariant, but **not selection-invariant**. The val positives come from a Label Studio project whose images were selected for labelling (95.4% of val images contain a dog vs ~8–17% of the corpus) — selected on dog presence and almost certainly on dog *visibility*. Post-stratifying on is_pano and aspect corrects two axes; it cannot correct selection on apparent dog size. Publish a **2-D range over (P, R)** with the audited R, not a 1-D range over P with val R baked in, and add a size check: GT box-area distribution in val vs stored detection box-area distribution on the corpus.

**Honest headline template:**

> Between **X and Y** of 32.5M images contain a street dog at conf ≥ t. The spread is driven by a precision figure measured on N audited real detections and a recall figure measured on M audited zero-detection images. Val precision (0.8722) is **not** applicable: val false-positive density is 0.107 boxes/image, which exceeds the corpus's entire detection output of 0.091 boxes/image.

**Audit design:** two-way stratified sample, 5 confidence bands × {pano, non-pano} = 10 cells × 150 = **1,500 detections** (per-cell 95% CI ±6.4% at p=0.8, pooled ±2.0% `[D]`), plus **4,000 images from the zero-detection stratum** for the recall denominator (~80 found misses at an expected 2% rate → 95% CI ±0.43% `[D]`). Horvitz-Thompson estimator gives precision at **every** threshold from one labelling pass: `P(t) = Σ_{h: lo≥t} N_h p_h / Σ_{h: lo≥t} N_h`. Blind to confidence, randomised order, 10% double-reviewed with `κ ≥ 0.8` required. Effort ≈ 8 h. Written rubric with an explicit AMBIGUOUS bucket reported separately. Sample the negatives as whole inode-sorted directory slices, not uniformly at random — uniform sampling across 32.5M files defeats the locality every read rate depends on.

### 8.3 Silent-failure guards

The dangerous case is proven: **a JPEG truncated to 98% with a valid EOI marker decodes to a full-size, normal-looking array — cv2 returns it, PIL accepts it, dimensions match, bottom-5% std 25.55** `[M]`. Simulated bit-rot (40 flipped bytes) behaves identically. The **only** signal is libjpeg's `Corrupt JPEG data: premature end of data segment` written to C-level fd 2, which is not a Python exception.

**But fd 2 is process-global, not thread-local.** The naive per-image `os.dup2(w, 2)` wrapper, run at 4 decode threads, measured **58/80 real corruptions missed (72.5%), 35/80 clean images falsely flagged (43.75%), and then `OSError [Errno 32] Broken pipe` on the next stderr write** — killing every log line for the rest of the run `[M]`. At 43.75% false alarm a >1% auto-pause rule halts all four drives inside the first window.

**Resolved architecture — keep threaded decode, split the guards by attributability:**

| tier | guards | thread safety |
|---|---|---|
| **per-image** (bitmask into `img.guards`) | G1 SOI/EOI markers on the raw bytes · G2 `imdecode` returns None · G4 decoded dims × reduction vs the **SOF dims parsed from the same bytes** (exact) · G5 `ndim==3 and shape[2]==3` · G7 `arr.std() < 2.0` · G9 SOFn component count == 4 (CMYK) · G12 `file_bytes < 1024` · crc32 of the raw bytes | all pure-Python/numpy on local buffers |
| **drive-level rate only** | G3 libjpeg stderr text matching `{'Corrupt JPEG data','Premature end','Bogus','Invalid JPEG','extraneous bytes'}` | fd 2 redirected **once at process start** to a pipe drained by a dedicated thread; never per-image |

When the drive-level warning rate crosses threshold, the suspect window is re-decoded **single-threaded** in a quarantine helper to attribute it. Baseline measured on 400 real images: 0 greyscale, 0 CMYK, 0 EXIF rotation, 0 truncation, 0 decode failures `[M]` → 95% upper bound on any single failure rate = `1 − 0.05^(1/400)` = 0.75% `[D]`.

**G4 must compare against SOF, never against parquet metadata.** Comparing decoded dims to parquet `width`/`height` across 720 images on 4 drives found 11 mismatches (1.53%), **all height-only, off by 4–12 px**, because Mapillary records pre-crop/pre-MCU-alignment height `[M]`. Rates are cell-clustered (6/60 in one South_America cell, 0/60 in nine others), so a global 0.5%/1.0% alarm would fire in bursts an order of magnitude over threshold on ~500,000 images for an entirely benign reason — while detecting essentially none of the corruption it exists for, since truncation and bit-rot do **not** change decoded dimensions. Parquet `width` matched 717/717 non-null rows `[M]`, so it is safe as a cross-check; parquet height is not.

crc32 costs 4,004 MB/s single-thread and fully releases the GIL (4.50× at 4 threads) `[M]` → **~5% of one core** at 198 MB/s aggregate, not the 20% previously assumed. Take it.

**Alarm policy:** per rolling 10,000-image window **per drive**, investigate at `(decode_null + corrupt_marker) / n > 0.5%`, auto-pause at `> 1.0%`. **Auto-pause must have a resume path**: re-probe with backoff 30 s / 2 min / 10 min / 30 min using the sentinel + `st_dev` check, resume after two consecutive passes, park permanently only after N failures, and write pause/resume state into the dashboard sidecar. A single bus reset at hour 6 must not park 4.04M images for the rest of the run with nobody notified. **Never auto-pause on a guard whose false-alarm rate has not been measured at production concurrency** (P7).

### 8.4 Panoramas

Panos are 50.4% of the deduped corpus `[M, P10 recount]` and the model *was* trained on them — train 21.1% pano, val 21.6%, pano boxes labelled `[M]` — just under-weighted ~2.4×. Tag, never reject.

Distortion where dogs actually appear is mild: all 108 val pano GT boxes sit at `y_center ∈ (0.50, 0.85]`, median 0.594, **zero above the horizon**, where equirectangular stretch `1/cos(lat)` is only 1.05× at the median and 1.32× at p95 `[M]`.

Store `y_center_norm` as a **column**, and ship the `[0.48, 0.82]` horizon band as a **re-filterable post-hoc prior with a retention claim only** (108/108 val pano GT retained, 95% lower bound 96.6% `[D]`) and **no false-positive-reduction claim**. The "removes ~66% of pano FPs if uniform in y" figure is unsupported — pano FPs will be dog-shaped ground objects (bins, bollards, shrubs, statues, reflections) that concentrate in the same band as true dogs. The audit's pano strata measure the actual FP y-distribution. Separately, split "is_pano from parquet metadata" from "aspect is 2:1" as two distinct flags and apply the prior only to the former; a 2:1 crop from a non-equirectangular rig has no horizon at y=0.5.

### 8.5 Provenance

`runs/gen=NNNN/_manifest.json`, plus `run_id` + `shard_idx` on every row and the same fields in every parquet's key-value metadata:

```
model     path, sha256 a69139a74b1b610ba4a6d908ffda4a9f88fbebc85f1e88114985247a718b3012,
          118,476,320 bytes, mtime 2026-06-06, arch yolo26x, train-30, best_epoch 243,
          single_cls true, nc 1, class_map {"0":"target"},   /* HARD-CODED. model.names
          returns {0:'item'} because single_cls overrode it at trainer.py:612-615 */
          head Detect, end2end false      /* NMS stays OUTSIDE the graph */
engine    tensorrt-fp16-static, sha256, build_ts, shape (8,3,1280,1280), io_dtype float32
inference conf 0.05, iou 0.90, max_det 300, max_time_img 1e9, agnostic_nms false,
          classes null, imgsz 1280, letterbox square/pad 114/scaleup false/center true,
          interp INTER_LINEAR, box_space "original_px"
decode    adaptive_sof, reduce_max 8, min_long 1280, IMREAD_IGNORE_ORIENTATION
versions  python 3.11.13, torch 2.9.0.dev20250711+cu128, torchvision 0.24.0.dev20250711+cu128,
          ultralytics 8.3.165, tensorrt 10.12.0.36, opencv 4.12.0.88, numpy 2.2.6,
          onnx 1.17.0, onnxslim 0.1.95, pyarrow 21.0.0
hardware  RTX 5080 sm_120, 16,303 MiB, driver 570.211.01, CUDA 12.8
worklist  sha256, n_images 32542334, n_cells 665, n_shards <from enumerate>,
          catalog_snapshot_mtime, dedup_policy, n_cross_drive_dupes_removed 39985
git       commit sha, dirty flag
valcheck  P4/P5/P6/P9 results for THIS exact engine config
```

`requirements.txt` currently pins nothing and omits torch entirely; letterbox geometry, source-type handling and the `single_cls` name override are all version-dependent behaviours this pipeline silently depends on. Freeze `dnd` to `tools/detect/dnd.lock.txt` (tracked) and fail preflight with exit 70 on any pin mismatch. A TRT engine is additionally tied to the exact GPU arch, driver and TRT version — a driver upgrade mid-run invalidates it, so re-run P2/P4/P9 if the driver moves.

**Policy on mixing engines:** forbidden by default. Expose the query views filtered to one `run_id`; crossing runs requires an explicit `img_all`/`det_all` view.

---

## 9. Build plan

| # | step | deliverable | effort |
|---|---|---|---|
| 1 | Phase 0 P0: env prep, lockfile, bobcat mount, keep-alive timers, dashboard restart, swappiness | `tools/detect/dnd.lock.txt` tracked; clean mount table; 3 keep-alive timers active | 40 min + user approvals |
| 2 | `tools/detect/engine.py` — AutoBackend loader, SOF parse, adaptive decode, letterbox, coordinate transform, NMS wrapper with `max_time_img=1e9` and the log-string tripwire | importable module + a `bench` entry point | 1 d |
| 3 | Phase 0 P1–P4, P9 (engine export, throughput, preprocessing equivalence, numerics chain, determinism) | `valcheck.json`; iou 0.7-vs-0.9 decision; engine sha256 | 2 h machine, 0.5 d analysis |
| 4 | `tools/detect/guards.py` — per-image bitmask guards, process-level fd-2 drain, crc32, quarantine re-decode helper | module + P7 harness | 0.5 d |
| 5 | Phase 0 P5–P7 (adaptive-decode detection gate, coordinate gate, concurrency guard gate) | pass/fail per stratum; `decode_reduce_max` fixed | 3 h machine |
| 6 | `tools/detect/sweep.py enumerate` — catalog read, inode-sorted scandir, 16-cell dedup, frozen `ids.npy`, `_meta.json` + hash, region column | `worklist/gen=0001/` | 0.5 d |
| 7 | Phase 0 P10 (composition recount) + P8 (controller frontier) | corrected reweight factors; capybara pacing target; schedule bracket fixed | 2.5 h machine |
| 8 | `tools/detect/store.py` — parquet writer, part naming, durable rename + dirfd fsync, `_state.json`, adoption with footer row-count verification, invariant assertions, bootstrap partition | module + unit tests for the crash windows | 1 d |
| 9 | `tools/detect/sweep.py run` — 11 readers, 4 decoders, pinned ring with the **completion-counted** publish, preallocated per-slot device tensors, GPU consumer with one D2H/batch, per-shard outstanding counters, closed-loop water-fill pacing, sentinel probe + quarantine, flock, two-stage SIGINT | working single-lane run | 2 d |
| 10 | `tools/detect/status.py` — publisher with rehydration, `allow_nan=False`, long-lived nvidia-smi child, mountinfo re-resolution, `--demo` (explicit path only) | `detect_status.json` + `--demo` mode | 0.5 d |
| 11 | dashboard patch: `DETECT_FILE` consts, `detect_payload()`, `Cache-Control`, the `/api/detect` route, CSS, the `f-det` section, the poll IIFE, **the `:1081` resize selector edit** | patched `dashboard.py` (gitignored) validated against `--demo` before any restart | 0.5 d |
| 12 | `preflight.sh` (exit 70 vs 75 split), `dogdetect.service.in`, `install_service.sh` (sentinels + render), `detect.config.example.json` | installable unit | 0.5 d |
| 13 | **Phase 0 P11: 6-hour soak** with Ctrl+C, simulated unmount, restart | all invariants green; measured production img/s; peak RSS | 6.5 h |
| 14 | Re-derive the schedule from P11's measured rate; commit (`yapf8`, author Alyetama, no Claude trailer, `git add -f` the new `tools/detect/` scripts); launch | `systemctl --user enable --now dogdetect` | 1 h |
| 15 | Monitor. Run P12 (audit) **in parallel**. `sweep.py verify` daily. | | 3.5 d elapsed |
| 16 | `sweep.py compact` + `verify`; mirror to weasel; threshold table + audited precision/recall; headline with the 2-D (P, R) range | final report | 1 d |

**Total: ~9 working days of build + 3.5 days of run**, with steps 3/5/7/13 as hard gates. Steps 2, 4, 6, 8, 10 are independent and can be interleaved.

---

## 10. Open questions needing a human decision

**Q1 — Clean bobcat's stale `shutdown` mount?** `/proc/mounts` carries both `/dev/sdm ... rw,relatime,shutdown` (device gone) and the live `/dev/sde` on `<mounts>/bobcat` `[M]`. Needs `sudo umount` on a live mount holding 4,043,431 images.
**Recommendation: YES, before the run.** This is the single highest-severity hazard in the plan. Unmount the stale entry and remount `/dev/sde` with the same options as the other three (`rw,nosuid,nodev,relatime,errors=remount-ro`). The sentinel probe protects against the failure, but leaving the underlying hazard in place for 3.5 unattended days when a 2-minute fix exists is not a good trade. Ask before touching.

**Q2 — Stop the three `prediction_api.py` Label Studio workers?** PIDs 908146 / 1190108 / 4193111, holding 1,234 MiB VRAM `[M]`, and at least two watch directories on **bobcat** — one of the four sweep drives.
**Recommendation: YES for the duration.** Not for the VRAM (the pipeline uses 18% of the card) but because they share the GPU scheduler, occasionally run inference, and poll a sweep drive. It takes the ML backend offline for ~4 days.

**Q3 — Restart the dashboard with `--interval 86400 --images-every 999999`?** Otherwise ~84 hourly non-incremental cross-drive dedup scans compete with the sweep for the same four qd=1 drives while holding the catalog writer lock `[M]`.
**Recommendation: YES.** Zero cost — the detection panel is a sidecar and needs no build. Restore afterwards; record the change in the run manifest.

**Q4 — Ship at `iou=0.90` instead of the inherited `0.7`?** NMS runs *before* anything is stored, so boxes suppressed at 0.7 are gone forever; re-tightening to any lower iou downstream is always possible, re-loosening costs 3.5 GPU-days. `0.7` is an inherited `cfg/default.yaml:52` default nobody chose, on a corpus of street scenes where packs of overlapping dogs are exactly what a 0.7 cut suppresses.
**Recommendation: YES, gated on P4.** Ship 0.90 if the box-count inflation on val-fullres + 700 real images is ≤ 3× (worst case 24.5M rows ≈ 640 MB, still 0.4% of free space). Otherwise 0.7, recorded as an explicit irreversible decision rather than an inheritance. Downstream counts must then re-apply NMS — a torchvision call over ~25M rows, seconds.

**Q5 — Install `pyarrow` into `dnd`?** The alternative is a numpy-only binary WAL plus an out-of-env compactor.
**Recommendation: YES, in P0, then freeze the lockfile.** `pip --dry-run` resolves cleanly with no torch/numpy conflict and the wheel is cached `[M]`. The reproducibility concern is about `ultralytics`/`torch` geometry, which pyarrow does not touch. The two-env alternative costs a cross-env dtype contract, a compactor, a lost-update manifest bug, 1.5–6.25 h of cold-tier staleness, and a buffered-write durability bug that fabricates "scanned negative" rows.

**Q6 — Write `.detect_drive_id.json` sentinels into the four grid_runs roots?** Four tiny dotfiles at the root, outside any cell, invisible to the `.jpg` enumeration.
**Recommendation: YES.** They are what makes an *unmounted* drive distinguishable from an *empty* one. Without them a reader on an unmounted `<mounts>/jackal` sees a legitimately empty directory and marks 10M images MISSING. The write-free alternative (ext4 UUID reverse lookup via `/dev/disk/by-uuid`) is more brittle across replug.

**Q7 — Clamp `decode_reduce_max` to 4 preemptively?** r=8 covers 6.0% of images (~2M panos) at 0.76% post-letterbox pixel error `[M]`.
**Recommendation: let P5 decide, but clamp on anything marginal.** Cost of clamping is essentially zero — decode has 2.3× headroom and r=4 on those images is still 4× less work than full decode.

**Q8 — Run the human audit before the sweep or in parallel?** Before delays the start ~1 day but lets the threshold table ship with real precision; in parallel starts sooner but the operating threshold cannot be defended until the audit lands.
**Recommendation: in parallel.** The run is 3.5 days and re-filtering is seconds. P12's 5,000-image inference is 10 minutes of machine time and can be drawn from the sweep's own output once the first ~1M images land.

**Q9 — What exactly is the `target` class?** `single_cls=true` collapsed the label space and the checkpoint reports `item`; the semantic definition (free-roaming only, or any dog? leashed? owned?) exists only in the original Label Studio project.
**Recommendation: resolve before the audit, not before the sweep.** The sweep does not depend on it. But precision is meaningless if reviewers and the training annotators disagree on leashed/owned dogs, so the rubric cannot be written without it. Pull the original project's labelling instructions.

**Q10 — Does `--drives <name>` stay in the shipped CLI?** It is the documented path to running a second process alongside the systemd unit.
**Recommendation: keep it, but only behind the per-lane flock.** With `runs/gen=NNNN/.lock` plus `.lane-<drive>.lock`, `--drives jackal` and a full run are mutually exclusive per lane and the escape hatch is safe. Without the locks it silently produces last-writer-wins parquet and dropped `_state.json` entries.
---

# Addendum A — the pipeline is two-stage (detect → crop → classify)

Written after §1–§10. The sweep does not just detect dogs: every detection is
cropped and passed to a leash classifier. Where this addendum and §1–§10
disagree, this addendum wins.

## A.1 What changed

`yolo26x-cls` at imgsz 640 classifies each detected crop. It was a two-class
model (`leashed` / `unleashed`); it is being retrained to three
(`leashed` / `unleashed` / `not_a_dog`) on `leash_3class_v2`, built by
`build_crop_dataset.py` in the dogs_detection repo.

**Update (2026-08-02): the single 3-class classifier was retired.** Measured on
its own clean val set it called 15.5% of real dogs `not_a_dog` at a mean
confidence of 0.870 -- confidently wrong, so no confidence filter rescues it,
and at 9.34M crops that is ~1.4M dogs discarded with no way to recover them
short of a re-run. The cause is structural: argmax over three classes offers no
operating point to tune, and the errors here are wildly asymmetric (a dropped
dog is unrecoverable; an admitted non-dog is filterable later).

The shipped architecture is therefore two stages: a **binary dog gate**
(`--binary`, thresholded for ~97% dog recall) followed by the **existing
2-class leash model**, which was never the weak link -- it was 95.1% accurate
at leashed-vs-unleashed whenever it did say "dog". The gate's negatives are fed
by dashboard-flagged false positives, harvested automatically at full
resolution (`tools/detect/harvest_flagged.py`).

**Why a third class.** 562 detector-flagged images that annotators then labelled
give the detector's real precision: **84.2% contain a dog, 15.8% do not** `[M]`
(3.4% background, 3.2% other-animal, 9.1% ambiguous "no boxes, no flag" — so the
honest range is 7–16%). A two-class classifier has no way to say "not a dog", so
at 9.34M crops it would confidently label **0.65M–1.5M** non-dog crops as leashed
or unleashed `[D]`. Those are unrecoverable without a re-run.

**Why the negative class is harvested, not annotated.** The annotators' non-dog
boxes have a median short side of **36 px** against **139 px** for dogs `[M]` —
a classifier trained on those learns "small = not a dog", scores well in
validation, and then fails on distant street dogs, which is the case that
matters. Instead the negative class is the detector's own false positives: every
predicted box that does not overlap a dog ground-truth box at IoU ≥ 0.5. Result
on `leash_3class_v2`: `not_a_dog` p50 = **144 px** between `unleashed` 155 and
`leashed` 232 `[M]` — fully overlapping, so size carries no signal.

## A.2 Cost — the classifier is free

| stage | rate | time for the sweep |
|---|---|---|
| detection, TRT fp16 bs8 `[M 127.1 img/s]` | 127.1 img/s | 32,542,334 / 127.1 = 256,037 s |
| disk floor, jackal-bound `[D]` | — | 309,022 s |
| **GPU idle waiting on disk** | — | **52,985 s** |
| classification `[M 178 crops/s]`, 9.34M crops `[D]` | 178 crops/s | 52,472 s |

52,472 < 52,985, so classification lands inside the idle the GPU already spends
waiting on jackal. **Wall clock stays ~3.6 days, disk-bound.** Headroom is thin
(1%), so if the crop rate or the disk estimate moves, classification becomes the
binding constraint — export the classifier to TensorRT too and the margin is
comfortable (the ultralytics wrapper costs 47% of raw throughput `[M]`).

Crop volume is measured, not assumed: 0.287 boxes/image at conf 0.05 / NMS iou
0.90 on 700 real images → 9.34M `[D]`; 5.63M at conf 0.20 `[M]`.

## A.3 Design consequences

1. **Classify inline, on crops already in RAM.** A second pass means re-reading
   41 TB — another 3.5 days for a stage that is otherwise free. The crop is a
   numpy slice of the already-decoded full-res image; it must never be written
   to disk and re-read.
2. **Crops come from the full-res decode**, not a downscaled copy. Cropping from
   a 1280px-wide image yields a crop ~3x smaller than the classifier was
   trained for.
3. **Store the detection confidence on every leash label.** The classifier sees
   crops down to conf 0.05, so `not_a_dog` is not the only defence against junk
   — the detection conf must remain available as a post-hoc filter, exactly as
   §5 does for detections.
4. **Do not shrink the classifier's imgsz below 640 without retraining.** Real
   crops are 88 px median `[M]`; 640 already upscales ~7x. Inferring at a size
   the model was not trained at is what cost the detector 62% of its detections
   at 640 (§3).
5. **Retraining the classifier at 1280 is not worth it** — crops are 88 px, so
   1280 is pure interpolation at 4x the compute. Accuracy is also flat across
   crop size (0.906–0.935 across every bucket from <64 px to 250+ px `[M]`), so
   there is no size-driven failure to fix.

## A.4 Schema delta (extends §5)

Add to the detection row: `leash_class` (enum `leashed`/`unleashed`/`not_a_dog`),
`leash_conf` (float32), and `leash_model_sha` in the provenance record alongside
the detector's. Keep every detection row even when `leash_class = not_a_dog` —
that is the measured non-dog rate, and discarding it destroys the ability to
audit the detector's precision after the fact.

## A.5 Dashboard delta (extends §7)

The live panel gains: crops classified, class split (`leashed` / `unleashed` /
`not_a_dog`), and the running `not_a_dog` rate. That last one is the single most
useful health signal in the whole run — the labelled prior is 7–16%, so a
sustained reading far outside that band means the detector is behaving
differently on unseen geography than it did on the labelled sample, and it is
visible within minutes instead of after 3.5 days.

## A.6 Open questions

**A-Q1 — operating threshold for the classifier.** The detector stores at conf
0.05; does the classifier run on every box, or only above some detection conf?
**Recommendation: classify every box ≥ 0.05.** It fits in the idle, and the
alternative needs a second targeted pass to fill in the gap later.

**A-Q2 — `not_a_dog` recall is the number to check after training.** A real dog
classified `not_a_dog` is a silently discarded true detection. Weight the review
toward that cell of the confusion matrix rather than top-1 accuracy. If recall
is weak, 443 other-animal images unused in detector training plus 100
`background=yes` images are available as extra negatives.
