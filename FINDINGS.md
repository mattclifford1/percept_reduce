# Findings

Analysis of the committed runs, the bugs that affect them, and what to run next.

Everything below was recomputed from the CSVs (`~/anaconda3/envs/percept/bin/python`), not read
off the figures. Unless stated, the headline metric is **best-epoch MLP probe accuracy**.

> **Where the runs live now.** §1–§3 analyse the original 90 runs, which are no longer in
> `saves/` — they were archived to `saves_legacy/gen1_original/` (pre-fix) and
> `saves_legacy/gen2_lossfix_oldprobe/` (post-B1/B2, pre-probe-v2). `saves/` now holds the
> **gen3** re-run on the current protocol (probe v2, seeded, checkpointed) plus the 40 unmigrated
> `IMAGENET64_VAL` runs. Gen3 results are in **§4**; note that `saves/readme.md` still describes
> the old 90-run layout and has not been updated.

> **Read the tables in §1 with this caveat.** "Best-epoch" is the maximum over ~16 evaluations of
> a metric computed on the probe's own eval set. That is selection on the test set: it inflates
> every number, and it inflates *noisy* runs (1% data, `DISTS`) more than stable ones — which is
> exactly where the comparisons of interest sit. `plots/summarise_runs.py` now defaults to the
> **final** epoch and prints how much best-epoch selection would have added; `--select best`
> reproduces the numbers below. Regenerate §1 on the final-epoch basis before quoting any of it.
>
> If you want a *stopped* number rather than the last one, the probe now carves a separate
> selection split: `--select early_stop` picks the epoch on it and reports on a disjoint split,
> which is unbiased. Runs predating that split (everything in §1) cannot produce it.
>
> The "±0.032 noise floor" quoted throughout §1 and §3 is **superseded** — it conflated init
> variance with probe under-fitting. On the current probe it is **±0.007 MLP** (B13). Both
> figures appear below; the ±0.032 ones are left in place as the record of what was believed at
> the time, not as current numbers.

---

## 1. Results

> **Historical — probe v1.** Every number in this section was measured on the unstandardised
> probe (B13) and with `MSSIM`/`LPIPS` still buggy (B1/B2). It is kept as the record that
> motivated the fixes. **Do not compare anything here to a current number**; the untrained MLP
> baseline alone moved from 0.128 to 0.411. §3 supersedes the `MSSIM`/`LPIPS`/`DISTS` rows on the
> old probe; §4 supersedes the whole section on the current one.

### CIFAR-10, `conv_big_z`, best-epoch MLP accuracy

Chance = 0.100. Untrained random encoder (epoch 0) = 0.128 ± 0.032. Raw-pixel RF baseline = 0.424.

| loss | 1% | 10% | 50% | 100% | uniform noise |
|---|---|---|---|---|---|
| **LPIPS** | 0.222 | 0.483 | 0.594 | **0.605** | 0.172 |
| **DISTS** | 0.206 † | 0.107 † | 0.170 † | 0.527 | 0.163 † |
| **MSE** | 0.352 | 0.445 | 0.452 | 0.459 | **0.418** |
| **SSIM** | 0.417 | 0.431 | 0.425 | 0.421 | 0.323 |
| **NLPD** | **0.422** | 0.417 | 0.396 | 0.398 | 0.221 |
| **MSSIM** | 0.296 | 0.359 | 0.377 | 0.380 | 0.385 |

† collapsed run — see §2.

### The one genuinely interesting result

**Hand-designed perceptual losses are flat in data size; learned ones are steep.**

- `SSIM` 0.417 → 0.421 and `NLPD` 0.422 → 0.398 going from 1% to 100% of the data. Both
  changes are inside the ±0.032 noise floor. **These losses reach their ceiling on 240 images.**
- `LPIPS` goes 0.222 → 0.605 over the same range, and `MSE` 0.352 → 0.459.

So the project's original hypothesis ("perceptual loss ⇒ need less data") is **supported for
hand-designed metrics and inverted for learned ones**. SSIM/NLPD encode a fixed visual prior
that substitutes for data but caps out around 0.42; LPIPS carries a much stronger prior (VGG
features trained on ImageNet) but needs data to exploit it, and then wins outright.

That framing — *perceptual priors trade data efficiency against ceiling* — is a better paper
than "perceptual is better than MSE", and the existing data already shows it.

### Training on pure noise recovers most of MSE's performance

`MSE`+`uniform` reaches **0.418** vs **0.459** for MSE on all 24,000 real images — **91%**, from
an autoencoder that has never seen a photograph. `MSSIM`+`uniform` (0.385) actually *beats*
`MSSIM` on real data (0.380).

This is the most under-reported number in the repo. It says a large fraction of what these
autoencoders contribute is architectural, not learned from the data distribution. Any claim
of the form "loss X produces better representations" has to clear this bar, and MSE, MSSIM
and (arguably) SSIM barely do.

Caveat: the `uniform` condition is not matched to anything. It draws a **fresh** noise image on
every `__getitem__`, so a 30-epoch run sees ~150,000 distinct images and never repeats one,
against a hardcoded 5,000/epoch that corresponds to no real-data condition.

### The untrained encoder is a strong baseline, and training often makes KNN worse

At epoch 0 (random init) the CIFAR probe already gets KNN ≈ 0.268 ± 0.014 and NB ≈ 0.287 ± 0.019.
**15 of 30 CIFAR runs finish with worse KNN accuracy than they started with.** A randomly
initialised conv stack is a decent random projection, and reconstruction training frequently
degrades local neighbourhood structure even while it improves the MLP-separable structure.

Worth reporting explicitly rather than leaving implicit in the epoch-0 row.

### Reconstruction quality is a weak proxy for representation quality

Spearman correlation between final val MSE and final MLP accuracy across the 30 CIFAR runs:
**−0.70** (ImageNet64: −0.46). Directionally right, but far from a substitute — which is the
whole justification for running the probe rather than just reporting MSE.

### ImageNet64 results are at or near chance and should not be used

Both ImageNet64 grids (60 runs) evaluate **1,000 classes on a 4,950-instance probe eval set** —
about 5 examples per class, with the probe fit on ~10 per class. Chance is 0.001. Only 20 of
60 runs exceed 0.01, and the best result across all 60 is **0.0218**.

The figures in `plots/figs/IMAGENET64_VAL/` used to look like they contained signal only because
`plot_training_runs.py` hardcoded `ylim=[0, 0.1]` for ImageNet — an 8× zoom relative to the
CIFAR panels. **That is fixed** — the plotting scripts now set only `ylim(bottom=0)` and let the
top autoscale, so the panels no longer flatter the results. The measurement problem is
untouched: nothing in these runs is interpretable as it stands. (Run counts have since moved —
40 sit in `saves/IMAGENET64_VAL/`, still in the legacy layout, with 60 more archived.)

---

## 2. Confirmed bugs

Ordered by how much they distort the committed results. Each was verified by running the code,
not by inspection alone.

> **Status.** Fixed: **B1, B2, B8, B9, B10, B13**, and the best-epoch item under B12. **B3 is
> settled** (see its entry — normalisation, not the loss). Partially addressed: **B7** (recorded
> and warned about, not prevented). Still open and re-verified as present: **B2's secondary**
> (stateful LPIPS), **B4, B5, B6, B11, B12**. `saves_legacy/gen1_original/` holds the runs B1/B2
> produced. §1 above describes the *pre-fix* results; it is retained as the record that motivated
> the fixes, and is superseded for `MSSIM`/`LPIPS` by §3 and entirely by §4.

### B1 — `MSSIM` is very nearly a no-op — FIXED

`percept_loss/losses/perceptual.py:31` — `MS_SSIM(..., win_size=1)`.

`pytorch_msssim` hard-asserts `min(H,W) > (win_size-1) * 2**4`, i.e. 161 px at the default
`win_size=11`. CIFAR is 32 px, so `win_size=1` was the only setting that would run. That
collapses the Gaussian window to a single pixel and destroys the structural term.

Measured, blanking a 10×10 patch of a 32×32 image:

| metric | identical | corrupted |
|---|---|---|
| real SSIM (win 11) | 1.0000 | **0.7455** |
| `MSSIM` as configured | 1.0000 | **0.9992** |

~300× weaker gradient for the same distortion. Visible in the runs: `MSSIM-1` val MSE **rises
monotonically 0.064 → 0.139** across training because the loss barely constrains the decoder.

The whole `MSSIM` row of the results table measures a near-constant loss, not multi-scale SSIM.

**Fix (verified):** `torchmetrics.image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0,
betas=(0.5, 0.5))` runs correctly at 32 px and gives 1.0000 → 0.3834 on the same test.
`pytorch_msssim` cannot be made to work at this resolution — its assertion ignores the `weights`
length.

### B2 — `LPIPS` is fed the wrong input range — FIXED

`percept_loss/losses/perceptual.py:22` — `LearnedPerceptualImagePatchSimilarity(net_type='vgg')`
defaults to `normalize=False`, meaning it expects **`[-1, 1]`**. It is given `[0, 1]`
(`NORMALISE = (0,1)`, decoders end in `Sigmoid`). torchmetrics 1.0.0 does not raise, so this
passed silently. The network only ever exercises half the dynamic range VGG was calibrated on.

**Fix:** `normalize=True` (verified equivalent to manually scaling to `[-1,1]`).

Note LPIPS still wins by a wide margin *despite* this, which makes the result more interesting,
not less — but the LPIPS column needs re-running before it is quotable.

Secondary — **still open**: `LearnedPerceptualImagePatchSimilarity` is a stateful
`torchmetrics.Metric`. Calling it in the training loop accumulates `sum_scores`/`total`
indefinitely and does roughly double the necessary work per step (verified: `total` grows by 2×
batch size per call). `losses/perceptual.py` still returns the metric object directly, for both
`LPIPS` and `LPIPS1`.

### B3 — three `DISTS` cells are optimisation failures, not data-size effects — SETTLED, AND THE CAUSE IS NOW MEASURED

> **The cause is the missing normalisation, not the loss and not the learning rate.** Interim
> count from the big re-run (265 of 417 trained cells finished, collapse detector on, everything
> at Adam 1e-3):
>
> | network | normalisation | collapsed |
> |---|---|---|
> | `dcgan` | BatchNorm | **0 / 51** |
> | `vae` | BatchNorm | **0 / 48** |
> | `resnet18` | BatchNorm | 3 / 48 |
> | `conv_big_z` | none | **20 / 70** |
> | `vit` | LayerNorm (pre-LN) | 33 / 48 |
>
> `dcgan` is `conv_big_z` plus BatchNorm and LeakyReLU and nothing else, so this is close to a
> controlled comparison: **29% collapse → 0%**. By loss, the collapses concentrate in `DISTS`
> (22/59) and `SSIM` (13/58) but appear even in `MSE` (4/59), which is what you would expect
> from an optimisation failure rather than a property of any loss.
>
> Consequence for `TODO.md` T1.1: the LR sweep may not be the fix. Normalising the backbone
> already removes the failure entirely, and it does so without changing the optimiser, so the
> loss axis stays comparable to the committed grid.
>
> The `vit` number is a *different* failure and should not be read as an architecture verdict:
> pre-LN transformers at 1e-3 with no warmup are a known divergence case, and `vit` was
> registered with that caveat already recorded in `networks/vit_autoencoder.py`.


`DISTS` at 1%, 10% and 50% collapse to a constant output within one epoch and never recover:
MLP pinned at exactly **0.1007** (single-class prediction) for 10–13 of 16 evals, val MSE flat
at ~0.09. At 100% data the same config escapes after ~7 epochs and reaches 0.527.

This is a dead-unit / degenerate-solution failure at Adam's default LR, not a statement about
data efficiency. **The `DISTS` row of §1's results table is meaningless.**

**Resolved by the gen3 `dcgan` runs, and the cause was normalisation rather than the LR.** All
five `DISTS`×data-size cells on `dcgan` — which is `conv_big_z` plus BatchNorm/LeakyReLU and
nothing else — trained the full 30 epochs at the *same* 1e-3 LR with no collapse, `out std`
~0.23 throughout, and a monotone data-size curve (§4). `LPIPS` at 1%, the other suspected case,
likewise shows no peak-then-degrade there. So the mechanism is a degenerate solution that batch
normalisation prevents, and the LR sweep is no longer a prerequisite for reading the loss axis —
it is now only worth running to confirm the mechanism on the unnormalised `conv_big_z`.

### B4 — data-size and optimisation-budget are confounded

`percept_loss/pipeline/generic.py:41-43`:

```python
scaled_epochs = int(epochs/data_percent)
scaled_epochs = epochs              # <- immediately overwrites the line above
```

The intended compensation is dead code. Every run gets 30 passes over its own training set, so:

| data | images | steps/epoch | total gradient steps |
|---|---|---|---|
| 100% | 24,000 | 750 | 22,500 |
| 50% | 12,000 | 375 | 11,250 |
| 10% | 2,400 | 75 | 2,250 |
| 1% | 240 | 8 | **240** |

**Every "less data" result is also a "~100× less optimisation" result.** This does not overturn
the SSIM/NLPD flatness finding — if anything it strengthens it, since those losses reach their
ceiling with 240 steps — but it fatally weakens any quantitative claim about data efficiency.

### B5 — no seeding, one seed per cell, no error bars

Only the train/val/test split is seeded (`proportions.get_indicies`, seed 42). Network init and
shuffle order are free-running. Measured spread at epoch 0 across nominally identical
random-init networks:

| dataset | KNN | MLP | NB |
|---|---|---|---|
| CIFAR (n=30) | 0.268 ± 0.014 | 0.128 ± **0.032** | 0.287 ± 0.019 |
| ImageNet64 (n=60) | 0.0041 ± 0.0008 | 0.0015 ± 0.0009 | 0.0121 ± 0.0032 |

**±0.032 on CIFAR MLP is larger than several of the differences being compared** (e.g. the
entire SSIM and NLPD data-size trends). Every cell is n=1.

**Partly fixed, and the ±0.032 figure is superseded.** Seeding now exists (`set_seed` is called
after the loss is built and before the network, so cells differing only in loss share an init)
and `seed` is a run axis that lands in the directory name. Re-measured on the current probe from
5 seeds of the `RANDOM` control, the floor is **±0.007 MLP** (B13) — the old figure was mostly
probe under-fitting, not init variance. What has *not* changed: the committed grid is still
n=1 per trained cell. The gen3 sweep is running 3 seeds, not the ≥5 the error bars want.

### B6 — `NLPD` uses 1 of its 6 pyramid levels

`NLPD(nlpd_k=1)`, but `DN_filters()` defines six levels of divisive-normalisation filters and
sigmas. Five are unused, so this is a single-scale metric wearing a multi-scale name.

Verified maximum usable `k`: **3 at 32 px, 5 at 64 px**; `k=6` fails at both (padding exceeds
the downsampled dimension). The reference implementation uses `k=6`.

Related: NLPD peaks at epoch 1–3 and then *degrades* at 50%/100% data (0.398 @ ep1 → 0.339 final
at 100%). Its "flat across data size" result is partly "its best number is roughly its epoch-1
number".

### B7 — split proportions are selected by `validate_every` — PARTIALLY ADDRESSED

`training/run_and_test.py:86` uses `validate_every == 1` as a proxy for "this is the big
ImageNet64 train split" and switches props from `[0.4,0.3,0.3]` to `[0.89,0.1,0.01]`. Changing a
validation *frequency* silently changes the dataset *split*, and the split is not recorded in
the run directory name. Runs with different `validate_every` are not comparable and will
collide on the same save path.

The mechanism is unchanged, but it is no longer silent: `config.json` now records `split_props`
and `validate_every`, and `write_config` prints a warning when it overwrites a config whose
`split_props`, `validate_every` or `data_percent` differ from the run about to start. The
collision itself is still possible — fixing that means putting the split in the run path.

### B8 — `batch_size` is never forwarded — FIXED

`pipeline.generic.run(batch_size=...)` was used only to name the save directory. It was not
passed to `train()`, which did not pass it to `get_all_loaders()`, which defaults to 32. Every
committed run is therefore BS32 and the `BS32` in those directory names is true by coincidence.

`train()` now takes `batch_size` and forwards it, and takes `lr` as well (Adam's default was
hardcoded, which made the LR sweep T1.1 asks for impossible without editing the trainer). Both
are recorded in `config.json`. **No committed result changes** — they really were all BS32.

### B9 — no `torch.no_grad()` / `net.eval()` in evaluation — FIXED

`validate()` and `make_encodings()` both ran a full split with autograd live and discarded the
graph. Pure waste and the dominant memory cost of eval. `net.eval()` was never called either —
harmless while no registered net had BatchNorm or Dropout, and a silent correctness bug the
moment one did.

The moment arrived: `dcgan`, `resnet18`, `vit` and `vae` all use BatchNorm. In train mode a
BatchNorm encoder normalises each batch by its own statistics, so an image's encoding would
depend on whichever images happened to share its probe batch — the probe would be scored on
encodings that are not a function of the image alone.

Both functions now save the mode, call `.eval()`, run under `torch.no_grad()`, and restore the
previous mode. This also makes the VAE decode from mu rather than a sample during validation,
so val MSE is not sampling noise. **No pre-existing result changes** — every run made before the
fix used the norm-free, dropout-free `conv_big_z`/`bigger_z`/`standard`, where `.eval()` is a
no-op. The BatchNorm backbones now in `saves/` were all trained after it landed, so no run
anywhere was probed in train mode.

### B10 — a crashed run is skipped forever — FIXED

`train_saver.previously_done` was just `os.path.exists(csv_file)`. A killed run left a partial
CSV and every subsequent invocation printed `Passing` and moved on. There was no row-count or
completeness check. (All 90 committed runs do have the full 16 rows — checked.)

Completion is now a `done.json` written after the last epoch, carrying wall time, row count, and
whether the run tripped the collapse detector. A results CSV with no `done.json` is treated as a
crashed attempt: it is moved aside to `training_results.csv.partial-*` and the cell re-runs.
Legacy directories (which predate `done.json`) are still honoured as complete, so migrating is
optional and an un-migrated grid is never silently re-run.

### B11 — stale scripts that no longer import

- `training/benchmark.py` — imports `get_all_loaders_CIFAR` and `random_forest_test`, neither
  exists.
- `testing/baseline_performance.py` — same broken import, plus it calls `test_all_classifiers`
  with the pre-refactor positional signature.

The second matters: it is the only source of the raw-pixel baseline in `saves/readme.md`, and
that baseline's 10%-data figure (0.0421) is *below* chance for a 10-class problem, so it is
almost certainly wrong and cannot currently be rechecked.

### B12 — smaller latent issues

All re-verified as still present except the last, which is fixed.

- Mutable default `image_dict={}` on both real loader classes (`CIFAR_10/loader.py:18`,
  `IMAGENET/ImageNet64.py:22`); masked today because all call sites pass an explicit dict, but a
  live trap.
- `percept_loss/datasets/CIFAR_10/__init__,py` — **comma instead of a dot**. Works only via
  implicit namespace packages plus editable install; `find_packages()` does not see the
  directory, so a non-editable `pip install .` ships a broken package.
- ImageNet64 `_get_labels` writes `one_hot[label-1]` (`ImageNet64.py:94`) but returns `label`
  unshifted. Consistent today only because nothing reads `data[1]` — the probe uses `data[2]`.
- `validate()` averages per-batch means unweighted (`sum(scores)/len(scores)`), over-weighting
  the final short batch.
- `get_indicies` documents accepting a list for `total_instances`, and branches on it, then
  computes `int(total_instances * prop)` — which raises for a list. The branch is dead.
- ~~Reporting best-epoch accuracy selects over 16 evaluations **on the same eval set**, which
  inflates every number in §1 by an unmeasured amount.~~ **FIXED**, both ways suggested: the
  probe carves a dedicated selection split — 67% fit / 16.5% select / 16.5% report — so
  `--select early_stop` is unbiased, and `summarise_runs.py` defaults to final-epoch and prints
  the size of the inflation. §1 is still best-epoch and predates both.

### B13 — probe features were never standardised, and the MLP probe was under-fit — FIXED

`test_all_classifiers` fed raw latents straight to `KNeighborsClassifier` (pure Euclidean
distance) and `MLPClassifier(alpha=1)` (heavy L2). Both are scale-sensitive; the latents are
`Tanh`-bounded, roughly [-1, 1], and small. A `StandardScaler` fit on the probe's train split is
now applied before every classifier.

The effect is not marginal. Untrained `dcgan`, same seed, same weights, same split — the only
difference is the scaler:

| probe | unscaled | standardised |
|---|---|---|
| KNN | 0.2703 | 0.2793 |
| **MLP** | **0.1619** | **0.4207** |
| NB | 0.2944 | 0.2944 |

`NB` is unchanged (Gaussian NB estimates a per-feature variance, so it is already scale-free) and
`KNN` barely moves, which is what makes the `MLP` jump attributable to scaling rather than to
anything else. The **MLP probe is the headline metric of the whole project**, and on unscaled
features it was under-fitting to the point of near-chance on an untrained encoder.

**The baseline has now been re-measured on `conv_big_z` itself.** Five seeds of the `RANDOM`
control (`saves/CIFAR_10/conv_big_z/RANDOM/1/`), standardised three-way probe, untrained network:

| probe | mean ± sd over 5 seeds | §1's figure |
|---|---|---|
| KNN | 0.2727 ± 0.0100 | 0.268 ± 0.014 |
| Linear | 0.3852 ± 0.0097 | *(did not exist)* |
| **MLP** | **0.4112 ± 0.0071** | **0.128 ± 0.032** |
| NB | 0.2806 ± 0.0206 | 0.287 ± 0.019 |

`KNN` and `NB` reproduce §1 almost exactly, which is the control that makes the `MLP` row
believable: the split, the network and the seeds are behaving as before, and only the
scale-sensitive probe moved. **The untrained MLP baseline is 0.411, not 0.128.**

Consequences:

- §1's trained `MLP` numbers were measured on the *old* probe, so they cannot be compared to
  0.411 — that comparison would be cross-protocol. What can be said is that the bar has moved by
  ~0.28 and no trained number exists on the current protocol. Several §1 rows (`SSIM` 0.421,
  `NLPD` 0.398, `MSSIM` 0.380, and `MSE` at 0.459) are close enough to 0.411 that whether they
  beat an untrained encoder is now an **open question**, not a settled one.
- The noise floor is **±0.007 on MLP**, not ±0.032. The old figure came from unseeded runs on an
  unstandardised probe, so it was conflating init variance with probe under-fitting. Per-metric
  spreads are now in `plots/run_index.py` and drive the shaded band in the figures. Note this is
  init variance only — it does not include shuffling variance during training.
- Because under-fitting penalises whichever latents are hardest to fit, the loss *ranking* in §1
  is not safe either — this is not a constant offset.
- The gen1/gen2 runs cannot be re-probed (no checkpoints), so only a re-run puts trained numbers
  on the current protocol. **That re-run is under way** — `pipeline_CIFAR_BIG.py` — and every run
  it produces saves `checkpoint.pt`, so the next probe change costs seconds via
  `testing/reprobe.py` rather than a full retrain. First trained numbers on the current protocol
  are in §4, and they answer the "open question" above.

### B14 — a legacy directory could satisfy a run of a different protocol — FIXED

The skip rule treated a pre-migration directory as proof a cell was done. A legacy directory has
no `config.json`, so it cannot say which seed, LR **or probe protocol** produced it. Narrowing it
to `LEGACY_SEED = 42` / `LEGACY_LR = 1e-3` fixed the seed and LR halves but not the protocol half.

**Measured consequence.** In the big re-run, all 35 `conv_big_z` seed-42 cells were skipped: seed
and LR matched a legacy directory, so each reported "done" on the strength of probe-v1 numbers
that the re-run existed to replace. The grid finished with seed 42 at **1 completed cell against
36 for each of seeds 1 and 2**, and the equal-steps grid's seed-42 partition was caught the same
way. Nothing was printed — the skip path was silent whenever seed and LR agreed.

Two fixes:

- A legacy CSV now only stands in for a current run if it was produced by the current probe.
  `savers.legacy_matches_current_probe` reads the header: v2 runs carry a `<metric> select`
  column per classifier, v1 runs do not. When a legacy directory is rejected, the reason is
  printed. Verified against all five cases (v1/v2 × matching/non-matching seed and LR, plus
  `done.json`).
- `pipeline.generic.run` counts skips by reason and prints a summary; a partition that trains
  **zero** cells now prints a warning instead of exiting quietly, which is what made the original
  incident invisible.

Consequence worth knowing: the 40 legacy `IMAGENET64_VAL` directories no longer count as done, so
an ImageNet re-run will now actually re-run them rather than silently skipping. That is correct —
those runs are probe v1 — but it means the ImageNet grid is *not* "already done" any more.

### B15 — two processes could run the same cell and corrupt its CSV — FIXED

`previously_done` is a check-then-act with no lock, so two processes aimed at the same cell both
decide to run it. Both then write `training_results.csv`, and `save_and_merge_df_as_csv` is an
**outer join** — the result is a file with duplicated epochs rather than an error.

**Measured consequence.** Running the fixed-budget and equal-steps grids concurrently, five
seed-42 cells collided (`equal_steps` at `data_percent=1` resolves to the same 30 epochs, hence
the same directory as the fixed-budget run). All five CSVs came out corrupted — one had 30 rows
and 14 duplicate epochs where 16 rows were expected. The five were deleted and re-run.

Fix: `train_saver` claims a cell with an `O_CREAT|O_EXCL` lock file (`running.lock`) recording
the pid; a second process finds the lock and skips with `skip_reason='locked'`. A lock whose
process is gone is reclaimed, so a crash cannot wedge a cell forever (which would have been B10
in a new costume). `write_done` releases it. Verified: concurrent claim, stale-lock reclaim, and
release-on-done.

Note the collision itself is legitimate and expected — at 100% data and on `uniform` the two
budgets *are* the same run, so they share one cell. Only the concurrent write was the bug.

### B16 — the readers pooled seeds and both optimisation budgets into one cell — FIXED

`summarise_runs.summarise` grouped on `('dataset', 'net', 'loss', 'datasize')` and
`plot_data_efficiency` selected cells with `runs[runs['datasize'] == size]`. Both were correct
while the grid had **one seed and one budget** — that key was then unique per run. The big
re-run made it a 6-to-1 key (3 seeds × 2 budgets) and nothing complained, because the readers
never carried `epochs` or `epoch_scaling`: `load_long` copied a fixed list of metadata columns
onto each row and neither was in it.

**Measured consequence.** The concatenated rows were sorted by epoch and reduced:

- `final` = `.iloc[-1]` of six runs → **whichever run had the highest epoch number**, which is
  always the 3000-epoch `equal_steps` cell. A table labelled "fixed budget" was silently
  reporting equal-steps numbers at 0.01/0.1/0.5.
- `best` = `max` over six runs, compounding the selection bias B13/T1.5 already warn about.
- `epoch0` and `n_evals` summed across seeds — `n_evals` read 48 (3 × 16) instead of 16, which
  is the visible tell that the pooling was happening.
- The two budgets, whose *difference* is the entire point of T1.2, were averaged into each other.

Fix: `load_long` now carries `epochs` and `epoch_scaling` (defaulting to `'fixed'`, which every
pre-budget run was). `summarise` groups by `run_dir` — the only key guaranteed unique per run —
and aggregation across seeds is now explicit: **mean ± sd over seeds**, with the per-cell run
count printed. `plot_data_efficiency` reduces per `run_dir` and draws ±1 sd error bars, and both
readers emit **one table/figure per budget**.

`run_index.budget_view` pulls the shared cells (`data_percent=1`, `uniform`) into both budget
views: the scaling factor there is exactly 1, so they are one run belonging to both grids, and
without them the equal-steps curve loses the full-data anchor it is measured against.

**Every accuracy read off `summarise_runs.py` or `plot_data_efficiency.py` between the big
re-run landing and this fix is wrong** — not noisy, wrong — for `conv_big_z` at sizes 0.01/0.1/
0.5. Single-seed, single-budget trees (everything in `saves_legacy/`, the ImageNet grid) are
unaffected, since the old key was still unique there.

Unrelated but found in the same pass: the plot scripts imported `matplotlib.pyplot` without
selecting a backend, so with a display advertised but not answering they block in `poll()`
forever — observed at 10 minutes elapsed for 1 second of CPU. Both now call
`matplotlib.use('Agg')`; neither ever called `show()`.

---

## 3. Post-fix results

> **Historical — probe v1.** Still the unstandardised probe, so these numbers are comparable to
> §1 and to each other but **not** to §4. The B1/B2/B3 conclusions below are about the fixes and
> survive the probe change; the absolute accuracies do not.

The full CIFAR grid (35 cells) was re-run after fixing B1/B2, now **seeded** — every cell shares
an identical network init, so cells differing only in loss are a *paired* comparison. Results in
`saves_legacy/gen2_lossfix_oldprobe/` (moved there when the probe changed); the pre-fix runs are in `saves_legacy/gen1_original/`.

Headline metric here is **final-epoch** MLP accuracy. Best-epoch (used in §1) is a max over ~16
evals on the probe's own eval set and is optimistically biased by **+0.026 on average** — and
biased more for unstable runs than stable ones, which is exactly where the interesting
comparisons are. Both are shown.

### B1 fix — `MSSIM` improved a lot, and the noise control moved the right way

| datasize | pre-fix best | fixed best | fixed final | Δ best |
|---|---|---|---|---|
| 1% | 0.2965 | 0.4136 | 0.4086 | **+0.117** |
| 10% | 0.3591 | 0.4150 | 0.4057 | +0.056 |
| 50% | 0.3773 | 0.4350 | 0.4269 | +0.058 |
| 100% | 0.3796 | 0.4290 | 0.4190 | +0.049 |
| uniform noise | 0.3854 | 0.2939 | 0.2148 | **−0.091** |

The real-data gains are well outside the ±0.032 noise floor. **The `uniform` row is the real
confirmation:** before the fix, MSSIM trained on pure noise (0.385) *beat* MSSIM trained on
24,000 real images (0.380) — the loss was so weak that real data bought nothing. After the fix,
real data (0.42) clearly beats noise (0.21). That is the signature of a loss that actually
constrains the decoder, and it is exactly what a no-op being repaired should look like.

Fixed `MSSIM` now sits alongside `SSIM` (~0.41–0.43) instead of well below it, which is the
sane result — they are the same metric at different scale counts.

### B2 fix — the correction made LPIPS *worse*, which is why `LPIPS1` was worth keeping

`LPIPS` (`normalize=True`, correct) vs `LPIPS1` (`normalize=False`, the original bug), identical
init and identical shuffle order — the loss is the only difference:

| datasize | LPIPS final | LPIPS1 final | Δ | LPIPS best | LPIPS1 best | Δ |
|---|---|---|---|---|---|---|
| 1% | 0.1439 | 0.1574 | −0.014 | 0.2084 | 0.2103 | −0.002 |
| 10% | 0.5072 | 0.4988 | +0.008 | 0.5072 | 0.5118 | −0.005 |
| 50% | 0.5785 | 0.5993 | −0.021 | 0.5843 | 0.5993 | −0.015 |
| 100% | 0.5736 | **0.6285** | **−0.055** | 0.5796 | 0.6301 | −0.051 |
| uniform | 0.1007 | 0.1007 | 0.000 | 0.1673 | 0.1673 | 0.000 |

**The uncorrected loss wins at 50% and 100% data**, by more than the noise floor at 100%. The
fix is still correct — `normalize=False` genuinely feeds VGG half the dynamic range it was
calibrated on — but "correct" did not mean "better here".

Most likely explanation, and it is a *confound rather than a finding*: halving the input
contrast shrinks the LPIPS gradient, which at Adam's untuned default LR acts as an implicit
learning-rate reduction. Given that `DISTS` and `LPIPS` both collapse outright at this LR
(B3 below), a weaker perceptual gradient being better-conditioned is entirely plausible. So
this is evidence that **the LR is wrong**, not that the bug was good.

*Update.* B3 turned out to be fixable with BatchNorm at an unchanged LR, which weakens the
"the LR is wrong" reading — the conditioning problem was the unnormalised architecture. The
`LPIPS`/`LPIPS1` comparison has not been re-run on `dcgan`, so whether the gap survives a
well-conditioned backbone is untested and is the cheap way to settle it.

Caveats, both real: n=1 per cell, and the ±0.032 noise floor comes from the *old unseeded*
runs — in the new grid every cell shares an init, so epoch-0 spread is exactly 0.0000 and the
grid can no longer estimate its own noise. Resolving this needs the multi-seed sweep (T1.4) and
an LR sweep (T1.1), in that order.

### B3 confirmed — the `DISTS` collapse moved when only the seed changed

Final-epoch MLP, same grid, same data, different init:

| datasize | unseeded (old) | seeded (new) |
|---|---|---|
| 1% | 0.1007 collapsed | 0.1392 |
| 10% | 0.1007 collapsed | 0.1007 collapsed |
| 50% | 0.1007 collapsed | **0.5007 worked** |
| 100% | **0.5202 worked** | 0.1007 collapsed |
| uniform | 0.1045 | 0.1067 |

The collapse pattern **completely rearranged**: the cell that worked before now fails, and one
that failed now works. This settles B3 — `DISTS` collapse is initialisation-dependent
optimisation instability, not a property of the data size. Any `DISTS` row read as a
data-efficiency curve is reading noise. T1.1 in `TODO.md`.

### What did not change

`MSE`, `SSIM` and `NLPD` reproduce their pre-fix shape closely under the new seed, including the
§1 headline: hand-designed losses stay flat across data size (`SSIM` 0.413 → 0.407 from 1% to
100%; `NLPD` 0.403 → 0.341) while `LPIPS` climbs steeply (0.144 → 0.574). That result survives
the fixes.

## 4. Gen3 — first results on the current protocol

Probe v2 (standardised, three-way split), seeded, checkpointed, `saves/CIFAR_10/`. **This section
is the only part of the file whose numbers are comparable to each other and to the untrained
control.** It is partial — the sweep is still running — and every trained cell below is n=1
(seed 42). Metric is **final-epoch MLP**, `dcgan`, 30 epochs, lr 1e-3.

### Untrained baselines, per architecture

`RANDOM` control, no gradient steps. `conv_big_z` is 5 seeds, the rest 3:

| network | MLP | Linear | KNN | NB |
|---|---|---|---|---|
| `conv_big_z` | 0.4112 ± 0.0071 | 0.3852 ± 0.0097 | 0.2727 ± 0.0100 | 0.2806 ± 0.0206 |
| `dcgan` | 0.4227 ± 0.0103 | 0.3961 ± 0.0023 | 0.2728 ± 0.0106 | 0.2878 ± 0.0025 |
| `resnet18` | **0.4430 ± 0.0078** | 0.4067 ± 0.0049 | **0.3028 ± 0.0047** | 0.2809 ± 0.0039 |
| `vit` | 0.3441 ± 0.0058 | **0.4071 ± 0.0108** | 0.2376 ± 0.0075 | 0.2269 ± 0.0043 |
| `vae` | 0.4064 ± 0.0125 | 0.3886 ± 0.0060 | 0.2660 ± 0.0104 | 0.2790 ± 0.0058 |

An **untrained ResNet-18 encoder scores 0.443** — higher than most trained cells in §1. The bar
is architecture-dependent, so every comparison has to be against its *own* network's control,
not a single global baseline. `vit` is the interesting one: worst MLP, best Linear, which says
its latent is more linearly separable and less nonlinearly rich than the conv stacks'.

### `dcgan` loss × data size, final-epoch MLP (untrained control = 0.4227)

| loss | 1% | 10% | 50% | 100% | uniform |
|---|---|---|---|---|---|
| **LPIPS** | 0.4552 | 0.5138 | 0.5684 | **0.5980** | 0.4539 |
| **DISTS** | 0.4444 | 0.5027 | 0.5384 | 0.5434 | 0.4421 |
| **MSE** | 0.4613 | 0.4673 | 0.4768 | 0.4593 | 0.4394 |

`Linear` tells the same story (LPIPS 0.4229 → 0.5832; MSE 0.4212 → 0.4145).

**Three things change relative to §1:**

1. **`DISTS` does not collapse anywhere**, at the same LR that killed three cells on
   `conv_big_z`. It is a well-behaved, monotone data-size curve. This is what settles B3.
2. **`MSE` is flat and barely above its own untrained control** — 0.459 at 100% data against
   0.423 untrained, and its 1% cell (0.461) is no worse than its 100% cell. On probe v1 `MSE`
   looked like a real data-size trend (0.352 → 0.459); on probe v2 that trend is gone, because
   most of what the old numbers were measuring was the probe failing to fit the untrained
   latents. The learned perceptual losses keep their steep trend.
3. **The uniform-noise result mostly dissolves.** §1 read `MSE`+`uniform` = 0.418 vs 0.459 as
   "91% of the performance from an autoencoder that never saw a photograph". Here `MSE`+`uniform`
   is 0.439 against 0.423 untrained — training on noise moves the encoder ~0.016, and training
   `MSE` on 24,000 real images moves it ~0.037. The right reading is not "noise is nearly as good
   as data" but **"`MSE` reconstruction barely changes the representation at all"**. Against
   `LPIPS`, whose uniform cell (0.454) sits ~0.14 below its real-data cell (0.598), the noise
   control behaves exactly as it should.

So §1's headline — *hand-designed perceptual priors are data-efficient, learned ones are
data-hungry but win* — survives only in its second half. The first half needs restating: on the
current probe, the non-learned losses (and the pixel loss) are not data-efficient so much as
**close to doing nothing**, and the untrained encoder was always most of their score.

Caveats before quoting any of this: n=1 per trained cell against n=3 controls; `conv_big_z`,
`SSIM`, `MSSIM`, `NLPD` and `LPIPS1` cells were still running at the time of writing; the ±0.007
floor is init variance only and so a lower bound; and the B4 data/optimisation-budget confound
is untouched — every cell is still 30 passes over its own training set.

### `vit` on the completed grid — it reconstructs fine and encodes almost nothing

Three seeds, MLP probe, unbiased early stopping. `vit` loses on every axis that matters:

| network | untrained | best cell | gain | collapsed | corr(val MSE, MLP) |
|---|---|---|---|---|---|
| `conv_big_z` | 0.4112 | 0.6108 (LPIPS1@1) | +0.200 | 43/168 (26%) | −0.32 |
| `dcgan` | 0.4227 | 0.5963 (LPIPS@1) | +0.174 | 0/51 (0%) | −0.32 |
| `resnet18` | 0.4430 | **0.6889** (LPIPS@1) | +0.246 | 3/48 (6%) | −0.41 |
| `vae` | 0.4064 | 0.5933 (LPIPS@1) | +0.187 | 0/48 (0%) | −0.24 |
| `vit` | 0.3441 | 0.4781 (MSE@1) | +0.134 | **33/48 (69%)** | **+0.008** |

The last column is the interesting one. For every convolutional architecture, better
reconstruction goes with a better latent (−0.24 to −0.41, as expected). For `vit` the
correlation is **+0.008 — no relationship at all**. It reconstructs CIFAR-10 perfectly well and
the latent still carries nothing the probe can use, which is the cleanest evidence in this repo
that **val MSE is not a proxy for the metric we care about**. It also means `vit`'s low numbers
are not an optimisation failure that a better LR would fix.

Two more `vit`-specific facts:

- **Training can make the representation worse than the initialisation.** At final epoch,
  `MSE`@1% reaches 0.2532 and `SSIM`@100% reaches 0.2767 against an untrained control of 0.3441
  — training destroys 0.07–0.09 of accuracy. Unbiased early stopping rescues both by selecting
  epoch 0, which is why the early-stopped table shows no `vit` cell below its control.
- **69% collapse** is not the loss's fault: `dcgan` and `vae` collapse 0% on the same losses and
  data. It is architectural, and consistent with B3's conclusion that collapse tracks
  normalisation rather than the loss.

`vit` was added expecting it to lose on data grounds (ViTs are data-hungry, and this is 50k
32×32 images with no augmentation and no masking objective). It does lose — but the mechanism is
not the expected one, and the decoupling of reconstruction from probe accuracy is worth more
than the ranking.

## 5. What to run next

Moved to **[TODO.md](TODO.md)** so it can be worked through and ticked off. That list carries
the same items, grouped by priority:

- **Tier 1** (blocks any quotable result) — decoupling data size from optimisation budget (B4),
  fixing the ImageNet64 probe, and multi-seed error bars. *(The `CIFAR`/`CIFAR_10` save-path
  mismatch is resolved bar an assertion, and the `DISTS` collapse is settled — see B3.)*
- **Tier 2** (one small change away) — a proper baseline panel, matching the `uniform` control,
  re-sweeping the architecture axis, longer training for the learned perceptual losses, and
  sweeping the `NLPD` pyramid depth.
- **Tier 3** (new directions) — loss combinations, more probe types, encoder transfer, and
  logging every metric for every run.
- **Engineering** — the stale scripts (B11), the B12 list, the stateful-LPIPS inefficiency, and a
  set of assertions that would have caught B1–B3. *(`no_grad`/`eval`, saving weights, config
  sidecars and skip-if-exists completeness are all done.)*

This file stays as the evidence: what was measured, and what it implies.
