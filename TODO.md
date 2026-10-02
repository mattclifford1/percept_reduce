# TODO

Open items from the `FINDINGS.md` audit. Evidence and measurements for each are in that file —
this is the action list. Ordered by how much each blocks a publishable claim.

**Done and not repeated here:** the `MSSIM` no-op (B1) and `LPIPS` input-range (B2) bugs are
fixed, `LPIPS1` was added to preserve the pre-fix LPIPS behaviour as a measurable control, and
per-run seeding was added so runs differing only in loss share an identical network init. The
full 35-cell CIFAR grid was re-run and lives in `saves/CIFAR_10/`; all pre-fix runs are
archived in `saves_legacy/gen1_original/`; that re-run is itself now superseded and sits in `saves_legacy/gen2_lossfix_oldprobe/`. Results in `FINDINGS.md` §3.

Also done since this list was written, and no longer repeated below: checkpoints
(`checkpoint.pt`) + `testing/reprobe.py`, the per-run `config.json` sidecar, `done.json` as the
completion marker, `batch_size` actually forwarded to the loaders, `eval()`/`no_grad()` around
`validate` and `make_encodings`, a linear (logistic-regression) probe, a separate
fit/select/report probe split, and the collapse detector.

**The gen3 sweep is finished (September 2026).** `saves/CIFAR_10/` holds **599 completed runs**:
7 architectures (`conv_big_z`, `dcgan`, `dcgan_gdn`, `resnet18`, `vae`, `vit`, `vit_wu`), the
full loss axis on `conv_big_z` and `dcgan`, 3 seeds per trained cell, both optimisation budgets,
`RANDOM` and `uniform` controls throughout, plus the AdaBN re-probes. **The results are in
`FINDINGS.md` §5** and in the Overleaf write-up `percept-reduce-encoders`; §5 supersedes §1, §3
and §4 of that file. Items below that the sweep closed are marked and kept for the record.

**Two results from the first half of that re-run changed the priorities below, and are kept
because they explain the ordering:**

1. **The `LPIPS` fix made LPIPS *worse*** — the uncorrected `LPIPS1` beats it by 0.055 at 100%
   data under an identical init. The most likely reason is that halving the input contrast
   shrank the gradient and acted as an implicit LR reduction. That makes **T1.1 (LR) a
   prerequisite for interpreting the loss axis at all**, not just a `DISTS` fix. Promote it.
   *(Update: partly overtaken. The `dcgan` runs show the conditioning problem is fixed by
   BatchNorm rather than by tuning the LR — see T1.1. An LR sweep is still the way to confirm the
   mechanism on `conv_big_z`, but it is no longer blocking the loss axis.)*
2. **The `DISTS` collapse pattern completely rearranged when only the seed changed** (the cell
   that worked now fails; one that failed now works). B3 is settled as initialisation-dependent
   optimisation instability. This also means **T1.4 (multi-seed) is not optional polish** — with
   n=1 the grid cannot distinguish a loss effect from an init effect, and it can no longer even
   estimate its own noise floor, because seeding makes every cell's epoch-0 identical (spread
   exactly 0.0000, vs ±0.032 measured across the old unseeded runs).

---

## Tier 1 — blocks any quotable result

### T1.0 — the CIFAR save path does not match the dataset key

Found while re-running. The committed CIFAR results live in **`saves/CIFAR/`**, but the
dataset key is `CIFAR_10`, so `train_saver` writes to **`saves/CIFAR_10/`**. The `CIFAR` path
predates the rename of the `DATA_LOADER` key (commits `d5e420c` / `ef1534e`).

Consequence: **skip-if-exists never matches for CIFAR.** Anyone running `pipeline_CIFAR.py`
today silently re-runs the entire grid from scratch instead of resuming, and writes it to a
second directory. `IMAGENET64_VAL` is unaffected — its key and save path agree.

- [x] The current re-run went to `saves/CIFAR_10/` (correct). The legacy `saves/CIFAR/` grid
      was archived to `saves_legacy/gen1_original/`.
- [ ] Add an assertion that the save-path dataset component is a key of `DATA_LOADER`, so a
      future rename fails loudly instead of orphaning a results directory.

### T1.1 — the learning rate is wrong, and it is contaminating the loss axis (bug B3)

*Answered — the cause is normalisation, not the learning rate. `FINDINGS.md` §5.7 has the
evidence at three seeds; the remaining `lr0.0001` box below confirms a mechanism rather than
unblocking anything. Demoted from the top of Tier 1.*

Originally: three of five CIFAR `DISTS` cells collapse to a constant output within one epoch,
MLP pinned at exactly 0.1007 (single-class prediction), val MSE flat at ~0.09.

**Confirmed as an optimisation failure, not a data effect.** Re-running with only the seed
changed rearranged the pattern entirely — 50% went from collapsed to 0.5007, 100% went from
0.5202 to collapsed. Nothing about the data changed.

Two further symptoms point at the same cause, all at Adam's untuned default LR of 1e-3:

- `LPIPS` also collapses at 1% and on `uniform`, and peaks at epoch 3 before degrading at 1%.
- The *weaker-gradient* `LPIPS1` beats the correct `LPIPS` at 50% and 100% data — consistent
  with the correct loss simply being too strong for this LR.

So the loss comparison currently confounds "which loss gives better representations" with
"which loss happens to be well-conditioned at 1e-3". An LR sweep is a prerequisite for the
headline table, not a side quest.

**Answered by the `dcgan` runs — and the answer was normalisation, not LR.** Over the full
sweep, `dcgan` collapses **0 of 168 runs** against `conv_big_z`'s 43 of 168 on the same losses,
data and LR; the two differ only in BatchNorm and LeakyReLU. Replacing BatchNorm with GDN brings
collapse back (4/48) *and* lowers the LPIPS ceiling to 0.491 against `dcgan`'s 0.586, so
normalisation is not merely suppressing an instability — it raises what the encoder can reach
(§5.7). The original single-seed evidence: all five
`DISTS`×data-size cells on `dcgan` (`saves/CIFAR_10/dcgan/DISTS/*`, seed 42, same 1e-3 LR)
trained to 30 epochs with no collapse, `out std` ~0.23 throughout, and a clean monotone
data-size trend: MLP **0.444 / 0.503 / 0.538 / 0.543** at 1 / 10 / 50 / 100%, `uniform` 0.442.
`LPIPS` at 1% likewise shows no epoch-3-peak-then-degrade — it rises to 0.455 and ends at its
best. So B3 was an optimisation failure that BatchNorm fixes, and the LR sweep is no longer a
prerequisite for the loss axis on a normalised backbone.

- [x] Try `dcgan` — it is `conv_big_z` plus BatchNorm, and a first-epoch collapse is what
      normalisation prevents. Done, and it prevents it (numbers above).
- [ ] Still open for `conv_big_z` itself: re-run `DISTS`/`LPIPS` there with a lower LR (try 1e-4)
      and/or a short warmup, to confirm the mechanism rather than just routing around it.
      **Unblocked**: `lr` is an argument to `train()` and a run axis (`'lr': [1e-3, 1e-4]` in the
      runs dict), landing in separate directories. No `lr0.0001` run exists yet.
- [x] Add a collapse detector: if the loss is flat and the output variance across a batch
      collapses, abort and mark the run rather than writing 30 epochs of a dead network. Done —
      `train()` aborts when the batch-wise output std stays below `collapse_tol` (1e-4) for
      `collapse_patience` (3) epochs, `done.json` records `collapsed: true`, and `out std` is a
      CSV column so a collapse is visible without opening the images.
- [x] Consider whether the same fragility affects `LPIPS` at 1% — it also peaks at epoch 3
      (0.222) and then degrades to 0.195, which looks like the beginning of the same failure.
      Same answer: on `dcgan` the degradation is gone (1%: 0.439 @ ep1 → 0.455 final, best at the
      end). Same fragility, same fix. Still unverified on `conv_big_z`.

### T1.2 — data size is confounded with optimisation budget (bug B4) — CLOSED

`percept_loss/pipeline/generic.py`:

```python
scaled_epochs = int(epochs/data_percent)
scaled_epochs = epochs              # <- immediately overwrites the line above
```

The intended compensation is dead code, so every run gets 30 passes over its own training set:

| data | images | steps/epoch | total gradient steps |
|---|---|---|---|
| 100% | 24,000 | 750 | 22,500 |
| 50% | 12,000 | 375 | 11,250 |
| 10% | 2,400 | 75 | 2,250 |
| 1% | 240 | 8 | **240** |

Every "less data" result is also a "~100× less optimisation" result.

- [x] Add an explicit `budget` option: `'epochs'` (current) vs `'steps'` (fixed gradient steps
      across data sizes). Done as `run(..., epoch_scaling=)`: `'fixed'` (the default, and what
      every committed run used) or `'equal_steps'` (`int(epochs/data_percent)`). The dead line is
      gone. **Note the pathology the old comment warns about is real** — `equal_steps` at 1% is
      3000 epochs over 240 images, so treat it as a second condition to compare against, not as
      the corrected version.
- [x] Run the headline CIFAR grid under both budgets. **The difference between the two grids is
      the data-efficiency result** — this is the single most valuable experiment on this list.
      Both grids are now in `saves/CIFAR_10/conv_big_z/` at three seeds: 35 fixed-budget cells
      and 21 equal-steps cells per seed (`0.01`/`0.1`/`0.5` only — at `1` and `uniform` the
      scaling factor is 1, so those cells are shared, see B15/B16).
- [x] **Read the comparison. Done — the confound is real but small.** Paired by seed, equalising
      gradient steps moves every `dcgan` cell by **at most 0.020**, and the largest single move
      (MSSIM at 10%, +0.020) is inside its own seed spread of 0.019. LPIPS given 10× the epochs
      on 10% of the data reaches 0.527, still 0.059 short of the 0.586 it gets from all of it.
      **So the data-size effect is distinct images, not optimisation budget.** Full table in
      `FINDINGS.md` §5.4. This closes B4 and T1.2. (The readers pooled the two budgets until B16
      was fixed — any comparison made before that fix needs redoing.)
- [x] Record the budget mode in the run directory name or a config sidecar. The scaled epoch
      count is in the directory (`_ep30` vs `_ep3000`) so the two budgets cannot collide, and
      `config.json` records the resolved value.

### T1.3 — the ImageNet64 probe is measuring nothing

The ImageNet64 grids evaluate **1,000 classes on a 4,950-instance probe eval set** — ~5 examples
per class, probe fit on ~10 per class. Chance is 0.001; only 20 of 60 runs exceed 0.01 and the
best across all 60 is 0.0218. (Run counts have moved since: `saves/IMAGENET64_VAL/` now holds
**40** runs, all still in the legacy directory layout, with 60 more archived under
`saves_legacy/`. The measurement problem is unchanged.)

The figures used to look like they contained signal because `plot_training_runs.py` hardcoded
`ylim=[0, 0.1]` for ImageNet, an 8× zoom relative to the CIFAR panels.

- [ ] Pick a fix, cheapest first: (a) subsample to 50–100 ImageNet classes; (b) use
      `IMAGENET64_TRAIN` so the probe gets a real number of examples per class; (c) report
      top-5 and class-balanced accuracy.
- [ ] Until one lands, treat every committed ImageNet64 run as null — the 40 in
      `saves/IMAGENET64_VAL/` and the 60 archived. Do not cite them.
- [x] Fix the hardcoded y-limits so the panels are not visually misleading. Done — both
      `plot_training_runs.py` and `plot_data_efficiency.py` now only set `ylim(bottom=0)` and let
      the top autoscale; no dataset-specific limit anywhere in `plots/`.

### T1.4 — seeds and error bars (bug B5)

Per-run seeding now exists, but there is still exactly **one seed per cell**. Measured spread at
epoch 0 across nominally identical random-init networks: CIFAR MLP **0.128 ± 0.032**. That noise
floor is larger than the entire SSIM and NLPD data-size trends.

- [ ] Sweep ≥5 seeds per cell for the headline CIFAR loss × data-size grid; report mean ± sd.
      **Still open, and now the main Tier 1 item.** The sweep finished at `SEEDS = [42, 1, 2]`,
      i.e. n=3, not the ≥5 asked for here; only the `conv_big_z` `RANDOM` control reached 5.
      n=3 is enough for the large effects — the loss, data-size and normalisation results in
      `FINDINGS.md` §5.2–§5.4 and §5.7 all exceed 0.1 — but **not** to rank losses differing by
      0.02–0.04, which is exactly the SSIM/MSSIM/NLPD band. Extend the seed list before quoting
      any ranking in that range.
- [x] Put the seed in the run directory name (it was invisible, so multi-seed runs would have
      overwritten each other). Seeds are a run axis — `'seed': [1, 2, 3]` — and land in
      `.../bs32_seed1_lr0.001_ep30/`, with the seed also in `config.json`.
- [x] Re-measure the ±0.032 noise floor. Done via the 5-seed `RANDOM` control on the current
      probe: **±0.007 MLP**, ±0.010 KNN/Linear, ±0.021 NB. The old figure mixed init variance with
      probe under-fitting. Per-metric values live in `plots/run_index.py` and drive the shaded
      band in the figures. Caveat: this is *init* variance only — it does not include the
      shuffling variance a trained run also carries, so it is a lower bound for trained cells.
- [ ] Re-measure the band on a second architecture. It is drawn on every figure for all seven
      backbones but measured on `conv_big_z` alone, which is the least stable of them — this is
      the weakest number the figures rely on.

### T1.5 — every §1 number predates probe standardisation (bug B13)

Probe features were never standardised, and `MLPClassifier(alpha=1)` on raw `Tanh` latents was
badly under-fit. On an untrained `dcgan`, adding a `StandardScaler` moves MLP accuracy from
**0.162 to 0.421** while leaving `NB` identical and `KNN` almost unchanged. The MLP probe is the
headline metric of the project.

The scaler is now in `test_all_classifiers`, so this affects interpretation rather than code.

- [x] Re-measure the untrained baseline on `conv_big_z`. Done, 5 seeds: **MLP 0.4112 ± 0.0071**
      against §1's 0.128 ± 0.032, with `KNN` and `NB` reproducing §1 almost exactly. The bar moved
      by ~0.28 and `SSIM`/`NLPD`/`MSSIM`/`MSE` are all now close enough to it that beating an
      untrained encoder is an open question. Table in `FINDINGS.md` B13.
- [x] Re-read every "beats the untrained baseline" claim in `FINDINGS.md` against the new number.
      Done by replacement rather than by patching: `FINDINGS.md` §5 is recomputed from all 599
      runs and supersedes §1, §3 and §4 wholesale, with the per-architecture untrained control
      as the bar in every table.
- [x] Do not assume the loss *ranking* survives. It did not. On the corrected probe the
      hand-designed losses (SSIM, MSSIM, NLPD) sit within ~0.02 of their own untrained control
      at every data size, so §1's "hand-designed priors substitute for data" reading is gone —
      a flat curve there means the loss is barely moving the representation, not that it is
      data-efficient (§5.3). The learned losses keep their steep curves and win outright.
- [x] The committed runs cannot be re-probed — they have no checkpoints. Only re-running gives
      standardised numbers for them; runs from here on can be re-probed with
      `percept_loss/testing/reprobe.py`. Resolved by decision rather than by fix: the gen2 grid
      was archived to `saves_legacy/gen2_lossfix_oldprobe/` and is being re-run from scratch on
      probe v2 with checkpoints (`pipeline_CIFAR_BIG.py`). `config.json` records `probe_version`,
      so a table can tell it is mixing protocols.

### T1.6 — is LPIPS's advantage ImageNet supervision leaking through the loss?

*New, and the highest-value single experiment on this list — it decides how every result in
`FINDINGS.md` §5 is read.*

The completed sweep says only the VGG-feature losses clearly beat a pixel loss (§5.2), and that
they are the ones that convert data into representation quality (§5.3). But LPIPS and DISTS are
computed on features from a network **trained to classify ImageNet**, and our probe then measures
class information. So the advantage may not be perceptual at all: it may be supervision reaching
an unsupervised model through its loss function, from data that model never trains on. Nothing
run so far separates the two, and §5.8 sharpens the question rather than answering it — LPIPS
trained on *pure noise* still gains +0.053 over untrained after AdaBN, and there the loss network
is the only route natural-image information can take.

- [ ] Train with LPIPS computed on a **randomly initialised** VGG — same architecture, no
      ImageNet training. One entry in the loss registry (**no `-` in the name**) and ~24 runs on
      `dcgan`. If the advantage mostly survives, it is a patch-statistics prior and the claim is
      "learned perceptual losses make better use of every image". If it mostly vanishes, it is a
      supervision channel — the more interesting result.
- [ ] Add two reference probes on the same axes: frozen ImageNet-VGG features of the images
      themselves (an upper bound on what could leak), and the same on a random VGG.
- [ ] Only if leakage survives, characterise the channel: supervised vs self-supervised VGG of
      the same architecture (does it follow the *labels*?); early vs late VGG layers (dose);
      probe classes with no ImageNet counterpart (overlap); and shorter routes for the same
      teacher signal — a decoder that outputs VGG features instead of pixels, and a FitNets-style
      head on the latent.

Either outcome is publishable, and §5.1, §5.6 and §5.7 carry over as a methods contribution on
auditing the frozen-probe protocol regardless. The full fork, with what each account predicts for
each follow-up, is in the Overleaf write-up `percept-reduce-encoders` (Appendix B).

---

## Tier 2 — one small change away

### T2.1 — a proper baseline panel

Every headline claim needs these on the same axes and none is currently plotted:

- [x] untrained random encoder — now runnable as a first-class condition: put `'RANDOM'` in the
      loss slot and `pipeline/generic.py` takes zero gradient steps, writing
      `.../RANDOM/{datasize}/bs32_seed{s}_lr0.001_ep0/` with one row at epoch 0. Verified to
      reproduce the epoch-0 row of a trained run of the same network exactly (same seed, same
      init). Now also plotted correctly — `plot_data_efficiency.py` draws it as a dotted
      `axhline` at the mean, not a one-point series;
- [ ] raw pixels, no encoder — requires fixing `testing/baseline_performance.py` (see **B11** in
      Engineering; the old cross-reference to T3.3 was wrong). It still fails at import on
      `get_all_loaders_CIFAR`;
- [ ] random projection to the same `latent_dim` — ~2 lines of sklearn.

Motivation, restated on the corrected probe: the untrained encoder is **0.34–0.44** depending on
architecture, which is most of what any trained cell scores (§5.1), and on three of seven
backbones training on pure noise is indistinguishable from not training at all (§5.5). Both
controls now exist. What is still missing is the *floor beneath them* — raw pixels and a random
projection — so "untrained encoder" cannot yet be separated from "no encoder at all". Until it
is, the architectural prior is measured but not decomposed.

*(The original motivation quoted here — `MSE`+`uniform` reaching 91% of MSE on real data — was a
probe-v1 number and is superseded. The effect was mostly MSE barely moving the representation;
see §5.5.)*

### T2.2 — match the `uniform` control

`UNIFORM_LOADER` draws a **fresh** `torch.rand` on every `__getitem__`, so a 30-epoch run sees
~150,000 distinct noise images and never repeats one, against a hardcoded 5,000/epoch that
corresponds to no real-data condition.

- [ ] Make it a fixed, seeded set of N images, sized to match each real-data condition.
- [ ] Add intermediate controls: pixel-shuffled real images, and real images from a *different*
      dataset. Turns the noise finding from an anomaly into a measurement of how much of the
      representation is distribution-specific.

### T2.3 — sweep the architecture axis again — LARGELY DONE

`plots/legacy_figs/` shows a four-architecture CIFAR sweep whose CSVs no longer exist. Latent
size directly changes *probe* capacity, so it is confounded with representation quality in every
current comparison.

- [x] Re-run the architecture sweep with the fixed losses. Done: seven configurations at three
      seeds — `conv_big_z`, `dcgan`, `dcgan_gdn`, `resnet18`, `vae`, `vit`, `vit_wu` — all at a
      384-dim latent, in `FINDINGS.md` §5. The two normalisation variants (`dcgan_gdn`,
      `vit_wu`) were added specifically to vary normalisation while holding the backbone fixed,
      and they are what turned B3 from "settled" into a measured mechanism (§5.7).
- [x] Include a fixed-latent-size comparison across losses to separate the two effects — five
      literature backbones (`dcgan`, `resnet18`, `resnet18_thin`, `vit`, `vae`) are registered,
      **all at a 384-dim latent**, matching `conv_big_z`. Citations and the reasoning for each are
      in `percept_loss/networks/README.md`. Run them with `percept_loss/pipeline_CIFAR_ARCH.py`
      (kept separate from `pipeline_CIFAR.py` so the committed grid is untouched and the two can
      run concurrently).
- [x] `dcgan` is `conv_big_z` + BatchNorm/LeakyReLU and nothing else, so `dcgan` vs `conv_big_z`
      answers T1.1 as a side effect: if `DISTS` no longer collapses there, B3 was optimisation
      rather than the loss, and no LR sweep is needed. **It does not collapse** — all five
      `DISTS` cells trained cleanly. See T1.1 for the numbers. Note this also makes `dcgan` the
      better default backbone for the headline grid than `conv_big_z`.
- [x] `resnet18` is the point of the exercise — it is the backbone SimCLR/BYOL/SimSiam report
      CIFAR-10 probe accuracy on, so it is what makes our numbers comparable to published ones.
      Done, 45 non-collapsed runs: it has both the **strongest untrained control (0.443)** and
      the **best trained cell in the study (LPIPS at 100%, 0.683 ± 0.005)**. `resnet18_thin` is
      registered but unrun.
- [ ] The loss axis is still only complete on `conv_big_z` and `dcgan`; the other five run
      MSE/SSIM/LPIPS/DISTS only, and 50% data is `conv_big_z`/`dcgan` only. Fill those in before
      any claim that needs a full loss × architecture table.

### T2.6 — the VAE's `beta` is a guess

`networks/vae.py` stores the KL divided by the pixel count so it lands ~0.02 at init, the same
scale as the reconstruction losses, and sets `beta = 1.0` on that basis. That is a defensible
starting point and nothing more — the reconstruction/KL balance is exactly the knob that decides
whether the latent is informative or posterior-collapsed, and it interacts with the loss axis
(the perceptual losses do not all return the same magnitude).

- [ ] Sweep `beta` over ~3 decades on one loss before reading anything into the VAE rows.
      The completed sweep makes this more interesting, not less: the `vae` is the **only**
      architecture whose reconstruction/probe correlation stays strongly positive with data size
      held fixed (+0.47, against ≤|0.30| everywhere else, §5.6), and it is the only one where
      the best-reconstructing loss is SSIM rather than MSE. Both could be a β artefact.
- [x] Log the KL term separately in the CSV — a collapsed posterior and a working one are
      indistinguishable from probe accuracy alone. Done: `train()` accumulates `net.kl` per epoch
      and writes a `KL` column for any net that has the attribute; `beta` is in `config.json`.

### T2.4 — longer training for the learned perceptual losses

`LPIPS` at 100% peaks at epoch 23 of 30 and `DISTS` at epoch 27 of 30 — both still improving
when the run ends — while `SSIM`/`NLPD`/`MSE` peak early. The 30-epoch budget probably
understates the learned losses.

- [ ] Run the top configs to 100+ epochs. Weaker on `dcgan` than it was on `conv_big_z`: at 100%
      data `LPIPS` plateaus by epoch ~19 (0.606) and ends at 0.598, and `DISTS` is flat from ~23.
      Check this again on the gen3 `conv_big_z` runs before spending the compute. Note the
      completed sweep gives this a reason to exist that it did not have before: on `resnet18`,
      LPIPS is the one cell still climbing steeply with data (+0.241 from 1% to 100%, §5.3), so
      30 epochs is most likely to be understating *that* cell specifically.

### T2.5 — `NLPD` uses 1 of its 6 pyramid levels (bug B6)

`NLPD(nlpd_k=1)`, but `DN_filters()` defines six levels of divisive-normalisation filters and
sigmas. Verified maximum usable `k`: **3 at 32px, 5 at 64px**; `k=6` fails at both. The
reference implementation uses `k=6`.

- [ ] Sweep `k` ∈ {1, 2, 3} on CIFAR. Register as separate loss names (`NLPD`, `NLPD2`,
      `NLPD3`) so the runs are distinguishable on disk — **no `-` in the names**. Re-verified
      untouched: `losses/__init__.py` registers only `'NLPD'`, still `NLPD(nlpd_k=1)`.
- [ ] Note NLPD currently peaks at epoch 1–3 then *degrades* at 50%/100% data (0.398 @ ep1 →
      0.339 final at 100%). Its "flat across data size" result is partly "its best number is
      roughly its epoch-1 number". Re-check this with a working `k`.
- [ ] This matters for how §5.2 is stated. NLPD never beats its own initialisation on
      `conv_big_z`, and the hand-designed losses as a group sit within ~0.02 of untrained. That
      is currently reported as a property of hand-designed losses, but for NLPD it is confounded
      with running a single-scale version of a six-scale metric. Either fix `k` or say so
      explicitly wherever the group claim is made.

---

## Tier 3 — new directions the codebase supports cheaply

### T3.1 — loss combinations

- [ ] `α·MSE + (1−α)·perceptual` over a small α sweep. ~10-line change to the loss registry.
      **The original rationale is void** — it rested on the probe-v1 reading that hand-designed
      perceptual losses were data-efficient but capped around 0.42, and on the corrected probe
      they are not capped so much as barely moving the representation at all (§5.3). A mixture
      with a loss that does nothing is unlikely to beat either. The version still worth running
      is `α·MSE + (1−α)·LPIPS`: LPIPS wins on the probe while reconstructing worst (§5.6), so
      the sweep measures what that trade actually costs.

### T3.2 — vary the probe, not just the encoder

KNN and MLP already disagree sharply: on the probe-v1 grid, 15 of 30 CIFAR runs finished with
worse KNN accuracy than the untrained network while MLP improved. "Representation quality" is
probe-dependent and that disagreement is itself a result. The corrected probe adds a second
case: `vit` is the one architecture whose untrained latent is read **better linearly (0.407)
than by the MLP (0.344)**, so which probe you pick changes the architecture ranking, not just
the scores (§5.1). Worth recomputing the KNN-degradation count over the 599-run set.

- [x] Add a linear probe (the standard self-supervised metric). Done — `Linear` is
      `LogisticRegression(max_iter=2000)` in `test_all_classifiers`, reported as its own column
      (and `Linear select`). This is the column that makes results comparable to SimCLR/BYOL.
- [ ] k-NN at several k — still a single `KNeighborsClassifier()` at the sklearn default k=5.
- [ ] Report the probe's own train/test gap, not just accuracy.

### T3.3 — encoder transfer

- [ ] Train on CIFAR, probe on a different dataset. If perceptual losses encode a general
      visual prior they should transfer better than MSE. The loaders already make this easy.

### T3.4 — report every metric for every run

Only val MSE is logged, so a run trained with SSIM is never evaluated by SSIM.

- [ ] Log all metrics at eval time (they are cheap). The full cross-metric table comes free,
      including a direct check on whether the losses even agree about reconstruction quality.

---

## Engineering

- [x] **B9** — wrap `validate()` and `make_encodings()` in `torch.no_grad()`, and add
      `net.eval()` / `net.train()`. Done, and no longer hypothetical: every backbone added since
      (`dcgan`, `resnet18`, `vit`, `vae`) has BatchNorm. Both restore the previous mode.
- [x] **B5** — save network weights. Done — `saver.save_checkpoint()` writes `checkpoint.pt`
      (state dict + network name + latent dim), and `testing/reprobe.py` consumes it.
- [x] **B7/B8** — write a JSON config sidecar per run. Done: `config.json` carries dataset,
      network, loss, datasize, batch size, seed, lr, resolved epochs, split props,
      `validate_every`, latent dim, param count, `probe_version`, `beta`, train-set size, plus
      git sha / torch / python for provenance. `batch_size` is now actually forwarded to
      `get_all_loaders`. **Partial**: split proportions are still *selected* by `validate_every`
      and are still not part of the run path, so two runs with different splits share a
      directory — `write_config` prints a warning when it sees that rather than preventing it.
- [x] **B10** — `previously_done` checks only that `training_results.csv` *exists*. Done
      differently and better: `done.json` is written last and is what "done" means; a CSV with no
      `done.json` is moved to `training_results.csv.partial-*` and the cell re-runs. Legacy
      directories are still trusted by CSV alone, but only for `seed=42, lr=1e-3`.
- [ ] **B11** — fix or delete `training/benchmark.py` and `testing/baseline_performance.py`;
      both fail at import (`get_all_loaders_CIFAR`, `random_forest_test` no longer exist). The
      second is the only source of the raw-pixel baseline in `saves/readme.md`, whose 10%-data
      figure (0.0421) is *below* chance for 10 classes and so is certainly wrong. It also blocks
      T2.1, which the untrained-encoder result (§5.1) has made worth having.
- [ ] **B12** — all three re-verified as still present: mutable default `image_dict={}`
      (`CIFAR_10/loader.py:18`, `IMAGENET/ImageNet64.py:22`); `datasets/CIFAR_10/__init__,py`
      still has the comma (no longer a packaging bug — the hatchling build in `pyproject.toml`
      takes the whole tree where `find_packages()` skipped the directory — but still misnamed);
      ImageNet64 `_get_labels` still writes `one_hot[label-1]`
      (`ImageNet64.py:94`) but returns `label` unshifted. The last one is harmless only because
      the probe uses `data[2]`, never the one-hot.
- [ ] LPIPS is a stateful `torchmetrics.Metric`; calling it in the training loop accumulates
      state and does roughly double the necessary work per step. Numerically harmless, but it
      is the slowest loss in the grid.
- [x] Best-epoch selection happens over 16 evals **on the same eval set**, inflating every
      reported number by an unmeasured amount. Done, both ways: the probe now cuts its encodings
      67/16.5/16.5 into fit/select/report and writes a `{name} select` column, and
      `summarise_runs.py` defaults to final-epoch (`--select early_stop` uses the disjoint select
      split). Runs predating this have no `select` column and fall back to `NaN` for that mode.
- [ ] Add assertions: split sizes match expectation, encoder output shape matches `latent_dim`,
      no `-` in any registry key, and a 2-batch overfit test per loss (loss must go to ~0).
      **The last one alone would have caught B1, B2 and B3.** Still nothing — there is not one
      `assert` in first-party code outside the vendored `NLPD_torch/`. Fold T1.0's
      dataset-key assertion in here.
