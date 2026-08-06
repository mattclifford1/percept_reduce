# TODO

Open items from the `FINDINGS.md` audit. Evidence and measurements for each are in that file —
this is the action list. Ordered by how much each blocks a publishable claim.

**Done and not repeated here:** the `MSSIM` no-op (B1) and `LPIPS` input-range (B2) bugs are
fixed, `LPIPS1` was added to preserve the pre-fix LPIPS behaviour as a measurable control, and
per-run seeding was added so runs differing only in loss share an identical network init. The
full 35-cell CIFAR grid was re-run and lives in `saves/CIFAR_10/`; all pre-fix runs are
archived in `saves_pre_lossfix/`. Results in `FINDINGS.md` §3.

**Two results from that re-run change the priorities below:**

1. **The `LPIPS` fix made LPIPS *worse*** — the uncorrected `LPIPS1` beats it by 0.055 at 100%
   data under an identical init. The most likely reason is that halving the input contrast
   shrank the gradient and acted as an implicit LR reduction. That makes **T1.1 (LR) a
   prerequisite for interpreting the loss axis at all**, not just a `DISTS` fix. Promote it.
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
      was archived to `saves_pre_lossfix/`.
- [ ] Add an assertion that the save-path dataset component is a key of `DATA_LOADER`, so a
      future rename fails loudly instead of orphaning a results directory.

### T1.1 — the learning rate is wrong, and it is contaminating the loss axis (bug B3)

*Promoted — this is now the highest-value single experiment. See the note at the top.*

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

- [ ] Re-run `DISTS` with a lower LR (try 1e-4) and/or a short warmup. **Unblocked**: `lr` is now
      an argument to `train()` and a run axis (`'lr': [1e-3, 1e-4]` in the runs dict), landing in
      separate directories. Try `dcgan` at the same time — it is `conv_big_z` plus BatchNorm, and
      a first-epoch collapse is what normalisation prevents.
- [x] Add a collapse detector: if the loss is flat and the output variance across a batch
      collapses, abort and mark the run rather than writing 30 epochs of a dead network. Done —
      `train()` aborts when the batch-wise output std stays below `collapse_tol` (1e-4) for
      `collapse_patience` (3) epochs, `done.json` records `collapsed: true`, and `out std` is a
      CSV column so a collapse is visible without opening the images.
- [ ] Consider whether the same fragility affects `LPIPS` at 1% — it also peaks at epoch 3
      (0.222) and then degrades to 0.195, which looks like the beginning of the same failure.

### T1.2 — data size is confounded with optimisation budget (bug B4)

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
- [ ] Run the headline CIFAR grid under both budgets. **The difference between the two grids is
      the data-efficiency result** — this is the single most valuable experiment on this list.
- [x] Record the budget mode in the run directory name or a config sidecar. The scaled epoch
      count is in the directory (`_ep30` vs `_ep3000`) so the two budgets cannot collide, and
      `config.json` records the resolved value.

### T1.3 — the ImageNet64 probe is measuring nothing

Both ImageNet64 grids (60 runs) evaluate **1,000 classes on a 4,950-instance probe eval set** —
~5 examples per class, probe fit on ~10 per class. Chance is 0.001; only 20 of 60 runs exceed
0.01 and the best across all 60 is 0.0218.

The figures look like they contain signal only because `plot_training_runs.py` hardcodes
`ylim=[0, 0.1]` for ImageNet, an 8× zoom relative to the CIFAR panels.

- [ ] Pick a fix, cheapest first: (a) subsample to 50–100 ImageNet classes; (b) use
      `IMAGENET64_TRAIN` so the probe gets a real number of examples per class; (c) report
      top-5 and class-balanced accuracy.
- [ ] Until one lands, treat the 60 committed ImageNet64 runs as null. Do not cite them.
- [ ] Fix the hardcoded y-limits so the panels are not visually misleading.

### T1.4 — seeds and error bars (bug B5)

Per-run seeding now exists, but there is still exactly **one seed per cell**. Measured spread at
epoch 0 across nominally identical random-init networks: CIFAR MLP **0.128 ± 0.032**. That noise
floor is larger than the entire SSIM and NLPD data-size trends.

- [ ] Sweep ≥5 seeds per cell for the headline CIFAR loss × data-size grid; report mean ± sd.
      This is now only a compute question.
- [x] Put the seed in the run directory name (it was invisible, so multi-seed runs would have
      overwritten each other). Seeds are a run axis — `'seed': [1, 2, 3]` — and land in
      `.../bs32_seed1_lr0.001_ep30/`, with the seed also in `config.json`.
- [ ] Re-measure the ±0.032 noise floor. It was measured on **unstandardised** probe features and
      is not a number about the current probe (B13). The multi-seed `RANDOM` control gives it
      directly and costs seconds per cell.

### T1.5 — every §1 number predates probe standardisation (bug B13)

Probe features were never standardised, and `MLPClassifier(alpha=1)` on raw `Tanh` latents was
badly under-fit. On an untrained `dcgan`, adding a `StandardScaler` moves MLP accuracy from
**0.162 to 0.421** while leaving `NB` identical and `KNN` almost unchanged. The MLP probe is the
headline metric of the project.

The scaler is now in `test_all_classifiers`, so this affects interpretation rather than code.

- [ ] Re-measure the untrained baseline on `conv_big_z` — one `RANDOM` cell, seconds of compute.
      §1 quotes 0.128 ± 0.032 and the scaled figure will be far higher, which may put the
      baseline within reach of the trained `MSE` result (0.459).
- [ ] Re-read every "beats the untrained baseline" claim in `FINDINGS.md` against the new number.
- [ ] Do not assume the loss *ranking* survives. Under-fitting penalises whichever latents are
      hardest to fit, so this is not a constant offset across the grid.
- [ ] The committed runs cannot be re-probed — they have no checkpoints. Only re-running gives
      standardised numbers for them; runs from here on can be re-probed with
      `percept_loss/testing/reprobe.py`.

---

## Tier 2 — one small change away

### T2.1 — a proper baseline panel

Every headline claim needs these on the same axes and none is currently plotted:

- [x] untrained random encoder — now runnable as a first-class condition: put `'RANDOM'` in the
      loss slot and `pipeline/generic.py` takes zero gradient steps, writing
      `RANDOM-{datasize}-BS32/` with one row at epoch 0. Verified to reproduce the epoch-0 row of
      a trained run of the same network exactly (same seed, same init). Still needs plotting as a
      horizontal reference line rather than a one-point series;
- [ ] raw pixels, no encoder — requires fixing `testing/baseline_performance.py` (see T3.3);
- [ ] random projection to the same `latent_dim` — ~2 lines of sklearn.

Motivation: `MSE`+`uniform` (trained only on noise) reaches **0.418** vs **0.459** for MSE on all
24,000 real images — **91%**, from an autoencoder that has never seen a photograph.
`MSSIM`+`uniform` (0.385) actually beat `MSSIM` on real data (0.380). Some of the loss ranking
may be architecture rather than loss.

### T2.2 — match the `uniform` control

`UNIFORM_LOADER` draws a **fresh** `torch.rand` on every `__getitem__`, so a 30-epoch run sees
~150,000 distinct noise images and never repeats one, against a hardcoded 5,000/epoch that
corresponds to no real-data condition.

- [ ] Make it a fixed, seeded set of N images, sized to match each real-data condition.
- [ ] Add intermediate controls: pixel-shuffled real images, and real images from a *different*
      dataset. Turns the noise finding from an anomaly into a measurement of how much of the
      representation is distribution-specific.

### T2.3 — sweep the architecture axis again

`plots/legacy_figs/` shows a four-architecture CIFAR sweep whose CSVs no longer exist. Latent
size directly changes *probe* capacity, so it is confounded with representation quality in every
current comparison.

- [ ] Re-run the architecture sweep with the fixed losses.
- [x] Include a fixed-latent-size comparison across losses to separate the two effects — five
      literature backbones (`dcgan`, `resnet18`, `resnet18_thin`, `vit`, `vae`) are registered,
      **all at a 384-dim latent**, matching `conv_big_z`. Citations and the reasoning for each are
      in `percept_loss/networks/README.md`. Run them with `percept_loss/pipeline_CIFAR_ARCH.py`
      (kept separate from `pipeline_CIFAR.py` so the committed grid is untouched and the two can
      run concurrently).
- [ ] `dcgan` is `conv_big_z` + BatchNorm/LeakyReLU and nothing else, so `dcgan` vs `conv_big_z`
      answers T1.1 as a side effect: if `DISTS` no longer collapses there, B3 was optimisation
      rather than the loss, and no LR sweep is needed.
- [ ] `resnet18` is the point of the exercise — it is the backbone SimCLR/BYOL/SimSiam report
      CIFAR-10 probe accuracy on, so it is what makes our numbers comparable to published ones.
      15M params; use `resnet18_thin` (4M) if the 35-cell wall time is too long.

### T2.6 — the VAE's `beta` is a guess

`networks/vae.py` stores the KL divided by the pixel count so it lands ~0.02 at init, the same
scale as the reconstruction losses, and sets `beta = 1.0` on that basis. That is a defensible
starting point and nothing more — the reconstruction/KL balance is exactly the knob that decides
whether the latent is informative or posterior-collapsed, and it interacts with the loss axis
(the perceptual losses do not all return the same magnitude).

- [ ] Sweep `beta` over ~3 decades on one loss before reading anything into the VAE rows.
- [ ] Log the KL term separately in the CSV — a collapsed posterior and a working one are
      indistinguishable from probe accuracy alone.

### T2.4 — longer training for the learned perceptual losses

`LPIPS` at 100% peaks at epoch 23 of 30 and `DISTS` at epoch 27 of 30 — both still improving
when the run ends — while `SSIM`/`NLPD`/`MSE` peak early. The 30-epoch budget probably
understates the learned losses.

- [ ] Run the top configs to 100+ epochs.

### T2.5 — `NLPD` uses 1 of its 6 pyramid levels (bug B6)

`NLPD(nlpd_k=1)`, but `DN_filters()` defines six levels of divisive-normalisation filters and
sigmas. Verified maximum usable `k`: **3 at 32px, 5 at 64px**; `k=6` fails at both. The
reference implementation uses `k=6`.

- [ ] Sweep `k` ∈ {1, 2, 3} on CIFAR. Register as separate loss names (`NLPD`, `NLPD2`,
      `NLPD3`) so the runs are distinguishable on disk — **no `-` in the names**.
- [ ] Note NLPD currently peaks at epoch 1–3 then *degrades* at 50%/100% data (0.398 @ ep1 →
      0.339 final at 100%). Its "flat across data size" result is partly "its best number is
      roughly its epoch-1 number". Re-check this with a working `k`.

---

## Tier 3 — new directions the codebase supports cheaply

### T3.1 — loss combinations

- [ ] `α·MSE + (1−α)·perceptual` over a small α sweep. The observed pattern (perceptual =
      data-efficient but capped ~0.42; MSE = reliable; LPIPS = data-hungry but reaches 0.605)
      suggests a mixture beats either. ~10-line change to the loss registry.

### T3.2 — vary the probe, not just the encoder

KNN and MLP already disagree sharply: **15 of 30 CIFAR runs finish with worse KNN accuracy than
the untrained network**, while MLP improves. "Representation quality" is probe-dependent and
that disagreement is itself a result.

- [ ] Add a linear probe (the standard self-supervised metric) and k-NN at several k.
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

- [ ] **B9** — wrap `validate()` and `make_encodings()` in `torch.no_grad()`, and add
      `net.eval()` / `net.train()`. Both currently build an autograd graph over a whole split
      and discard it. Harmless today (no BatchNorm/Dropout) but a silent correctness bug the
      moment either is added.
- [ ] **B5** — save network weights. A run is currently unreproducible *and* unrecoverable.
- [ ] **B7/B8** — write a JSON config sidecar per run (epochs, LR, seed, split props, batch
      size, budget mode). Two things are currently invisible and dangerous:
      split proportions are selected by `validate_every` (`training/run_and_test.py:86`), and
      `batch_size` is accepted by `pipeline.run()` but never forwarded, so every run is BS32
      regardless of what the pipeline says.
- [ ] **B10** — `previously_done` checks only that `training_results.csv` *exists*. A crashed
      run leaves a partial CSV and is skipped forever. Check row count against expected evals.
- [ ] **B11** — fix or delete `training/benchmark.py` and `testing/baseline_performance.py`;
      both fail at import (`get_all_loaders_CIFAR`, `random_forest_test` no longer exist). The
      second is the only source of the raw-pixel baseline in `saves/readme.md`, whose 10%-data
      figure (0.0421) is *below* chance for 10 classes and so is certainly wrong.
- [ ] **B12** — mutable default `image_dict={}` on both loader classes; rename
      `datasets/CIFAR_10/__init__,py` (comma, not dot — `find_packages()` misses it, so a
      non-editable `pip install .` ships a broken package); ImageNet64 `_get_labels` writes
      `one_hot[label-1]` but returns `label` unshifted.
- [ ] LPIPS is a stateful `torchmetrics.Metric`; calling it in the training loop accumulates
      state and does roughly double the necessary work per step. Numerically harmless, but it
      is the slowest loss in the grid.
- [ ] Best-epoch selection happens over 16 evals **on the same eval set**, inflating every
      reported number by an unmeasured amount. Carve a separate split for epoch selection, or
      report final-epoch only.
- [ ] Add assertions: split sizes match expectation, encoder output shape matches `latent_dim`,
      no `-` in any registry key, and a 2-batch overfit test per loss (loss must go to ~0).
      **The last one alone would have caught B1, B2 and B3.**
