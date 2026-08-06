# Findings

Analysis of the 90 committed runs in `saves/`, the bugs that affect them, and what to run next.

Everything below was recomputed from the CSVs (`~/anaconda3/envs/percept/bin/python`), not read
off the figures. Unless stated, the headline metric is **best-epoch MLP probe accuracy**.

---

## 1. Results

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

The figures in `plots/figs/IMAGENET64_VAL/` look like they contain signal only because
`plot_training_runs.py` hardcodes `ylim=[0, 0.1]` for ImageNet — an 8× zoom relative to the
CIFAR panels. Nothing in these runs is interpretable as it stands.

---

## 2. Confirmed bugs

Ordered by how much they distort the committed results. Each was verified by running the code,
not by inspection alone.

### B1 — `MSSIM` is very nearly a no-op

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

### B2 — `LPIPS` is fed the wrong input range

`percept_loss/losses/perceptual.py:22` — `LearnedPerceptualImagePatchSimilarity(net_type='vgg')`
defaults to `normalize=False`, meaning it expects **`[-1, 1]`**. It is given `[0, 1]`
(`NORMALISE = (0,1)`, decoders end in `Sigmoid`). torchmetrics 1.0.0 does not raise, so this
passed silently. The network only ever exercises half the dynamic range VGG was calibrated on.

**Fix:** `normalize=True` (verified equivalent to manually scaling to `[-1,1]`).

Note LPIPS still wins by a wide margin *despite* this, which makes the result more interesting,
not less — but the LPIPS column needs re-running before it is quotable.

Secondary: `LearnedPerceptualImagePatchSimilarity` is a stateful `torchmetrics.Metric`. Calling
it in the training loop accumulates `sum_scores`/`total` indefinitely and does roughly double
the necessary work per step (verified: `total` grows by 2× batch size per call).

### B3 — three `DISTS` cells are optimisation failures, not data-size effects

`DISTS` at 1%, 10% and 50% collapse to a constant output within one epoch and never recover:
MLP pinned at exactly **0.1007** (single-class prediction) for 10–13 of 16 evals, val MSE flat
at ~0.09. At 100% data the same config escapes after ~7 epochs and reaches 0.527.

This is a dead-unit / degenerate-solution failure at Adam's default LR, not a statement about
data efficiency. **The `DISTS` row of the results table is currently meaningless.** It needs
re-running with a lower LR and/or warmup before it can be included.

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

### B6 — `NLPD` uses 1 of its 6 pyramid levels

`NLPD(nlpd_k=1)`, but `DN_filters()` defines six levels of divisive-normalisation filters and
sigmas. Five are unused, so this is a single-scale metric wearing a multi-scale name.

Verified maximum usable `k`: **3 at 32 px, 5 at 64 px**; `k=6` fails at both (padding exceeds
the downsampled dimension). The reference implementation uses `k=6`.

Related: NLPD peaks at epoch 1–3 and then *degrades* at 50%/100% data (0.398 @ ep1 → 0.339 final
at 100%). Its "flat across data size" result is partly "its best number is roughly its epoch-1
number".

### B7 — split proportions are selected by `validate_every`

`training/run_and_test.py:86` uses `validate_every == 1` as a proxy for "this is the big
ImageNet64 train split" and switches props from `[0.4,0.3,0.3]` to `[0.89,0.1,0.01]`. Changing a
validation *frequency* silently changes the dataset *split*, and the split is not recorded in
the run directory name. Runs with different `validate_every` are not comparable and will
collide on the same save path.

### B8 — `batch_size` is never forwarded

`pipeline.generic.run(batch_size=...)` is used only to name the save directory. It is not passed
to `train()`, which does not pass it to `get_all_loaders()`, which defaults to 32. Every run is
BS32; the `BS32` in the directory names is true by coincidence.

### B9 — no `torch.no_grad()` / `net.eval()` in evaluation

`validate()` and `make_encodings()` both run a full split with autograd live and discard the
graph. Pure waste and the dominant memory cost of eval. `net.eval()` is never called either —
harmless today (no BatchNorm/Dropout anywhere) but a silent correctness bug the moment either
is added.

### B10 — a crashed run is skipped forever

`train_saver.previously_done` is just `os.path.exists(csv_file)`. A killed run leaves a partial
CSV and every subsequent invocation prints `Passing` and moves on. There is no row-count or
completeness check. (All 90 committed runs do have the full 16 rows — checked.)

### B11 — stale scripts that no longer import

- `training/benchmark.py` — imports `get_all_loaders_CIFAR` and `random_forest_test`, neither
  exists.
- `testing/baseline_performance.py` — same broken import, plus it calls `test_all_classifiers`
  with the pre-refactor positional signature.

The second matters: it is the only source of the raw-pixel baseline in `saves/readme.md`, and
that baseline's 10%-data figure (0.0421) is *below* chance for a 10-class problem, so it is
almost certainly wrong and cannot currently be rechecked.

### B12 — smaller latent issues

- Mutable default `image_dict={}` on both real loader classes; masked today because all call
  sites pass an explicit dict, but a live trap.
- `percept_loss/datasets/CIFAR_10/__init__,py` — **comma instead of a dot**. Works only via
  implicit namespace packages plus editable install; `find_packages()` does not see the
  directory, so a non-editable `pip install .` ships a broken package.
- ImageNet64 `_get_labels` writes `one_hot[label-1]` but returns `label` unshifted. Consistent
  today only because nothing reads `data[1]`.
- `validate()` averages per-batch means unweighted, over-weighting the final short batch.
- `get_indicies` documents accepting a list for `total_instances` then computes
  `int(total_instances * prop)`, which raises for a list.
- Reporting best-epoch accuracy selects over 16 evaluations **on the same eval set**, which
  inflates every number in §1 by an unmeasured amount. Fix by carving a proper validation set
  for epoch selection, or by reporting final-epoch only.

---

## 3. What to run next

### Tier 1 — required before any of §1 is quotable

1. **Fix and re-run `MSSIM`, `LPIPS`, `NLPD`, `DISTS`** (B1, B2, B6, B3). Four of six loss rows
   are currently measuring something other than the named metric. Delete the affected `saves/`
   directories first. Cheap: 20 CIFAR runs, and CIFAR runs are fast.

2. **Seeds.** Seed torch/numpy per run and sweep ≥5 seeds per cell, at minimum for the
   headline CIFAR loss × data-size grid. Report mean ± sd. With a ±0.032 noise floor and n=1,
   several current comparisons are not resolvable. Put the seed in the run directory name.

3. **Decouple data size from optimisation budget** (B4). Run the grid twice: once at fixed
   epochs (what exists now) and once at **fixed gradient steps** across data sizes. The
   difference between those two grids *is* the data-efficiency result. This is the single most
   valuable experiment in this list.

4. **Fix the ImageNet64 probe** (§1). At 1,000 classes with ~5 eval examples per class the
   measurement is dead. Options, cheapest first: (a) subsample to 50–100 ImageNet classes;
   (b) use `IMAGENET64_TRAIN` so the probe gets a real number of examples per class; (c) report
   top-5 or class-balanced accuracy. Until one of these lands, the 60 committed ImageNet64 runs
   should be treated as null.

### Tier 2 — experiments the current setup is one small change away from

5. **A proper baseline panel.** Every headline claim needs these three lines on the same axes,
   and none of them is currently reported alongside the results:
   - untrained random encoder (already logged as epoch 0 — just plot it);
   - raw pixels, no encoder (fix `baseline_performance.py`, B11);
   - random projection to the same `latent_dim` (2 lines of sklearn).

   Given uniform-noise MSE reaches 91% of full-data MSE, it is entirely possible some of the
   loss ranking is architecture, not loss.

6. **Match the `uniform` control.** Make the noise dataset a *fixed* set of N images
   regenerated with a seed, sized to match each real-data condition, so "trained on 240 noise
   images" is comparable to "trained on 240 real images". Also add intermediate controls:
   shuffled-pixel real images, and real images from a *different* dataset. This turns the
   noise finding from an anomaly into a measurement of how much of the representation is
   distribution-specific.

7. **Sweep the architecture axis again.** `plots/legacy_figs/` shows a four-architecture CIFAR
   sweep whose CSVs no longer exist. Latent size directly changes probe capacity, so it is
   confounded with representation quality in every current comparison — a fixed-latent-size
   comparison across losses would settle it.

8. **Longer training for LPIPS/DISTS.** LPIPS at 100% peaks at epoch 23 of 30 and DISTS at
   epoch 27 of 30 — both are still improving when the run ends, while SSIM/NLPD/MSE peak early.
   The current 30-epoch budget probably understates the learned-perceptual losses. Run the top
   configs to 100+ epochs.

### Tier 3 — new directions the codebase supports cheaply

9. **Loss combinations.** `α·MSE + (1−α)·perceptual` over a small α sweep. The observed pattern
   (perceptual = data-efficient but capped; MSE = reliable) suggests a mixture beats either.
   This is a ~10-line change to the loss registry.

10. **Vary the probe, not just the encoder.** Add linear probe accuracy (the standard
    self-supervised-learning metric) and k-NN at several k. KNN and MLP already disagree
    sharply — KNN gets *worse* with training in half the runs while MLP improves — which means
    "representation quality" is probe-dependent and that disagreement is itself a result.

11. **Encoder transfer.** Train on CIFAR, probe on a different dataset. If perceptual losses
    really encode a general visual prior, they should transfer better than MSE. The loaders
    already make this easy.

12. **Report the loss's own value.** Only val MSE is logged, so a run trained with SSIM is
    never evaluated by SSIM. Log all metrics for all runs (they are cheap at eval time) and the
    cross-metric table comes for free — including a direct check on whether the losses even
    disagree about reconstruction quality.

### Small engineering wins

- Wrap eval in `torch.no_grad()` and add `net.eval()`/`net.train()` (B9).
- Save network weights — currently a run is unreproducible *and* unrecoverable (B5).
- Record the full config (epochs, LR, seed, split props, batch size) as a JSON sidecar in each
  run directory instead of encoding a subset in the directory name (B7, B8).
- Validate completeness, not existence, in `previously_done` (B10).
- Fix or delete `benchmark.py` and `baseline_performance.py` (B11).
- Add a handful of assertions: split sizes match expectation, latent shape matches
  `latent_dim`, no `-` in registry keys, loss is finite and decreases on a 2-batch overfit
  test. Most of B1–B3 would have been caught by the last one.
