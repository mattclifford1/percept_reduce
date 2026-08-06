# Findings

Analysis of the 90 committed runs in `saves/`, the bugs that affect them, and what to run next.

Everything below was recomputed from the CSVs (`~/anaconda3/envs/percept/bin/python`), not read
off the figures. Unless stated, the headline metric is **best-epoch MLP probe accuracy**.

> **Read the tables in §1 with this caveat.** "Best-epoch" is the maximum over ~16 evaluations of
> a metric computed on the probe's own eval set, and the measured run-to-run noise floor is
> ±0.032. That is selection on the test set: it inflates every number, and it inflates *noisy*
> runs (1% data, `DISTS`) more than stable ones — which is exactly where the comparisons of
> interest sit. `plots/summarise_runs.py` now defaults to the **final** epoch and prints how much
> best-epoch selection would have added; `--select best` reproduces the numbers below. Regenerate
> §1 on the final-epoch basis before quoting any of it.

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

> **Status:** B1 and B2 are **fixed** and their runs re-done — see `TODO.md` for what is still
> open, and `saves_pre_lossfix/` for the runs those two bugs produced. Everything from B3 down
> is outstanding. §1 above describes the *pre-fix* results; it is retained as the record that
> motivated the fixes, and is superseded for `MSSIM`/`LPIPS` by §4.

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
so val MSE is not sampling noise. **No committed result changes** — every run in `saves/` used a
norm-free, dropout-free network, where `.eval()` is a no-op.

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
  for epoch selection, or by reporting final-epoch only. *(Now surfaced: `summarise_runs.py`
  defaults to final-epoch and prints the size of the inflation. §1 is still best-epoch.)*

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

Consequences, all still open:

- The untrained-encoder baseline in §1 (**0.128 ± 0.032**) is an unscaled number. The scaled
  equivalent is far higher — 0.42 on this architecture — which would put it within reach of
  `MSE` at 100% data (0.459). *Every* "beats the untrained baseline" claim needs re-checking.
- The measured ±0.032 noise floor is likewise an unscaled-probe number.
- Because under-fitting penalises whichever latents are hardest to fit, the loss *ranking* in §1
  is not safe either — this is not a constant offset.
- The numbers above are `dcgan` at 1% data, not `conv_big_z`. Re-measure with the `RANDOM`
  control on `conv_big_z` before quoting anything: that is one cell and takes seconds.

---

## 3. Post-fix results

The full CIFAR grid (35 cells) was re-run after fixing B1/B2, now **seeded** — every cell shares
an identical network init, so cells differing only in loss are a *paired* comparison. Results in
`saves/CIFAR_10/`; the pre-fix runs are in `saves_pre_lossfix/`.

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

## 4. What to run next

Moved to **[TODO.md](TODO.md)** so it can be worked through and ticked off. That list carries
the same items, grouped by priority:

- **Tier 1** (blocks any quotable result) — the `CIFAR`/`CIFAR_10` save-path mismatch, the
  `DISTS` collapse, decoupling data size from optimisation budget, fixing the ImageNet64 probe,
  and multi-seed error bars.
- **Tier 2** (one small change away) — a proper baseline panel, matching the `uniform` control,
  re-sweeping the architecture axis, longer training for the learned perceptual losses, and
  sweeping the `NLPD` pyramid depth.
- **Tier 3** (new directions) — loss combinations, more probe types, encoder transfer, and
  logging every metric for every run.
- **Engineering** — `no_grad`/`eval`, saving weights, config sidecars, completeness checks on
  skip-if-exists, the stale scripts, and a set of assertions that would have caught B1–B3.

This file stays as the evidence: what was measured, and what it implies.
