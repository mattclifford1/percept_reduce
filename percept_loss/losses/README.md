# `losses`

The training-signal registry. `LOSS[name]()` returns a callable object that takes
`(x1, x2)` and returns a **scalar to minimise**, plus a `.to(device)`.

All inputs are in **`[0, 1]`** (`NORMALISE = (0, 1)`), and every decoder ends in `Sigmoid`.
Any loss added here must be correct on that range.

**No `-` in a loss name.** Run directories are `{LOSS}-{datasize}-BS{bs}` and
`plots/plot_training_runs.py` parses them by splitting on `-`.

## Registry

| key | backend | notes |
|---|---|---|
| `MSE` | `nn.MSELoss` | the pixel baseline |
| `MAE` | `nn.L1Loss` | registered, never used in a committed run |
| `SSIM` | `pytorch_msssim.SSIM(data_range=1, channel=3)` | 11×11 Gaussian window, default |
| `SSIM_torchmetrics` | `torchmetrics` functional SSIM | registered, never used in a committed run |
| `MSSIM` | `torchmetrics` MS-SSIM, `betas=(0.5, 0.5)` | fixed — was a near-no-op, see below |
| `LPIPS` | `torchmetrics` LPIPS, `net_type='vgg'`, `normalize=True` | fixed — was fed the wrong range |
| `LPIPS1` | same but `normalize=False` | **deliberately kept unfixed**, see below |
| `DISTS` | `DISTS_pytorch` | **see warning below** |
| `NLPD` | vendored `NLPD_torch`, `k=1` | **see warning below** |

`sim_to_loss` wraps a similarity metric `s` as `1 - s`. `SSIM` and `MSSIM` go through it;
`LPIPS`, `DISTS` and `NLPD` are already distances.

## Fixed bugs — and why `LPIPS1` exists

Measurements in `FINDINGS.md`. Runs produced *before* these fixes are archived in
`saves_legacy/gen1_original/` and must not be compared against current `saves/`.

### `MSSIM` was very nearly a no-op — FIXED

`pytorch_msssim.MS_SSIM` hard-asserts `min(H, W) > (win_size - 1) * 2**4`, i.e. **161 px for
the default `win_size=11`**. CIFAR is 32 px, so `win_size=1` was used to get past the
assertion. That collapses the Gaussian window to a single pixel and destroys the structural
term. Measured: a 10×10 blanked patch on a 32×32 image moves real SSIM from 1.0000 to
**0.7455** but moves `MSSIM` as configured from 1.0000 to only **0.9992**. The gradient is
~300× weaker. This is visible in the runs — `MSSIM-1` val MSE *rises* monotonically from
0.064 to 0.139 over training because the loss barely constrains the output.

Now `torchmetrics` MS-SSIM, which has no such assertion (1.0000 → 0.3834 on the same test,
and a **31× stronger gradient** on a realistic near-reconstruction pair). **Only 2 scales are
usable at 32 px** — 3 `betas` returns NaN — so `MSSIM_BETAS = (0.5, 0.5)` is used at every
resolution to keep CIFAR and ImageNet64 comparable. 64 px could support 3 scales if that
consistency is dropped.

The functional form is used rather than the class, which also sidesteps the stateful-`Metric`
double-compute described below.

### `LPIPS` was fed the wrong input range — FIXED, with `LPIPS1` kept as a control

`LearnedPerceptualImagePatchSimilarity` defaults to `normalize=False`, which means it expects
inputs in **`[-1, 1]`**. It was being given `[0, 1]`. torchmetrics 1.0.0 does not raise, so
this passed silently — the network only ever saw half of the dynamic range VGG was
calibrated on.

- **`LPIPS`** now passes `normalize=True` (correct: declares the input as `[0, 1]`).
- **`LPIPS1`** keeps `normalize=False` on purpose, so the pre-fix behaviour survives as a
  first-class loss and the size of the fix can be *measured* rather than inferred. Because
  `pipeline.generic.run` seeds immediately before building the network, `LPIPS` and `LPIPS1`
  share an identical initialisation — it is a paired comparison.

LPIPS is still a stateful `torchmetrics.Metric`, so calling it in the training loop
accumulates `sum_scores`/`total` and does roughly double the necessary work per step.
Numerically harmless; it just makes LPIPS the slowest loss in the grid. Tracked in `TODO.md`.

### `NLPD` uses 1 of its 6 pyramid levels

`NLPD(nlpd_k=1)` builds a `LaplacianPyramid` with `k=1`, but `DN_filters()` defines six levels
of divisive-normalisation filters and sigmas. Five are unused. `k` is capped by image size:
**`k<=3` works at 32 px, `k<=5` at 64 px**, `k=6` fails at both (padding exceeds the
downsampled dimension). The reference implementation uses `k=6` on larger images.

### `DISTS` collapses at reduced data

Three of five CIFAR `DISTS` runs collapse to a constant output within one epoch and never
recover (MLP accuracy pinned at 0.1007 = one class, val MSE flat at ~0.09). This is an
optimisation failure with the default Adam LR, not a data-size effect, and it makes the
`DISTS` row of the results table meaningless as it stands.

## `NLPD_torch/`

Vendored from Alex Hepburn's `expert` repo
(<https://github.com/alexhepburn/expert/tree/bb3ec766a7242961e9a47b399b8ab6b554c51fa6>),
BSD licensed. `pyramids.py` also contains `LaplacianPyramidGDN`, `SteerablePyramid` and
`SteerableWavelet` — none of them are wired into the `LOSS` registry, and `SteerableWavelet`
needs the Fourier utils that were deliberately left out. Only `LaplacianPyramid` is used.
