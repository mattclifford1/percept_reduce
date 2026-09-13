# `networks`

## Contract

**Autoencoders** must be an `nn.Module` with methods `encoder_forward` and `decoder_forward`
(plus `forward`), and a **`latent_dim`** attribute — `testing/make_encodings` preallocates the
encoding matrix from `latent_dim`, so a wrong value silently corrupts every probe result.

(An earlier version of this file asked for a `name` attribute. Nothing reads `name`; only
`simple_autoencoder` still sets one. `latent_dim` is the attribute that is actually required.)

**Classifiers** are any sklearn-style object with `.fit` and `.predict`. The `CLASSIFIER` dict
in `__init__.py` is an empty placeholder; the classifiers actually used live in
`testing/benchmark_encodings.py`.

**No `-` in a registered network name** — run directories are parsed by splitting on `-`.

## Registry

### `CIFAR_AUTOENCODERS` (3×32×32) — `cifar_autoencoder.py`

All four are the same shape: 4×4 stride-2 convs down, `ConvTranspose2d` back up, `ReLU`
between, `Tanh` on the bottleneck, `Sigmoid` on the output (matching the `[0, 1]` data range).

| key | class | latent | shape |
|---|---|---|---|
| `conv_small_z` | `Autoencoder_mini` | 48 | 12×2×2, channels 3→6→12→12→12 |
| `conv_big_z` | `Autoencoder2` | **384** | 96×2×2, channels 3→12→24→48→96 |
| `conv_bigger_z` | `Autoencoder_small` | 96 | 24×2×2, channels 3→6→12→24→24 |
| `conv_biggest_z` | `Autoencoder_big` | 768 | 48×4×4, channels 3→12→24→48 (3 downsamples) |

The names are misleading — `conv_bigger_z` (96) has a *smaller* latent than `conv_big_z`
(384). Ordered by actual latent size: `conv_small_z` < `conv_bigger_z` < `conv_big_z` <
`conv_biggest_z`. All committed CIFAR results use **`conv_big_z`**.

`Autoencoder_big` has its fourth conv layer commented out, which is why it stops at 4×4.

### Literature backbones (3×32×32)

Added so the loss axis can be run on architectures that exist in the literature rather than only
on the hand-rolled stack above. **All five carry the same 384-dim latent** — see "Fixed latent
budget" below for why that is not a coincidence.

| key | class | file | params | latent | one-line reason |
|---|---|---|---|---|---|
| `dcgan` | `DCGAN_AE` | `dcgan_autoencoder.py` | 0.15M | 384 | `conv_big_z` + BatchNorm/LeakyReLU |
| `resnet18` | `ResNet18_AE` | `resnet_autoencoder.py` | 15.2M | 384 | the frozen-probe standard backbone |
| `resnet18_thin` | `ResNet18_AE(width=32)` | `resnet_autoencoder.py` | 4.0M | 384 | half width, ~4× cheaper |
| `vit` | `ViT_AE` | `vit_autoencoder.py` | 4.6M | 384 | transformer comparison, expected to lose |
| `vae` | `DCGAN_VAE` | `vae.py` | 0.60M | 384 | KL term on the `dcgan` backbone |
| `dcgan_gdn` | `DCGAN_GDN_AE` | `dcgan_gdn.py` | 0.16M | 384 | `dcgan` with GDN in place of BatchNorm |
| `vit_wu` | `ViT_AE_warmup` | `vit_autoencoder.py` | 4.6M | 384 | `vit` with a 10% linear LR warm-up |

Each file's docstring carries the full citation and reasoning. Summary:

**`dcgan`** — Radford, Metz & Chintala (2016), *Unsupervised Representation Learning with Deep
Convolutional GANs*, ICLR, arXiv:1511.06434. Deliberately the **same channel plan as
`conv_big_z`** (3→12→24→48→96) plus BatchNorm after every conv and LeakyReLU(0.2) in the
encoder, so `dcgan` vs `conv_big_z` is a clean ablation of normalisation alone. The reason to
want it: three `DISTS` cells collapse to a constant output inside one epoch (`FINDINGS.md` B3),
and a first-epoch collapse at Adam's default LR is precisely the failure mode normalisation
prevents. If `DISTS` survives on `dcgan`, B3 was optimisation and not the loss — cheaper than
the LR sweep in `TODO.md` T1.1.

**`resnet18`** — He et al. (2016), *Deep Residual Learning*, CVPR, arXiv:1512.03385, in the
CIFAR variant (3×3 stem, stride 1, no max-pool). This is the backbone the frozen-probe
literature reports on: SimCLR (arXiv:2002.05709), BYOL (arXiv:2006.07733) and SimSiam
(arXiv:2011.10566) all fit CIFAR-10 probes on frozen ResNet-18 features. Running our loss axis
on it makes the accuracies in `FINDINGS.md` comparable to published self-supervised numbers
instead of interpretable only against our own grid. At 15M params it is ~80× `conv_big_z`;
`resnet18_thin` exists for when that is too slow.

**`vit`** — Dosovitskiy et al. (2021), arXiv:2010.11929, at ViT-Tiny width (dim 192, 3 heads,
per DeiT arXiv:2012.12877), patch 4 → 64 tokens. **Two caveats, both load-bearing:** (1) it is
*not* MAE (arXiv:2111.06377) — MAE's representation quality comes from the masked-patch
objective, which cannot be used here because every perceptual loss in the grid is defined on a
whole image, not a subset of patches; masking would silently change the objective per-loss. (2)
It is expected to lose on data grounds — ViTs lack the convolutional prior and the 1% cell is
240 images — so a poor result is evidence about data scale, not about perceptual losses. It is
registered because it was worth seeing.

**`vae`** — Kingma & Welling (2014), arXiv:1312.6114, with Hou et al. (2017), *Deep Feature
Consistent VAE*, WACV, arXiv:1610.00291, as the direct precedent: a VAE whose reconstruction
term is a perceptual feature loss, judged on latent usability. That is this experiment with a
KL term added. `encoder_forward` returns **mu**, not a sample, so the probe sees a deterministic
encoding; `forward` samples in train mode and uses mu in eval. `self.kl` is set by every forward
and `training/run_and_test.py` adds `beta * net.kl` — the only VAE-specific line in the trainer.
**`beta = 1.0` is a starting point, not a tuned value** (T2.x in `TODO.md`): the KL is stored
divided by the pixel count so it lands around 0.02 at init, the same scale as the reconstruction
losses.

**`dcgan_gdn`** — Ballé, Laparra & Simoncelli (2016), *Density Modeling of Images Using a
Generalized Normalization Transformation*, ICLR, arXiv:1511.06281. `dcgan` with every
BatchNorm + activation replaced by GDN (inverse GDN in the decoder), so `dcgan` vs `dcgan_gdn`
isolates the normalisation. GDN is the divisive normalisation of learned image codecs and of
NLPD, i.e. perceptually motivated rather than an optimisation device, and it is **per-sample**:
no batch or running statistics, so train and eval mode are identical and a noise-trained net
carries no noise statistics into the probe. `gdn.py` is adapted from CompressAI (via
H-Test-IQM) so `compressai` is not a dependency.

**`vit_wu`** — `vit` plus a linear learning-rate warm-up over the first 10% of optimiser steps,
read by the trainer from the class attribute `warmup_frac`. Registered as its own network so
readers never pool it with `vit`. Caveat in its docstring: pre-LN transformers usually train
without warm-up, and DeiT's LR scaling puts batch 32 near 3e-5, so warm-up alone may not rescue
1e-3.

Two shared implementation choices:

- **Decoders upsample with `Upsample` + 3×3 conv, never `ConvTranspose2d`.** Transposed convs
  leave checkerboard artefacts (Odena, Dumoulin & Olah 2016, *Deconvolution and Checkerboard
  Artifacts*, Distill), and a perceptual loss scores those artefacts directly — with
  `ConvTranspose2d` the artefact would surface in the results as a loss-axis effect. (The four
  original `conv_*` nets do use `ConvTranspose2d`; they are left alone so their committed runs
  stay valid.)
- **Fixed latent budget of 384.** The probe is fit on the flattened latent, so latent
  dimensionality changes probe capacity independently of representation quality. A ResNet-18's
  natural output is 512×4×4 = 8192, which would hand the probe 21× `conv_big_z`'s input and win
  on capacity alone. Every new encoder ends with a projection to 96×2×2 (conv) or a 384-wide
  linear (ViT/VAE), so an architecture comparison is not secretly a latent-size comparison.

### `RANDOM` — the untrained-encoder control

Not an architecture: `'RANDOM'` in the **loss** slot of a run dict means *take zero gradient
steps* (`pipeline/generic.py`). The probe then reads the initialisation itself, and the run
directory is `RANDOM-{datasize}-BS32` with a single CSV row at epoch 0.

Why it matters more than any of the architectures above: `FINDINGS.md` reports that training on
pure uniform noise recovers 91% of MSE's full-data accuracy. The competing explanation is that
the *convolutional architecture* is doing the work and training is nearly irrelevant — random-weight
conv features are a known-strong baseline (Saxe et al. 2011, *On Random Weights and Unsupervised
Feature Learning*, ICML) and the deep image prior (Ulyanov et al. 2018, arXiv:1711.10925) is the
same phenomenon. If `RANDOM` ≈ `uniform`, that finding is an architecture prior, which is both
cleaner and more defensible than what `FINDINGS.md` currently reaches for. It costs one eval pass.

Because no training data is touched, the result is independent of `data_percent` — run **one
cell per network**, not one per data size.

### `IMAGENET64_AUTOENCODERS` (3×64×64) — `image_net_64_autoencoder.py`

| key | class | latent | shape |
|---|---|---|---|
| `standard` | `noraml_64` *(sic)* | 384 | 24×4×4, 4 downsamples, `ReLU` |
| `bigger_z` | `big_64` | 1536 | 24×8×8, 3 downsamples, **`ELU`** |

`bigger_z` differs from `standard` in *two* ways — latent size **and** activation function —
so the pair is not a clean ablation of either.

`noraml_64` is a typo for `normal_64`. Renaming the class is safe; renaming the registry key
`standard` is not, since that key is what appears in `saves/IMAGENET64_VAL/standard/`.

The layer comments in this file were copy-pasted from the CIFAR version and still say
`[batch, 3, 32, 32]` / `[batch, 12, 16, 16]`. They are wrong — these are 64×64 networks.

### Unregistered

- `simple_autoencoder.auto_encoder` — a `Flatten` + `Linear` bottleneck design. Commented out
  of the registry as "not working": `encoder_forward` ends with a `Linear` producing a flat
  vector, but `decoder_forward` skips `self.linear`/reshape and feeds that vector straight
  into a `ConvTranspose2d`. Fixable by uncommenting the two lines in `decoder_forward`.
- `linear_autoencoder.linear_AE` — MLP autoencoder, also marked "not working". Its
  `encoder_forward` reshapes to `(-1, 3072)`, which looks correct; worth retesting rather than
  assuming it is broken.
- `classifier.simple_image_clf` — a small conv classifier operating directly on images.
  Unused by the pipeline, and identical to the network in the top-level `load_cifar.py`
  scratch script.

## Note on capacity

Every committed autoencoder is tiny (≲100k parameters) and every one is trained for exactly
30 epochs at Adam's default LR. Encoder capacity is a confound that has never been swept
alongside the loss axis — the CIFAR grid varies loss and data size at a single fixed
architecture. Since the probe is fit on the latent vector, latent *size* alone changes the
probe's capacity independently of representation quality.
