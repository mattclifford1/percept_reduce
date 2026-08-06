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
