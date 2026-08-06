# `saves` — committed run outputs

90 runs. One directory per grid cell:

```
saves/<dataset>/<network>/<LOSS>-<datasize>-BS<batch_size>/
    training_results.csv     one row per eval: epoch, KNN, MLP, NB, val MSE
    images/<epoch>-.png      2×4 grid, top row inputs / bottom row reconstructions
```

The directory name is the *only* record of a run's configuration. Epochs, learning rate,
`validate_every`, split proportions and seed are not stored anywhere. See
`percept_loss/utils/README.md`.

**Trained weights are not saved.** Re-running a config means retraining from scratch, and
because network init is unseeded you will not get the same numbers back.

## What is here

| dataset | network | cells | latent |
|---|---|---|---|
| `CIFAR` | `conv_big_z` | 6 losses × 5 sizes = 30 | 384 |
| `IMAGENET64_VAL` | `bigger_z` | 30 | 1536 |
| `IMAGENET64_VAL` | `standard` | 30 | 384 |

All at 30 epochs, batch size 32, `validate_every=2` → 16 rows per CSV (epochs 0, 1, 3, …, 29).
Epoch 0 is the **untrained network**, i.e. the random-projection baseline.

Rows are written in reverse-chronological order — **sort by `epoch` before analysing.**

Not present: any `IMAGENET64_TRAIN` results (that pipeline has never completed), and the
multi-architecture CIFAR sweep that produced `plots/legacy_figs/` (those CSVs were lost; the
PNGs are the only record).

## Baseline: raw pixels, no autoencoder

Random Forest fit directly on flattened CIFAR-10 images, no encoder:

| data | accuracy |
|---|---|
| 1 (60,000) | 0.4237 |
| 0.1 (6,000) | 0.0421 |

Produced by `percept_loss/testing/baseline_performance.py`, which **no longer runs** (broken
imports — see `percept_loss/testing/README.md`). The 0.1 figure of 0.0421 is *below* the 0.1
chance rate for 10 classes and is almost certainly wrong; treat it as unverified.

Note this baseline used Random Forest, which is commented out of the current classifier set
(`KNN`, `MLP`, `NB`), so it is not directly comparable to the numbers in the CSVs.

## Before trusting these numbers

`FINDINGS.md` documents confirmed bugs that affect specific rows: `MSSIM` is close to a no-op
as configured, `NLPD` uses 1 of 6 pyramid levels, `LPIPS` is fed the wrong input range, and
three `DISTS` cells are collapsed optimisation failures rather than data-size effects. The
ImageNet64 probe evaluates 1000 classes on ~5 examples each and sits at or near chance
throughout.

Re-running a fixed loss requires **deleting the affected run directories** — the pipeline
skips any cell whose `training_results.csv` already exists.
