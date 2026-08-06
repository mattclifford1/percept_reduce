# CLAUDE.md

Guidance for Claude Code when working in this repo.

## What this project is

**`percept_reduce`** — a research codebase testing one hypothesis:

> If you train an autoencoder with a *perceptual* loss instead of a pixel loss, does the
> learned latent space become more semantically useful, and does that let you get away with
> **less training data**?

The experiment is a grid: `{6 losses} × {5 training-set sizes} × {autoencoder} × {dataset}`.
For every cell we train an autoencoder, periodically freeze it, encode a held-out set, and
fit cheap sklearn classifiers (KNN / MLP / GaussianNB) on those encodings. **Downstream
classification accuracy from frozen encodings is the metric**, not reconstruction quality.
Reconstruction MSE on the val split is logged alongside as a sanity/reference number.

The `uniform` "dataset size" is a control condition: train the autoencoder on **pure uniform
noise** (never sees a real image). It is a deliberately strong null hypothesis, and it does
alarmingly well — see `FINDINGS.md`.

Author: Matt Clifford <matt.clifford@bristol.ac.uk> (University of Bristol).

## Repo map

```
percept_loss/
  pipeline/generic.py        the experiment driver: cartesian product over the run grid
  pipeline_CIFAR.py          entry point — CIFAR-10 grid
  pipeline_IMAGENET64.py     entry point — ImageNet64 *val* split as the whole dataset
  pipeline_IMAGENET64_TRAIN.py  entry point — full ImageNet64 train split (never committed results)
  training/run_and_test.py   the actual training loop + async eval harness
  training/dev_loop.py       tiny 2-epoch smoke test / async-vs-sync timing benchmark
  training/benchmark.py      STALE — broken imports, do not use
  networks/                  autoencoder definitions (see networks/README.md)
  losses/                    loss registry: MSE, MAE, SSIM, MSSIM, LPIPS, DISTS, NLPD
  losses/NLPD_torch/         vendored Laplacian-pyramid NLPD from Alex Hepburn's `expert` repo
  datasets/                  CIFAR-10 + ImageNet64 loaders, split logic, uniform-noise loader
  testing/                   encode-then-probe evaluation (make_encodings + sklearn classifiers)
  testing/baseline_performance.py  STALE — broken imports, do not use
  utils/savers.py            run-directory naming, CSV merge, reconstruction image grids
saves/<dataset>/<net>/<LOSS>-<size>-BS<bs>/
  training_results.csv       one row per eval epoch: KNN, MLP, NB, val MSE, epoch
  images/<epoch>-.png        2×4 grid: top row inputs, bottom row reconstructions
plots/plot_training_runs.py  scrapes all saves/ CSVs into plots/figs/
load_cifar.py                unrelated scratch script (plain CIFAR classifier tutorial)
FINDINGS.md                  results analysis, confirmed bugs, proposed experiments
```

## Key conventions (these are load-bearing)

- **Run identity is encoded in the directory name**: `saves/{dataset}/{network}/{LOSS}-{datasize}-BS{batch_size}/`.
  `plots/plot_training_runs.py` parses this by splitting on `-`. **Never put a `-` in a loss
  name or a network name** — both `losses/__init__.py` and `networks/__init__.py` carry this
  warning. Adding a hyphenated key silently corrupts every plot.
- **Runs are skipped if `training_results.csv` already exists** (`train_saver.previously_done`).
  To re-run a config you must delete its directory. A crashed run leaves a partial CSV and will
  be skipped forever — check this first if a run "does nothing".
- **Losses are zero-arg factories** in the `LOSS` dict; they must expose `__call__(x1, x2)`
  returning a scalar to minimise, and a `.to(device)`. Similarity metrics are wrapped by
  `sim_to_loss` (`1 - sim`).
- **Autoencoders** must expose `encoder_forward`, `decoder_forward`, `forward`, and a
  `latent_dim` attribute (used to preallocate the encoding matrix in `make_encodings`).
- **Data are normalised to `[0, 1]`** (`NORMALISE = (0, 1)` in `datasets/torch_loaders.py`) and
  every decoder ends in `Sigmoid`. Any new loss must be correct on that range.
- Dataset items are `(image, one_hot, numerical_label)`. Training uses `data[0]`; the probe
  uses `data[0]` and `data[2]`.

## Environment

```bash
conda activate percept          # python 3.10, torch 2.0.1, torchmetrics 1.0.0
```

There is no `python`/`pandas` on the system PATH — the interpreter lives at
`~/anaconda3/envs/percept/bin/python`. Use that (or activate the env) for any analysis script.

Data locations:
- CIFAR-10: auto-downloaded to `percept_loss/datasets/CIFAR_10/raw_data/` (gitignored). Present.
- ImageNet64: expected at `~/datasets/ImageNet64/{train,val}/` with `images/` + `meta_data.csv`.
  Present. Built by `~/datasets/ImageNet64/process.py` from the downsampled-ImageNet pickles.

## Running things

```bash
python percept_loss/pipeline_CIFAR.py         # ~30 runs, the committed CIFAR grid
python percept_loss/pipeline_IMAGENET64.py    # ImageNet64 val-split grid
python percept_loss/training/dev_loop.py      # 2-epoch smoke test
python plots/plot_training_runs.py            # regenerate plots/figs/ from saves/
```

Pipelines must be run from the repo root (`plot_training_runs.py` hardcodes `./saves`).

## Things to know before changing code

- **There are no tests.** Nothing in the repo verifies a loss is sane, a split is the size you
  think, or a run didn't silently collapse. Several of the confirmed bugs in `FINDINGS.md`
  would have been caught by a five-line assertion.
- **Nothing is seeded except the train/val/test split** (`proportions.get_indicies`, seed 42).
  Network init and shuffling are unseeded, so runs are not reproducible and there is exactly
  one seed per grid cell. The measured noise floor is ~±0.03 absolute MLP accuracy on CIFAR
  (see `FINDINGS.md`), which is large relative to several of the effects being claimed.
- **`FINDINGS.md` lists confirmed bugs that affect the committed results.** Read it before
  trusting any number in `saves/`. In particular `MSSIM` is very nearly a no-op as configured,
  `NLPD` uses 1 of 6 pyramid levels, `LPIPS` is fed the wrong input range, and three `DISTS`
  runs are collapsed optimisation failures rather than data-size effects.
- **Changing a loss config invalidates its saved runs.** If you fix `MSSIM`/`NLPD`/`LPIPS`,
  delete the corresponding `saves/` directories or the skip-if-exists logic will hide the fix.
- `saves/` and `plots/figs/` are committed to git. Regenerating them produces large diffs —
  that is normal for this repo (see commit `3f775c1 "complete redo of figs"`).
- `percept_loss/datasets/CIFAR_10/__init__,py` has a **comma instead of a dot**. It works today
  only because of implicit namespace packages + editable install. `find_packages()` does not
  see that directory, so a non-editable `pip install .` would ship a broken package.

## Style

Plain PyTorch + sklearn, no config framework, no Hydra, no lightning. Registries are plain
dicts. Match that — do not introduce abstraction layers. Comments are sparse and lowercase.
