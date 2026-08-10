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
  pipeline_CIFAR_ARCH.py     entry point — loss axis on the literature backbones + RANDOM control
  training/run_and_test.py   the actual training loop + async eval harness
  training/dev_loop.py       tiny 2-epoch smoke test / async-vs-sync timing benchmark
  training/benchmark.py      STALE — broken imports, do not use
  networks/                  autoencoder definitions (see networks/README.md)
  networks/{dcgan,resnet,vit}_autoencoder.py, networks/vae.py
                             literature backbones — DCGAN / CIFAR-ResNet-18 / ViT-Tiny / VAE,
                             all at a 384-dim latent so probe capacity is held fixed
  losses/                    loss registry: MSE, MAE, SSIM, MSSIM, LPIPS, LPIPS1, DISTS, NLPD
  losses/NLPD_torch/         vendored Laplacian-pyramid NLPD from Alex Hepburn's `expert` repo
  datasets/                  CIFAR-10 + ImageNet64 loaders, split logic, uniform-noise loader
  testing/                   encode-then-probe evaluation (make_encodings + sklearn classifiers)
  testing/reprobe.py         re-run probes from a saved checkpoint, no retraining
  testing/baseline_performance.py  STALE — broken imports, do not use
  utils/savers.py            run directories, config.json/done.json, CSV merge, checkpoints
  utils/seeding.py           set_seed() — called before net construction so runs are paired
  utils/migrate_saves.py     legacy save layout → current layout (dry run by default)
saves/<dataset>/<net>/<loss>/<size>/bs<bs>_seed<s>_lr<lr>_ep<n>/
  config.json                full run config + git sha — the source of truth for a run
  training_results.csv       one row per eval epoch: KNN, Linear, MLP, NB (+ a `<name> select`
                             column each), val MSE, train loss, out std, (KL for VAEs),
                             probe secs, epoch
  checkpoint.pt              final weights
  done.json                  written last; this is what "already done" means
  images/<epoch>-.png        2×4 grid: top row inputs, bottom row reconstructions
saves/<dataset>/<net>/<LOSS>-<size>-BS<bs>/   LEGACY layout — every committed run. still read
                             and still skipped correctly; convert with utils/migrate_saves.py
saves_legacy/                superseded runs by generation — never mix with saves/ (see its README)
plots/run_index.py           finds runs in either layout — every reader goes through it
plots/plot_training_runs.py  per-run training curves into plots/figs/
plots/plot_data_efficiency.py  accuracy vs data size, one line per loss — the headline figure.
                             one figure per optimisation budget, mean ± sd over seeds
plots/summarise_runs.py      every run in one table (final-epoch headline), one table per budget
load_cifar.py                unrelated scratch script (plain CIFAR classifier tutorial)
FINDINGS.md                  results analysis + confirmed bugs (the evidence lives here)
TODO.md                      the open action list (DISTS collapse, data budget, ImageNet probe)
```

## Current state of the experiment

- **`MSSIM` and `LPIPS` bugs are fixed** (B1, B2 in `FINDINGS.md`). `MSSIM` is now torchmetrics
  MS-SSIM with `betas=(0.5, 0.5)` — the only scale count that works at 32 px. `LPIPS` now
  passes `normalize=True`.
- **`LPIPS1` is a deliberately unfixed LPIPS** (`normalize=False`), kept so the effect of the
  fix is measurable rather than assumed. Do not "fix" it.
- **Runs are now seeded** — `pipeline.generic.run` calls `set_seed(seed)` after building the
  loss (LPIPS/DISTS draw from the RNG when constructing their backbone) and before building
  the network, so cells differing only in loss share an identical init. `seed` is a run axis, so
  multi-seed sweeps (T1.4) are a compute question, not a plumbing one — but every *committed*
  run is still a single seed.
- **The probe now standardises its features, and this changed the numbers a lot** (B13). On an
  untrained `dcgan`, adding `StandardScaler` moved MLP accuracy from 0.162 to 0.421 with `NB`
  unchanged. Every accuracy in `FINDINGS.md` §1 — including the untrained baseline and the
  ±0.032 noise floor — predates it. Do not compare a new number to an old one.
- **`FINDINGS.md` §1 quotes best-epoch accuracy**, which selects on the probe's own eval set over
  ~16 evals. `summarise_runs.py` now defaults to final-epoch and prints how much best-epoch
  would have added. See T1.5.
- **The probe splits the test-split encodings three ways**: 67% fit, 16.5% *select*, 16.5%
  *report*. `MLP` is the report split; `MLP select` is for choosing an epoch and must never be
  quoted. `--select early_stop` in `summarise_runs.py` / `plot_data_efficiency.py` is therefore
  unbiased early stopping — unlike `best`, which collapses the two splits into one. Runs made
  before this have no `select` columns and fall back to `NaN` for that mode.
- **CIFAR results now live in `saves/CIFAR_10/`**, matching the `DATA_LOADER` key. The legacy
  `saves/CIFAR/` path predates the key rename, which meant skip-if-exists never matched and
  the whole CIFAR grid silently re-ran. See T1.0 in `TODO.md`.
- Everything else in `FINDINGS.md` is still open — most importantly the `DISTS` collapse, the
  data-size/optimisation-budget confound, and the near-chance ImageNet64 probe.

## Key conventions (these are load-bearing)

- **A run's identity lives in `config.json`**, not in its path. Each path component is one field
  (`saves/{dataset}/{net}/{loss}/{datasize}/bs32_seed42_lr0.001_ep30/`) and nothing parses the
  leaf name. The old layout packed three fields into one directory name and every reader split
  it on `-`, which is what forced the "no `-` in a loss or network name" rule. **That rule still
  applies to legacy directories**, and `losses/__init__.py` / `networks/__init__.py` still carry
  the warning, but new code should read `config.json` via `plots/run_index.py`.
- **Runs are skipped if `done.json` exists** (`train_saver.previously_done`). A results CSV with
  no `done.json` is a crashed run: it gets moved to `training_results.csv.partial-*` and the cell
  re-runs. To force a re-run, delete the run directory.
- **A legacy directory only counts as done if it could have produced the same numbers** — same
  seed (42), same LR (1e-3), *and* the same probe protocol, detected from the CSV header (B14).
  It has no `config.json`, so none of that is recorded and all three have to be inferred. When a
  legacy run is rejected the reason is printed, and `run()` reports skips by reason and warns if a
  partition trained nothing. That warning exists because 35 cells once vanished silently.
- **Seeds, LR and epoch scaling are run axes.** `'seed': [1, 2, 3]` or `'lr': [...]` in the runs
  dict works and lands in separate directories. `run(..., epoch_scaling='equal_steps')` scales
  epochs by `1/data_percent` so every cell gets the same gradient-step budget; the default
  `'fixed'` is what every committed run used (and is `FINDINGS.md` B4).
- **`run_dir` is the only key unique per run — reduce on it before aggregating.** With three
  seeds and two budgets in one tree, `(dataset, net, loss, datasize)` matches six runs. Grouping
  on it and taking `.iloc[-1]` as "final" returns whichever ran longest (the 3000-epoch
  equal-steps cell) under a fixed-budget heading, and turns `best` into a max over six runs —
  that is B16, and it was silent. Reduce per `run_dir`, then aggregate seeds explicitly
  (mean ± sd). **Never put both budgets in one table or figure**: their difference *is* the
  data-efficiency result (T1.2), so averaging them destroys the measurement. `run_index.
  budget_view` gives one budget's rows plus the cells both share (`data_percent=1`, `uniform`,
  where the scaling factor is 1 and the two grids are by definition the same run).
- **Evaluation writes diagnostics, not just accuracies.** `train loss`, `out std` and (for VAEs)
  `KL` are logged per eval, so a collapsed run is visible in the CSV. Training aborts if the
  batch-wise output std stays under `collapse_tol` for `collapse_patience` epochs, and
  `done.json` records `collapsed: true`.
- **Losses are zero-arg factories** in the `LOSS` dict; they must expose `__call__(x1, x2)`
  returning a scalar to minimise, and a `.to(device)`. Similarity metrics are wrapped by
  `sim_to_loss` (`1 - sim`).
- **Autoencoders** must expose `encoder_forward`, `decoder_forward`, `forward`, and a
  `latent_dim` attribute (used to preallocate the encoding matrix in `make_encodings`).
  `encoder_forward` must be **deterministic** — the VAE returns mu, not a sample, or probe
  accuracy would pick up sampling noise. A net may set `self.kl` and `self.beta`; the trainer
  adds `beta * kl` to the loss when `kl` is present, and that is the only VAE-aware line in it.
- **New architectures hold the latent at 384** to match `conv_big_z`. The probe is fit on the
  flattened latent, so latent size changes probe capacity independently of representation
  quality — an unconstrained ResNet-18 (512×4×4 = 8192) would win on capacity alone.
- **`'RANDOM'` in the loss slot is not a loss** — it is the untrained-encoder control, zero
  gradient steps, one CSV row at epoch 0. It touches no training data, so one cell per network
  is enough; more `data_percent` values just duplicate it.
- **Evaluation runs under `net.eval()` + `torch.no_grad()`**, restoring the previous mode
  (`make_encodings`, `validate`). Every net added since the originals has BatchNorm, and in
  train mode an image's encoding would depend on its probe batch-mates.
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
python percept_loss/pipeline_CIFAR_ARCH.py    # architecture sweep + untrained controls
python percept_loss/testing/reprobe.py saves --all   # re-run probes from checkpoints, no retrain
python percept_loss/utils/migrate_saves.py saves     # legacy → current layout (dry run)
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
- **`FINDINGS.md` lists confirmed bugs; `TODO.md` tracks what is still open.** Read them before
  trusting any number in `saves/`. `MSSIM` and `LPIPS` are fixed; still open are `NLPD` using
  1 of 6 pyramid levels, three `DISTS` runs that are collapsed optimisation failures rather
  than data-size effects, and the data-size/optimisation-budget confound.
- **Changing a loss config invalidates its saved runs.** If you fix `NLPD`/`DISTS`, move the
  corresponding run directories out of `saves/` or skip-if-exists will hide the fix. Prefer
  *archiving* to deleting (see `saves_legacy/`), and consider keeping the old behaviour as
  a separate named loss the way `LPIPS1` does — it turns "we fixed it" into a measurement.
- `saves/` and `plots/figs/` are committed to git. Regenerating them produces large diffs —
  that is normal for this repo (see commit `3f775c1 "complete redo of figs"`).
- `percept_loss/datasets/CIFAR_10/__init__,py` has a **comma instead of a dot**. It works today
  only because of implicit namespace packages + editable install. `find_packages()` does not
  see that directory, so a non-editable `pip install .` would ship a broken package.

## Style

Plain PyTorch + sklearn, no config framework, no Hydra, no lightning. Registries are plain
dicts. Match that — do not introduce abstraction layers. Comments are sparse and lowercase.
