# `plots`

```bash
python plots/plot_training_runs.py       # per-run training curves
python plots/plot_data_efficiency.py     # accuracy vs data size — the headline figure
python plots/summarise_runs.py           # every run in one table
```

All three must be run from the repo root (`./saves` is the default path) and all three read runs
through `run_index.py`, which handles **both** save layouts:

```
saves/CIFAR_10/conv_big_z/LPIPS/0.5/bs32_seed42_lr0.001_ep30/     current
saves/CIFAR_10/conv_big_z/LPIPS-0.5-BS32/                         legacy
```

For current-layout runs the metadata comes from `config.json` and nothing parses a directory
name. For legacy runs the middle component is still split on `-`, so **a `-` in a loss or network
name still breaks those silently** — points get attributed to the wrong series rather than
erroring. `percept_loss/utils/migrate_saves.py` converts legacy → current.

## `plot_training_runs.py` — per-run curves

One figure per (dataset, network, data size) at `figs/<dataset>/<network>/<size>.png`, all losses
overlaid as metric-vs-epoch curves. `data_size` is multiplied by 100 for the filename, so `0.5` →
`50.0.png`; `uniform` is passed through as-is.

The metric grid sizes itself from whichever columns are present (`METRIC_ORDER` sets the
preference; `probe secs` is bookkeeping and never plotted), so adding a probe no longer indexes
out of range.

Every accuracy panel carries three reference marks, all of which exist to stop the figures being
over-read:

- **chance** (0.1 for CIFAR-10, 0.001 for ImageNet64), drawn as a dashed line;
- **y starts at 0.** The old code zoomed ImageNet64 to `[0, 0.1]`, which made near-chance results
  look like signal — 8× the CIFAR zoom, against a chance level of 0.001;
- **the untrained-encoder band**: epoch-0 accuracy ± the measured ±0.032 run-to-run noise
  (`NOISE_FLOOR` in `run_index.py`). A curve inside that band is not distinguishable from an
  untrained network.

`RANDOM` (the untrained control) has a single row at epoch 0, so it is drawn as a horizontal
reference line rather than a lone point.

## `plot_data_efficiency.py` — the headline figure

Probe accuracy vs training-set size, one line per loss, per (dataset, network). This is the shape
the hypothesis is about, and until it existed the claim could only be read off a table by hand.

```bash
python plots/plot_data_efficiency.py --metric Linear    # the SSL-protocol linear probe
python plots/plot_data_efficiency.py --select val       # epoch chosen by val MSE
```

`--select` is the important flag:

| mode | what it does | use it? |
|---|---|---|
| `final` | last eval | default, honest |
| `early_stop` | epoch picked on the probe's `select` split, reported on the disjoint `report` split | **the right choice when you want a stopped number** |
| `val` | epoch with lowest val MSE | unbiased but a weak selector |
| `best` | max over all evals | optimistically biased — see below |

`early_stop` needs the three-way probe split, so it is `NaN` for runs made before that landed.

## `summarise_runs.py` — cross-run table

```bash
python plots/summarise_runs.py                    # ./saves
python plots/summarise_runs.py saves_legacy/gen2_lossfix_oldprobe   # any saves-shaped dir
python plots/summarise_runs.py saves --metric Linear --select best
```

Prints a loss × datasize pivot per (dataset, network), the untrained epoch-0 baseline with its
spread, a flag for any run that never beat that baseline, and a flag for any run whose output
collapsed (`out std` ≈ 0).

**The headline is the final epoch, not the best one.** Best-epoch is a maximum over ~16 evals of a
metric measured on the probe's own eval set, against a ±0.032 noise floor: it inflates every
number and inflates noisy runs more than stable ones. The table prints how much best-epoch
selection would have added — that quantity is selection bias, not signal. Every number in
`FINDINGS.md` §1 is currently a best-epoch number; `--select best` reproduces them.

Keep reading `best` for one thing only: **`best − final` measures peak-then-degrade
instability**, which is a real phenomenon here (`LPIPS` at 1% peaks at epoch 3 and then slides).
As a performance number it is an oracle that stops using the reported scores themselves; as an
instability diagnostic it is exactly right. If you want a legitimately stopped score, that is
what `early_stop` is for.

Still missing: error bars, because there is one seed per cell. Seeds are a run axis now
(`'seed': [1, 2, 3]` in the runs dict), so this is a matter of compute rather than plumbing —
see T1.4 in `TODO.md`.

## Directories

- `figs/` — current figures, regenerated wholesale by the scripts (`3f775c1 "complete redo of
  figs"`). Committed.
- `legacy_figs/` — figures from an older run of the CIFAR grid that swept **four** autoencoder
  architectures (`conv_small_z`, `conv_big_z`, `conv_bigger_z`, `conv_biggest_z`) at datasizes
  1/10/50/100%. The underlying CSVs for those runs are **not** in `saves/` any more — these
  PNGs are the only surviving record of the architecture sweep. Do not delete them without
  re-running that grid.
