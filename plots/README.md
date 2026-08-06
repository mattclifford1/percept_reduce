# `plots`

```bash
python plots/plot_training_runs.py     # must be run from the repo root
```

Scrapes every `saves/*/*/*/training_results.csv`, groups by
`dataset → network → data size → metric`, and writes one figure per
(dataset, network, data size) to `plots/figs/<dataset>/<network>/<size>.png`.

Each figure is a 2×2 grid of subplots — one per CSV column (`KNN`, `MLP`, `NB`, `val MSE`) —
with all six losses overlaid as accuracy-vs-epoch curves.

## How runs are identified

Entirely from the directory path:

```
saves/CIFAR/conv_big_z/LPIPS-0.5-BS32/
      ^^^^^ ^^^^^^^^^^ ^^^^^ ^^^ ^^^^
      dataset  network  loss size batch
```

`info[2].split('-')` recovers loss / size / batch size. **A `-` anywhere in a loss or network
name breaks this silently** — points get attributed to the wrong series rather than erroring.

`data_size` is multiplied by 100 for the filename, so `0.5` → `50.0.png` and `1` → `100.0.png`.
`uniform` is passed through as-is.

## Hardcoded things to watch

- `colours = {'SSIM', 'LPIPS', 'MSE', 'MSSIM', 'NLPD', 'DISTS'}` — **a new loss raises
  `KeyError` here.** Add it to the dict before plotting a new grid.
- Y-limits are hardcoded per dataset: `[0, 0.1]` if `'IMAGENET'` is in the dataset name, else
  `[0, 0.8]`. Applied to every subplot except `val MSE`. This is why the ImageNet64 figures
  look like they have signal — the axis is zoomed 8× relative to CIFAR, and chance for a
  1000-class problem is 0.001, which is invisible at that scale. Do not read the ImageNet64
  figures without that in mind.
- Subplot layout assumes exactly 4 metric columns (2 rows × `ceil(4/2)`). Adding a fifth
  metric will index out of range; dropping to 3 leaves a blank axis.
- `save_dir` is `./saves`, so the script only works from the repo root.

## Directories

- `figs/` — current figures, regenerated wholesale by the script (`3f775c1 "complete redo of
  figs"`). Committed.
- `legacy_figs/` — figures from an older run of the CIFAR grid that swept **four** autoencoder
  architectures (`conv_small_z`, `conv_big_z`, `conv_bigger_z`, `conv_biggest_z`) at datasizes
  1/10/50/100%. The underlying CSVs for those runs are **not** in `saves/` any more — these
  PNGs are the only surviving record of the architecture sweep. Do not delete them without
  re-running that grid.

## Not covered by this script

Nothing aggregates *across* runs — there is no summary table, no loss-vs-datasize curve, no
error bars (there is only one seed per cell). Every comparison in `FINDINGS.md` was computed
ad hoc from the CSVs rather than from these figures.
