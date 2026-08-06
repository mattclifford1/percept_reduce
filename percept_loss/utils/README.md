# `utils`

`savers.py` — `train_saver` owns run identity, the results CSV, and reconstruction image grids.

## Run identity

```python
self.unique_name = [dataset, network, f'{loss}-{datasize}-BS{batch_size}']
save_dir = saves/<dataset>/<network>/<LOSS>-<datasize>-BS<bs>/
```

This directory name **is** the experiment record — nothing else stores the configuration.
Consequences:

- **No `-` in a loss or network name.** `plots/plot_training_runs.py` recovers loss, datasize
  and batch size by `split('-')` on the third component. A hyphenated name silently
  misattributes every point in every figure.
- `loss` and `network` must be passed as **strings**. `pipeline/generic.py` does this
  correctly; `training/dev_loop.py` passes the objects and gets a directory named after
  `repr()`.
- Anything not in the name is invisible: epochs, learning rate, `validate_every`, split
  proportions, seed. Two runs with different split proportions collide on the same path.

## Skip-if-exists

```python
self.previously_done = os.path.exists(self.csv_file)
```

`pipeline.generic.run` uses this to skip completed cells, which makes the grid resumable.
The failure mode: **a crashed or killed run leaves a partial CSV and is skipped forever.** If
a config seems to "do nothing", check the row count in its CSV before anything else. Deleting
the run directory is the only way to force a re-run.

Note the flag is computed in `__init__`, i.e. the directory is created as a side effect of
asking whether the run was already done.

## `write_scores` / `save_and_merge_df_as_csv`

Each eval appends one row keyed on `epoch`. The merge is
`pd.merge(new, saved, how='outer')` over all shared columns — so re-scoring the *same* epoch
with *different* values appends a second row rather than replacing the first. Combined with
skip-if-exists this rarely fires, but a partially-deleted CSV can produce duplicate epochs
that the plotting script will happily draw.

Rows land in reverse-chronological order in the file; sort by `epoch` before analysing.

## `write_images`

2×4 grid: top row inputs, bottom row reconstructions, from the last training batch of that
epoch. Saved to `<run>/images/<epoch>-.png`. The `extra_name` suffix is currently always
empty — there is a commented-out line that used to tag the filename with MLP accuracy.

Wrapped in `try/except RuntimeError` because matplotlib backends throw when a run is launched
without a display (commit `b85ace3 "catch ploting crash with tkinter"`). It prints and
continues. If images are missing from a run, this is why.

Must stay on the main thread — matplotlib is not thread-safe, which is why
`test_and_saver.run_and_save_async` writes images before spawning the probe thread.
