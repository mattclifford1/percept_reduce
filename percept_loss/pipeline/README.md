# `pipeline`

`generic.py` is the experiment driver. It takes a dict of lists, expands the cartesian product,
and trains one autoencoder per cell.

```python
runs = {
    'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
    'loss':         ['SSIM', 'MSE', 'LPIPS', 'MSSIM', 'DISTS', 'NLPD'],
    'network':      ['conv_big_z'],
}
run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32,
    preload_data=True, dataset='CIFAR_10', validate_every=2)
```

## `get_all_dict_permutations`

Reverses the dict before taking the product so that the **last** key in the dict varies
slowest. In practice this means: put the axis you most want to finish first at the bottom of
the `runs` dict. (The comment `# want eval first` refers to this ordering intent.)

## `preload_data`

`preload_data=True` calls `get_preloaded()` once and passes the resulting
`{filename: tensor}` dict into every loader, so images are read from disk and decoded exactly
once for the whole grid. For CIFAR this is float32-on-device; for ImageNet64 it caches raw
uint8 tensors instead (~20 GB for the full train split) and converts per access.

## Gotchas

- **`data_percent`-aware epoch scaling is dead code.**

  ```python
  scaled_epochs = int(epochs/data_percent)
  scaled_epochs = epochs          # <- immediately overwrites the line above
  ```

  As it stands, every run gets exactly `epochs` passes over its own (differently sized)
  training set. A 1%-data run therefore takes ~100× fewer gradient steps than a 100% run, so
  "less data" and "less optimisation" are confounded throughout the committed results. See
  `FINDINGS.md`.

- **The net and loss are constructed before the skip check**, so an already-completed
  `LPIPS` cell still pays for loading VGG weights before printing `Passing`.

- `run` reports skips by printing `Passing` with no indication of which cell was skipped.

- `batch_size` is accepted by `run()` but never forwarded — `get_all_loaders` uses its own
  default of 32. Every committed run is therefore BS32 regardless of what the pipeline says.
