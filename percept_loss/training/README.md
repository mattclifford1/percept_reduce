# `training`

## `run_and_test.py` — the real training loop

`train(network, loss, epochs, device, saver, data_percent, ...)`

Plain PyTorch: Adam at its **default LR (1e-3)**, no scheduler, no weight decay, no grad
clipping, no early stopping, no checkpointing. The trained weights are never saved — only the
metrics CSV and reconstruction image grids survive a run.

Evaluation happens at epoch 0 (untrained network — this is the random-encoder baseline) and
then every `validate_every` epochs. For CIFAR/`validate_every=2` that gives 16 rows per CSV:
epochs `0, 1, 3, 5, ..., 29`.

### Split proportions are chosen by `validate_every`

```python
if validate_every == 1:   # know we have a big dataset so decrease validation/test set size
    props = [0.89, 0.1, 0.01]
else:
    props = [0.4, 0.3, 0.3]
```

This is a proxy for "is this the big ImageNet64 train split". It works but it is fragile and
invisible from the pipeline file — changing `validate_every` silently changes the dataset
split, which changes what the accuracy numbers even mean. Do not treat runs with different
`validate_every` as comparable.

### `test_and_saver` — the async eval harness

sklearn probe fitting (especially the MLP, `max_iter=1000`) is slow and CPU-bound, so it runs
on a background thread while the GPU carries on training.

The ordering is deliberate and worth preserving:

1. `make_encodings` + `validate` run **on the main thread**, so `X, y` and `val_MSE` are a
   snapshot of the network at this epoch. Subsequent training cannot corrupt them.
2. Image writing stays on the main thread (matplotlib is not thread-safe).
3. Before launching a new probe thread, the previous one is `join`ed — so at most one probe is
   in flight and CSV writes are serialised.
4. `tester.wait_to_finish()` at the end of `train()` blocks until the last probe lands.

`dev_loop.py` measured this at ~30–38 s vs ~35–41 s for 2 epochs, i.e. a modest win.

### Things that are missing / wrong

- **No `torch.no_grad()`** in `validate()` or `make_encodings()`. Both build autograd graphs
  for a whole split that are immediately discarded. Pure waste, and the dominant memory cost
  of eval.
- **No `net.eval()` / `net.train()`.** Harmless today (the autoencoders have no BatchNorm or
  Dropout) but it will silently corrupt results the moment someone adds either.
- **`validate()` averages per-batch means unweighted**, so the final short batch is
  over-weighted. Small effect, but it makes val MSE not exactly the dataset MSE.
- **Nothing is seeded** except the train/val/test split. Network init and shuffle order are
  free-running, so runs are not reproducible and there is one seed per grid cell.
- `LOSS['MSE']()` is reconstructed on every eval call.

## `dev_loop.py`

2-epoch smoke test on `conv_biggest_z` + MSE. Run it directly to time async vs sync eval.

Note it passes the **loss and network objects** (not their names) into `train_saver`, so it
writes to a directory named after `repr()` of those objects rather than a clean name. Its
output goes under `saves/dev/`. Fine for a smoke test, not something to copy.

## `benchmark.py` — STALE, DO NOT USE

Imports `get_all_loaders_CIFAR` and `random_forest_test`, neither of which exists any more.
It will fail at import. It also mixes up `lr`/`train_total` in the `train_saver` positional
arguments. Superseded by `run_and_test.py`; kept only for reference.
