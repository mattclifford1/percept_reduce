# `datasets`

Loaders, split logic, and the uniform-noise control.

## Contract

A loader class takes `indicies_to_use`, `image_dict`, `normalise`, `device` and yields
`(image, one_hot, numerical_label)`. Training uses `data[0]`; the probe uses `data[0]` and
`data[2]`. `get_images_dict()` returns `{filename: tensor}` for the preload path.

Register in `__init__.py` under `DATA_LOADER` **and** `TOTAL_INSTANCES`, and add the image
shape to the `uniform` branch of `torch_loaders.py`.

## Registered datasets

| key | source | size | image |
|---|---|---|---|
| `CIFAR_10` | auto-downloaded, unpacked to PNG | 60,000 | 3×32×32 |
| `IMAGENET64_VAL` | `~/datasets/ImageNet64/val/` | 50,000 | 3×64×64 |
| `IMAGENET64_TRAIN` | `~/datasets/ImageNet64/train/` | 1,281,167 | 3×64×64 |
| `UNIFORM` | generated | n/a | configurable |

Images are normalised to `[0, 1]` (`NORMALISE` in `torch_loaders.py`).

### CIFAR-10

`downloader.py` fetches via torchvision, unpickles the batches, writes 60,000 individual PNGs
plus `meta_data.csv` (filename, label, numerical_label) into
`CIFAR_10/raw_data/`, then deletes the tarball and the pickle directory. Idempotent — it
short-circuits if `raw_data/` exists. `loader.py` calls it on construction, so CIFAR "just
works" on a clean checkout. Labels are 0-indexed.

### ImageNet64

Not downloaded automatically. Paths are hardcoded to `~/datasets/ImageNet64/{train,val}/`,
each expected to contain `images/` and `meta_data.csv`. Built by the `process.py` that lives
alongside the data, from the downsampled-ImageNet pickles
(<https://patrykchrabaszcz.github.io/Imagenet32/>).

Labels are **1-indexed** (1..1000). `_get_labels` accordingly writes `one_hot[label - 1]`,
but returns `label` unshifted. Only the unshifted `label` reaches the probe, so this is
consistent today — but the one_hot and the label disagree by one, which will bite anyone who
starts using `data[1]`.

Unlike the CIFAR loader, the ImageNet64 cache stores **raw uint8** tensors and converts to
float on each access (`get_images_dict(preprocess=False)`), keeping the full train split near
~20 GB of RAM instead of ~60 GB.

### `uniform` — the noise control

`UNIFORM_LOADER` generates a fresh `torch.rand(size)` on **every `__getitem__`**. It is not a
fixed dataset of 5000 images; `length=5000` only sets the epoch boundary. Over a 30-epoch run
the network therefore sees ~150,000 distinct noise images and never repeats one. Worth
remembering when comparing it against the real-data conditions, which do repeat.

Note also that `train_total` for `uniform` is hardcoded to 5000 in `torch_loaders.py`
regardless of the dataset — it is not matched to any of the real-data sizes.

## Splits — `proportions.get_indicies`

Shuffles `range(total_instances)` with **seed 42** (this is the only seeded thing in the repo)
and slices it into consecutive proportions. Because the order is already random, reducing the
training set is just `train_inds[:n]` — a random subsample, and crucially a **nested** one:
the 1% set is a subset of the 10% set is a subset of the 50% set. Good for the comparison.

Resulting sizes with the proportions actually used:

| dataset | props | train | val | test | probe fit / probe eval |
|---|---|---|---|---|---|
| `CIFAR_10` | 0.4 / 0.3 / 0.3 | 24,000 | 18,000 | 18,000 | 12,060 / 5,940 |
| `IMAGENET64_VAL` | 0.4 / 0.3 / 0.3 | 20,000 | 15,000 | 15,000 | 10,050 / 4,950 |
| `IMAGENET64_TRAIN` | 0.89 / 0.1 / 0.01 | 1,140,238 | 128,116 | 12,811 | 8,583 / 4,228 |

The `data_percent` axis scales the **train** column only:

| `data_percent` | CIFAR train images | steps/epoch @ BS32 | steps over 30 epochs |
|---|---|---|---|
| `1` | 24,000 | 750 | 22,500 |
| `0.5` | 12,000 | 375 | 11,250 |
| `0.1` | 2,400 | 75 | 2,250 |
| `0.01` | 240 | 8 | 240 |
| `uniform` | 5,000/epoch (fresh) | 157 | 4,710 |

**The probe eval sets on the ImageNet64 rows are the problem.** 4,950 evaluation examples
across 1,000 classes is ~5 per class, and the probe is fit on ~10 per class. Chance is 0.001
and the committed ImageNet64 results top out around 0.02. See `FINDINGS.md`.

## Gotchas

- `image_dict={}` is a **mutable default argument** on both real loaders. Any loader
  constructed without an explicit dict shares one process-global cache, and `cache_data=True`
  writes into it. The call sites always pass an explicit dict, so this is currently masked —
  but it is a live trap for anything new.
- `torch_loaders.get_all_loaders` uses `pre_loaded_images == True` / `== None` to distinguish
  "load it for me" / "no cache" / "here is a cache". Passing a non-empty dict works, passing
  an empty dict works, but the idiom is comparing a dict against a bool.
- `get_all_loaders` has its own `batch_size=32` default and the pipeline never forwards its
  `batch_size` argument, so **every committed run is BS32** regardless of the pipeline setting.
- `get_indicies` claims to accept a list for `total_instances` but then computes
  `int(total_instances * prop)`, which raises for a list.
