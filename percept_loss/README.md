# `percept_loss` package

The experiment code. Top-level `pipeline_*.py` files are entry points; everything else is a
subpackage with its own README.

## Flow of a single run

```
pipeline_CIFAR.py
  └─ pipeline.generic.run(runs, AUTOENCODERS, ...)          # expand the grid
       └─ for each cell:
            utils.savers.train_saver(...)                    # names the save dir, skip-if-exists
            losses.LOSS[name]()                              # build loss
            AUTOENCODERS[name]()                             # build net
            training.run_and_test.train(...)
                 ├─ datasets.torch_loaders.get_all_loaders() # train/val/test dataloaders
                 └─ loop epochs:
                      forward → loss(inputs, outputs) → backward → step
                      every `validate_every` epochs:
                        testing.encoded_dataset.make_encodings()   # freeze, encode test split
                        validate()                                 # val reconstruction MSE
                        [background thread] testing.benchmark_encodings.test_all_classifiers()
                        saver.write_scores() / saver.write_images()
```

## Entry points

| file | dataset | net | `validate_every` | splits |
|---|---|---|---|---|
| `pipeline_CIFAR.py` | `CIFAR_10` (60,000) | `conv_big_z` | 2 | 0.4 / 0.3 / 0.3 |
| `pipeline_IMAGENET64.py` | `IMAGENET64_VAL` (50,000) | `bigger_z` | 2 | 0.4 / 0.3 / 0.3 |
| `pipeline_IMAGENET64_TRAIN.py` | `IMAGENET64_TRAIN` (1,281,167) | `standard` | 1 | 0.89 / 0.1 / 0.01 |

Note the split proportions are **selected by `validate_every`**, not by dataset — see
`training/run_and_test.py:86`. That coupling is accidental and is flagged in `FINDINGS.md`.
It means the two ImageNet64 pipelines are not comparable with each other.

Only `CIFAR` and `IMAGENET64_VAL` results are committed under `saves/`. The
`IMAGENET64_TRAIN` grid has never been (successfully) run to completion.

## Registries

Three plain dicts define the search space. **None of their keys may contain a `-`** — run
directories are named `{LOSS}-{datasize}-BS{bs}` and the plotting script parses that by
splitting on `-`.

- `losses.LOSS` — loss name → zero-arg factory
- `networks.CIFAR_AUTOENCODERS` / `networks.IMAGENET64_AUTOENCODERS` — net name → class
- `datasets.DATA_LOADER` + `datasets.TOTAL_INSTANCES` — dataset name → loader class / size

`networks.CLASSIFIER` is an empty placeholder dict; the classifiers actually used live in
`testing/benchmark_encodings.py`.

## Known-stale files

- `training/benchmark.py` — imports `get_all_loaders_CIFAR` and `random_forest_test`, neither
  of which exists any more. Will not import.
- `testing/baseline_performance.py` — same broken import, plus it calls
  `test_all_classifiers` with the old positional signature. Will not run.

Both predate the `get_all_loaders` rename and the `test_all_classifiers` signature change.
They are the only place the "raw pixels, no autoencoder" baseline is computed (result recorded
in `saves/readme.md`), so fixing them is worthwhile — see `FINDINGS.md`.
