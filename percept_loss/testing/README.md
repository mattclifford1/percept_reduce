# `testing`

The evaluation half of the experiment: freeze the encoder, encode a held-out split, fit cheap
classifiers on the latents, report accuracy. This is the number the whole project turns on.

## `encoded_dataset.make_encodings(data_loader, autoencoder, device)`

Runs `autoencoder.encoder_forward` over the loader and returns `(X, y)` as numpy arrays,
`X` shaped `(n_instances, latent_dim)`. This is why every autoencoder must carry a
`latent_dim` attribute — `X` is preallocated from it.

`X`/`y` are preallocated to `100`, a deliberate poison value: any row left at 100 means the
loader produced fewer instances than `len(data_loader.dataset)` claimed. It is not asserted,
just left as a smell you would notice in a debugger.

No `torch.no_grad()` — the whole split is encoded with autograd live and the graph thrown
away. Wasteful; the single easiest performance fix in the repo.

## `benchmark_encodings.test_all_classifiers(data=(X, y))`

Splits the encodings `train_test_split(test_size=0.33, random_state=42)` and fits:

- `KNeighborsClassifier()` — defaults, k=5
- `MLPClassifier(alpha=1, max_iter=1000, random_state=42)` — one hidden layer of 100
- `GaussianNB()`

Returns `{name: accuracy}`, which becomes the `KNN` / `MLP` / `NB` columns of
`training_results.csv`. `val MSE` is added by the caller.

Six more classifiers (SVM, GP, tree, random forest, AdaBoost, QDA) are present but commented
out — they were too slow at this call frequency. The multiprocessing pool is likewise
commented out in favour of a plain `map`, because this function is already called from a
background thread (see `training/README.md`).

`random_GaussianNB_test` is the fast dev-only variant; `test_and_saver` has a commented line
to swap to it.

### What these numbers actually mean

They are **linear-probe-style accuracies on a frozen encoder**, fit and evaluated inside the
held-out test split. They are not comparable to a supervised classifier trained end to end,
and they are sensitive to latent dimensionality — `conv_big_z` gives a 384-d latent, the
CIFAR probe eval set is 5,940 instances over 10 classes.

Two baselines matter and are easy to miss:

- **Epoch 0 is the untrained network.** A randomly initialised conv encoder is a random
  projection, and it is a genuinely strong baseline here (CIFAR KNN ≈ 0.27). Roughly half the
  committed CIFAR runs finish with *worse* KNN accuracy than they started with.
- **Raw pixels, no autoencoder at all** — see `saves/readme.md`. Computed by
  `baseline_performance.py`, which no longer runs.

## `baseline_performance.py` — STALE, DO NOT USE

Uses a `dummy_encoder` that returns the input unchanged (`latent_dim = 32*32*3`) to measure
the raw-pixel baseline. Broken in two ways:

1. imports `get_all_loaders_CIFAR`, which was renamed to `get_all_loaders`;
2. calls `test_all_classifiers(test_dataloader, dummy_encoder(), device, verbose=True)` —
   positionally, `data=test_dataloader`, which is not the current signature.

Worth fixing: it is the only source of the "no autoencoder" reference point, and every claim
in this project is relative to it.
