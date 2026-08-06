# percept_reduce

Does training an autoencoder with a **perceptual** loss instead of a pixel loss make its latent
space more semantically useful — and does that let you train on **less data**?

## The experiment

For every combination of (loss × training-set size × autoencoder × dataset):

1. Train an autoencoder from scratch for 30 epochs with that loss on that much data.
2. Every 2 epochs, freeze it, run a held-out split through `encoder_forward`, and fit cheap
   sklearn classifiers (KNN, MLP, GaussianNB) on the resulting latent vectors.
3. Log downstream accuracy + val reconstruction MSE to a CSV.

**Downstream classification accuracy from a frozen encoder is the metric.** Reconstruction
quality is only a reference number — a perceptual loss is allowed to reconstruct "worse" in
pixel terms if its latent space is more useful.

| axis | values |
|---|---|
| loss | `MSE`, `SSIM`, `MSSIM`, `LPIPS`, `LPIPS1`, `DISTS`, `NLPD` (also `MAE`, `SSIM_torchmetrics`) |
| training-set size | `1`, `0.5`, `0.1`, `0.01` (fraction of the train split) and `uniform` |
| dataset | `CIFAR_10`, `IMAGENET64_VAL`, `IMAGENET64_TRAIN` |
| autoencoder | 4 CIFAR variants, 2 ImageNet64 variants |

`uniform` is a **control**: the autoencoder is trained on pure uniform-random noise and never
sees a real image. It exists to answer "how much of this is the loss function and how much is
just a randomly-shaped conv stack?" It does much better than you would hope.

## Results

**[FINDINGS.md](FINDINGS.md)** — full analysis of the runs and the confirmed bugs.
**[TODO.md](TODO.md)** — the open action list and proposed follow-up experiments.

`LPIPS1` is not a typo: it is LPIPS with its original (incorrect) `normalize=False` input
handling, kept as a first-class loss so the effect of fixing that bug is *measured* against the
corrected `LPIPS` under an identical seed rather than assumed. Runs made before the `MSSIM` and
`LPIPS` fixes are archived in `saves_pre_lossfix/` and must not be mixed with `saves/`.

Short version of the results:

- On CIFAR-10, **LPIPS at full data is the clear winner** (0.60 MLP accuracy vs 0.46 for MSE).
- **Hand-designed perceptual losses (SSIM, NLPD) are flat across data size** — they hit their
  ceiling at 1% of the data. Learned perceptual losses (LPIPS, DISTS) are steeply data-hungry
  but go much higher. That trade-off is the most interesting thing in the results.
- **Training on uniform noise recovers ~91% of full-data MSE performance.** The strong
  baselines here are load-bearing and under-reported.
- The ImageNet64 results are **at or near chance** and should not be interpreted as they stand
  — the probe evaluates 1000 classes on ~5 examples each.

## Setup

```bash
conda create -n percept python=3.10 -y
conda activate percept
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

Or in one line (make sure you're in the repo):

```bash
conda create -n percept python=3.10 -y && conda activate percept && conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y && pip install -r requirements.txt && pip install -e .
```

### Data

- **CIFAR-10** downloads and unpacks itself on first use into
  `percept_loss/datasets/CIFAR_10/raw_data/` (gitignored). Nothing to do.
- **ImageNet64** must be prepared by hand. Download the 64×64 downsampled ImageNet pickles
  from <https://patrykchrabaszcz.github.io/Imagenet32/>, extract into
  `~/datasets/ImageNet64/{train,val}/`, and run the `process.py` that lives alongside them.
  The loader expects `~/datasets/ImageNet64/<split>/images/` plus `meta_data.csv`.
  Paths are hardcoded in `percept_loss/datasets/IMAGENET/ImageNet64.py`.

## Running

All commands from the repo root.

```bash
python percept_loss/pipeline_CIFAR.py            # the committed CIFAR-10 grid (30 runs)
python percept_loss/pipeline_IMAGENET64.py       # ImageNet64 val-split grid
python percept_loss/pipeline_IMAGENET64_TRAIN.py # full ImageNet64 train split
python percept_loss/training/dev_loop.py         # 2-epoch smoke test
python plots/plot_training_runs.py               # regenerate plots/figs/ from saves/
```

Edit the `runs` dict at the top of a pipeline file to change the grid.

**Runs already on disk are skipped.** `saves/.../training_results.csv` existing means
"previously done" — delete the run directory to force a re-run. This also means a crashed run
leaves a partial CSV that will be skipped forever.

## Layout

| path | what |
|---|---|
| [`percept_loss/`](percept_loss/README.md) | the package |
| [`percept_loss/pipeline/`](percept_loss/pipeline/README.md) | grid driver |
| [`percept_loss/training/`](percept_loss/training/README.md) | training loop + async eval |
| [`percept_loss/losses/`](percept_loss/losses/README.md) | loss registry |
| [`percept_loss/networks/`](percept_loss/networks/README.md) | autoencoders |
| [`percept_loss/datasets/`](percept_loss/datasets/README.md) | loaders + splits |
| [`percept_loss/testing/`](percept_loss/testing/README.md) | encode-then-probe evaluation |
| [`percept_loss/utils/`](percept_loss/utils/README.md) | saving / run naming |
| [`saves/`](saves/readme.md) | committed run outputs |
| [`plots/`](plots/README.md) | figure generation |
| `load_cifar.py` | unrelated scratch file — a plain CIFAR classifier tutorial, not part of the experiment |

## Adding things

**A new loss** — write a zero-arg factory in `losses/standard.py` or `losses/perceptual.py`
returning an object with `__call__(x1, x2) -> scalar` and `.to(device)`. Wrap similarity
metrics in `sim_to_loss`. Register it in `losses/__init__.py`. Inputs are `[0, 1]`.
**No `-` in the name** — the saved-run directory name is parsed by splitting on `-`.

**A new autoencoder** — subclass `nn.Module` with `encoder_forward`, `decoder_forward`,
`forward`, and a `latent_dim` attribute. Register in `networks/__init__.py`. **No `-` in the
name.**

**A new dataset** — write a loader class matching the shape of
`datasets/CIFAR_10/loader.py` (`indicies_to_use`, `image_dict`, `get_images_dict`,
`__getitem__ -> (image, one_hot, label)`), register it in `datasets/__init__.py` along with its
size in `TOTAL_INSTANCES`, and add its image shape to the `uniform` branch of
`datasets/torch_loaders.py`.
