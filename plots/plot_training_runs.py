'''
per-run training curves: one figure per (dataset, network, datasize).

changes from the original version, all of them things the old figures made it easy to misread:
  - reads both save layouts via run_index (nothing here parses directory names any more)
  - the metric grid sizes itself, so adding a probe (e.g. 'Linear') no longer indexes out of range
  - chance level is drawn on every accuracy panel, and y always starts at 0. the old code zoomed
    ImageNet64 to [0, 0.1], which made near-chance results look like signal
  - the untrained-encoder baseline +/- the measured run-to-run noise is shaded: any curve inside
    that band is not distinguishable from an untrained network
  - RANDOM (the untrained control) is one point at epoch 0, so it is drawn as a horizontal line
'''
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from run_index import load_long, chance_level, NOISE_FLOOR

# both paths are relative to the working directory -- identical to the old behaviour when run
# from the repo root, but pointing the script at another saves tree no longer writes its figures
# back into the repo's committed plots/figs/
DEFAULT_SAVE_DIR = os.path.join('.', 'saves', '')
DEFAULT_PLOT_DIR = os.path.join('plots', 'figs')

# plotted in this order when present. 'probe secs' is bookkeeping, never plotted.
METRIC_ORDER = ['KNN', 'Linear', 'MLP', 'NB', 'val MSE', 'train loss', 'out std', 'KL']
ACCURACY_METRICS = {'KNN', 'Linear', 'MLP', 'NB'}

colours = {'SSIM': 'blue', 'LPIPS': 'green', 'MSE': 'red', 'MSSIM': 'orange', 'NLPD': 'yellow',
           'DISTS': 'pink',
           'LPIPS1': 'darkgreen',   # LPIPS1 = pre-fix LPIPS, plotted next to the fixed LPIPS
           'RANDOM': 'black'}       # RANDOM = untrained control, drawn as a horizontal line


def plot_metric(ax, panel, metric, dataset):
    baselines = []
    for loss, run in panel.groupby('loss'):
        run = run.sort_values('epoch')
        values = run[metric].to_numpy()
        epochs = run['epoch'].to_numpy()
        if np.all(np.isnan(values)):
            continue
        baselines.append(values[0])
        colour = colours.get(loss, 'grey')
        if loss == 'RANDOM' or len(epochs) == 1:
            # a single epoch-0 measurement: a reference level, not a curve
            ax.axhline(values[0], color=colour, linestyle=':', label=loss)
        else:
            ax.plot(epochs, values, label=loss, color=colour)

    ax.set_title(metric)
    ax.set_xlabel('epoch')
    if metric in ACCURACY_METRICS:
        ax.set_ylabel('accuracy')
        ax.set_ylim(bottom=0)
        chance = chance_level(dataset)
        ax.axhline(chance, color='grey', linestyle='--', linewidth=1)
        ax.annotate('chance', (0, chance), fontsize=7, color='grey',
                    va='bottom', ha='left')
        # anything inside this band is indistinguishable from an untrained encoder
        if len(baselines):
            base = float(np.nanmean(baselines))
            ax.axhspan(base - NOISE_FLOOR, base + NOISE_FLOOR, color='grey', alpha=0.12)
            ax.annotate(f'untrained +/-{NOISE_FLOOR}', (0, base + NOISE_FLOOR), fontsize=7,
                        color='grey', va='bottom', ha='left')
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default=DEFAULT_SAVE_DIR)
    ap.add_argument('--out', default=DEFAULT_PLOT_DIR)
    args = ap.parse_args()
    plot_dir = args.out
    os.makedirs(plot_dir, exist_ok=True)

    long_df = load_long(args.save_dir)
    groups = long_df.groupby(['dataset', 'net', 'datasize'])
    for (dataset, net, data_size), panel in tqdm(groups, total=len(groups)):
        metrics = [m for m in METRIC_ORDER
                   if m in panel.columns and not panel[m].isna().all()]
        n_cols = min(3, len(metrics))
        n_rows = int(np.ceil(len(metrics)/n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, squeeze=False)
        fig.suptitle(f'{dataset} / {net} with datasize {data_size}')

        flat = axs.flatten()
        for ax, metric in zip(flat, metrics):
            plot_metric(ax, panel, metric, dataset)
        for ax in flat[len(metrics):]:
            ax.axis('off')

        plot_exact_dir = os.path.join(plot_dir, dataset, net)
        os.makedirs(plot_exact_dir, exist_ok=True)
        name = data_size if data_size == 'uniform' else float(data_size)*100
        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout()
        plt.savefig(os.path.join(plot_exact_dir, f'{name}.png'), bbox_inches='tight', dpi=100)
        plt.close(fig)


if __name__ == '__main__':
    main()
