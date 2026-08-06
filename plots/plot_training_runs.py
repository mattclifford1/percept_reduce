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

from run_index import (load_long, chance_level, noise_floor, caption, probe_version,
                       MIXED_PROTOCOL_WARNING)

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
            spread = noise_floor(metric)
            ax.axhspan(base - spread, base + spread, color='grey', alpha=0.12)
            ax.annotate(f'untrained +/-{spread}', (0, base + spread), fontsize=7,
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

        size_text = ('trained on uniform noise (control -- the net never sees a photograph)'
                     if data_size == 'uniform' else
                     f'trained on {float(data_size)*100:g}% of the training split')
        n_runs = panel['run_dir'].nunique()
        seeds = sorted(set(panel['seed'].dropna().tolist()))
        seed_text = f'seed {int(seeds[0])}' if len(seeds) == 1 else f'{len(seeds)} seeds'
        fig.suptitle(f'{dataset} / {net} -- {size_text}\n'
                     f'{n_runs} runs, {seed_text}, probe accuracy vs training epoch',
                     fontsize=13)

        flat = axs.flatten()
        for ax, metric in zip(flat, metrics):
            plot_metric(ax, panel, metric, dataset)
        for ax in flat[len(metrics):]:
            ax.axis('off')

        # everything a reader needs to interpret the panel without opening the repo
        headline = 'MLP' if 'MLP' in metrics else metrics[0]
        collapsed = ''
        if 'out std' in panel.columns and not panel['out std'].isna().all():
            dead = sorted(panel[panel['out std'] < 1e-4]['loss'].unique())
            if len(dead):
                collapsed = ('collapsed to a constant output (batch std ~ 0), not a data-size '
                             'effect: ' + ', '.join(dead))
        text = caption(panel, dataset, headline, extra=collapsed)
        if panel.groupby('run_dir').apply(lambda g: probe_version(g, headline)).nunique() > 1:
            text = MIXED_PROTOCOL_WARNING + '\n' + text
        fig.text(0.01, 0.005, text, fontsize=9, va='bottom', ha='left', color='dimgrey')

        plot_exact_dir = os.path.join(plot_dir, dataset, net)
        os.makedirs(plot_exact_dir, exist_ok=True)
        name = data_size if data_size == 'uniform' else float(data_size)*100
        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(os.path.join(plot_exact_dir, f'{name}.png'), bbox_inches='tight', dpi=100)
        plt.close(fig)


if __name__ == '__main__':
    main()
