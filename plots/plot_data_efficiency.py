'''
the figure that states the result: probe accuracy vs training-set size, one line per loss.

the hypothesis is "a perceptual loss lets you get away with less training data". that is a claim
about the *shape* of accuracy-vs-data, and until now it could only be read off a table by hand --
every existing figure is a training curve for a single cell.

    python plots/plot_data_efficiency.py                      # ./saves, MLP probe, final epoch
    python plots/plot_data_efficiency.py --metric Linear       # the linear probe (SSL protocol)
    python plots/plot_data_efficiency.py --select val          # epoch chosen by val MSE

epoch selection matters more than it looks (see --select):
  final  the last eval. the honest default.
  val    the epoch with the lowest val MSE. the only unbiased early-stopping rule available,
         but FINDINGS reports val MSE is a weak proxy for probe accuracy, so it can pick a poor
         epoch.
  best   the maximum over all evals. this is what FINDINGS.md currently quotes and it is
         **optimistically biased**: it selects on the probe's own eval set, over ~16 draws of a
         metric with a +/-0.032 noise floor, and it flatters noisy runs (small data, DISTS)
         more than stable ones.
'''
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from run_index import load_long, sorted_sizes, chance_level, NOISE_FLOOR

# relative to the working directory, so pointing this at another saves tree does not write into
# the repo's committed plots/figs/
DEFAULT_PLOT_DIR = os.path.join('plots', 'figs')

colours = {'SSIM': 'blue', 'LPIPS': 'green', 'MSE': 'red', 'MSSIM': 'orange', 'NLPD': 'yellow',
           'DISTS': 'pink', 'LPIPS1': 'darkgreen', 'RANDOM': 'black'}


def select_epoch(run, metric, how):
    # runs predating a probe have no column for it -- an all-NaN group, not a zero
    run = run.sort_values('epoch').dropna(subset=[metric])
    if len(run) == 0:
        return np.nan
    if how == 'final':
        return run.iloc[-1][metric]
    if how == 'best':
        return run[metric].max()
    if how == 'val':
        return run.loc[run['val MSE'].idxmin(), metric]
    raise SystemExit(f'unknown --select {how}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default=os.path.join('.', 'saves', ''))
    ap.add_argument('--metric', default='MLP', help='KNN, Linear, MLP or NB')
    ap.add_argument('--select', default='final', choices=['final', 'val', 'best'])
    ap.add_argument('--out', default=DEFAULT_PLOT_DIR)
    args = ap.parse_args()
    plot_dir = args.out

    long_df = load_long(args.save_dir)
    if args.metric not in long_df.columns:
        raise SystemExit(f'no {args.metric} column -- older runs predate it')

    for (dataset, net), g in long_df.groupby(['dataset', 'net']):
        sizes = sorted_sizes(g['datasize'].unique())
        x = np.arange(len(sizes))
        fig, ax = plt.subplots()

        untrained = []
        for loss, runs in g.groupby('loss'):
            ys = []
            for size in sizes:
                cell = runs[runs['datasize'] == size]
                ys.append(select_epoch(cell, args.metric, args.select) if len(cell) else np.nan)
                if len(cell):
                    untrained.append(cell.sort_values('epoch').iloc[0][args.metric])
            if loss == 'RANDOM':
                # no training data involved, so it is a level not a curve
                ax.axhline(np.nanmean(ys), color='black', linestyle=':', label='RANDOM (untrained)')
                continue
            ax.plot(x, ys, marker='o', label=loss, color=colours.get(loss, 'grey'))

        chance = chance_level(dataset)
        ax.axhline(chance, color='grey', linestyle='--', linewidth=1)
        ax.annotate('chance', (0, chance), fontsize=8, color='grey', va='bottom')
        if len(untrained):
            base = float(np.nanmean(untrained))
            ax.axhspan(base - NOISE_FLOOR, base + NOISE_FLOOR, color='grey', alpha=0.12)
            ax.annotate(f'untrained encoder +/-{NOISE_FLOOR} (run-to-run noise)',
                        (0, base + NOISE_FLOOR), fontsize=8, color='grey', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel('training set size (fraction of dataset; "uniform" = noise control)')
        ax.set_ylabel(f'{args.metric} probe accuracy')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{dataset} / {net} -- {args.metric}, {args.select}-epoch')
        ax.legend(fontsize=8)

        out_dir = os.path.join(plot_dir, dataset, net)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'data_efficiency-{args.metric}-{args.select}.png')
        fig.set_size_inches(8, 5.5)
        fig.tight_layout()
        fig.savefig(out, bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
