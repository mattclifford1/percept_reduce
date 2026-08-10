'''
the figure that states the result: probe accuracy vs training-set size, one line per loss.

the hypothesis is "a perceptual loss lets you get away with less training data". that is a claim
about the *shape* of accuracy-vs-data, and until now it could only be read off a table by hand --
every existing figure is a training curve for a single cell.

    python plots/plot_data_efficiency.py                      # ./saves, MLP probe, final epoch
    python plots/plot_data_efficiency.py --metric Linear       # the linear probe (SSL protocol)
    python plots/plot_data_efficiency.py --select val          # epoch chosen by val MSE

epoch selection matters more than it looks (see --select):
  final       the last eval. the honest default.
  early_stop  epoch picked on the probe's 'select' split and reported on the disjoint 'report'
              split. unbiased early stopping -- use this when you want a stopped number. only
              available for runs made after the three-way probe split landed.
  val         the epoch with the lowest val MSE. also unbiased, but FINDINGS reports val MSE is
              a weak proxy for probe accuracy, so it can pick a poor epoch.
  best        the maximum over all evals. this is what FINDINGS.md currently quotes and it is
              **optimistically biased**: it selects on the reported numbers themselves, over ~16
              draws of a noisy metric, and it flatters noisy runs (small data, DISTS) most.
              read `best - final` as instability, not as an achievable score.
'''
import argparse
import os

import matplotlib
matplotlib.use('Agg')    # these scripts only savefig -- never show(). without this matplotlib
                         # tries to open a display and blocks on the socket indefinitely if one
                         # is advertised but not answering (seen: 10 min, 1s of CPU, stuck in poll)
import matplotlib.pyplot as plt
import numpy as np

from run_index import (load_long, sorted_sizes, chance_level, noise_floor, caption,
                       probe_version, MIXED_PROTOCOL_WARNING, budget_view, SHARED_SIZES)

SELECT_TEXT = {
    'final': 'value at the final epoch',
    'early_stop': 'value at the epoch chosen on the probe select split (unbiased early stopping)',
    'val': 'value at the epoch with the lowest validation MSE',
    'best': 'MAXIMUM over all epochs -- optimistically biased, not an achievable score',
}

# relative to the working directory, so pointing this at another saves tree does not write into
# the repo's committed plots/figs/
DEFAULT_PLOT_DIR = os.path.join('plots', 'figs')

colours = {'SSIM': 'blue', 'LPIPS': 'green', 'MSE': 'red', 'MSSIM': 'orange', 'NLPD': 'yellow',
           'DISTS': 'pink', 'LPIPS1': 'purple', 'RANDOM': 'black'}
# LPIPS1 was 'darkgreen' next to LPIPS's 'green' -- the two curves that most need telling apart
# (the fix vs the deliberately unfixed version) were the two hardest to distinguish

# one figure per optimisation budget -- they are different experiments. the fixed-budget grid
# confounds data size with gradient steps (FINDINGS B4); the equal-steps grid holds steps fixed
# and varies only the number of distinct images. the difference is the result (TODO T1.2).
BUDGET_TEXT = {'fixed': 'fixed budget: 30 epochs over each cell\'s own training set '
                        '(data size and gradient steps are confounded)',
               'equal_steps': 'equal optimisation budget: epochs scaled by 1/data_percent '
                              '(only the number of distinct images varies)'}


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
    if how == 'early_stop':
        # epoch picked on the disjoint 'select' split -- unbiased early stopping
        sel_col = f'{metric} select'
        if sel_col not in run.columns or run[sel_col].isna().all():
            return np.nan          # run predates the three-way probe split
        return run.loc[run[sel_col].idxmax(), metric]
    raise SystemExit(f'unknown --select {how}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default=os.path.join('.', 'saves', ''))
    ap.add_argument('--metric', default='MLP', help='KNN, Linear, MLP or NB')
    ap.add_argument('--select', default='final',
                    choices=['final', 'early_stop', 'val', 'best'])
    ap.add_argument('--out', default=DEFAULT_PLOT_DIR)
    args = ap.parse_args()
    plot_dir = args.out

    long_df = load_long(args.save_dir)
    if args.metric not in long_df.columns:
        raise SystemExit(f'no {args.metric} column -- older runs predate it')

    panels = []
    for (dataset, net), whole in long_df.groupby(['dataset', 'net']):
        for scaling in sorted(whole['epoch_scaling'].dropna().unique()):
            panels.append((dataset, net, scaling, budget_view(whole, scaling)))

    for dataset, net, scaling, g in panels:
        sizes = sorted_sizes(g['datasize'].unique())
        x = np.arange(len(sizes))
        fig, ax = plt.subplots()

        untrained = []
        n_seeds = set()
        for loss, runs in g.groupby('loss'):
            ys, errs = [], []
            for size in sizes:
                cell = runs[runs['datasize'] == size]
                # one value per run, then mean +/- sd over seeds. reducing the concatenated
                # cell directly would take .iloc[-1] across every seed at once and report the
                # longest-running one as if it were the cell.
                vals = [select_epoch(r, args.metric, args.select)
                        for _, r in cell.groupby('run_dir')]
                vals = [v for v in vals if not np.isnan(v)]
                ys.append(np.mean(vals) if vals else np.nan)
                errs.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                n_seeds.add(len(vals))
                for _, r in cell.groupby('run_dir'):
                    untrained.append(r.sort_values('epoch').iloc[0][args.metric])
            if loss == 'RANDOM':
                # no training data involved, so it is a level not a curve
                ax.axhline(np.nanmean(ys), color='black', linestyle=':', label='RANDOM (untrained)')
                continue
            ax.errorbar(x, ys, yerr=errs, marker='o', capsize=3, label=loss,
                        color=colours.get(loss, 'grey'))

        chance = chance_level(dataset)
        ax.axhline(chance, color='grey', linestyle='--', linewidth=1)
        ax.annotate('chance', (0, chance), fontsize=8, color='grey', va='bottom')
        if len(untrained):
            base = float(np.nanmean(untrained))
            spread = noise_floor(args.metric)
            ax.axhspan(base - spread, base + spread, color='grey', alpha=0.12)
            ax.annotate(f'untrained encoder +/-{spread} (seed-to-seed spread)',
                        (0, base + spread), fontsize=8, color='grey', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel('autoencoder training set size (fraction of the training split)\n'
                      '"uniform" = trained on pure noise, never sees a photograph')
        ax.set_ylabel(f'{args.metric} probe accuracy on frozen encodings')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{dataset} / {net}: does a perceptual loss buy data efficiency?\n'
                     f'{args.metric} probe, {SELECT_TEXT[args.select]}\n'
                     f'{BUDGET_TEXT.get(scaling, scaling)}', fontsize=11)
        ax.legend(fontsize=8, loc='best')

        # the figure has to stand on its own -- it is the one most likely to be read alone
        note = ('a line that is flat in x reached its ceiling on the smallest training set; '
                'a steep line needs data. differences smaller than the shaded band are noise. '
                'error bars are +/-1 sd over seeds'
                + (f' (n={min(n_seeds - {0})}-{max(n_seeds)} per point).' if n_seeds - {0}
                   else '.'))
        shared = [s for s in sizes if s in SHARED_SIZES]
        if shared and long_df['epoch_scaling'].nunique() > 1:
            note += (f' the {", ".join(shared)} points are one run shared by both budgets -- '
                     'the scaling factor there is 1, so the two grids cannot differ.')
        text = caption(g, dataset, args.metric, extra=note)
        if g.groupby('run_dir').apply(lambda r: probe_version(r, args.metric)).nunique() > 1:
            text = MIXED_PROTOCOL_WARNING + '\n' + text
        fig.text(0.01, 0.01, text, fontsize=7.5, va='bottom', ha='left', color='dimgrey')

        out_dir = os.path.join(plot_dir, dataset, net)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir,
                           f'data_efficiency-{args.metric}-{args.select}-{scaling}.png')
        fig.set_size_inches(9, 6.6)
        fig.tight_layout(rect=[0, 0.14, 1, 1])   # leave room for the caption block
        fig.savefig(out, bbox_inches='tight', dpi=120)
        plt.close(fig)
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
