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
import textwrap

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
# x layout. the controls sit on the left, then real data on a log axis, so reading left to right
# is "how much natural-image signal has the encoder had": none (untrained), training but no
# photographs (uniform noise), then 1% -> 100% of the training split.
UNTRAINED_X, UNIFORM_X, DATA_X0 = 0.0, 1.0, 4.2    # 1% lands at 2.2, 100% at 4.2
TRAIN_IMAGES = {'CIFAR_10': 24000}                  # size of the 100% training split


def xpos(size):
    return UNIFORM_X if str(size) == 'uniform' else DATA_X0 + np.log10(float(size))


def size_label(size, dataset):
    if str(size) == 'uniform':
        return 'uniform\nnoise'
    frac = float(size)
    n = TRAIN_IMAGES.get(dataset)
    if n is None:
        return f'{frac:.0%}'
    count = frac*n
    return f'{frac:.0%}\n{count/1000:g}k' if count >= 1000 else f'{frac:.0%}\n{count:.0f}'


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
        fracs = [s for s in sizes if str(s) != 'uniform']
        fig, ax = plt.subplots()
        n_losses = g['loss'].nunique()

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
                # no training data involved: a point in the control column, and a level across
                # the plot so every trained point can be read against it
                i = next(k for k, s in enumerate(sizes) if not np.isnan(ys[k]))
                ax.axhline(ys[i], color='black', linestyle=':', linewidth=1)
                ax.errorbar([UNTRAINED_X], [ys[i]], yerr=[errs[i]], marker='s', capsize=3,
                            color='black', linestyle='none', label='RANDOM (untrained)')
                continue
            colour = colours.get(loss, 'grey')
            by_size = dict(zip([str(s) for s in sizes], zip(ys, errs)))
            # real data: a curve on the log axis
            xd = [xpos(s) for s in fracs]
            ax.errorbar(xd, [by_size[str(s)][0] for s in fracs],
                        yerr=[by_size[str(s)][1] for s in fracs],
                        marker='o', capsize=3, label=loss, color=colour)
            # uniform noise: a control, so a point not joined to the curve. jittered so the
            # losses, which all sit near the untrained level, do not hide each other
            if 'uniform' in by_size:
                k = sorted(g['loss'].unique()).index(loss)
                jitter = (k - (n_losses - 1)/2)*0.05
                ax.errorbar([UNIFORM_X + jitter], [by_size['uniform'][0]],
                            yerr=[by_size['uniform'][1]], marker='o', capsize=3,
                            color=colour, linestyle='none')

        chance = chance_level(dataset)
        ax.axhline(chance, color='grey', linestyle='--', linewidth=1)
        ax.annotate('chance', (0, chance), fontsize=8, color='grey', va='bottom')
        if len(untrained):
            base = float(np.nanmean(untrained))
            spread = noise_floor(args.metric)
            ax.axhspan(base - spread, base + spread, color='grey', alpha=0.12)

        ticks = [UNTRAINED_X] + [xpos(s) for s in sizes]
        labels = ['untrained\n(no training)'] + [size_label(s, dataset) for s in sizes]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=9)
        ax.axvline((UNIFORM_X + xpos(fracs[0]))/2 if fracs else 1.8, color='lightgrey',
                   linewidth=1)
        ax.set_xlim(UNTRAINED_X - 0.5, DATA_X0 + 0.4)
        ax.set_xlabel('controls  |  natural images the autoencoder is trained on '
                      '(fraction of the training split, log scale)')
        ax.set_ylabel(f'{args.metric} probe accuracy on frozen encodings')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{dataset} / {net}: does a perceptual loss buy data efficiency?\n'
                     f'{args.metric} probe, {SELECT_TEXT[args.select]}\n'
                     f'{BUDGET_TEXT.get(scaling, scaling)}', fontsize=11)
        ax.legend(fontsize=8, loc='lower right')

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
        # wrap before drawing: savefig(bbox_inches='tight') grows the canvas to contain this
        # block, so one long line silently doubles the figure width and shrinks the axes
        text = '\n'.join(textwrap.fill(line, 118) for line in text.split('\n'))
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
