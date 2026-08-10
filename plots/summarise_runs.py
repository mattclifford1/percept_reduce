'''
aggregate every run in a saves directory into one table.

plot_training_runs.py draws per-run curves but nothing compares across runs, so every
comparison in FINDINGS.md had to be recomputed ad hoc. this does it once.

usage:
    python plots/summarise_runs.py                    # summarise ./saves
    python plots/summarise_runs.py saves_legacy/gen1_original  # or any saves-shaped dir
    python plots/summarise_runs.py saves --metric Linear
    python plots/summarise_runs.py saves --select best   # the old, biased, headline

**the headline is now the final epoch, not the best one.** best-epoch is the maximum over ~16
evals of a metric measured on the probe's own eval set, with a measured +/-0.032 noise floor: it
is optimistically biased, and biased *more* for noisy runs (small data, DISTS) than stable ones,
which is precisely where the comparisons of interest are. every number in FINDINGS.md section 1
is a best-epoch number and should be re-read with that in mind. all four columns are printed:

    final       the last eval. the honest default.
    early_stop  epoch picked on the probe's *select* split, reported on the disjoint report
                split -- unbiased early stopping. only exists for runs made after the three-way
                probe split landed.
    val_sel     epoch picked by lowest val MSE. unbiased, but a weak selector.
    best        max over all evals. **optimistic** -- an oracle that early-stops using the
                reported numbers themselves. read `best - final` as a measure of peak-then-
                degrade instability, not as an achievable score.
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from run_index import load_long, sorted_sizes, budget_view

# the two optimisation budgets are different experiments and must never share a table: the
# fixed-budget grid confounds data size with gradient steps (FINDINGS B4), the equal-steps grid
# holds steps constant. the *difference* between them is the data-efficiency result (TODO T1.2).
BUDGET_TEXT = {'fixed': 'fixed budget: 30 epochs over each cell\'s own data',
               'equal_steps': 'equal steps: epochs scaled by 1/data_percent'}


def summarise(long_df, metric='MLP'):
    '''
    one row per run: untrained baseline, final, val-selected and best epoch

    grouped by `run_dir`, which is the only key guaranteed unique per run. grouping on
    (dataset, net, loss, datasize) was correct while there was one seed and one budget, and
    became silently wrong the moment the grid had three seeds and two budgets: it concatenated
    six runs and took .iloc[-1] as 'final', which returns whichever ran longest -- always the
    3000-epoch equal_steps cell -- while 'best' became a maximum over six runs.
    '''
    out = []
    keys = ['dataset', 'net', 'loss', 'datasize', 'epoch_scaling', 'seed', 'run_dir']
    for key, g in long_df.groupby(keys, dropna=False):
        # runs predating a probe have no column for it -- an all-NaN group, not a zero
        g = g.sort_values('epoch').dropna(subset=[metric])
        if len(g) == 0:
            continue
        row = dict(zip(keys, key), **{
            'epoch0': g.iloc[0][metric],            # untrained net = the random-encoder baseline
            'final': g.iloc[-1][metric],
            'best': g[metric].max(),                # biased -- see module docstring
            'best_epoch': int(g.loc[g[metric].idxmax(), 'epoch']),
            'val_MSE': g.iloc[-1]['val MSE'],
            'n_evals': len(g),
            'layout': g.iloc[0]['layout'],
        })
        # epoch chosen by val MSE: unbiased w.r.t. the probe, but a weak selector (FINDINGS
        # reports reconstruction quality correlates poorly with probe accuracy)
        row['val_sel'] = g.loc[g['val MSE'].idxmin(), metric]
        # epoch chosen on the probe's own *select* split and reported on the disjoint report
        # split. this is the honest early-stopped number -- 'best' is the same idea with the
        # two splits collapsed into one, which is what makes 'best' biased.
        sel_col = f'{metric} select'
        if sel_col in g.columns and not g[sel_col].isna().all():
            row['early_stop'] = g.loc[g[sel_col].idxmax(), metric]
            row['early_stop_epoch'] = int(g.loc[g[sel_col].idxmax(), 'epoch'])
        else:
            row['early_stop'] = np.nan          # runs predating the three-way split
        # probe protocol. the select columns only exist under v2, so this works for runs whose
        # config.json predates probe_version (and for legacy runs, which have no config at all).
        row['probe'] = 2 if sel_col in g.columns and not g[sel_col].isna().all() else 1
        if 'out std' in g.columns and not g['out std'].isna().all():
            row['min_out_std'] = g['out std'].min()   # ~0 means the run collapsed
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default='saves')
    ap.add_argument('--metric', default='MLP', help='KNN, Linear, MLP or NB')
    ap.add_argument('--select', default='final',
                    choices=['final', 'early_stop', 'val_sel', 'best'])
    args = ap.parse_args()

    long_df = load_long(args.save_dir)
    if args.metric not in long_df.columns:
        raise SystemExit(f'no {args.metric} column in {args.save_dir} -- older runs predate it')
    summary = summarise(long_df, args.metric)

    pd.set_option('display.width', 200)
    panels = []
    for (dataset, net), whole in summary.groupby(['dataset', 'net']):
        for scaling in sorted(whole['epoch_scaling'].dropna().unique()):
            panels.append((dataset, net, scaling, budget_view(whole, scaling)))

    for dataset, net, scaling, g in panels:
        budget = BUDGET_TEXT.get(scaling, scaling)
        print(f'\n=== {dataset} / {net} — {args.select} {args.metric} accuracy [{budget}] ===')
        # mean +/- sd over seeds. one cell is now several runs, and collapsing them by pivoting
        # would just pick an arbitrary one; the spread across seeds is also the only honest
        # error bar this experiment has.
        agg = (g.groupby(['loss', 'datasize'], dropna=False)[args.select]
                .agg(['mean', 'std', 'count']).reset_index())
        cells = agg.apply(lambda r: f'{r["mean"]:.4f}' if r['count'] < 2
                          else f'{r["mean"]:.4f}+/-{r["std"]:.4f}', axis=1)
        agg = agg.assign(cell=cells)
        pivot = agg.pivot(index='loss', columns='datasize', values='cell')
        pivot = pivot[sorted_sizes(pivot.columns)]
        print(pivot.fillna('-').to_string())
        seeds = sorted({s for s in g['seed'] if pd.notna(s)})
        n = agg['count']
        print(f'  seeds: {seeds if seeds else "unrecorded (legacy)"}; '
              f'{n.min()}-{n.max()} runs per cell'
              + ('  (+/- is sd over seeds)' if n.max() > 1 else ''))

        # a table that mixes probe protocols is not a table. v1 numbers were measured on
        # unstandardised features with an under-fit MLP (FINDINGS B13); v2 numbers are not
        # comparable to them, in either direction.
        if g['probe'].nunique() > 1:
            counts = g['probe'].value_counts().sort_index()
            print('  !! MIXED PROBE PROTOCOLS -- rows above are not comparable to each other.')
            print('     ' + ', '.join(f'v{v}: {n} runs' for v, n in counts.items())
                  + '  (v1 = pre-standardisation, FINDINGS B13)')
            minority = counts.idxmin()
            if counts.min() <= 8:
                names = sorted(f'{r.loss}@{r.datasize}' for r in
                               g[g['probe'] == minority].itertuples())
                print(f'     v{minority}: ' + ', '.join(names))

        # the untrained-network spread is the run-to-run noise floor for this grid
        base = g['epoch0']
        print(f'  untrained baseline (epoch 0): {base.mean():.4f} +/- {base.std():.4f} '
              f'over {len(base)} runs')
        if args.select != 'best':
            inflation = (g['best'] - g[args.select]).mean()
            print(f'  best-epoch selection would add {inflation:+.4f} on average '
                  f'(this is selection bias, not signal)')

        # runs that never got off the ground
        floor = g[g[args.select] <= base.mean() + base.std()]
        if len(floor):
            print('  did not beat the untrained baseline: '
                  + ', '.join(f'{r.loss}@{r.datasize}' for r in floor.itertuples()))
        if 'min_out_std' in g.columns:
            dead = g[g['min_out_std'] < 1e-4]
            if len(dead):
                print('  collapsed output (batch std ~ 0): '
                      + ', '.join(f'{r.loss}@{r.datasize}' for r in dead.itertuples()))

    print(f'\n=== all runs, {args.metric} ===')
    cols = ['dataset', 'net', 'loss', 'datasize', 'epoch_scaling', 'seed', 'epoch0', 'final',
            'early_stop', 'val_sel', 'best', 'best_epoch', 'val_MSE', 'n_evals', 'layout']
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
