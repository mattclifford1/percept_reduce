'''
aggregate every run in a saves directory into one table.

plot_training_runs.py draws per-run curves but nothing compares across runs, so every
comparison in FINDINGS.md had to be recomputed ad hoc. this does it once.

usage:
    python plots/summarise_runs.py                    # summarise ./saves
    python plots/summarise_runs.py saves_pre_lossfix  # or any other saves-shaped dir
    python plots/summarise_runs.py saves --metric Linear
    python plots/summarise_runs.py saves --select best   # the old, biased, headline

**the headline is now the final epoch, not the best one.** best-epoch is the maximum over ~16
evals of a metric measured on the probe's own eval set, with a measured +/-0.032 noise floor: it
is optimistically biased, and biased *more* for noisy runs (small data, DISTS) than stable ones,
which is precisely where the comparisons of interest are. every number in FINDINGS.md section 1
is a best-epoch number and should be re-read with that in mind. both columns are printed.
'''
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from run_index import load_long, sorted_sizes


def summarise(long_df, metric='MLP'):
    '''one row per run: untrained baseline, final, val-selected and best epoch'''
    out = []
    keys = ['dataset', 'net', 'loss', 'datasize']
    for key, g in long_df.groupby(keys):
        g = g.sort_values('epoch')
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
        if 'out std' in g.columns and not g['out std'].isna().all():
            row['min_out_std'] = g['out std'].min()   # ~0 means the run collapsed
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default='saves')
    ap.add_argument('--metric', default='MLP', help='KNN, Linear, MLP or NB')
    ap.add_argument('--select', default='final', choices=['final', 'val_sel', 'best'])
    args = ap.parse_args()

    long_df = load_long(args.save_dir)
    if args.metric not in long_df.columns:
        raise SystemExit(f'no {args.metric} column in {args.save_dir} -- older runs predate it')
    summary = summarise(long_df, args.metric)

    pd.set_option('display.width', 200)
    for (dataset, net), g in summary.groupby(['dataset', 'net']):
        print(f'\n=== {dataset} / {net} — {args.select} {args.metric} accuracy ===')
        pivot = g.pivot(index='loss', columns='datasize', values=args.select)
        pivot = pivot[sorted_sizes(pivot.columns)]
        print(pivot.round(4).to_string())

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
    cols = ['dataset', 'net', 'loss', 'datasize', 'epoch0', 'final', 'val_sel', 'best',
            'best_epoch', 'val_MSE', 'n_evals', 'layout']
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
