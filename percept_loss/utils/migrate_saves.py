'''
move legacy run directories into the current layout.

    saves/{dataset}/{net}/{LOSS}-{datasize}-BS{bs}/
      ->
    saves/{dataset}/{net}/{LOSS}/{datasize}/bs{bs}_seed{seed}_lr{lr}_ep{epochs}/

and write the config.json those runs never had. **dry run by default**:

    python percept_loss/utils/migrate_saves.py saves           # show what would move
    python percept_loss/utils/migrate_saves.py saves --apply   # actually move

DO NOT run this while a sweep is writing to the same saves/ tree -- a live run holds its output
path open and will simply recreate the old directory, splitting its results across two places.

## What is guessed

legacy directories record only loss, datasize and batch size. seed, lr and epoch count were
never written down anywhere, so they are taken from --seed/--lr/--epochs (defaulting to what the
committed runs used: 42, Adam's 1e-3 default, 30 epochs) and the config is marked
`"inferred": true`. a migrated run therefore claims less provenance than a new one, which is
accurate -- that information genuinely does not exist for those runs.

`done.json` is written with `"migrated": true` so the runs are not re-run; nothing about a
legacy CSV can prove it is complete beyond its row count, which is also recorded.
'''
import argparse
import json
import os
import shutil
import time
from glob import glob

import pandas as pd

from percept_loss.utils.savers import variant_name


def find_legacy(save_dir):
    out = []
    for results in sorted(glob(os.path.join(save_dir, '*', '*', '*', 'training_results.csv'))):
        run_dir = os.path.dirname(results)
        cfg = os.path.basename(run_dir)
        if cfg.count('-') != 2:
            continue
        loss, datasize, batch = cfg.split('-')
        parts = run_dir.split(os.sep)
        out.append({'run_dir': run_dir, 'dataset': parts[-3], 'net': parts[-2],
                    'loss': loss, 'datasize': datasize,
                    'batch_size': int(batch[2:]) if batch[2:].isdigit() else 32})
    return out


def target_dir(save_dir, run, seed, lr, epochs):
    return os.path.join(save_dir, run['dataset'], run['net'], run['loss'], run['datasize'],
                        variant_name(run['batch_size'], seed, lr, epochs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('save_dir', nargs='?', default='saves')
    ap.add_argument('--apply', action='store_true', help='without this, only prints the plan')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--epochs', type=int, default=30)
    args = ap.parse_args()

    runs = find_legacy(args.save_dir)
    if len(runs) == 0:
        print(f'no legacy run directories under {args.save_dir}')
        return

    moved, skipped = 0, 0
    for run in runs:
        dest = target_dir(args.save_dir, run, args.seed, args.lr, args.epochs)
        if os.path.exists(dest):
            print(f'SKIP  {run["run_dir"]} -> {dest} (target exists)')
            skipped += 1
            continue
        print(f'MOVE  {run["run_dir"]} -> {dest}')
        if args.apply == False:
            continue

        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.move(run['run_dir'], dest)
        n_rows = len(pd.read_csv(os.path.join(dest, 'training_results.csv')))
        # data_percent must come back as a number ('uniform' stays a string) -- testing/reprobe.py
        # hands it straight to get_all_loaders, which branches on isinstance(..., str)
        try:
            data_percent = float(run['datasize'])
        except ValueError:
            data_percent = run['datasize']
        with open(os.path.join(dest, 'config.json'), 'w') as f:
            json.dump({'dataset': run['dataset'], 'network': run['net'], 'loss': run['loss'],
                       'datasize': run['datasize'], 'data_percent': data_percent,
                       'batch_size': run['batch_size'], 'seed': args.seed, 'lr': args.lr,
                       'epochs': args.epochs, 'inferred': True,
                       'note': 'migrated from the legacy layout; seed/lr/epochs were never '
                               'recorded and are defaults, not measurements'}, f, indent=2)
        with open(os.path.join(dest, 'done.json'), 'w') as f:
            json.dump({'migrated': True, 'n_eval_rows': n_rows,
                       'migrated_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        moved += 1

    verb = 'moved' if args.apply else 'would move'
    print(f'\n{verb} {len(runs) - skipped} runs, skipped {skipped}')
    if args.apply == False:
        print('dry run -- pass --apply to do it')


if __name__ == '__main__':
    main()
