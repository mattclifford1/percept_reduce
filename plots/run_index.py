'''
find runs in a saves/ tree and load them into one long dataframe.

handles **both** layouts, because the two coexist until utils/migrate_saves.py is run:

    legacy  saves/{dataset}/{net}/{LOSS}-{datasize}-BS{bs}/training_results.csv
    current saves/{dataset}/{net}/{loss}/{datasize}/{variant}/training_results.csv

for current-layout runs the metadata comes from config.json, which is authoritative. the leaf
directory name is only parsed as a fallback, and the legacy path is parsed by splitting on '-'
the way it always was -- which is exactly the fragility the new layout exists to remove.

every reader (plot_training_runs, plot_data_efficiency, summarise_runs) goes through here, so
none of them re-implement the parsing.
'''
import json
import os
from glob import glob

import pandas as pd

RESULTS = 'training_results.csv'


def _read_config(run_dir):
    path = os.path.join(run_dir, 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _parse_variant(name):
    '''bs32_seed42_lr0.001_ep30 -> dict. fallback only; config.json is the real source'''
    out = {}
    for field, key, cast in (('bs', 'batch_size', int), ('seed', 'seed', int),
                             ('lr', 'lr', float), ('ep', 'epochs', int)):
        for part in name.split('_'):
            if part.startswith(field):
                try:
                    out[key] = cast(part[len(field):])
                except ValueError:
                    pass
    return out


def iter_runs(save_dir):
    '''yield one dict of metadata per run directory found, newest layout first'''
    for results in sorted(glob(os.path.join(save_dir, '*', '*', '*', '*', '*', RESULTS))):
        run_dir = os.path.dirname(results)
        parts = results.split(os.sep)
        dataset, net, loss, datasize, variant = parts[-6], parts[-5], parts[-4], parts[-3], parts[-2]
        meta = {'dataset': dataset, 'net': net, 'loss': loss, 'datasize': datasize,
                'layout': 'current', 'run_dir': run_dir, 'results': results,
                'done': os.path.exists(os.path.join(run_dir, 'done.json')),
                'checkpoint': os.path.exists(os.path.join(run_dir, 'checkpoint.pt'))}
        meta.update(_parse_variant(variant))
        meta.update(_read_config(run_dir))   # config wins where it exists
        yield meta

    for results in sorted(glob(os.path.join(save_dir, '*', '*', '*', RESULTS))):
        run_dir = os.path.dirname(results)
        parts = results.split(os.sep)
        dataset, net, cfg = parts[-4], parts[-3], parts[-2]
        if cfg.count('-') != 2:
            continue          # not a legacy run directory
        loss, datasize, batch = cfg.split('-')
        yield {'dataset': dataset, 'net': net, 'loss': loss, 'datasize': datasize,
               'batch_size': int(batch[2:]) if batch[2:].isdigit() else batch[2:],
               'seed': None, 'lr': None, 'epochs': None,
               'layout': 'legacy', 'run_dir': run_dir, 'results': results,
               'done': True, 'checkpoint': False}


def load_long(save_dir, require_done=False):
    '''one row per (run, epoch), with the run metadata attached to every row'''
    frames = []
    for meta in iter_runs(save_dir):
        if require_done and meta['done'] == False:
            continue
        df = pd.read_csv(meta['results']).sort_values('epoch')
        for key in ('dataset', 'net', 'loss', 'datasize', 'batch_size', 'seed', 'lr',
                    'layout', 'run_dir'):
            df[key] = meta.get(key)
        frames.append(df)
    if len(frames) == 0:
        raise SystemExit(f'no {RESULTS} found under {save_dir}')
    return pd.concat(frames, ignore_index=True)


# datasizes in a sensible order rather than alphabetical
SIZE_ORDER = ['0.01', '0.1', '0.5', '1', 'uniform']


def sorted_sizes(sizes):
    return sorted(sizes, key=lambda s: SIZE_ORDER.index(str(s)) if str(s) in SIZE_ORDER else 99)


def chance_level(dataset):
    '''accuracy of always predicting one class -- the line every probe number must clear'''
    if 'IMAGENET' in dataset:
        return 1/1000
    return 1/10


# measured run-to-run spread of the untrained-encoder probe on CIFAR (FINDINGS B5).
# until multi-seed runs exist this is the only honest error bar available.
# two caveats, both meaning this is the best available number rather than the right one:
#   - it was measured on *unstandardised* probe features, i.e. before FINDINGS B13
#   - it came from the old unseeded runs. seeding makes every cell's epoch-0 identical, so the
#     current grid has a measured spread of exactly 0 and cannot estimate its own noise floor
# re-measure with the multi-seed RANDOM control (TODO T1.4).
NOISE_FLOOR = 0.032
