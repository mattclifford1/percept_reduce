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


def norm_size(size):
    '''
    datasize as a canonical string. legacy runs carry it as the string parsed out of the
    directory name; config.json carries whatever type was passed to run(). without this, '1' and
    1 land in two separate columns of the same pivot table.
    '''
    try:
        value = float(size)
    except (TypeError, ValueError):
        return str(size)          # 'uniform'
    return str(int(value)) if value == int(value) else str(value)


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
        meta['datasize'] = norm_size(meta['datasize'])
        yield meta

    for results in sorted(glob(os.path.join(save_dir, '*', '*', '*', RESULTS))):
        run_dir = os.path.dirname(results)
        parts = results.split(os.sep)
        dataset, net, cfg = parts[-4], parts[-3], parts[-2]
        if cfg.count('-') != 2:
            continue          # not a legacy run directory
        loss, datasize, batch = cfg.split('-')
        yield {'dataset': dataset, 'net': net, 'loss': loss, 'datasize': norm_size(datasize),
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


def probe_version(df, metric):
    '''
    which probe protocol produced these rows. the '<metric> select' column only exists under v2,
    so this works for legacy runs and for configs written before probe_version was recorded.
      1  raw features, single 67/33 eval split      -- FINDINGS.md section 1
      2  StandardScaler + 67/16.5/16.5 fit/select/report
    '''
    col = f'{metric} select'
    if col in df.columns and not df[col].isna().all():
        return 2
    return 1


def caption(df, dataset, metric, extra=''):
    '''
    the block of text that makes a figure readable without the repo open. every figure carries
    it: which probe produced the numbers, what the reference marks mean, and a loud warning if
    the panel is mixing protocols that are not comparable.
    '''
    lines = []
    v2 = probe_version(df, metric) == 2
    if v2:
        lines.append(f'{metric} = accuracy on the probe report split (probe v2: StandardScaler, '
                     f'67% fit / 16.5% select / 16.5% report of the held-out test split).')
    else:
        lines.append(f'{metric} = accuracy on a 33% eval split of the held-out test split '
                     f'(probe v1: raw features, no standardisation -- see FINDINGS.md B13).')
    lines.append(f'dashed grey = chance ({chance_level(dataset):.3g}).  '
                 f'shaded band = untrained encoder +/-{noise_floor(metric)} '
                 f'(seed-to-seed spread of the RANDOM control, 5 seeds).  '
                 f'dotted = RANDOM, the untrained-encoder control.')
    if extra != '':
        lines.append(extra)
    return '\n'.join(lines)


MIXED_PROTOCOL_WARNING = ('!! this panel mixes probe v1 and v2 runs -- they are NOT comparable '
                          '(FINDINGS.md B13: standardising moved untrained MLP 0.162 -> 0.421)')


def chance_level(dataset):
    '''accuracy of always predicting one class -- the line every probe number must clear'''
    if 'IMAGENET' in dataset:
        return 1/1000
    return 1/10


# run-to-run spread of the untrained-encoder probe, measured directly: 5 seeds of the RANDOM
# control on CIFAR-10 / conv_big_z, standardised three-way probe.
#   saves/CIFAR_10/conv_big_z/RANDOM/1/bs32_seed{1,2,3,4,42}_lr0.001_ep0/
# this replaces the old ±0.032, which came from unseeded runs on an unstandardised probe and so
# mixed init variance with probe under-fitting (FINDINGS B13). it is init variance only -- it
# does not cover run-to-run variation from shuffling during training.
NOISE_FLOOR_BY_METRIC = {'KNN': 0.010, 'Linear': 0.010, 'MLP': 0.007, 'NB': 0.021}
NOISE_FLOOR = 0.021    # the widest of them, for a metric-agnostic band


def noise_floor(metric):
    return NOISE_FLOOR_BY_METRIC.get(metric, NOISE_FLOOR)
