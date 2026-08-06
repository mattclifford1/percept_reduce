'''
re-run the probes on a saved checkpoint, without retraining.

this is what checkpointing buys. changing a probe used to mean re-running the whole grid --
30 epochs per cell to answer a question about a five-second sklearn fit. now:

    python percept_loss/testing/reprobe.py saves --all       # every run with a checkpoint
    python percept_loss/testing/reprobe.py saves/CIFAR_10/conv_big_z/MSE/1/bs32_seed42_lr0.001_ep30

writes probe_results.csv next to the checkpoint. it is deliberately a *separate* file from
training_results.csv: those rows were produced by the probe configuration in force at training
time, and overwriting them would erase the comparison.

only current-layout runs have checkpoints -- legacy runs were trained before weights were saved
and cannot be re-probed at all.
'''
import argparse
import json
import os
import time
from glob import glob

import pandas as pd
import torch

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.networks import CIFAR_AUTOENCODERS, IMAGENET64_AUTOENCODERS
from percept_loss.testing.benchmark_encodings import test_all_classifiers
from percept_loss.testing.encoded_dataset import make_encodings


def registry_for(dataset):
    return IMAGENET64_AUTOENCODERS if 'IMAGENET' in dataset else CIFAR_AUTOENCODERS


def reprobe_run(run_dir, device, verbose=True):
    with open(os.path.join(run_dir, 'config.json')) as f:
        config = json.load(f)

    net = registry_for(config['dataset'])[config['network']]()
    checkpoint = torch.load(os.path.join(run_dir, 'checkpoint.pt'), map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(device)
    net.eval()

    # same split call as training -- get_indicies is seeded, so the test split is identical
    _, _, test_dataloader, _ = get_all_loaders(train_percept_reduce=config['data_percent'],
                                               device=device,
                                               batch_size=config.get('batch_size', 32),
                                               props=config.get('split_props', [0.4, 0.3, 0.3]),
                                               dataset=config['dataset'])

    X, y = make_encodings(test_dataloader, net, device)
    scores = test_all_classifiers(data=(X, y), verbose=verbose)
    scores['probed'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out = os.path.join(run_dir, 'probe_results.csv')
    pd.DataFrame([scores]).to_csv(out, index=False)
    return scores, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='a run directory, or a saves dir with --all')
    ap.add_argument('--all', action='store_true', help='every run under path that has a checkpoint')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.all:
        run_dirs = [os.path.dirname(p) for p in
                    sorted(glob(os.path.join(args.path, '*', '*', '*', '*', '*', 'checkpoint.pt')))]
    else:
        run_dirs = [args.path]
    if len(run_dirs) == 0:
        raise SystemExit(f'no checkpoints under {args.path} (legacy runs have none)')

    failed = []
    for run_dir in run_dirs:
        # one unreadable run must not take out a batch of 35
        try:
            scores, out = reprobe_run(run_dir, device)
        except Exception as e:
            print(f'{run_dir}: FAILED ({type(e).__name__}: {e})')
            failed.append(run_dir)
            continue
        print(f'{run_dir}: ' + ', '.join(f'{k}={v:.4f}' for k, v in scores.items()
                                         if isinstance(v, float)))
        print(f'  -> {out}')
    if len(failed):
        print(f'\n{len(failed)} of {len(run_dirs)} runs failed to re-probe:')
        for run_dir in failed:
            print(f'  {run_dir}')


if __name__ == '__main__':
    main()
