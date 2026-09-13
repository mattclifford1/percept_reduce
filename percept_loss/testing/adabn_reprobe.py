'''
AdaBN re-probe: on a BatchNorm network, does the uniform-noise result measure the noise, or the
noise's BatchNorm statistics?

a BatchNorm encoder is probed in eval mode, where it normalises with running statistics
accumulated during training. for a net trained on uniform noise those are *noise* statistics,
then applied to real images at probe time. this recomputes the statistics on real training
images -- no gradient steps, weights untouched -- and re-probes (the AdaBN trick: Li et al.
2016, "Revisiting Batch Normalization for Practical Domain Adaptation").

run on, for each BatchNorm backbone:
  uniform    every loss -- the question
  RANDOM     untrained: default statistics make BatchNorm the identity at eval, so this is the
             fair baseline for an AdaBN'd noise net
  MSE, LPIPS at 100% data -- a sanity check; trained on real images, they should barely move

    python percept_loss/testing/adabn_reprobe.py           # every candidate run
    python percept_loss/testing/adabn_reprobe.py --list    # just show which runs

writes adabn_results.csv next to each checkpoint: a 'plain' row (the checkpoint probed as it was
trained) and an 'adabn' row, from the same probe code, so the comparison is like for like.
runs that already have the file are skipped, so re-running after new runs land is cheap.
'''
import argparse
import json
import os
import time
from glob import glob

import pandas as pd
import torch
import torch.nn as nn

from percept_loss.datasets.torch_loaders import get_all_loaders, get_preloaded
from percept_loss.networks import CIFAR_AUTOENCODERS
from percept_loss.testing.benchmark_encodings import test_all_classifiers
from percept_loss.testing.encoded_dataset import make_encodings

BN_NETS = ['dcgan', 'resnet18', 'vae']
PATTERNS = ['{net}/*/uniform/*_ep30/checkpoint.pt',
            '{net}/RANDOM/1/*/checkpoint.pt',
            '{net}/MSE/1/*_ep30/checkpoint.pt',
            '{net}/LPIPS/1/*_ep30/checkpoint.pt']
OUT = 'adabn_results.csv'


def candidates(root):
    found = []
    for net in BN_NETS:
        for pattern in PATTERNS:
            found += sorted(glob(os.path.join(root, pattern.format(net=net))))
    return [os.path.dirname(p) for p in found]


def probe(net, loader, device):
    X, y = make_encodings(loader, net, device)
    return test_all_classifiers(data=(X, y))


def recompute_bn_stats(net, loader, device):
    '''fresh running statistics from real images, as a plain average over one pass'''
    bns = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.train()
    with torch.no_grad():
        for data in loader:
            net.encoder_forward(data[0].to(device))
    net.eval()
    return len(bns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=os.path.join('saves', 'CIFAR_10'))
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    run_dirs = candidates(args.root)
    todo = [r for r in run_dirs if not os.path.exists(os.path.join(r, OUT))]
    print(f'{len(run_dirs)} candidate runs, {len(todo)} without {OUT}')
    if args.list:
        for r in todo:
            print(' ', r)
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre = get_preloaded(dataset='CIFAR_10', device=device)
    # the real images the statistics are re-estimated on: the full training split
    real_train, _, _, _ = get_all_loaders(train_percept_reduce=1, device=device,
                                          pre_loaded_images=pre, dataset='CIFAR_10')
    failed = []
    for run_dir in todo:
        try:
            with open(os.path.join(run_dir, 'config.json')) as f:
                config = json.load(f)
            net = CIFAR_AUTOENCODERS[config['network']]()
            checkpoint = torch.load(os.path.join(run_dir, 'checkpoint.pt'), map_location=device)
            net.load_state_dict(checkpoint['state_dict'])
            net.to(device)
            net.eval()
            # the same test split the run was probed on during training
            _, _, test_loader, _ = get_all_loaders(
                train_percept_reduce=config['data_percent'], device=device,
                batch_size=config.get('batch_size', 32),
                props=config.get('split_props', [0.4, 0.3, 0.3]),
                pre_loaded_images=pre, dataset='CIFAR_10')
            rows = [dict(mode='plain', **probe(net, test_loader, device))]
            n_bn = recompute_bn_stats(net, real_train, device)
            rows.append(dict(mode='adabn', **probe(net, test_loader, device)))
            for r in rows:
                r.update(n_bn_layers=n_bn, probed=time.strftime('%Y-%m-%d %H:%M:%S'))
            pd.DataFrame(rows).to_csv(os.path.join(run_dir, OUT), index=False)
            print(f'{run_dir}: MLP plain {rows[0]["MLP"]:.4f} -> adabn {rows[1]["MLP"]:.4f}',
                  flush=True)
        except Exception as e:
            print(f'{run_dir}: FAILED ({type(e).__name__}: {e})', flush=True)
            failed.append(run_dir)
    print(f'done, {len(failed)} failed')


if __name__ == '__main__':
    main()
