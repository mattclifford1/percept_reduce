'''
run directories, per-run metadata, results CSV, checkpoints.

## Layout

    saves/{dataset}/{network}/{loss}/{datasize}/bs{bs}_seed{seed}_lr{lr}_ep{epochs}/
        config.json            everything needed to rebuild the run (see write_config)
        training_results.csv   one row per eval epoch
        checkpoint.pt          final weights
        done.json              written last -- the completion marker
        images/{epoch}-.png

each path component is now one field. the old layout packed three fields into one directory
name (`{LOSS}-{datasize}-BS{bs}`) and every reader re-parsed it by splitting on '-', which is
what forced the "no '-' in a loss or network name" rule and left nowhere to put the seed. that
rule no longer applies to anything the writer does: identity lives in config.json, and the
directory name only has to be unique and readable.

## Legacy runs

everything committed before this change is in the old layout. `previously_done` checks it too,
so an un-migrated grid is still skipped rather than silently re-run (which is exactly the
accident T1.0 documents). `utils/migrate_saves.py` converts old -> new; readers handle both.

## Completion (FINDINGS B10)

"done" used to mean "training_results.csv exists", so a crashed run was skipped forever with a
partial CSV. done.json is written only after the last epoch, and a CSV without one is treated as
a failed attempt: it is moved aside to training_results.csv.partial-* and the run starts again.
'''

import json
import os
import platform
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _git_sha():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return f'{sha}-dirty' if dirty else sha
    except Exception:      # not a repo, no git, whatever -- never fail a run over provenance
        return 'unknown'


def variant_name(batch_size, seed, lr, epochs):
    '''leaf directory name. nothing parses it -- config.json is the source of truth'''
    return f'bs{batch_size}_seed{seed}_lr{lr:g}_ep{epochs}'


def legacy_dir(base_save, dataset, network, loss, datasize, batch_size):
    return os.path.join(base_save, dataset, network, f'{loss}-{datasize}-BS{batch_size}')


class train_saver:
    def __init__(self, epochs, loss, network, batch_size, datasize, dataset='dev',
                 seed=42, lr=1e-3, base_save='saves'):
        self.epochs = epochs
        self.loss = loss
        self.network = network
        self.batch_size = batch_size
        self.datasize = datasize
        self.dataset = dataset
        self.seed = seed
        self.lr = lr

        self.unique_name = [f'{dataset}', f'{network}', f'{loss}', f'{datasize}',
                            variant_name(batch_size, seed, lr, epochs)]
        self.save_dir = os.path.join(base_save, *self.unique_name)
        self.image_dir = os.path.join(self.save_dir, 'images')
        self.csv_file = os.path.join(self.save_dir, 'training_results.csv')
        self.config_file = os.path.join(self.save_dir, 'config.json')
        self.done_file = os.path.join(self.save_dir, 'done.json')
        self.checkpoint_file = os.path.join(self.save_dir, 'checkpoint.pt')

        self.legacy_dir = legacy_dir(base_save, dataset, network, loss, datasize, batch_size)
        self.legacy_csv = os.path.join(self.legacy_dir, 'training_results.csv')

        self.previously_done = self._check_done()
        self.image_save_counter = 0
        self.start_time = time.time()

    def _check_done(self):
        if os.path.exists(self.done_file):
            return True
        if os.path.exists(self.legacy_csv):
            # pre-migration run. trust it: it was written before done.json existed
            return True
        if os.path.exists(self.csv_file):
            # a CSV with no done.json is a crashed run (B10). move it aside and start over
            # rather than skipping this cell forever or merging into half a table.
            partial = f'{self.csv_file}.partial-{int(time.time())}'
            os.rename(self.csv_file, partial)
            print(f'incomplete run found, moved {self.csv_file} -> {partial}, re-running')
        return False

    def _ensure_dirs(self):
        # lazily -- a skipped run should not leave an empty directory behind
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def write_config(self, extra=None):
        '''
        everything needed to rebuild the run, plus provenance. written at the start so a
        crashed run still says what it was trying to do.
        '''
        self._ensure_dirs()
        config = {
            'dataset': self.dataset,
            'network': self.network,
            'loss': self.loss,
            'datasize': self.datasize,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'lr': self.lr,
            'epochs': self.epochs,
            'git_sha': _git_sha(),
            'torch': torch.__version__,
            'python': platform.python_version(),
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        if extra != None:
            config.update(extra)
        # B7: validate_every silently selects the train/val/test proportions, and the split is
        # not part of the run path -- two runs with different splits still collide here. can't
        # fix that without putting the split in the directory name, but it can be made loud.
        if os.path.exists(self.config_file):
            with open(self.config_file) as f:
                old = json.load(f)
            for key in ('split_props', 'validate_every', 'data_percent'):
                if key in old and old.get(key) != config.get(key):
                    print(f'WARNING {self.save_dir}: {key} was {old.get(key)}, now '
                          f'{config.get(key)} -- these runs are not comparable and share a path')
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        return config

    def write_done(self, extra=None):
        '''written last. this -- not the CSV -- is what previously_done means'''
        self._ensure_dirs()
        info = {
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
            'wall_seconds': round(time.time() - self.start_time, 1),
        }
        if os.path.exists(self.csv_file):
            info['n_eval_rows'] = len(pd.read_csv(self.csv_file))
        if extra != None:
            info.update(extra)
        with open(self.done_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        return info

    def save_checkpoint(self, net):
        '''
        final weights. the whole grid used to be discarded at the end of every run, so
        re-probing with a different classifier, encoder transfer (TODO T3.3), or reproducing any
        published number all meant retraining the cell. see testing/reprobe.py for the payoff.
        '''
        self._ensure_dirs()
        torch.save({'state_dict': net.state_dict(),
                    'network': self.network,
                    'latent_dim': getattr(net, 'latent_dim', None)}, self.checkpoint_file)

    def write_images(self, torch_images:list, epoch, num_ims=4, extra_name=''):
        self._ensure_dirs()
        try:
            _, axes_list = plt.subplots(len(torch_images), num_ims)
            for i, torch_tensor in enumerate(torch_images):
                for j in range(num_ims):
                    np_im = torch_tensor[j, :, :, :].detach().cpu().numpy()
                    np_im = np.squeeze(np_im)
                    np_im = np.transpose(np_im, (1, 2, 0))
                    np_im = np.clip(np_im, 0, 1)
                    axes_list[i, j].imshow(np_im)

            plt.savefig(os.path.join(self.image_dir, f'{epoch}-{extra_name}.png'))
            self.image_save_counter += 1
            plt.close()
        except RuntimeError:
            print(f"fig {os.path.join(self.image_dir, f'{epoch}-{extra_name}.png')} failed to save")

    def write_scores(self, scores, epoch):
        self._ensure_dirs()
        scores['epoch'] = epoch
        for name, item in scores.items():
            scores[name] = [item]
        df = pd.DataFrame.from_dict(scores)
        df.set_index('epoch', inplace=True)
        save_and_merge_df_as_csv(df, self.csv_file)


def save_and_merge_df_as_csv(df, file):
    '''df need to be indexed'''
    if os.path.exists(file):
        df_saved = pd.read_csv(file)
        index_name = df.index.name
        df = pd.merge(df.reset_index(), df_saved, how='outer')
        df.set_index(index_name, inplace=True)
    df.to_csv(file, index=True)
