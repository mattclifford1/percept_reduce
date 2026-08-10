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


# the old layout had nowhere to put a seed or an lr, so a legacy directory can only stand in
# for the one configuration that generation actually ran. without this, a legacy CSV answered
# "done" for *every* seed and *every* lr of that cell, and a sweep silently produced no runs at
# all -- verified: 8/8 sampled (seed, lr) variants returned previously_done=True.
LEGACY_SEED = 42
LEGACY_LR = 1e-3


def legacy_matches_current_probe(csv_file):
    '''
    can this legacy CSV stand in for a run of the *current* probe?

    a legacy directory has no config.json, so it cannot say which probe protocol produced it --
    and a run is only "already done" if the numbers in it are the numbers we would produce now.
    the probe rewrite (FINDINGS B13) is exactly such a change: standardising the features moved
    untrained MLP accuracy from 0.162 to 0.421, and the three-way split added a '<metric> select'
    column per classifier. the header is therefore a direct, dependency-free test of protocol:
    v2 runs have select columns, v1 runs do not.

    this is the fix for a real incident, not a hypothetical. every conv_big_z seed-42 cell of the
    big re-run was skipped -- seed and lr matched a legacy directory, so 35 cells reported "done"
    on the strength of probe-v1 numbers that the re-run existed to replace. the whole seed-42
    column was missing before anyone noticed.
    '''
    try:
        with open(csv_file) as f:
            return ' select' in f.readline()
    except OSError:
        return False


def _process_alive(pid):
    try:
        os.kill(pid, 0)          # signal 0 tests existence without touching the process
        return True
    except (OSError, TypeError):
        return False


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
        self.lock_file = os.path.join(self.save_dir, 'running.lock')

        self.previously_done = self._check_done()
        if self.previously_done == False and self._claim() == False:
            # another process is already on this cell
            self.previously_done = True
            self.skip_reason = 'locked'
        self.image_save_counter = 0
        self.start_time = time.time()

    def _check_done(self):
        # why a run is (not) being skipped, so a caller can report it. a silent skip is how the
        # seed-42 incident stayed invisible: 35 cells "passed" and nothing said why.
        self.skip_reason = None
        if os.path.exists(self.done_file):
            self.skip_reason = 'done.json'
            return True
        if os.path.exists(self.legacy_csv):
            # pre-migration run. trust it only for a configuration that generation could
            # actually have produced -- same seed, same lr, *and* the same probe protocol.
            wrong_variant = self.seed != LEGACY_SEED or self.lr != LEGACY_LR
            old_probe = legacy_matches_current_probe(self.legacy_csv) == False
            if wrong_variant == False and old_probe == False:
                self.skip_reason = 'legacy'
                return True
            why = []
            if wrong_variant:
                why.append(f'it only covers seed={LEGACY_SEED} lr={LEGACY_LR:g}')
            if old_probe:
                why.append('it predates the current probe (FINDINGS B13), so its numbers are '
                           'not the numbers this run would produce')
            print(f'legacy run at {self.legacy_dir} ignored for seed={self.seed} '
                  f'lr={self.lr:g} -- ' + '; '.join(why) + '. running this cell')
        if os.path.exists(self.csv_file):
            # a CSV with no done.json is a crashed run (B10). move it aside and start over
            # rather than skipping this cell forever or merging into half a table.
            partial = f'{self.csv_file}.partial-{int(time.time())}'
            os.rename(self.csv_file, partial)
            print(f'incomplete run found, moved {self.csv_file} -> {partial}, re-running')
        return False

    def _claim(self):
        '''
        take exclusive ownership of this cell, or report that someone else has it.

        `previously_done` is a check-then-act, so two processes pointed at the same cell both
        decide to run it and then both write training_results.csv. the merge in
        save_and_merge_df_as_csv is an outer join, so the file ends up with duplicated epochs
        rather than an error. **this happened**: running the fixed-budget and equal-steps grids
        at the same time, five seed-42 cells collide (equal_steps at data_percent=1 resolves to
        the same 30 epochs, hence the same directory), and all five CSVs came out with duplicate
        rows -- one had 30 rows where 16 were expected.

        O_CREAT|O_EXCL is atomic, so exactly one process wins. the lock records the pid, and a
        lock whose process is gone is reclaimed -- otherwise a crashed run would wedge the cell
        forever, which is the B10 trap in a new costume.
        '''
        self._ensure_dirs()
        try:
            fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(self.lock_file) as f:
                    holder = json.load(f)
            except (OSError, ValueError):
                holder = {}
            if _process_alive(holder.get('pid')):
                print(f'{self.save_dir} is being run by pid {holder.get("pid")} -- skipping')
                return False
            print(f'stale lock at {self.lock_file} (pid {holder.get("pid")} is gone) -- reclaiming')
            os.unlink(self.lock_file)
            fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            json.dump({'pid': os.getpid(), 'host': platform.node(),
                       'started': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
        return True

    def _release(self):
        try:
            os.unlink(self.lock_file)
        except OSError:
            pass

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
        self._release()      # done.json is the record now; the lock has served its purpose
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
