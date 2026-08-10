'''
simple trainer to test pipeline is working
'''
import itertools
import torch
from tqdm import tqdm

from percept_loss.training.run_and_test import train
from percept_loss.datasets.torch_loaders import get_preloaded
from percept_loss.utils.savers import train_saver
from percept_loss.utils.seeding import set_seed
from percept_loss.losses import LOSS

def get_all_dict_permutations(dict_):
    '''
    given a dict with list values, return all the possible permuations of
    single values for each key item
    '''
    dict_ = dict(reversed(dict_.items()))  # want eval first
    keys, values = zip(*dict_.items())
    dict_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return dict_permutations

def run(runs, AUTOENCODERS, epochs=30, batch_size=32, preload_data=False, dataset='CIFAR_10',
        validate_every=2, n_validations=None, props=None, seed=42, lr=1e-3,
        epoch_scaling='fixed'):
    '''
    cartesian product over the run grid.

    `seed` and `lr` may also be given as *axes* in the runs dict (e.g. 'seed': [1, 2, 3]) --
    they are part of the run directory, so a multi-seed sweep no longer collides with itself
    (TODO T1.4). the values here are the fallbacks.

    epoch_scaling:
      'fixed'       every run gets `epochs` passes over its own training set. small-data cells
                    therefore get proportionally fewer gradient steps, so data size and
                    optimisation budget are confounded -- this is FINDINGS B4, and it is the
                    behaviour of every committed run.
      'equal_steps' epochs are scaled by 1/data_percent so every cell takes roughly the same
                    number of gradient steps. this is what the dead `scaled_epochs` line in the
                    old code was trying to do. 'uniform' is unscaled -- it has no data_percent.

    with 'equal_steps', pass `n_validations` too: it fixes the number of evaluations per run
    instead of the interval between them. 1% data becomes 3000 epochs, and at validate_every=2
    that is 1500 probes -- the probe is 95% of wall time here, so the cadence has to scale with
    the epoch count or the eval cost swamps the experiment.

    pass `props` explicitly alongside it. the default split is derived from validate_every
    (FINDINGS B7), so a computed cadence can move the split by arithmetic accident.
    '''
    # fixed things for all runs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # DATA
    if preload_data == True:
        pre_loaded_images = get_preloaded(dataset=dataset, device=device)
    else:
        pre_loaded_images = {}

    # all variations
    all_runs = get_all_dict_permutations(runs)
    skipped = {'done.json': 0, 'legacy': 0}
    trained = 0

    # # run
    for run in tqdm(all_runs, desc='All Runs'):
        loss = run['loss']
        network = run['network']
        data_percent = run['data_percent']
        run_seed = run.get('seed', seed)
        run_lr = run.get('lr', lr)
        if epoch_scaling == 'equal_steps' and not isinstance(data_percent, str):
            scaled_epochs = int(epochs/data_percent)
        else:
            scaled_epochs = epochs
        # 'RANDOM' is the untrained control -- zero gradient steps, so the directory has to say
        # ep0 rather than inheriting the trained epoch count
        run_epochs = 0 if loss == 'RANDOM' else scaled_epochs
        # saver
        saver = train_saver(run_epochs, loss, network, batch_size, data_percent, dataset=dataset,
                            seed=run_seed, lr=run_lr) # saver
        # run
        if saver.previously_done == False:
            # build the loss first: LPIPS/DISTS construct a backbone and so draw from the RNG.
            # seeding after that but before the net means runs differing only in loss share an
            # identical initialisation -- makes e.g. LPIPS vs LPIPS1 a paired comparison.
            if loss == 'RANDOM':
                # not a loss -- the untrained-encoder control. zero gradient steps, so the probe
                # reads the initialisation itself. it still needs *a* metric for the val MSE
                # reference column, hence MSE. see FINDINGS 'the untrained encoder is a strong
                # baseline' and the uniform-noise result: if random init ~= uniform-trained then
                # that finding is an architecture prior (random conv features, Saxe et al. 2011;
                # deep image prior, Ulyanov et al. 2018), not something training discovered.
                # the result does not depend on data_percent -- no training data is touched --
                # so one data_percent per network is enough.
                loss_func = LOSS['MSE']()
            else:
                loss_func = LOSS[loss]()
            set_seed(run_seed)
            network_func = AUTOENCODERS[network]()
            train(network=network_func,
                  loss=loss_func,
                  epochs=run_epochs,
                  device=device, 
                  saver=saver, 
                  data_percent=data_percent, 
                  pre_loaded_images=pre_loaded_images, 
                  verbose=False,
                  validate_every=validate_every,
                  n_validations=n_validations,
                  props=props,
                  extra_config={'epoch_scaling': epoch_scaling},
                  dataset=dataset,
                  batch_size=batch_size,
                  lr=run_lr)
            trained += 1
        else:
            skipped[saver.skip_reason] = skipped.get(saver.skip_reason, 0) + 1
            print(f'Passing ({saver.skip_reason}): {saver.save_dir}')

    # a partition that skips everything is a real failure mode, not a fast success -- it is how
    # 35 seed-42 cells went missing from the big re-run without anyone noticing. say so loudly.
    print(f'\n{trained} cells trained, {sum(skipped.values())} skipped '
          f'({skipped["done.json"]} already done, {skipped["legacy"]} matched a legacy run, '
          f'{skipped.get("locked", 0)} held by another process)')
    if trained == 0 and len(all_runs) > 0:
        print('WARNING: every cell in this partition was skipped -- nothing was run. if you '
              'expected work, check whether legacy directories are standing in for it.')
    if skipped['legacy'] > 0:
        print(f'NOTE: {skipped["legacy"]} cells were satisfied by legacy (pre-migration) '
              f'directories rather than by runs of the current code.')


if __name__ == '__main__':
    from percept_loss.networks import CIFAR_AUTOENCODERS
    # run configs
    runs = {
        'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
        # 'data_percent': ['uniform', 0.5, 0.1],
        'loss': ['SSIM', 'MSE', 'LPIPS', 'LPIPS1', 'MSSIM', 'DISTS', 'NLPD'],
        # 'loss': ['DISTS', 'NLPD'],
        # 'network': ['conv_small_z', 'conv_bigger_z', 'conv_big_z'],
        'network': ['conv_big_z'],
    }

    run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32, preload_data=True, dataset='CIFAR_10')

    
