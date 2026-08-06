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
        validate_every=2, seed=42, lr=1e-3, epoch_scaling='fixed'):
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
                  dataset=dataset,
                  batch_size=batch_size,
                  lr=run_lr)
        else:
            print('Passing')


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

    
