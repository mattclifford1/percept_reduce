'''
simple trainer to test pipeline is working
'''
import itertools
import torch
from tqdm import tqdm

from percept_loss.training.run_and_test import train
from percept_loss.datasets.torch_loaders import get_preloaded
from percept_loss.utils.savers import train_saver
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

def run(runs, AUTOENCODERS, epochs=30, batch_size=32, preload_data=False, dataset='CIFAR_10', validate_every=2):
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
        if not isinstance(data_percent, str):
            scaled_epochs = int(epochs/data_percent)
            scaled_epochs = epochs
        else:
            scaled_epochs = epochs
        # saver
        saver = train_saver(scaled_epochs, loss, network, batch_size, data_percent, dataset=dataset) # saver
        loss_func = LOSS[loss]()
        network_func = AUTOENCODERS[network]()
        # run
        if saver.previously_done == False:
            train(network_func, loss_func, scaled_epochs, device, saver, data_percent, pre_loaded_images, validate_every)


if __name__ == '__main__':
    from percept_loss.networks import CIFAR_AUTOENCODERS
    # run configs
    runs = {
        'data_percent': [1, 0.5, 0.1, 0.01, 'uniform'],
        # 'data_percent': ['uniform', 0.5, 0.1],
        'loss': ['SSIM', 'MSE', 'LPIPS', 'MSSIM', 'DISTS', 'NLPD'],
        # 'loss': ['DISTS', 'NLPD'],
        # 'network': ['conv_small_z', 'conv_bigger_z', 'conv_big_z'],
        'network': ['conv_big_z'],
    }

    run(runs, CIFAR_AUTOENCODERS, epochs=30, batch_size=32, preload_data=True, dataset='CIFAR_10')

    
