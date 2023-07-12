'''
simple trainer to test pipeline is working
'''
import itertools
import torch
from tqdm import tqdm

from percept_loss.training.run_and_test import train
from percept_loss.datasets.torch_loaders import get_preloaded
from percept_loss.utils.savers import train_saver

def get_all_dict_permutations(dict_):
    '''
    given a dict with list values, return all the possible permuations of
    single values for each key item
    '''
    dict_ = dict(reversed(dict_.items()))  # want eval first
    keys, values = zip(*dict_.items())
    dict_permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return dict_permutations

if __name__ == '__main__':
    # fixed things for all runs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    batch_size = 32
    # DATA
    pre_loaded_images = get_preloaded(device=device)

    # run configs
    runs = {
        'loss': ['SSIM', 'MSE', 'LPIPS', 'MSSIM'],
        'data_percent': [1, 0.5, 0.1, 0.01],
        'network': ['conv_small_z', 'conv_bigger_z', 'conv_big_z'],
    }
    all_runs = get_all_dict_permutations(runs)

    # # run
    for run in tqdm(all_runs, desc='All Runs'):
        loss = run['loss']
        network = run['network']
        data_percent = run['data_percent']
        saver = train_saver(epochs, loss, network, batch_size, data_percent) # saver
        train(network, loss, epochs, device, saver, data_percent, pre_loaded_images)





    
