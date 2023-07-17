'''
simple trainer to test pipeline is working
'''
import time
import timeit
import torch
from tqdm import tqdm
import numpy as np

from percept_loss.training.run_and_test import train
from percept_loss.datasets.torch_loaders import get_preloaded
from percept_loss.utils.savers import train_saver

def run(pre_loaded_images=None, async_test=True):
    # fixed things for all runs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    batch_size = 32
    # DATA
    if pre_loaded_images ==  None:
        pre_loaded_images = get_preloaded(device=device)
    loss = 'LPIPS'
    network = 'conv_biggest_z'
    data_percent = 1
    saver = train_saver(epochs, loss, network, batch_size, data_percent) # saver
    train(network, loss, epochs, device, saver, data_percent, pre_loaded_images, verbose=False, async_test=async_test)



if __name__ == '__main__':
    pre_loaded_images = get_preloaded()
    reps =timeit.repeat(stmt=lambda: run(pre_loaded_images, True), repeat=10)
    avg = np.mean(reps)
    print(f'Asunc true: {avg}')

    reps =timeit.repeat(stmt=lambda: run(pre_loaded_images, False), repeat=10)
    avg = np.mean(reps)
    print(f'Asunc false: {avg}')
    


    '''
    # dev timings:
        no async: 35-41
        async:    30-38

    '''






    
