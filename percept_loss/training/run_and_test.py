'''
simple trainer to test pipeline is working
'''
import threading

import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.testing.benchmark_encodings import random_GaussianNB_test, test_all_classifiers
from percept_loss.testing.encoded_dataset import make_encodings
from percept_loss.utils.savers import train_saver
from percept_loss.losses import LOSS

def validate(net, loader, loss_metric, device):
    '''
    get loss on validation set
    '''
    scores = []
    for i, data in enumerate(loader, 0):
        inputs = data[0].to(device)
        # run through the network
        outputs = net(inputs)
        loss = loss_metric(inputs, outputs).detach().cpu().numpy()
        scores.append(loss)
    return sum(scores)/len(scores)


class test_and_saver():
    '''
    util to get test scores and save in an async manner after getting encodings
    '''
    def __init__(self, val_dataloader, test_dataloader, saver, device, verbose=False):
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.saver = saver
        self.device = device
        self.verbose = verbose

        self.test_func = test_all_classifiers
        # self.test_func = random_GaussianNB_test

        self.running_thread = None


    def run_and_save_async(self, net, epoch, save_ims=None, run_async=True):
        # set up the data we need
        X, y = make_encodings(self.test_dataloader, net, self.device)
        mse = LOSS['MSE']() # for eval
        val_MSE = validate(net, self.val_dataloader, mse, self.device)

        # matplotlib shiz needs to stay on the main thread
        if save_ims != None:
            self.saver.write_images(save_ims, epoch, extra_name=f"")
            # self.saver.write_images(save_ims, epoch, extra_name=f"MLP-acc-{save_data['MLP']}")

        # EVAL - wait for previous first
        if self.running_thread != None:
            self.running_thread.join()   # dont start a new eval if the previus one is still running

        # threading
        args = ((X, y), val_MSE, self.verbose, self.test_func, epoch, self.saver)
        if run_async == True:
            self.running_thread = threading.Thread(target=self._run, args=args)
            self.running_thread.start()
        else:
            self._run(*args)

    @staticmethod
    def _run(data, val_mse, verbose, test_func, epoch, saver):
        save_data = test_func(data=data, verbose=verbose)
        save_data['val MSE'] = val_mse
        saver.write_scores(save_data, epoch)
        # mse_training = mse(inputs, outputs)

    def wait_to_finish(self):
        if self.running_thread != None:
            self.running_thread.join()  
    

def train(network, loss, epochs, device, saver, data_percent, pre_loaded_images=None, verbose=False, async_test=True, validate_every=2, dataset='CIFAR_10'):
    '''
    main training loop
    '''
    if validate_every == 1: # know we have a big dataset so decrease validation/test set size
        props = [0.8, 0.1, 0.1]
    else:
        props = [0.4, 0.3, 0.3]
    train_dataloader, val_dataloader, test_dataloader, _ = get_all_loaders(train_percept_reduce=data_percent,
                                                                           device=device,
                                                                           pre_loaded_images=pre_loaded_images,
                                                                           props=props,
                                                                           dataset=dataset)

    # NETWORK
    net = network
    net.to(device)

    # TRAINING SIGNAL
    loss_metric = loss
    loss_metric.to(device)

    # optimiser = optim.SGD(net.parameters(), lr=1e-5)#, momentum=0.9)
    optimiser = optim.Adam(net.parameters())#, lr=lr)

    # get initial network stats on eval/test
    tester = test_and_saver(val_dataloader, test_dataloader, saver, device, verbose=verbose)
    tester.run_and_save_async(net, epoch=0, save_ims=None, run_async=async_test)
    
    for epoch in tqdm(range(epochs), desc='Epoch', leave=False):
        # test of classifier from encodings
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            # run through the network
            outputs = net(inputs)
            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            loss = loss_metric(inputs, outputs)

            loss.backward()
            optimiser.step()
        if epoch % validate_every == 0:
            tester.run_and_save_async(net, epoch=epoch+1, save_ims=[inputs, outputs], run_async=async_test)
    tester.wait_to_finish()


if __name__ == '__main__':
    from percept_loss.networks import CIFAR_AUTOENCODERS
    from percept_loss.losses import LOSS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up parameters
    network = CIFAR_AUTOENCODERS['conv_biggest_z']()
    loss_str = 'MSSIM'
    loss = LOSS[loss_str]()
    print(loss_str)

    epochs = 20
    batch_size = 32
    data_percent = 1
    print(data_percent)


    # DATASETS
    train_dataloader, val_dataloader, test_dataloader, train_total = get_all_loaders(train_percept_reduce=data_percent,
                                                                                    device=device)

    # TRAINING
    saver = train_saver(epochs, loss, network, batch_size, data_percent) # saver
    # run
    loaders = train_dataloader, val_dataloader, test_dataloader
    train(network, loss, epochs, device, loaders, saver)





    
