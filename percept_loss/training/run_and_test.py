'''
simple trainer to test pipeline is working
'''
import threading
import time

import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.testing.benchmark_encodings import (random_GaussianNB_test, test_all_classifiers,
                                                      PROBE_VERSION)
from percept_loss.testing.encoded_dataset import make_encodings
from percept_loss.utils.savers import train_saver
from percept_loss.losses import LOSS

def validate(net, loader, loss_metric, device):
    '''
    get loss on validation set

    eval() + no_grad() for the same reasons as make_encodings (FINDINGS B9): batch statistics
    must not leak between images, and the graph was being built and thrown away. eval() also
    makes the VAE decode from mu instead of a sample, so val MSE is not sampling noise.
    '''
    was_training = net.training
    net.eval()
    scores = []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs = data[0].to(device)
            # run through the network
            outputs = net(inputs)
            loss = loss_metric(inputs, outputs).detach().cpu().numpy()
            scores.append(loss)
    net.train(was_training)
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


    def run_and_save_async(self, net, epoch, save_ims=None, run_async=True, extra=None):
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
        args = ((X, y), val_MSE, self.verbose, self.test_func, epoch, self.saver, extra)
        if run_async == True:
            self.running_thread = threading.Thread(target=self._run, args=args)
            self.running_thread.start()
        else:
            self._run(*args)

    @staticmethod
    def _run(data, val_mse, verbose, test_func, epoch, saver, extra=None):
        start = time.time()
        save_data = test_func(data=data, verbose=verbose)
        save_data['val MSE'] = val_mse
        # training-side diagnostics: without these the CSV cannot tell a converged run from a
        # collapsed one, or a working VAE from a posterior-collapsed one, without opening images
        if extra != None:
            save_data.update(extra)
        save_data['probe secs'] = round(time.time() - start, 2)
        saver.write_scores(save_data, epoch)

    def wait_to_finish(self):
        if self.running_thread != None:
            self.running_thread.join()  
    

def batch_std(outputs):
    '''
    spread of the reconstructions across a batch. near zero means every image decodes to the
    same picture -- the DISTS failure mode in FINDINGS B3, where three cells collapsed inside
    one epoch and then wrote 30 epochs of a dead network that read as a data-size effect.
    '''
    return float(outputs.std(dim=0).mean())


def train(network, loss, epochs, device, saver, data_percent, pre_loaded_images=None, verbose=False,
          async_test=True, validate_every=2, n_validations=None, props=None, extra_config=None,
          dataset='CIFAR_10', batch_size=32, lr=1e-3,
          collapse_tol=1e-4, collapse_patience=3):
    '''
    main training loop

    batch_size is now actually forwarded to the loaders (FINDINGS B8 -- it used to be accepted
    here, never passed on, and defaulted to 32, so every `BS32` in a directory name was true by
    coincidence). lr is explicit for the same reason: T1.1 needs an LR sweep and Adam's default
    was hardcoded.
    '''
    if n_validations != None:
        # a fixed number of evaluations per run, whatever the epoch count. without this,
        # epoch_scaling='equal_steps' at 1% data means 3000 epochs, and at validate_every=2 that
        # is 1500 probes -- about six hours of sklearn for a single cell, against ~4 minutes of
        # training. the probe is 95% of wall time on this grid, so the cadence is the cost.
        validate_every = max(1, int(round(epochs/n_validations)))

    if props == None:
        # legacy derivation, kept so existing callers are unchanged. it couples the data split
        # to the eval cadence (FINDINGS B7), which is actively dangerous once the cadence is
        # computed: a scaled validate_every can land on 1 by arithmetic and silently move the
        # split from [0.4, 0.3, 0.3] to [0.89, 0.1, 0.01]. pass props explicitly to break that.
        if validate_every == 1: # know we have a big dataset so decrease validation/test set size
            props = [0.89, 0.1, 0.01]
        else:
            props = [0.4, 0.3, 0.3]
    train_dataloader, val_dataloader, test_dataloader, _ = get_all_loaders(train_percept_reduce=data_percent,
                                                                           device=device,
                                                                           batch_size=batch_size,
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
    optimiser = optim.Adam(net.parameters(), lr=lr)

    is_vae = hasattr(net, 'kl')
    config = {'data_percent': data_percent,
                              'split_props': props,
                              'validate_every': validate_every,
                              'n_validations': n_validations,
                              'latent_dim': getattr(net, 'latent_dim', None),
                              'n_parameters': sum(p.numel() for p in net.parameters()),
                              'optimiser': 'Adam',
                              'probe_version': PROBE_VERSION,
                              'beta': getattr(net, 'beta', None) if is_vae else None,
                              'train_images': len(train_dataloader.dataset)}
    if extra_config != None:
        config.update(extra_config)
    saver.write_config(extra=config)

    # get initial network stats on eval/test
    tester = test_and_saver(val_dataloader, test_dataloader, saver, device, verbose=verbose)
    nan = float('nan')
    epoch0 = {'train loss': nan, 'out std': nan}
    if is_vae:
        epoch0['KL'] = nan
    tester.run_and_save_async(net, epoch=0, save_ims=None, run_async=async_test, extra=epoch0)

    collapsed_epochs = 0
    collapsed = False
    epoch = -1   # epochs=0 is the RANDOM control -- no training, just the epoch-0 probe
    for epoch in tqdm(range(epochs), desc='Epoch', leave=False):
        # test of classifier from encodings
        epoch_loss, epoch_std, epoch_kl, n_batches = 0.0, 0.0, 0.0, 0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            # run through the network
            outputs = net(inputs)
            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            loss = loss_metric(inputs, outputs)
            if is_vae:
                # VAE: reconstruction + beta*KL. net.kl is set by forward() and is already
                # per-pixel, so it sits on the same scale as the reconstruction losses.
                epoch_kl += float(net.kl)
                loss = loss + net.beta*net.kl

            loss.backward()
            optimiser.step()

            epoch_loss += float(loss)
            epoch_std += batch_std(outputs)
            n_batches += 1

        epoch_loss, epoch_std = epoch_loss/n_batches, epoch_std/n_batches
        extra = {'train loss': epoch_loss, 'out std': epoch_std}
        if is_vae:
            extra['KL'] = epoch_kl/n_batches

        # collapse detector: abort instead of writing a dead network to disk for 30 epochs
        if epoch_std < collapse_tol:
            collapsed_epochs += 1
        else:
            collapsed_epochs = 0
        if collapsed_epochs >= collapse_patience:
            print(f'output collapsed (batch std {epoch_std:.2e} for {collapsed_epochs} epochs) '
                  f'-- stopping at epoch {epoch+1}/{epochs}')
            collapsed = True
            tester.run_and_save_async(net, epoch=epoch+1, save_ims=[inputs, outputs],
                                      run_async=async_test, extra=extra)
            break

        if epoch % validate_every == 0:
            tester.run_and_save_async(net, epoch=epoch+1, save_ims=[inputs, outputs],
                                      run_async=async_test, extra=extra)
    tester.wait_to_finish()

    saver.save_checkpoint(net)
    saver.write_done(extra={'collapsed': collapsed, 'epochs_run': epoch + 1})
    return net


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





    
