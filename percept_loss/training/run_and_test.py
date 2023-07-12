'''
simple trainer to test pipeline is working
'''
import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.testing.benchmark_encodings import random_GaussianNB_test, test_all_classifiers
from percept_loss.networks import AUTOENCODER
from percept_loss.losses import LOSS
from percept_loss.utils.savers import train_saver

test_func = test_all_classifiers
# test_func = random_GaussianNB_test

def validate(net, loader, loss_metric, device):
    scores = []
    for i, data in enumerate(loader, 0):
        inputs = data[0].to(device)
        # run through the network
        outputs = net(inputs)
        loss = loss_metric(inputs, outputs).detach().cpu().numpy()
        scores.append(loss)
    return sum(scores)/len(scores)

def train(network, loss, epochs, device, saver, data_percent, pre_loaded_images=None):

    train_dataloader, val_dataloader, test_dataloader, _ = get_all_loaders(train_percept_reduce=data_percent,
                                                                           device=device,
                                                                           pre_loaded_images=pre_loaded_images)

    # NETWORK
    net = AUTOENCODER[network]()
    net.to(device)

    mse = LOSS['MSE']() # for eval
    # TRAINING SIGNAL
    loss_metric = LOSS[loss]()
    loss_metric.to(device)

    # optimiser = optim.SGD(net.parameters(), lr=1e-5)#, momentum=0.9)
    optimiser = optim.Adam(net.parameters())#, lr=lr)

    save_data = test_func(test_dataloader, net, device)
    save_data['val MSE'] = validate(net, val_dataloader, mse, device)
    # save_data['val MSE'] = 1
    saver.write_scores(save_data, 0)
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

        # EVAL
        save_data = test_func(test_dataloader, net, device)
        save_data['val MSE'] = validate(net, val_dataloader, mse, device)
        saver.write_scores(save_data, epoch+1)
        mse_training = mse(inputs, outputs)
        saver.write_images([inputs, outputs], epoch, extra_name=f"NB-acc-{save_data['NB']}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up parameters
    network = 'conv'
    loss = 'MSSIM'
    print(loss)

    lr = 1e5
    epochs = 20
    batch_size = 32
    data_percent = 1
    print(data_percent)
    print_feq = 1000



    # DATASETS
    train_dataloader, val_dataloader, test_dataloader, train_total = get_all_loaders(train_percept_reduce=data_percent,
                                                                                    device=device)

    # TRAINING
    saver = train_saver(epochs, loss, network, batch_size, data_percent) # saver
    # run
    loaders = train_dataloader, val_dataloader, test_dataloader
    train(network, loss, epochs, device, loaders, saver)





    
