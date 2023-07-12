'''
simple trainer to test pipeline is working
'''
import torch
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.torch_loaders import get_all_loaders
from percept_loss.testing.benchmark_encodings import random_forest_test
from percept_loss.networks import AUTOENCODER
from percept_loss.losses import LOSS
from percept_loss.utils.savers import train_saver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up parameters
network = 'conv'
loss = 'MSSIM'
print(loss)

lr = 1e5
epochs = 100
batch_size = 32
data_percent = 1
print(data_percent)
print_feq = 1000



# DATASETS
train_dataloader, val_dataloader, test_dataloader, train_total = get_all_loaders(train_percept_reduce=data_percent,
                                                                                 device=device)


# NETWORK
net = AUTOENCODER[network]()
net.to(device)

mse = LOSS['MSE']() # for eval
# TRAINING SIGNAL
loss_metric = LOSS[loss]() # lr=1e-3
if loss == 'LPIPS':
    loss_metric.to(device)


# optimiser = optim.SGD(net.parameters(), lr=1e-5)#, momentum=0.9)
optimiser = optim.Adam(net.parameters())#, lr=lr)



# TRAINING
saver = train_saver(epochs, loss, network, batch_size, lr, train_total) # saver
# loop



acc = random_forest_test(test_dataloader, net, device)
print(f'Random Forest init Accuracy: {acc}')
for epoch in tqdm(range(epochs), desc='Epoch'):
    # test of classifier from encodings
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, one_hot_labels, numerical_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        inputs = data[0].to(device)
        # run through the network
        outputs = net(inputs)
        if i%print_feq == 0:
            
            mse_training = mse(inputs, outputs)
            # print('mse', mse_training.item())

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimise
        loss = loss_metric(inputs, outputs)

        loss.backward()
        optimiser.step()
        if i%print_feq == 0:
            # print('loss', loss.item(), '\n')
            pass
    acc = random_forest_test(test_dataloader, net, device)
    saver.write_images([inputs, outputs], epoch, extra_name=f'rf_acc-{acc}')

    
