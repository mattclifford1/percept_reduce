'''
simple trainer to test pipeline is working
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets.proportions import get_indicies
from percept_loss.testing.encoded_dataset import make_encodings
from percept_loss.datasets import DATA_LOADER
from percept_loss.networks import AUTOENCODER
from percept_loss.losses import LOSS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 10
batch_size = 32
print_feq = 250


# DATASETS
loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), 
                                 device=device, 
                                #  indicies_to_use=list(range(200))
                                 )
images = loader.get_images_dict()

# get inds
train_inds, val_inds, test_inds = get_indicies([0.4, 0.3, 0.3], total_instances=len(loader))
# get loaders
train_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=train_inds, image_dict=images)
val_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=val_inds, image_dict=images)
test_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=test_inds, image_dict=images)
# get torch loaders
train_dataloader = DataLoader(train_loader,  # type: ignore
                              batch_size=batch_size, 
                              shuffle=True)
val_dataloader = DataLoader(val_loader,  # type: ignore
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=False)
test_dataloader = DataLoader(test_loader,  # type: ignore
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=False)

# NETWORK
net = AUTOENCODER['simple']()
net.to(device)

# TRAINING SIGNAL
mse = LOSS['MSE']()

loss_metric = LOSS['MSE']() # lr=1e-3

# loss_metric = LOSS['LPIPS']() 
# loss_metric.to(device)

# loss_metric = LOSS['SSIM']()


# optimiser = optim.SGD(net.parameters(), lr=1e-5)#, momentum=0.9)
optimiser = optim.Adam(net.parameters(), lr=1e-5)


for epoch in tqdm(range(epochs), desc='Epoch'):
    # test of classifier from encodings
    make_encodings(val_dataloader, net, device)
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, one_hot_labels, numerical_labels = data[0].to(device), data[1].to(device), data[2].to(device)
        inputs = data[0].to(device)
        # run through the network
        outputs = net(inputs)
        if i%print_feq == 0:
            mse_training = mse(inputs, outputs)
            # print('mse', mse_training.item())
            pass

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimise
        loss = loss_metric(inputs, outputs)

        loss.backward()
        optimiser.step()
        if i%print_feq == 0:
            # print('loss', loss.item(), '\n')
            pass

    
