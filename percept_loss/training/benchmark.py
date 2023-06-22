'''
simple trainer to test pipeline is working
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from percept_loss.datasets import DATA_LOADER
from percept_loss.networks import AUTOENCODER
from percept_loss.losses import LOSS

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 10
batch_size = 32
print_feq = 250


# DATASETS
train_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1))

# images = train_loader.get_images_dict()
train_dataloader = DataLoader(train_loader, 
                              batch_size=batch_size, 
                              shuffle=True,
                            #   num_workers=22
                              )

# NETWORK
net = AUTOENCODER['simple']()
net.to(device)

# TRAINING SIGNAL
mse = LOSS['MSE']()

loss_metric = LOSS['MSE']() # lr=1e-3

# loss_metric = LOSS['LPIPS']() 
# loss_metric.to(device)

# loss_metric = LOSS['SSIM']()


optimiser = optim.SGD(net.parameters(), lr=1e-5)#, momentum=0.9)
optimiser = optim.Adam(net.parameters(), lr=1e-5)


for epoch in tqdm(range(epochs), desc='Epoch'):
    # for i, data in tqdm(enumerate(train_dataloader, 0), leave=False, total=len(train_loader)/batch_size):
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        mse_training = mse(inputs, outputs)
        if i%print_feq == 0:
            print('mse', mse_training.item())
            pass

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimise
        loss = loss_metric(inputs, outputs)

        loss.backward()
        optimiser.step()
        if i%print_feq == 0:
            print('loss', loss.item(), '\n')
            pass
