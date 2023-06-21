'''
simple trainer to test pipeline is working
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from percept_loss.datasets.CIFAR_10.loader import CIFAR_10_LOADER
from percept_loss.networks.classifier import simple_image_clf
from percept_loss.loss.losses import cross_entropy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 100
batch_size = 64


# DATASETS
train_loader = CIFAR_10_LOADER()
# train_loader = CIFAR_10_LOADER()

# images = train_loader.get_images_dict()
train_dataloader = DataLoader(train_loader, 
                              batch_size=batch_size, 
                              shuffle=True,
                            #   num_workers=22
                              )

# NETWORK
clf = simple_image_clf()
clf.to(device)

# TRAINING SIGNAL
criterion = cross_entropy()
optimiser = optim.SGD(clf.parameters(), lr=0.001, momentum=0.9)


for epoch in tqdm(range(epochs), desc='Epoch'):
    for i, data in tqdm(enumerate(train_dataloader, 0), leave=False, total=len(train_loader)/batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        print(torch.max(inputs))
        print(torch.min(inputs))

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimise
        outputs = clf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
