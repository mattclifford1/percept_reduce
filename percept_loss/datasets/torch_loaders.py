from torch.utils.data import DataLoader
from percept_loss.datasets import DATA_LOADER
from percept_loss.datasets.proportions import get_indicies


def get_preloaded(device='cpu'):
    # load the main dataset images etc.
    loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), 
                                    device=device)
    pre_loaded_images = loader.get_images_dict()
    return pre_loaded_images

def get_all_loaders_CIFAR(train_percept_reduce=0.5, props=[0.4, 0.3, 0.3], device='cpu', batch_size=32, pre_loaded_images=None):
    if pre_loaded_images == None:
        pre_loaded_images = get_preloaded(device=device)

    # get inds - split into train, val and test
    train_inds, val_inds, test_inds = get_indicies(props, total_instances=len(pre_loaded_images))
    # reduce the amount of training data
    train_total = max(min(int(train_percept_reduce*len(train_inds)), len(train_inds)), 1)
    train_inds = train_inds[:train_total]   # inds are shuffled already so we can take a random sample
    # get loaders
    train_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=train_inds, image_dict=pre_loaded_images)
    val_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=val_inds, image_dict=pre_loaded_images)
    test_loader = DATA_LOADER['CIFAR_10'](normalise=(0, 1), indicies_to_use=test_inds, image_dict=pre_loaded_images)
    # get torch loaders
    train_dataloader = DataLoader(train_loader,  # type: ignore
                                batch_size=batch_size, 
                                shuffle=True)
    val_dataloader = DataLoader(val_loader,  # type: ignore
                                batch_size=batch_size, 
                                shuffle=False,
                                drop_last=False)
    test_dataloader = DataLoader(test_loader,  # type: ignore
                                batch_size=batch_size, 
                                shuffle=False,
                                drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, train_total