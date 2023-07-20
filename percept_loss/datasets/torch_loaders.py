from torch.utils.data import DataLoader
from percept_loss.datasets import DATA_LOADER, TOTAL_INSTANCES
from percept_loss.datasets.proportions import get_indicies

NORMALISE = (0, 1)

def get_preloaded(dataset='CIFAR_10', device='cpu'):
    # load the main dataset images etc.
    loader = DATA_LOADER[dataset](normalise=NORMALISE, 
                                    device=device)
    pre_loaded_images = loader.get_images_dict()
    return pre_loaded_images

def get_all_loaders(train_percept_reduce=0.5, props=[0.4, 0.3, 0.3], device='cpu', batch_size=32, pre_loaded_images=None, dataset='CIFAR_10'):
    if pre_loaded_images == True:
        pre_loaded_images = get_preloaded(dataset=dataset, device=device)
    elif pre_loaded_images == None:
        pre_loaded_images = {}

    # get inds - split into train, val and test
    train_inds, val_inds, test_inds = get_indicies(props, total_instances=TOTAL_INSTANCES[dataset])
    # reduce the amount of training data
    if not isinstance(train_percept_reduce, str):
        train_total = max(min(int(train_percept_reduce*len(train_inds)), len(train_inds)), 1)
        train_inds = train_inds[:train_total]   # inds are shuffled already so we can take a random sample
    
    # get loaders
    val_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=val_inds, image_dict=pre_loaded_images)
    test_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=test_inds, image_dict=pre_loaded_images)
    if not isinstance(train_percept_reduce, str):
        train_loader = DATA_LOADER[dataset](normalise=NORMALISE, indicies_to_use=train_inds, image_dict=pre_loaded_images)
    elif train_percept_reduce == 'uniform': # get uniform noise loader
        train_total = 5000
        if dataset == 'CIFAR_10':
            size = (3, 32, 32)
        elif dataset == 'IMAGENET64_VAL' or dataset == 'IMAGENET64_TRAIN':
            size = (3, 64, 64)

        train_loader = DATA_LOADER['UNIFORM'](normalise=NORMALISE, length=train_total, size=size)
    else:
        raise ValueError(f'Cannot use trainset type/size: {train_percept_reduce}')
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