import os
import pandas as pd
import torch
from torchvision.io import read_image
from percept_loss.datasets.CIFAR_10.VARS import CIFAR_10_META_CSV, CIFAR_10_IMAGE_DIR


class CIFAR_10_LOADER:
    '''
    Generic data loader for CIFAR-10
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    def __init__(self, indicies_to_use='all', 
                 image_dict={}, 
                 cache_data=True,
                 dtype=torch.float32,
                 device='cpu'):
        self.indicies_to_use = indicies_to_use
        self.image_dict = image_dict
        self.cache_data = cache_data
        self.dtype = dtype
        self.device = device
    
        self.hot_one_cache = {}
        self.meta_data = pd.read_csv(CIFAR_10_META_CSV)
        self.filenames = self.meta_data['filename'].to_list()
        self.labels = self.meta_data['label'].to_list()
        self.numerical_label = self.meta_data['numerical_label'].to_list()
        self.num_classes = 10
        self.reduce_data()

    def reduce_data(self):
        if self.indicies_to_use == 'all':
            return
        else:
            self.filenames = [self.filenames[i] for i in self.indicies_to_use]
            self.labels = [self.labels[i] for i in self.indicies_to_use]
            self.numerical_label = [self.numerical_label[i] for i in self.indicies_to_use]

    def get_images_dict(self):
        '''get dict of all pre loaded and processed images - useful to pass to other dataset
        so that you only have to read from file and process once'''
        images = {}
        for ind in range(len(self.labels)):
            filename = self.filenames[ind]
            # get image
            image = self._get_image(filename)
            images[filename] = image
        return images

    def _get_image(self, filename):
        # use cache data if loaded already
        if filename in self.image_dict:
            image = self.image_dict[filename]
        else:
            # load if not cached
            path = os.path.join(CIFAR_10_IMAGE_DIR, filename)
            image = read_image(path)
            image = image.to(self.dtype)
            image.to(self.device)
            if self.cache_data == True:
                self.image_dict[filename] = image
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        filename = self.filenames[ind]
        # get image
        image = self._get_image(filename)
        # get one hot label
        if filename in self.hot_one_cache:
            one_hot_label = self.hot_one_cache[filename]
        else:
            one_hot_label = torch.zeros(self.num_classes, dtype=self.dtype)
            one_hot_label[self.numerical_label[ind]] = 1
            one_hot_label.to(self.device)
            if self.cache_data == True:
                self.hot_one_cache[filename] = one_hot_label

        return image, one_hot_label
    

if __name__ == '__main__':
    from tqdm import tqdm
    a = CIFAR_10_LOADER()
    ims = a.get_images_dict()

    b = CIFAR_10_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass