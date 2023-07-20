import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image

DEFAULT_VAL_DATASET = os.path.join(os.path.expanduser('~'), 'datasets', 'ImageNet64', 'val')
DEFAULT_TRAIN_DATASET = os.path.join(os.path.expanduser('~'), 'datasets', 'ImageNet64', 'train')


class IMAGENET_64_LOADER:
    '''
    Generic data loader for IMAGENET-64
    Args:
        indicies_to_use: list of in the dataset to use if you require to use a split of the dataset
        image_dict: dictionary of pre loaded images filename as keys and torch tensor as values 
    '''
    def __init__(self, 
                 data_dir=DEFAULT_VAL_DATASET,
                 indicies_to_use='all', 
                 image_dict={}, 
                 cache_data=True,
                 normalise=(0, 1),
                 dtype=torch.float32,
                 device='cpu'):
        self.image_dir = os.path.join(data_dir, 'images')
        self.indicies_to_use = indicies_to_use
        self.image_dict = image_dict
        self.cache_data = cache_data
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        self.label_cache = {}
        self.meta_data = pd.read_csv(os.path.join(data_dir, 'meta_data.csv'))
        self.filenames = self.meta_data['filename'].to_list()
        self.labels = self.meta_data['label'].to_list()
        self.numerical_label = self.meta_data['numerical_label'].to_list()
        self.num_classes = 1000
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
        for ind in tqdm(range(len(self.labels)), desc='Loading ImageNet64 to RAM', leave=True):
            filename = self.filenames[ind]
            # get image
            image = self._get_image(filename, preprocess=False) # dont save 32bit images as too big
            images[filename] = image
        return images
    
    def _process_image(self, image):
        image = image.to(self.dtype)
        image = image/255
        image = image*(self.normalise[1]-self.normalise[0])
        image = image + self.normalise[0]
        image = image.to(self.device)
        return image


    def _get_image(self, filename, preprocess=True):
        # use cache data if loaded already
        if filename in self.image_dict:
            image = self.image_dict[filename]
            image = self._process_image(image)  # cached image is unprocessed (8bit saved)
        else:
            # load if not cached
            path = os.path.join(self.image_dir, filename)
            image_raw = read_image(path)
            # cache raw image if required (8 bit so lower mem ~20GB for trainset)
            if self.cache_data == True:
                self.image_dict[filename] = image_raw
            if preprocess == False:
                return image_raw
            image = self._process_image(image_raw)
        return image
    
    def _get_labels(self, ind):
        if ind in self.label_cache:
            one_hot = self.label_cache[ind]['one_hot']
            label = self.label_cache[ind]['label']
        else:
            one_hot = torch.zeros(self.num_classes, dtype=self.dtype)
            one_hot[self.numerical_label[ind]-1] = 1
            # one_hot = one_hot.to(self.device)
            label = torch.tensor(self.numerical_label[ind])
            # label = label.to(self.device)
            if self.cache_data == True:
                self.label_cache[ind] = {'label':{}, 'one_hot':{}}
                self.label_cache[ind]['one_hot'] = one_hot
                self.label_cache[ind]['label'] = label
        return one_hot, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        filename = self.filenames[ind]
        # get image
        image = self._get_image(filename)
        # get labels
        one_hot, label = self._get_labels(ind)
        return image, one_hot, label


def IMAGENET_64_LOADER_VAL(*args, **kwargs):
    return IMAGENET_64_LOADER(*args, **kwargs, data_dir=DEFAULT_VAL_DATASET)

def IMAGENET_64_LOADER_TRAIN(*args, **kwargs):
    return IMAGENET_64_LOADER(*args, **kwargs, data_dir=DEFAULT_TRAIN_DATASET)


if __name__ == '__main__':
    from tqdm import tqdm
    a = IMAGENET_64_LOADER(data_dir=DEFAULT_TRAIN_DATASET)

    # ims = a.get_images_dict()
    # b = IMAGENET_64_LOADER(data_dir=DEFAULT_TRAIN_DATASET, image_dict=ims)

    b = IMAGENET_64_LOADER(data_dir=DEFAULT_VAL_DATASET, image_dict={}, cache_data=False)
    # benchmark
    print('preloaded')
    # for e in tqdm(range(10)):
    for i in tqdm(range(len(b)), leave=True):
        im = b[i]

    c = IMAGENET_64_LOADER(data_dir=DEFAULT_TRAIN_DATASET, cache_data=False)
    # benchmark
    print('not preloaded')
    for i in tqdm(range(len(c)), leave=True):
        im = c[i]