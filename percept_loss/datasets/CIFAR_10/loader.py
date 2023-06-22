import os
from tqdm import tqdm
import pandas as pd
import torch
from torchvision.io import read_image
from percept_loss.datasets.CIFAR_10.VARS import CIFAR_10_META_CSV, CIFAR_10_IMAGE_DIR
from percept_loss.datasets.CIFAR_10.downloader import download_CIFAR_10


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
                 normalise=(0, 1),
                 dtype=torch.float32,
                 device='cpu'):
        self.indicies_to_use = indicies_to_use
        self.image_dict = image_dict
        self.cache_data = cache_data
        self.normalise = normalise
        self.dtype = dtype
        self.device = device

        # make sure dataset is downloaded
        download_CIFAR_10()    

        self.label_cache = {}
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
        for ind in tqdm(range(len(self.labels)), desc='Loading CIFAR-10', leave=False):
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
            image = image/255
            image = image*(self.normalise[1]-self.normalise[0])
            image = image + self.normalise[0]
            image = image.to(self.device)
            # print(self.device)
            if self.cache_data == True:
                self.image_dict[filename] = image
        # print(type(image))
        return image
    
    def _get_labels(self, ind):
        if ind in self.label_cache:
            one_hot = self.label_cache[ind]['one_hot']
            label = self.label_cache[ind]['label']
        else:
            one_hot = torch.zeros(self.num_classes, dtype=self.dtype)
            one_hot[self.numerical_label[ind]] = 1
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
    

if __name__ == '__main__':
    from tqdm import tqdm
    a = CIFAR_10_LOADER()
    ims = a.get_images_dict()

    b = CIFAR_10_LOADER(image_dict=ims)

    # benchmark
    for e in tqdm(range(10)):
        for i in tqdm(b, leave=False):
            pass