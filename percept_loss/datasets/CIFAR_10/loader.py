import os
import pandas as pd
from torchvision.io import read_image
from percept_loss.datasets.CIFAR_10.VARS import CIFAR_10_META_CSV, CIFAR_10_IMAGE_DIR


class CIFAR_10_LOADER:
    def __init__(self, indicies_to_use='all', image_dict={}, cache_data=True):
        self.indicies_to_use = indicies_to_use
        self.image_dict = image_dict
        self.cache_data = cache_data
        self.meta_data = pd.read_csv(CIFAR_10_META_CSV)
        self.filenames = self.meta_data['filename'].to_list()
        self.labels = self.meta_data['label'].to_list()
        self.one_hot_labels = self.meta_data['one_hot_label'].to_list()
        self.reduce_data()

    def reduce_data(self):
        if self.indicies_to_use == 'all':
            return
        else:
            self.filenames = [self.filenames[i] for i in self.indicies_to_use]
            self.labels = [self.labels[i] for i in self.indicies_to_use]
            self.one_hot_labels = [self.one_hot_labels[i] for i in self.indicies_to_use]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, ind):
        filename = self.filenames[ind]
        # use cache data if loaded already
        if filename in self.image_dict:
            image = self.image_dict[filename]
        else:
            # load if not cached
            path = os.path.join(CIFAR_10_IMAGE_DIR, self.filenames[ind])
            image = read_image(path)
            if self.cache_data == True:
                self.image_dict[filename] = image

        # label = self.labels[ind]
        one_hot_label = self.one_hot_labels[ind]
        return image, one_hot_label
    
