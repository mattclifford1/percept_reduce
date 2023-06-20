import os
import shutil
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms

file_path = os.path.dirname(os.path.abspath(__file__))
download_path = os.path.join(file_path, 'raw_data')


def download_CIFAR_10(redo_download=False):
    if redo_download == True:
        shutil.rmtree(download_path)
    if not os.path.exists(download_path):
        # use torch vision to download CIFAR-10 and unzip
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=download_path, train=True,
                                            download=True, transform=None)
        testset = torchvision.datasets.CIFAR10(root=download_path, train=False,
                                            download=True, transform=None)
    # delete tar file
    comp = os.path.join(download_path, 'cifar-10-python.tar.gz')
    if os.path.exists(comp):
        os.remove(comp)
    
    # save to png
    ims_dir = os.path.join(download_path, 'images')
    if not os.path.exists(ims_dir):
        save_batches_as_png(ims_dir)

    bin_data = os.path.join(download_path, 'cifar-10-batches-py')
    if os.path.exists(bin_data):
        shutil.rmtree(bin_data)
        
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_batches_as_png(ims_dir):
    batches = ['data_batch_1',
               'data_batch_2',
               'data_batch_3',
               'data_batch_4',
               'data_batch_5',
               'test_batch']
    file_names = []
    labels = []
    one_hot_labels = []
    label_names = unpickle(os.path.join(download_path, 'cifar-10-batches-py', 'batches.meta'))
    label_names = [x.decode('utf-8') for x in label_names[b'label_names']]                    
    os.makedirs(ims_dir, exist_ok=True)
    for batch in tqdm(batches):
        batch_dict = unpickle(os.path.join(download_path, 'cifar-10-batches-py', batch))
        for i, file in tqdm(enumerate(batch_dict[b'filenames']), leave=False):
            filename = file.decode("utf-8")
            im = batch_dict[b'data'][i, :].reshape(3, 32, 32)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            im.save(os.path.join(ims_dir, filename))

            file_names.append(filename)
            label = int(batch_dict[b'labels'][i])
            one_hot_labels.append(label) 
            labels.append(label_names[int(label)])

    meta_dict = {'filename': file_names, 
                 'label': labels,
                 'one_hot_label': one_hot_labels}
    df = pd.DataFrame.from_dict(meta_dict)
    df.to_csv(os.path.join(download_path, 'meta_data.csv'), index=False, header=list(meta_dict.keys()))



if __name__ == '__main__':
    download_CIFAR_10(redo_download=True)
