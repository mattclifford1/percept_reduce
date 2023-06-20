import os
import shutil
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
from percept_loss.datasets.CIFAR_10.VARS import CIFAR_10_RAW_DATA_DIR, CIFAR_10_META_CSV, CIFAR_10_IMAGE_DIR


def download_CIFAR_10(redo_download=False):
    if redo_download == True:
        shutil.rmtree(CIFAR_10_RAW_DATA_DIR)
    if not os.path.exists(CIFAR_10_RAW_DATA_DIR):
        # use torch vision to download CIFAR-10 and unzip
        _ = torchvision.datasets.CIFAR10(root=CIFAR_10_RAW_DATA_DIR, train=True,
                                            download=True, transform=None)
        _ = torchvision.datasets.CIFAR10(root=CIFAR_10_RAW_DATA_DIR, train=False,
                                            download=True, transform=None)
    # delete tar file
    comp = os.path.join(CIFAR_10_RAW_DATA_DIR, 'cifar-10-python.tar.gz')
    if os.path.exists(comp):
        os.remove(comp)
    
    # save to png
    if not os.path.exists(CIFAR_10_IMAGE_DIR):
        save_batches_as_png(CIFAR_10_IMAGE_DIR)

    bin_data = os.path.join(CIFAR_10_RAW_DATA_DIR, 'cifar-10-batches-py')
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
    label_names = unpickle(os.path.join(CIFAR_10_RAW_DATA_DIR, 'cifar-10-batches-py', 'batches.meta'))
    label_names = [x.decode('utf-8') for x in label_names[b'label_names']]                    
    os.makedirs(ims_dir, exist_ok=True)
    for batch in tqdm(batches):
        batch_dict = unpickle(os.path.join(CIFAR_10_RAW_DATA_DIR, 'cifar-10-batches-py', batch))
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
    df.to_csv(CIFAR_10_META_CSV, index=False, header=list(meta_dict.keys()))


if __name__ == '__main__':
    download_CIFAR_10(redo_download=True)
