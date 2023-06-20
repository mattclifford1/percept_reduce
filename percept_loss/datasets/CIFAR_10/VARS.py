import os

file_path = os.path.dirname(os.path.abspath(__file__))
CIFAR_10_RAW_DATA_DIR = os.path.join(file_path, 'raw_data')
CIFAR_10_META_CSV = os.path.join(CIFAR_10_RAW_DATA_DIR, 'meta_data.csv')
CIFAR_10_IMAGE_DIR = os.path.join(CIFAR_10_RAW_DATA_DIR, 'images')
