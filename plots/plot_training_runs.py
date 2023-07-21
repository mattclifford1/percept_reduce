import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
plot_dir = os.path.join(dir_path, 'figs')
os.makedirs(plot_dir, exist_ok=True)

save_dir = os.path.join('.', 'saves', '')
# save_folders = [f.path for f in os.scandir(save_dir) if f.is_dir()]
save_folders = glob(os.path.join(save_dir, '*', '*', '*', ''), recursive=True)


# extract data from csv files
num_subplots = 1
all_results = {}
for dir in save_folders:
    results_file = os.path.join(dir, 'training_results.csv')
    df = pd.read_csv(results_file)

    # extract info
    info = dir[len(save_dir):].split(os.sep)
    train_info = info[2].split('-')
    dataset = info[0]
    net = info[1]
    data_size = train_info[1]
    # total_epochs = info[2]
    loss = train_info[0]
    batch_size = train_info[2][2:]

    if dataset not in all_results:
        all_results[dataset] = {}
    if net not in all_results[dataset]:
        all_results[dataset][net] = {}
    if data_size not in all_results[dataset][net]:
        all_results[dataset][net][data_size] = {}
    validations = df.columns.to_list()
    validations.remove('epoch')
    num_subplots = max(num_subplots, len(validations))
    for validation in validations:
        if validation not in all_results[dataset][net][data_size]:
            all_results[dataset][net][data_size][validation] = {}
        epochs = df['epoch'].to_list()
        scores = df[validation].to_list()
        all_results[dataset][net][data_size][validation][loss] = {'epoch': epochs, 'values': scores}

# now plot
colours = {'SSIM': 'blue', 'LPIPS': 'green', 'MSE': 'red', 'MSSIM': 'orange', 'NLPD': 'yellow', 'DISTS': 'pink'}
h_plots = int(np.ceil(num_subplots/2))
for dataset, networks in tqdm(all_results.items()):
    for net, ds, in networks.items():
        for data_size, v in ds.items():
            fig, axs = plt.subplots(2, h_plots)
            fig.suptitle(f'{net} with datasize {data_size}')
            plot_num = 0
            v_num = 0
            for val_name, losses in v.items():
                for loss, data in losses.items():
                    axs[v_num, plot_num].plot(data['epoch'], data['values'], label=loss, color=colours[loss])
                axs[v_num, plot_num].title.set_text(val_name)
                axs[v_num, plot_num].legend()
                axs[v_num, plot_num].set_xlabel('epoch')
                axs[v_num, plot_num].set_ylabel('accuracy')
                if 'IMAGENET' in dataset:
                    axs[v_num, plot_num].set_ylim([0, 0.1])
                else:
                    axs[v_num, plot_num].set_ylim([0, 0.8])

                plot_num += 1
                if plot_num == h_plots:
                    plot_num = 0
                    v_num += 1 

            plot_exact_dir = os.path.join(plot_dir, dataset, net)
            os.makedirs(plot_exact_dir, exist_ok=True)
            if data_size != 'uniform':
                data_size = float(data_size)*100
            file = os.path.join(plot_exact_dir, f'{data_size}.png')
            fig.set_size_inches(18.5, 10.5)
            fig.tight_layout()
            plt.savefig(file, bbox_inches='tight', dpi=100)
