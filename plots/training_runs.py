import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

save_dir = os.path.join('.', 'saves')
save_folders = [f.path for f in os.scandir(save_dir) if f.is_dir()]

# extract data from csv files
num_subplots = 1
all_results = {}
for dir in save_folders:
    results_file = os.path.join(dir, 'training_results.csv')
    df = pd.read_csv(results_file)
    info = dir.split('-')
    net = info[0].split('/')[-1]
    data_size = info[1]
    total_epochs = info[2]
    loss = info[3]
    batch_size = info[4]
    if net not in all_results:
        all_results[net] = {}
    if data_size not in all_results[net]:
        all_results[net][data_size] = {}
    validations = df.columns.to_list()
    validations.remove('epoch')
    num_subplots = max(num_subplots, len(validations))
    for validation in validations:
        if validation not in all_results[net][data_size]:
            all_results[net][data_size][validation] = {}
        epochs = df['epoch'].to_list()
        scores = df[validation].to_list()
        all_results[net][data_size][validation][loss] = {'epoch': epochs, 'values': scores}

# now plot
colours = {'SSIM': 'blue', 'LPIPS': 'green', 'MSE': 'red', 'MSSIM': 'orange'}
h_plots = int(np.ceil(num_subplots/2))
for net, ds, in all_results.items():
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
            axs[v_num, plot_num].set_ylim([0, 0.8])

            plot_num += 1
            if plot_num == h_plots:
                plot_num = 0
                v_num += 1 
        file = os.path.join('plots', f'{net} with datasize {float(data_size)*100}.png')
        fig.set_size_inches(18.5, 10.5)
        fig.tight_layout()
        plt.savefig(file, bbox_inches='tight', dpi=100)
