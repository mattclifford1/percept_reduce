import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class train_saver:
    def __init__(self, epochs, loss, network, batch_size, datasize, base_save='saves'):
        self.epochs = epochs
        self.loss = loss
        self.network = network
        self.batch_size = batch_size
        self.datasize = datasize

        self.unique_name = f"{network}-{datasize}-{epochs}-{loss}-{self.batch_size}"
        self.save_dir = os.path.join(base_save, self.unique_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        self.image_save_counter = 0

    def write_images(self, torch_images:list, epoch, num_ims=4, extra_name=''):
        _, axes_list = plt.subplots(len(torch_images), num_ims)
        for i, torch_tensor in enumerate(torch_images):
            for j in range(num_ims):
                np_im = torch_tensor[j, :, :, :].detach().cpu().numpy()
                np_im = np.squeeze(np_im)
                np_im = np.transpose(np_im, (1, 2, 0))
                np_im = np.clip(np_im, 0, 1)
                axes_list[i, j].imshow(np_im)
            
        plt.savefig(os.path.join(self.image_dir, f'{epoch}-{extra_name}.png'))
        self.image_save_counter += 1
        plt.close()

    def write_scores(self, scores, epoch):
        scores['epoch'] = epoch
        for name, item in scores.items():
            scores[name] = [item]
        df = pd.DataFrame.from_dict(scores)
        df.set_index('epoch', inplace=True)
        file = os.path.join(self.save_dir, f'training_results.csv')
        save_and_merge_df_as_csv(df, file)


def save_and_merge_df_as_csv(df, file):
    '''df need to be indexed'''
    if os.path.exists(file):
        df_saved = pd.read_csv(file)
        index_name = df.index.name
        df = pd.merge(df.reset_index(), df_saved, how='outer')
        df.set_index(index_name, inplace=True)
    df.to_csv(file, index=True)