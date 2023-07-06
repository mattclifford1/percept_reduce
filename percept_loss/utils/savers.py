import os
import matplotlib.pyplot as plt
import numpy as np

class train_saver:
    def __init__(self, epochs, loss, network, batch_size, lr, datasize, base_save='saves'):
        self.epochs = epochs
        self.loss = loss
        self.network = network
        self.batch_size = batch_size
        self.lr = lr
        self.datasize = datasize

        self.unique_name = f"{network}-bs{self.batch_size}-d{datasize}-e{epochs}-{loss}-lr{self.lr}"
        self.save_dir = os.path.join(base_save, self.unique_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        self.image_save_counter = 0

    def write_images(self, torch_images:list, epoch, num_ims=4):
        _, axes_list = plt.subplots(len(torch_images), num_ims)
        for i, torch_tensor in enumerate(torch_images):
            for j in range(num_ims):
                np_im = torch_tensor[j, :, :, :].detach().cpu().numpy()
                np_im = np.squeeze(np_im)
                np_im = np.transpose(np_im, (1, 2, 0))
                np_im = np.clip(np_im, 0, 1)
                axes_list[i, j].imshow(np_im)
            
        plt.savefig(os.path.join(self.image_dir, f'{self.image_save_counter}-{epoch}.png'))
        self.image_save_counter += 1
