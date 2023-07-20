import torch

class UNIFORM_LOADER:
    def __init__(self, normalise=(0, 1), length=60000, size=(3, 32, 32),
                 dtype=torch.float32,
                 device='cpu'):
        self.normalise = normalise
        self.length = length
        self.size = size
        self.dtype = dtype
        self.device = device

    def _process_image(self, image):
        image = image*(self.normalise[1]-self.normalise[0])
        image = image + self.normalise[0]
        return image

    def _get_image(self):
        image_raw = torch.rand(self.size, dtype=self.dtype, device=self.device)
        image = self._process_image(image_raw)
        return image
    
    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        # get image
        image = self._get_image()
        return image, 0, 0
    

