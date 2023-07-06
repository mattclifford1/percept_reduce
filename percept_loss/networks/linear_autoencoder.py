import torch

class linear_AE(torch.nn.Module):
    def __init__(self, dims=(3, 32, 32), latent_dim=64):
        super().__init__()
        self.dims = dims
        self.latent_dim = latent_dim
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(dims[0]*dims[1]*dims[2], self.latent_dim*4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim*4, self.latent_dim),
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim*4),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_dim*4, dims[0]*dims[1]*dims[2]),
            torch.nn.Sigmoid()
        )
 
    def encoder_forward(self, x):
        x = x.reshape(-1, self.dims[0]*self.dims[1]*self.dims[2])
        return self.encoder(x)
    
    def decoder_forward(self, z):
        x_hat = self.decoder(z)
        return x_hat.reshape(-1, self.dims[0], self.dims[1], self.dims[2])
    
    def forward(self, x):
        z = self.encoder_forward(x)
        return self.decoder_forward(z)