import torch
import torch.nn as nn

from .layers import Latent, Reshape

class Conv(nn.Module):
    def __init__(self, T, n, n_z):
        super().__init__()
        self.T = T
        self.n = n
        
        
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels = n, out_channels = 50, kernel_size = 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 50, out_channels = 100, kernel_size = 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(3100, 500),
            nn.LeakyReLU(0.2),
            Latent(500, n_z)
        )

        self.dec = nn.Sequential(
            nn.Linear(n_z, 500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 3100),
            nn.LeakyReLU(0.2),
            Reshape(100, 31),
            nn.ConvTranspose1d(in_channels = 100, out_channels = 50, kernel_size = 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels = 50, out_channels = n, kernel_size = 5, stride=1, padding=2)
        )


        self.mse = nn.MSELoss(reduction="sum")
        
    def encode(self, x, return_loss = False):
        z, mu, log_var =  self.enc(x)
        if return_loss:
            dkl = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            return z, mu, log_var, dkl
        return z, mu, log_var
    
    
    def decode(self,z):
        return self.dec(z)
    
    def _reg_loss(self):
        loss = 0
        lam = 0.05
        for m in self.parameters():
            loss += m.abs().sum()
        return loss*lam

    
    def forward(self, x, beta = 1.0):
        z, mu, log_var, dkl = self.encode(x, True)
        gen_x = self.decode(z)
        rcl = self.mse(gen_x, x)
        return gen_x, torch.mean(rcl + dkl*beta)