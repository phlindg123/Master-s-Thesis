import torch
import torch.nn as nn

from .layers import Latent, Reshape

class CondConv(nn.Module):
    def __init__(self, T, n_x, n_y, n_z):
        super().__init__()
        self.T = T
        self.n_x = n_x
        self.n_y = n_y
        
        
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels = n_y + n_x, out_channels = 10, kernel_size = 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels = 10, out_channels = 5, kernel_size = 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(155, 50),
            nn.LeakyReLU(0.2),
            Latent(50, n_z)
        )

        self.y_to_z = nn.Sequential(
            nn.Linear(T*n_y, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 50)
        )

        self.dec = nn.Sequential(
            nn.Linear(n_z+ 50, 50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 155),
            nn.LeakyReLU(0.2),
            Reshape(5, 31),
            nn.ConvTranspose1d(in_channels = 5, out_channels = 10, kernel_size = 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(in_channels = 10, out_channels = n_x, kernel_size = 5, stride=1, padding=2)
        )


        self.mse = nn.MSELoss(reduction="sum")
        
    def encode(self, x,y, return_loss = False):
        x = torch.cat([x,y], dim=1)
        z, mu, log_var =  self.enc(x)
        if return_loss:
            dkl = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            return z, mu, log_var, dkl
        return z, mu, log_var
    
    
    def decode(self,z, y):
        y = y.flatten(1)
        y = self.y_to_z(y)
        z = torch.cat([z,y], dim=1)
        return self.dec(z)
    
    def _reg_loss(self):
        loss = 0
        lam = 0.05
        for m in self.parameters():
            loss += m.abs().sum()
        return loss*lam

    
    def forward(self, x,y, beta = 1.0):
        z, mu, log_var, dkl = self.encode(x, y,True)
        gen_x = self.decode(z,y)
        rcl = self.mse(gen_x, x)
        return gen_x, torch.mean(rcl + dkl*beta)