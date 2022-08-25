import torch
import torch.nn as nn

from .layers import Latent

class Cond(nn.Module):
    def __init__(self, T, n_x, n_y, n_z):
        super().__init__()
        self.T = T
        self.n_x = n_x
        self.n_y = n_y
        
        self.enc = nn.Sequential(
            nn.Linear(T*(n_x + n_y), 250),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(0.2),
            nn.Linear(250, 125),
            nn.BatchNorm1d(125),
            nn.LeakyReLU(0.2),
            nn.Linear(125, 75),
            nn.BatchNorm1d(75),
            nn.LeakyReLU(0.2),
            Latent(75, n_z)
        )

        self.dec = nn.Sequential(
            nn.Linear(n_z + T*n_y, 75),
            nn.BatchNorm1d(75),
            nn.LeakyReLU(0.2),
            nn.Linear(75, 125),
            nn.BatchNorm1d(125),
            nn.LeakyReLU(0.2),
            nn.Linear(125, 250),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(0.2),
            nn.Linear(250, T*n_x)
        )
        self.mse = nn.MSELoss(reduction="sum")

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

        
    def encode(self, x,y, return_loss = False):
        #x = x.flatten(1)
        #y = y.flatten(1)
        x = torch.cat([x,y], dim=1)
        z, mu, log_var =  self.enc(x)
        if return_loss:
            dkl = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            return z, mu, log_var, dkl
        return z, mu, log_var

    def decode(self,z,y):
        z = torch.cat([z,y], 1)
        return self.dec(z)
    
    def _reg_loss(self):
        loss = 0
        lam = 0.05
        for m in self.parameters():
            loss += m.abs().sum()
        return loss*lam
    
    def _rcl_loss(self, x, dist):
        loss = 0
        for t in range(self.T):
            x_t = x[:, t, :]
            loss -= dist.log_prob(x_t).sum(dim=-1)
        return loss
    
    def forward(self, x,y, beta = 1.0):
        z, mu, log_var, dkl = self.encode(x,y, True)
        #gen_x, dist = self.decode(z)
        gen_x = self.decode(z,y)
        #rcl = self._rcl_loss(x,dist)
        rcl = self.mse(gen_x, x.flatten(1))
        #reg = self._reg_loss()
        return gen_x, torch.mean(rcl + dkl*beta)