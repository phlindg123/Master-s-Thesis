import torch
import torch.nn as nn

from .layers import Latent

class Simple(nn.Module):
    def __init__(self, T, n, n_z):
        super().__init__()
        self.T = T
        self.n = n
        
        self.enc = nn.Sequential(
            nn.Linear(T*n, 250),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(0.2),
            nn.Linear(250, 125),
            nn.BatchNorm1d(125),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            Latent(125, n_z)
        )

        self.dec = nn.Sequential(
            nn.Linear(n_z, 125),
            nn.BatchNorm1d(125),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(125, 250),
            nn.BatchNorm1d(250),
            #nn.Dropout(0.5),
            nn.LeakyReLU(0.2),
            nn.Linear(250, T*n)
        )
        self.mse = nn.MSELoss(reduction="sum")
        
    def encode(self, x, return_loss = False):
        x = x.flatten(1)
        z, mu, log_var =  self.enc(x)
        if return_loss:
            dkl = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            return z, mu, log_var, dkl
        return z, mu, log_var
    
    def decode2(self, z):
        x = self.dec(z)
        dist = self.dist(x)
        x_list = []
        for t in range(self.T):
            x_list.append(dist.sample().unsqueeze(1))
        gen_x = torch.cat(x_list, dim=1)
        return gen_x, dist
    
    def decode(self,z):
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
    
    def forward(self, x, beta = 1.0):
        z, mu, log_var, dkl = self.encode(x, True)
        #gen_x, dist = self.decode(z)
        gen_x = self.decode(z)
        #rcl = self._rcl_loss(x,dist)
        rcl = self.mse(gen_x, x.flatten(1))
        #reg = self._reg_loss()
        return gen_x, torch.mean(rcl + dkl*beta)