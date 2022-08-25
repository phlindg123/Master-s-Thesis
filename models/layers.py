import torch
import torch.nn as nn
import torch.distributions as td

class MVN(nn.Module):
    def __init__(self, i, n):
        super().__init__()
        self.n = n
        self.mu = nn.Linear(i, n)
        #self.log_var = nn.Linear(i, n)
        self.L = nn.Linear(i, n**2)
    
    def forward(self, x):
        mu = self.mu(x)
        #log_var = self.log_var(x)
        L_prim = self.L(x).view(-1, self.n, self.n)#.tril(-1) 
        #sigma = torch.exp(0.5 * log_var)
        eps = 0.05
        #L = L_prim + sigma.diag_embed()# + eps * torch.ones((x.size(0),self.n)).diag_embed()
        
        L = L_prim.bmm(L_prim.transpose(1,2)) + eps * torch.ones((x.size(0),self.n)).diag_embed()
        return td.MultivariateNormal(mu, scale_tril = L)

class Normal(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.mu = nn.Linear(i,o)
        self.log_var = nn.Linear(i,o)
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        sigma = torch.exp(0.5 * log_var) + 1e-4
        return td.Normal(mu, sigma)

class LogNormal(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.mu = nn.Linear(i,o)
        self.log_var = nn.Linear(i,o)
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        sigma = torch.exp(0.5 * log_var)
        return td.LogNormal(mu, sigma)


class ST(nn.Module):
    def __init__(self, i, n):
        super().__init__()
        self.n = n
        self.df = nn.Linear(i, n)
        self.mu = nn.Linear(i, n)
        self.log_var = nn.Linear(i, n)
    
    def forward(self, x):
        df = self.df(x)
        df = torch.exp(df)
        mu = self.mu(x)
        log_var = self.log_var(x)
        sigma = torch.exp(0.5 * log_var)
        return td.StudentT(df, mu, sigma)




class Latent(nn.Module):
    def __init__(self, i, n):
        super().__init__()
        self.mu = nn.Linear(i, n)
        self.log_var = nn.Linear(i, n)
    
    def _rep(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self._rep(mu, log_var)
        return z, mu, log_var


class Reshape(nn.Module):
    def __init__(self, s1, s2):
        super().__init__()
        self.s1 = s1
        self.s2 = s2
    def forward(self, x):
        return x.view(-1, self.s1, self.s2)