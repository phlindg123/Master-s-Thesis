

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.autograd import Variable

class CondTrainer:
    def __init__(self, vae, train_data, test_data, cuda=False, conv=False, T = None, n_x = None, n_y = None):
        self.train_data = train_data
        self.test_data = test_data
        
        if cuda:
            self.vae = vae.cuda()
            self.device = torch.device("cuda")
        else:
            self.vae = vae
            self.device = torch.device("cpu")
        
        self.conv = conv
        self.T = T
        self.n_x = n_x
        self.n_y = n_y
        
        
        self.opt = optim.Adam(vae.parameters(), lr=1e-3)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.opt, patience=10, verbose = True)
        #self.sched =  optim.lr_scheduler.StepLR(self.opt, 50, 0.1)


    
    def _train(self, e, burn_in = 0):
        self.vae.train()
        tot_loss = 0
        rcl_loss = 0
        dkl_loss = 0
        beta = 0
        if e > burn_in:
            beta = 1
        for i, (x,y, t) in enumerate(self.train_data.loader):
            x = Variable(x).to(self.device)
            y = Variable(y).to(self.device)
            if self.conv:
                x = x.view(-1, self.n_x, self.T)
                y = y.view(-1, self.n_y, self.T)
            self.opt.zero_grad()
            gen_x, loss, rcl, dkl = self.vae(x, y, beta)
            loss.backward()
            self.opt.step()
            tot_loss += loss.item()
            rcl_loss += rcl.item()
            dkl_loss += dkl.item()
        return tot_loss, rcl_loss, dkl_loss
    
    def _test(self, e, burn_in = 0):
        self.vae.eval()
        with torch.no_grad():
            tot_loss = 0
            beta = 0
            if e > burn_in:
                beta = 1
            for i, (x, y, t) in enumerate(self.test_data.loader):
                x = Variable(x).to(self.device)
                y = Variable(y).to(self.device)
                if self.conv:
                    x = x.view(-1, self.n_x, self.T)
                    y = y.view(-1, self.n_y, self.T)
                gen_x, loss, _, _ = self.vae(x, y, beta)
                tot_loss += loss.item()
            return tot_loss
    
    def fit(self, epochs, burn_in = 0, print_every = 10):
        losses = []
        for e in range(epochs):
            train_loss, rcl_loss, dkl_loss = self._train(e, burn_in)
            train_loss /= len(self.train_data)
            rcl_loss /= len(self.train_data)
            dkl_loss /= len(self.train_data)
            test_loss = self._test(e, burn_in)
            test_loss /= len(self.test_data)
            #self.sched.step(train_loss)
            if e % print_every == 0:
                print(f"Epoch: {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss: .2f}")
                print(f"RCL Loss: {rcl_loss:.2f}, DKL Loss: {dkl_loss:.2f}")
            losses.append(pd.DataFrame({"Test":test_loss, "Train":train_loss, "RCL":rcl_loss, "DKL":dkl_loss}, index=[e]))
        return pd.concat(losses)