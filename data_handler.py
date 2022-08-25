import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class SimpleDataset(Dataset):
    def __init__(self, data, t, shuffle=True):
        self.data = data
        self.t = t
        assert len(t) == len(data)
        self.loader = torch.utils.data.DataLoader(self, batch_size = 32, shuffle=shuffle)

    def __getitem__(self, index):
        return self.data[index].float(), self.t[index].float()

    def __len__(self):
        return len(self.data)


class CondDataset(Dataset):
    def __init__(self, X,y, t= None):
        self.X = X
        self.y = y
        self.t = t
        assert len(X) == len(y)
        if t is not None:
            assert len(t) == len(y)
        
        self.loader = torch.utils.data.DataLoader(self, batch_size = 32, shuffle=True)

    def __getitem__(self, index):
        if self.t is None:
            return self.X[index].float(), self.y[index].float()
        return self.X[index].float(), self.y[index].float(), self.t[index].float()

    def __len__(self):
        return len(self.X)


class Data(Dataset):
    def __init__(self, T, n, num_windows, num_classes = 5, df = None):
        self.T = T
        self.n = n
        self.df = df
        self.num_windows = num_windows
        self.num_classes = num_classes
        self.params = {}
        self.params[0] = (-0.2, 0.5)
        self.params[1] = (0.025, 0.1)
        self.params[2] = (0.0, 0.2)
        self.params[3] = (0.1, 0.2)
        self.params[4] = (0.2, 1.0)
        if df is None:
            self.X, self.y = self._gen_data()
        else:
            self.X, self.y = self._get_data()
        self.loader = DataLoader(self, batch_size=32, shuffle=True)

    def _get_data(self):
        X, y = [], []
        steps = self.df.shape[0] // self.T
        print("STEPS: ", steps)
        for t in range(steps):
            df_t = self.df.iloc[t*self.T:(t+1)*self.T].dropna(axis=1)
            if self.n > 1:
                for i in range(self.num_windows):
                
                    idx = np.random.choice(np.arange(df_t.shape[1]), self.n)
                    df_i = df_t.iloc[:, idx]
                    #df_i = df_i.div(df_i.iloc[0], axis=1)
                    X.append(df_i.values)
                    y.append(t)
            else:
                for i in range(df_t.shape[1]):
                    s = df_t.iloc[:, i]
                    #s = s.div(s.iloc[0])
                    X.append(s.values)
                    y.append(t)
        return np.array(X), np.array([y]).T
        
        
    def _gen_data(self):
        X, y = [], []
        dt = 1/252.
        for i in range(self.num_windows):
            cls = np.random.randint(0, self.num_classes)
            mu, sigma = self.params[cls]
            S = np.zeros((self.T,self.n))
            S[0,:] = 1
            for t in range(1,self.T):
                w = np.random.randn(self.n)
                #s = mu * dt + np.sqrt(dt) * sigma * w
                s = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + np.sqrt(dt)*sigma*w)
                S[t] = s
            #S = (1.0 + S).cumprod(axis=0)
            X.append(S)
            y.append(cls)
        return np.array(X), np.array([y]).T

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x = self.X[idx]
        #x = (x - x.mean(axis=0)) / x.std(axis=0)
        #print(x.shape)
        y = self.y[idx]

        return torch.Tensor(x), torch.Tensor(y)
        