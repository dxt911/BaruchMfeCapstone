import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


class _assis_net(nn.Module):
    def __init__(self, hiddenlayer:list, forward = 100, fit_period=200):
        super(_assis_net,self).__init__()
        layerNodes=[fit_period]+hiddenlayer+[forward]
        self.linear_layers=nn.ModuleList()
        for idx in range(len(layerNodes)-1):
            tempt = nn.Linear(layerNodes[idx],layerNodes[idx+1])
            self.linear_layers.append(tempt)


    def forward(self, x):
        for layer in self.linear_layers[:-1]:
            x = layer(x)
            x = F.selu(x)
        return self.linear_layers[-1](x)

class _loss(nn.Module):
    def __init__(self):
        super(_loss, self).__init__()

    def forward(self, y, y_pred):
        return torch.mean(torch.sum((y - y_pred) ** 2, dim=1))

class NeuralNetVol:
    def __init__(self, hiddenlayer:list = [100], forward=100, fitperiod=200):
        self.nnet = _assis_net(hiddenlayer, forward, fitperiod)
        self.forward=forward
        self.fitperiod=fitperiod

    def _organize_input_data(self, data, if_fit):

        if if_fit:
            return np.concatenate(np.array([data[idx:idx + self.fitperiod] for idx in range(len(data) - self.forward - self.fitperiod + 1)])[np.newaxis,:], axis=0)

        return np.concatenate(np.array([data[idx:idx + self.fitperiod] for idx in range(len(data) - self.fitperiod+1)])[np.newaxis,:], axis=0)

    def _organize_output_data(self, data):
        return np.concatenate(np.array([data[idx+self.fitperiod:idx+self.fitperiod+self.forward] for idx in range(len(data) - self.forward - self.fitperiod+1)])[np.newaxis,:], axis=0)

    def fit(self, tsdata: np.ndarray, lrs=[1, 1e-1, 1e-2, 1e-3, 1e-4], steps = 200, draw_loss = False, show_process=True, verbose=0):
        tsdata = np.log(tsdata)
        self.mean = np.mean(tsdata)
        self.std = np.std(tsdata)
        tsdata = (tsdata-self.mean) / self.std
        if verbose:
            print(tsdata, np.mean(tsdata), np.std(tsdata))

        X = self._organize_input_data(tsdata,True)
        if verbose>=1:
            print(X)
        Y = self._organize_output_data(tsdata)

        x=torch.tensor(X, dtype=torch.float)
        y=torch.tensor(Y, dtype=torch.float)

        optimizer = Adam(self.nnet.parameters(),lr=lrs[0])

        loss_record = []
        _lossm = _loss()
        def train(n):

            optimizer.zero_grad()
            y_pred = self.nnet(x)
            loss = _lossm(y, y_pred)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())


        for lr in lrs:
            if show_process:
                iters = tqdm(range(steps))
            else:
                iters = range(steps)
            optimizer = Adam(self.nnet.parameters(), lr=lr)
            for i in iters:
                train(i)

        if draw_loss:
            plt.figure(figsize=(8,8))
            plt.plot(np.log(loss_record))
            plt.ylabel("loss")
            plt.xlabel("training step")
            plt.show()
        return loss_record[-1]
    def predict(self, tsdata):
        tsdata = np.log(tsdata)
        tsdata = (tsdata-self.mean) / self.std
        X = self._organize_input_data(tsdata, if_fit=False)
        x = torch.tensor(X, dtype = torch.float)
        return (self.nnet(x).detach().numpy() * self.std) + self.mean