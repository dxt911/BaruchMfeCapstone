
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
        self.linear_layers=[]
        for idx in range(len(layerNodes)-1):
            self.linear_layers.append(nn.Linear(layerNodes[idx],layerNodes[idx+1]))

    def forward(self, x):
        for idx in range(len(self.linear_layers)-1):
            x = self.linear_layers[idx](x)
            x = F.relu(x)
        return self.linear_layers[-1](x)

class NeuralNetVol:
    def __init__(self, hiddenlayer:list=[100], forward=100, fitperiod=200):
        self.nnet = _assis_net(hiddenlayer, forward, fitperiod)
        self.forward=forward
        self.fitperiod=fitperiod

    def _organize_input_data(self, data, if_fit):

        if if_fit:
            return np.concatenate([data[idx:idx + self.fitperiod] for idx in range(len(data) - self.forward - self.fitperiod)])

        return np.concatenate([data[idx:idx + self.fitperiod] for idx in range(len(data) - self.fitperiod)])

    def _organize_output_data(self, data):
        np.concatenate([data[idx+self.fitperiod:idx+self.fitperiod+self.forward] for idx in range(len(data) - self.forward - self.fitperiod)])

    def fit(self, tsdata: np.ndarray, lr=1e-3, steps = 1000, draw_loss = False, show_process=True):
        tsdata = np.log(tsdata)
        X = self._organize_input_data(tsdata,True)
        Y = self._organize_output_data(tsdata)

        x=torch.tensor(X, dtype=torch.float)
        y=torch.tensor(Y, dtype=torch.float)

        y_pred = self.nnet(x)
        optimizer = Adam(self.nnet.parameters(),lr=lr)

        loss_record = []

        def train():

            optimizer.zero_grad()
            loss = torch.mean(torch.sum((y - y_pred) ** 2, dim=tuple(1)))
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())

        if show_process:
            iters = tqdm(range(steps))
        else:
            iters = range(steps)

        for _ in iters:
            train()

        if draw_loss:
            plt.figure(figsize=(8,8))
            plt.plot(loss_record)
            plt.ylabel("loss")
            plt.xlabel("training step")
            plt.show()

    def predict(self, tsdata):
        tsdata = np.log(tsdata)
        X = self._organize_input_data(tsdata, if_fit=False)
        x = torch.tensor(X, dtype = torch.float)
        return self.nnet(x).detach().numpy()