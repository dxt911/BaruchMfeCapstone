from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
class RoughVol:
    def __init__(self, forward=1, fit_period=200):
        self.forward = forward
        self.fit_period = fit_period
        self.time_period = [self.fit_period]

    def __str__(self):
        return("RoughVol")

    def fit(self, data, draw_power_plot=False, **args):
        # this is to estimate the H index in the model
        data = np.log(data)
        qs = np.arange(0.5, 2.5, 0.1)
        powers = []
        lm = LinearRegression()
        for q in qs:
            lags = list(range(1, 101))
            lag_q = [np.mean(np.abs(data[lag:] - data[:-lag]) ** q) for lag in lags]
            lm.fit(np.log(lags).reshape(-1, 1), np.log(lag_q))
            powers.append(lm.coef_[0])
        lm.fit(qs.reshape(-1, 1), powers)
        if draw_power_plot:
            plt.scatter(qs, powers)
        self.H = lm.coef_[0]
        return self.H

    @staticmethod
    @lru_cache(maxsize=2)
    def _A(fit_period, H, forward):
        gamma = 0.5 - H
        s_star = gamma ** (1 / (1 - gamma))
        A = np.sum(
            [(1 / ((j + 0.5 + forward) * ((j + .5) ** (H + 0.5)))) for j in range(1, fit_period + 1)])
        A += 1 / (s_star + forward) / ((s_star) ** (H + .5))
        return A

    @staticmethod
    @lru_cache(maxsize=2)
    def _sstar(H):
        gamma = 0.5 - H
        s_star = gamma ** (1 / (1 - gamma))
        return s_star

    def predict(self, data, **args):
        result = []
        for pos in range(self.fit_period, len(data) + 1):
            tempt_result = 0
            tempt_result += np.log(data[pos - 1]) / (self._sstar(self.H) + self.forward) / (self._sstar(self.H) ** (self.H + .5))
            tempt_result += np.sum(np.array(
                [np.log(data[pos - 1 - j]) / (j + 0.5 + self.forward) / ((j + .5) ** (self.H + 0.5)) for j in
                 range(1, self.fit_period + 1)], dtype=np.float64))
            result.append(tempt_result / self._A(self.fit_period,self.H, self.forward))
        return np.exp(result)

    def kernal(self, forward=1, lag=10, H=None):

        if H is None:
            H = self.H
        gamma = 0.5 - H
        s_star = gamma ** (1 / (1 - gamma))

        kernals = np.array([(1 / ((j + 0.5 + forward) * ((j + .5) ** (H + 0.5)))) for j in range(0, lag + 1)])
        kernals[0] = 1 / (s_star + forward) / ((s_star) ** (H + .5))
        kernals = kernals / np.sum(kernals)

        return kernals



