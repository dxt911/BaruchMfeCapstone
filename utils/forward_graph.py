import numpy as np
from .rough_vol import RoughVol
import matplotlib.pyplot as plt
from .har import HarModel,LogHarModelWithReturn

class forward_graph_rough:
    def __init__(self, model=RoughVol(), fit_period=200, lower_bound=1, higher_bound=100):
        self.model = model.__class__(fit_period=fit_period)
        self.fit_period = fit_period
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound

    def plot(self, data, verbose=False, log=True):
        unconditional_var = []
        conditional_var = []

        H = self.model.fit(data[:int(len(data) / 2)])

        for time_forward in range(self.lower_bound, self.higher_bound + 1):
            self.model.forward = time_forward
            pred_result = []
            true_result = []
            delta_true = []
            count = 0

            for pos in range(int(len(data) / 2), len(data) - time_forward + 1):
                true_result.append(data[pos + time_forward - 1])
                if log:
                    delta_true.append(np.log(data[pos + time_forward - 1]) - np.log(data[pos - 1]))
                else:
                    delta_true.append(data[pos + time_forward - 1] - data[pos - 1])
            pred_result = self.model.predict(data[(int(len(data) / 2) - self.fit_period):(len(data) - time_forward)])
            unconditional_var.append(np.mean(np.array(delta_true) ** 2))
            if verbose:
                print(true_result[:100])
                print(any(np.array(pred_result) <= 0))
            if log:
                conditional_var.append(np.mean((np.log(true_result) - np.log(pred_result)) ** 2))
            else:
                conditional_var.append(np.mean((np.array(true_result) - np.array(pred_result)) ** 2))
            if verbose:
                print(conditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), unconditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), conditional_var)
        plt.legend(['unconditional_var', 'conditional_var'])
        plt.ylim((0, 1.05 * max(max(conditional_var), max(unconditional_var))))
        plt.ylabel(f'variance; H: {H}')
        plt.xlabel('lag')


class forward_graph_har:
    def __init__(self, model=HarModel(), fit_period=200, lower_bound=1, higher_bound=100):
        self.model = model
        self.fit_period = fit_period
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound

    def plot(self, data, verbose=False, log=True):
        unconditional_var = []
        conditional_var = []

        for time_forward in range(self.lower_bound, self.higher_bound + 1):
            self.model.forward = time_forward
            pred_result = []
            true_result = []
            delta_true = []
            count = 0

            for pos in range(self.fit_period + time_forward + max(self.model.time_period),
                             len(data) - time_forward + 1):

                self.model.fit(data[:pos])
                pred_result.append(self.model.predict(data[pos - max(self.model.time_period) - 1:pos])[-1])
                true_result.append(data[pos + time_forward - 1])
                if log:
                    delta_true.append(np.log(data[pos + time_forward - 1]) - np.log(data[pos - 1]))
                else:
                    delta_true.append(data[pos + time_forward - 1] - data[pos - 1])

            unconditional_var.append(np.mean(np.array(delta_true) ** 2))
            if verbose:
                print(true_result[:100])
                print(any(np.array(pred_result) <= 0))
            if log:
                conditional_var.append(np.mean((np.log(true_result) - np.log(pred_result)) ** 2))
            else:
                conditional_var.append(np.mean((np.array(true_result) - np.array(pred_result)) ** 2))
            if verbose:
                print(conditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), unconditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), conditional_var)
        plt.legend(['unconditional_var', 'conditional_var'])
        plt.ylim((0, 1.05 * max(max(conditional_var), max(unconditional_var))))
        plt.ylabel('variance')
        plt.xlabel('lag')
#         return(conditional_var,unconditional_var)


class forward_graph_har_return:
    def __init__(self, model=LogHarModelWithReturn(), fit_period=200, lower_bound=1, higher_bound=100):
        self.model = model
        self.fit_period = fit_period
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound

    def plot(self, data, price_data, verbose=False, log=True):
        unconditional_var = []
        conditional_var = []

        for time_forward in range(self.lower_bound, self.higher_bound + 1):
            self.model.forward = time_forward
            pred_result = []
            true_result = []
            delta_true = []
            count = 0
            _001len = int(0.01 * len(data))
            for pos in range(self.fit_period + time_forward + max(self.model.time_period),
                             len(data) - time_forward + 1):
                if count % _001len == 0:
                    self.model.fit(data[:pos], price_data[:pos])
                count += 1
                pred_result.append(self.model.predict(data[pos - max(self.model.time_period) - 1:pos],
                                                      price_data[pos - max(self.model.time_period) - 1:pos])[-1])
                true_result.append(data[pos + time_forward - 1])
                if log:
                    delta_true.append(np.log(data[pos + time_forward - 1]) - np.log(data[pos - 1]))
                else:
                    delta_true.append(data[pos + time_forward - 1] - data[pos - 1])

            unconditional_var.append(np.mean(np.array(delta_true) ** 2))
            if verbose:
                print(true_result[:100])
                print(any(np.array(pred_result) <= 0))
            if log:
                conditional_var.append(np.mean((np.log(true_result) - np.log(pred_result)) ** 2))
            else:
                conditional_var.append(np.mean((np.array(true_result) - np.array(pred_result)) ** 2))
            if verbose:
                print(conditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), unconditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), conditional_var)
        plt.legend(['unconditional_var', 'conditional_var'])
        plt.ylim((0, 1.05 * max(max(conditional_var), max(unconditional_var))))
        plt.ylabel('variance')
        plt.xlabel('lag')

from tqdm import tqdm_notebook as tqdm
class CompareGraph:
    def __init__(self, models:list, fit_period=2000, lower_bound=1, higher_bound=100):
        self.models = models
        self.fit_period = fit_period
        self.lower_bound = lower_bound
        self.higher_bound = higher_bound

    def plot(self, data, price_data=None, verbose=False, log=True, time_bar = False):
        unconditional_var = []
        conditional_var = []
        if price_data is None:
            price_data = data
        itrs = tqdm(range(self.lower_bound, self.higher_bound + 1)) if time_bar else range(self.lower_bound, self.higher_bound + 1)
        for _ in range(len(self.models)):
            conditional_var.append([])
        for time_forward in itrs:
            pred_result = []
            for model in self.models:
                model.forward = time_forward
                pred_result.append([])

            true_result = []
            delta_true = []
            count = 0
            _001len = int(0.01 * len(data))
            for pos in range(self.fit_period + time_forward, len(data) - time_forward + 1):

                if count % _001len == 0:
                    for model in self.models:
                        if str(model)!='RoughVol':
                            model.fit(data=data[:pos], price_data=price_data[:pos])
                        else:
                            if count % (20*_001len)==0:
                                model.fit(data=data[:pos], price_data=price_data[:pos])
                count += 1

                for i in range(len(pred_result)):
                    if str(self.models[i])!='RoughVol':
                        pred_result[i].append(self.models[i].predict(data=data[pos - max(self.models[i].time_period) - 1:pos],
                                                          price_data=price_data[pos - max(self.models[i].time_period) - 1:pos])[-1])
                    else:
                        pred_result[i].append(
                            self.models[i].predict(data=data[pos - max(self.models[i].time_period):pos],
                                                   price_data=price_data[
                                                              pos - max(self.models[i].time_period):pos])[-1])
                true_result.append(data[pos + time_forward - 1])
                if log:
                    delta_true.append(np.log(data[pos + time_forward - 1]) - np.log(data[pos - 1]))
                else:
                    delta_true.append(data[pos + time_forward - 1] - data[pos - 1])

            unconditional_var.append(np.mean(np.array(delta_true) ** 2))
            if verbose:
                print(true_result[:100])
                print(any(np.array(pred_result) <= 0))
            if log:
                for i in range(len(conditional_var)):
                    conditional_var[i].append(np.mean((np.log(true_result) - np.log(pred_result[i])) ** 2))
            else:
                for i in range(len(conditional_var)):
                    conditional_var.append(np.mean((np.array(true_result) - np.array(pred_result[i])) ** 2))
            if verbose:
                print(conditional_var)
        plt.scatter(range(self.lower_bound, self.higher_bound + 1), unconditional_var)
        for i in range(len(conditional_var)):
            plt.scatter(range(self.lower_bound, self.higher_bound + 1), conditional_var[i])
        plt.legend(['unconditional_var']+ [f'conditional_var{model}' for model in self.models])
        plt.ylim((0, 1.05 * max(np.max(conditional_var), np.max(unconditional_var))))
        plt.ylabel('variance')
        plt.xlabel('lag')