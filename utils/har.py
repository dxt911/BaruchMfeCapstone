import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV

class HarModel:
    def __init__(self, time_period=(1, 5, 22), forward=1):
        self.time_period = time_period
        self.forward = forward

    def fit(self, data: np.ndarray, **args):
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T  # input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)-self.forward+1)])
        #         input_data=sm.add_constant(input_data)
        output_data = data[max(self.time_period) + self.forward - 1:]
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(input_data, output_data)

    def predict(self, data, **args):
        assert len(data) >= max(self.time_period)
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        #         input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)+1)])
        #         input_data=sm.add_constant(input_data)
        return self.model.predict(input_data)

    def summary(self, data, **args):
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T  # input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)-self.forward+1)])
        input_data = sm.add_constant(input_data)
        output_data = data[max(self.time_period + self.forward - 1):]
        model = sm.OLS(output_data, input_data).fit()
        print(model.summary())

    def __str__(self):
        return("HarModel")

class LogHarModel:
    def __init__(self, time_period=(1, 5, 22), forward=1):
        self.time_period = time_period
        self.forward = forward

    def __str__(self):
        return("LogHarModel")

    def fit(self, data: np.ndarray, **args):
        data = np.log(data)
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T  # input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)-self.forward+1)])
        #         input_data=sm.add_constant(input_data)
        output_data = data[max(self.time_period) + self.forward - 1:]
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(input_data, output_data)

    def predict(self, data, **args):
        data = np.log(data)
        assert len(data) >= max(self.time_period)
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        #         input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)+1)])
        #         input_data=sm.add_constant(input_data)
        return np.exp(self.model.predict(input_data))

    def summary(self, data, **args):
        data = np.log(data)
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T  # input_data = np.array([[np.mean(data[ele-t:ele]) for t in self.time_period] for ele in range(max(self.time_period),len(data)-self.forward+1)])
        input_data = sm.add_constant(input_data)
        output_data = data[max(self.time_period + self.forward - 1):]
        model = sm.OLS(output_data, input_data).fit()
        print(model.summary())


class LogHarModelWithReturnRidge:

    def __init__(self, time_period=(1, 5, 22), forward=1):
        self.time_period = time_period
        self.forward = forward

    def __str__(self):
        return("LogHarModelWithReturnRidge")

    def fit(self, data: np.ndarray, price_data):
        assert len(data) == len(price_data)
        data = np.log(data)
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T
        price_input_data = np.array(
            [np.convolve(price_data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T
        input_data = np.concatenate([input_data, price_input_data], axis=1)
        output_data = data[max(self.time_period) + self.forward - 1:]
        self.model = RidgeCV(alphas=[1e-3, 1e-2, 0.1, 1, 10])
        self.model.fit(input_data, output_data)

    def predict(self, data, price_data):
        assert len(data) == len(price_data)
        data = np.log(data)
        assert len(data) >= max(self.time_period)
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        price_input_data = np.array(
            [np.convolve(price_data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        input_data = np.concatenate([input_data, price_input_data], axis=1)
        return np.exp(self.model.predict(input_data))

class LogHarModelWithReturn:

    def __init__(self, time_period=(1, 5, 22), forward=1):
        self.time_period = time_period
        self.forward = forward

    def __str__(self):
        return("LogHarModelWithReturn")

    def fit(self, data: np.ndarray, price_data):
        assert len(data) == len(price_data)
        data = np.log(data)
        assert len(data) > max(self.time_period), "the length of the data should be longer than the longest time period"
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T
        price_input_data = np.array(
            [np.convolve(price_data[max(self.time_period) - t:-self.forward], np.ones(t) / t, 'valid') for t in
             self.time_period]).T
        input_data = np.concatenate([input_data, price_input_data], axis=1)
        output_data = data[max(self.time_period) + self.forward - 1:]
        self.model = LinearRegression(n_jobs=3)
        self.model.fit(input_data, output_data)

    def predict(self, data, price_data):
        assert len(data) == len(price_data)
        data = np.log(data)
        assert len(data) >= max(self.time_period)
        input_data = np.array(
            [np.convolve(data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        price_input_data = np.array(
            [np.convolve(price_data[max(self.time_period) - t:], np.ones(t) / t, 'valid') for t in self.time_period]).T
        input_data = np.concatenate([input_data, price_input_data], axis=1)
        return np.exp(self.model.predict(input_data))

