from statsmodels.tsa.ar_model import AR
import numpy as np
class ArVol:
    def __init__(self, p=100):
        self.p=p

    def fit(self, data):
        data = np.log(data)
        self.fit = AR(data).fit(self.p)

    def params(self, show_intercept = True):
        if show_intercept:
            return self.fit.params
        return self.fit.params[1:]

    def predict(self, data):
        data = np.log(data)
        params = self.params()
        return np.exp(np.convolve(data,params[1::-1],'valid')+params[0])