from numpy import np


class MlWrapper:
    def __init__(self, model, lag=10, forward=1):
        self.model = model
        self.lag = lag
        self.forward = forward

    def __organize_input_data(self, data: np.ndarray, if_fit: bool) -> np.ndarray:

        if if_fit:
            return np.concatenate([data[idx:idx + self.lag] for idx in range(len(data) - self.forward - self.lag)])

        return np.concatenate([data[idx:idx + self.lag] for idx in range(len(data) - self.lag)])

    def __organize_output_data(self, data: np.array) -> np.ndarray:
        return np.convolve(data[self.lag:], np.ones(self.forward), 'valid')

    def fit(self, tsdata):
        pass

    def predict(self, tsdata):
        pass

    def shap_analysis(self, tsdata):
        pass

