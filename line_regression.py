import logging
import numpy as np
import pandas as pd
import random
from typing import Union, Callable
from sklearn.datasets import make_regression


class MyLineReg:

    # x - features in Pandas Dataframe,
    # y - targets in Pandas Series
    # verbose - number of iteration for log output
    # n_iter - number of iteration for end

    def __init__(self,
                 n_iter: int = 100,
                 learning_rate: Union[float, Callable] = 0.01,
                 metric: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0,
                 reg = None,
                 sgd_sample: float = None,
                 random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.metric_result = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    @staticmethod
    def _mse(y: pd.Series, y_pred: pd.Series) -> float:
        return np.mean((y_pred - y) ** 2)
        # return ((y - y_pred) ** 2).mean()

    @staticmethod
    def _mae(y: pd.Series, y_pred: pd.Series) -> float:
        return np.mean(np.abs(y - y_pred))
        # return (y - y_pred).abs().mean()

    @staticmethod
    def _rmse(y: pd.Series, y_pred: pd.Series) -> float:
        return np.sqrt(np.mean((y - y_pred) ** 2))
        # return np.sqrt(((y - y_pred) ** 2).mean())

    @staticmethod
    def _mape(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Mean absolute percentage error.
        Measures the average magnitude of error produced by a model,
        or how far off predictions are on average.

        :param y:
        :param y_pred:
        :return:
        """
        return 100 * np.mean(np.abs((y - y_pred) / y))
        # return 100 * ((y - y_pred) / y).abs().mean()

    @staticmethod
    def _r2(y: pd.Series, y_pred: pd.Series) -> float:
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        # return 1 - ((y - y_pred) ** 2).mean() / ((y - y.mean()) ** 2).mean()

    @staticmethod
    def _l1(coef1: float, coef2: float, weights: np.ndarray) -> list:
        # Regularization L1 - Lasso regression [loss, gradient]
        return [coef1 * np.sum(abs(weights)), coef1 * np.sign(weights)]

    @staticmethod
    def _l2(coef1: float, coef2: float, weights: np.ndarray) -> list:
        # Regularization L2 - Ridge regression [loss, gradient]
        return [coef2 * np.sum(weights ** 2), coef2 * 2 * weights]

    @staticmethod
    def _elasticnet(coef1: float, coef2: float, weights: np.ndarray) -> list:
        # Regularization ElasticNet [loss, gradient]
        return [coef1 * np.sum(abs(weights)) + coef2 * np.sum(weights ** 2),
                coef1 * np.sign(weights) + coef2 * 2 * weights]

    @staticmethod
    def sgd(sgd_sample, x):
        if isinstance(sgd_sample, int):
            return random.sample(range(x.shape[0]), sgd_sample)
        elif isinstance(sgd_sample, float):
            return random.sample(range(x.shape[0]), round(sgd_sample * x.shape[0]))

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose: int = None):
        random.seed(self.random_state)
        x.insert(0, 'ones', np.ones(x.shape[0], dtype=int))
        number_observ, number_features = x.shape
        self.weights = pd.Series(np.ones(number_features, dtype=int))
        rloss, rgrad = 0, 0
        print(np.dot(x, self.weights).shape)

        n = 1
        while n <= self.n_iter:
            # sample rows
            if self.sgd_sample:
                sample_rows_idx = self.sgd(self.sgd_sample, x)
            else:
                sample_rows_idx = x.shape[0]

            # predictions value
            y_pred = np.dot(x, self.weights)

            if self.reg is not None:
                rloss, rgrad = getattr(self, '_' + self.reg)(self.l1_coef, self.l2_coef, self.weights)

            # Mean square error + regularization
            loss_mse = np.mean((y_pred - y) ** 2) + rloss

            # Gradient + regularization
            if self.sgd_sample:
                grad_mse = (np.dot((2 * (y_pred[sample_rows_idx] - y.iloc[sample_rows_idx]) / len(sample_rows_idx)),
                            x.iloc[sample_rows_idx]) + self.l1_coef * np.sign(self.weights)
                            + self.l2_coef * 2 * self.weights)
            else:
                grad_mse = np.dot((2 * (y_pred - y) / number_observ), x.to_numpy()) + rgrad
                # grad_mse = (np.dot((2 * (y_pred - y) / number_observ), x.to_numpy())
                #             + self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights)

            # New weights
            if callable(self.learning_rate):
                self.weights -= self.learning_rate(n) * grad_mse
            else:
                self.weights -= self.learning_rate * grad_mse

            # self.weights -= grad_mse * self.learning_rate(n) if callable(self.learning_rate) \
            #     else grad_mse * self.learning_rate

            n += 1

            if self.metric:
                self.metric_result = getattr(self, '_' + self.metric)(y, y_pred)

            if verbose and n % verbose == 0:
                if self.metric:
                    logging.info(f'{n} | loss: {loss_mse}')
                else:
                    logging.info(f'{n} | loss: {loss_mse} | {self.metric}: {self.metric_result}')

    def get_coef(self):
        return self.weights[1:]

    def predict(self, x: pd.DataFrame):
        x.insert(0, 'ones', np.ones(x.shape[0], dtype=int))
        return np.dot(x.to_numpy(), self.weights.to_numpy())

    def get_best_score(self):
        return self.metric_result

    def __str__(self):
        # print(self.__dict__.items())
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    __repr__ = __str__

    # def __repr__(self):
    #     params = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
    #     return f'{__class__.__name__} class: {params}'


# MyLineReg_1 = MyLineReg(0.1)
# x, y = make_regression(400, 5, noise=5)
# m = MyLineReg_1.fit(pd.DataFrame(x), y)
# print(m)
# print(sum(MyLineReg_1.predict(pd.DataFrame(x))))
# print(MyLineReg_1)
# print(MyLineReg_1.get_coef())
# print(MyLineReg_1.get_best_score())