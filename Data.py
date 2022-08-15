import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
np.random.seed(2)


# loading the data
def load_data(path):
    df = pd.read_csv(path)
    return df


# adds the noise to our data as given
def add_noise(data):
    """
    :param data: dataset as np.array of shape (n, d) with n observations and d features
    :return: data + noise, where noise~N(0,0.001^2)
    """
    noise = np.random.normal(loc=0, scale=0.001, size=data.shape)
    return data + noise


# gets folds as given
def get_folds():
    """
    :return: sklearn KFold object that defines a specific partition to folds
    """
    return KFold(n_splits=5, shuffle=True, random_state=34)


def adjust_labels(y):
    """
    :param y: the column 'season' from the data frame we loaded
    :return: the column labeled by 0 or 1
    """
    for i, label in enumerate(y):
        if label == 0 or label == 1:
            y[i] = 0
        else:
            y[i] = 1
    return y


class StandardScaler:
    def __init__(self):
        """ object instantiation """
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        fit scaler by learning mean and standard deviation per feature
        :param X: the data frame
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, ddof=1, axis=0)

    def transform(self, X):
        """
        transform X by learned mean and standard deviation, and return it
        :param X: the data frame
        :return: the data frame we sent scaled
        """
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        """ fit scaler by learning mean and standard deviation per feature, and then transform X """
        self.fit(X)
        return self.transform(X)

