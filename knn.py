import numpy as np
from scipy import stats
from abc import abstractmethod
from Data import StandardScaler


class KNN:
    def __init__(self, k):
        """
        object instantiation, save k and define a scaler object.
        :param k: k neighbors value
        """
        self.X_train = None
        self.y_train = None
        self.k = k
        self.scaler = StandardScaler()
        self.X_test = None

    def fit(self, X_train, y_train):
        """
        fit scaler and save X_train and y_train
        """
        self.X_train = self.scaler.fit_transform(X=X_train)
        self.y_train = y_train

    @abstractmethod
    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        """

    def neighbours_indices(self, x):
        """
        for a given point x, find indices of k closest points in the training set
        """
        distances = [self.dist(x, x_train) for x_train in self.X_train]
        sorted_dist_indices = np.argsort(distances)
        return sorted_dist_indices[:self.k]

    @staticmethod
    def dist(x1, x2):
        """
        returns Euclidean distance between x1 and x2
        """

        x1 = np.asarray(x1)
        x2 = np.asarray(x2)
        return np.linalg.norm(x1 - x2)


class ClassificationKnn(KNN):
    def __init__(self, k):
        """
        object instantiation, save k and define a scaler object
        :param k: k neighbors value
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        """
        self.X_test = self.scaler.transform(X_test)
        predictions = np.zeros(len(X_test))
        for i in range(len(X_test)):
            k_closest = self.neighbours_indices(self.X_test[i])
            predictions[i] = stats.mode(self.y_train[k_closest])[0][0]
        return predictions


class RegressionKnn(KNN):
    def __init__(self, k):
        """
        object instantiation, save k and define a scaler object
        :param k: k neighbors value
        """
        super().__init__(k)

    def predict(self, X_test):
        """
        predict labels for X_test and return predicted labels
        """
        self.X_test = self.scaler.transform(X_test)
        averages = np.zeros(len(X_test))
        for i in range(len(X_test)):
            k_closest = self.neighbours_indices(self.X_test[i])
            averages[i] = np.mean(self.y_train[k_closest])
        return averages


