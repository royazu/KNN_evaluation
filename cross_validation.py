import numpy as np


def cross_validation_score(model, X, y, folds, metric):
    """
    run cross validation on X and y with specific model by given folds. Evaluate by given metric.
    :param model: model object after instantiation process - ClassificationKNN / RegressionKNN
    :param X: Matrix, each row is a date element, each column represent feature.
    :param y: numpy array, each element contains label of data element.
    :param folds: KFOLD.sklearn object, the output of "data.get_folds".
                  contains the data separation by the required folds number.
    :param metric: function, get y_true and y_prediction and returns a scalar.
    :return: list (size by number of folds). each element contains the metric value of a validation set.
    """
    metric_values = []
    for train_indices, validation_indices in folds.split(X):
        current_x_train, current_y_train = X[train_indices], y[train_indices]
        current_x_validation, current_y_validation = X[validation_indices], y[validation_indices]
        model.fit(current_x_train, current_y_train)
        y_pred = model.predict(current_x_validation)
        metric_values.append(metric(current_y_validation, y_pred))
    return metric_values


def model_selection_cross_validation(model, k_list, X, y, folds, metric):
    """
    run cross validation on X and y for every model induced by values from k_list by given folds.
    Evaluate each model by given metric.

    :param model: model - before instantiation process - just KNN.
    :param k_list: different k-values for KNN model.
    :param X: Matrix, each row is a date element, each column represent feature.
    :param y: numpy array, each element contains label of data element.
    :param folds: KFOLD.sklearn object, the output of "data.get_folds".
                  contains the data separation by the required folds number.
    :param metric: function, get y_true and y_prediction and returns a scalar.
    :return: list (size by number of folds). each element contains the metric value of a validation set.
    """
    model_metric_values = []
    model_metric_std = []
    for k in k_list:
        metric_values = cross_validation_score(model(k), X, y, folds, metric)
        model_metric_values.append(np.mean(metric_values))
        model_metric_std.append(np.std(metric_values, ddof=1))
    return model_metric_values, model_metric_std