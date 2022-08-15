import numpy as np
import matplotlib.pyplot as plt


def f1_score(y_true, y_pred):
    """
    returns f1_score of binary classification task with true labels y_true and predicted labels y_pred.
    :param y_true: true labels
    :param y_pred: predicted labels by classification
    :return f1_score
    """
    TP, FP, FN, TN = 0, 0, 0, 0
    for (prediction, true_label) in zip(y_pred, y_true):
        if prediction == 1:
            if true_label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if true_label == 0:
                TN += 1
            else:
                FN += 1
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score_ = (2 * precision * recall) / (precision + recall)
    return f1_score_


def rmse(y_true, y_pred):
    """
        returns RMSE of regression task with true labels y_true and predicted labels y_pred.
        :param y_true: true labels
        :param y_pred: predicted labels by classification
        :return rmse
    """
    y_sub = np.subtract(y_true, y_pred)
    return np.sqrt(np.mean(np.square(y_sub)))


# plotting the results
def visualize_results(k_list, scores, metric_name, title, path):
    """
     plot a results graph of cross validation scores

    :param k_list: list of k-cross validation values
    :param scores: list in the same size of k_list, each element refers the mean of the cross-validation process with
                   the correlated k value.
    :param metric_name: string - "rmse" / "f1_score"
    :param title: string - "classification" / "regression"
    :param path: path for saving.
    """
    plt.plot(k_list, scores)
    plt.title(f"{title} cross validation {metric_name} results by different k values")
    plt.xlabel("k")
    plt.ylabel(metric_name)
    plt.savefig(path)
    plt.show()