import sys
import Data
import evaluation
import cross_validation
from knn import ClassificationKnn, RegressionKnn


def main(argv):

    """ starting the data to be initialized - loading the csv file"""
    df = Data.load_data(argv[1])  # reads the csv file
    folds = Data.get_folds()  # defininf the folds
    features_list = ["t1", "t2", "wind_speed", "hum"]  # specified features as supposed to be
    X = df[features_list].to_numpy()
    y = Data.adjust_labels(df["season"].to_numpy())
    X = Data.add_noise(X)  # adding noise

    k_list = [3, 5, 11, 25, 51, 75, 101]
    mean_metric_scores, std_metric_scores = cross_validation.model_selection_cross_validation(ClassificationKnn, k_list,
                                                                     X, y, folds, evaluation.f1_score)

    """ Classification of the data as shown in part 1"""

    print('Part1 - Classification')
    for i in range(len(k_list)):
        print(f"k={k_list[i]}, mean score: {mean_metric_scores[i]:.04f}, std of scores: {std_metric_scores[i]:.04f}")

    evaluation.visualize_results(k_list, mean_metric_scores, "f1_score", "Classification", "./plot_Classification")
    print()
    X = df[features_list[:3]].to_numpy()
    y = df["hum"].to_numpy()
    # adding the noise
    X = Data.add_noise(X)

    """ Regression as shown in part 2 """

    print("Part2 - Regression")
    mean_metric_scores, std_metric_scores = cross_validation.model_selection_cross_validation(RegressionKnn, k_list, X, y, folds, evaluation.rmse)
    for i in range(len(k_list)):
        print(f"k={k_list[i]}, mean score: {mean_metric_scores[i]:.4f}, std of scores: {std_metric_scores[i]:.4f}")
    evaluation.visualize_results(k_list, mean_metric_scores, "rmse", "Regression", "./plot_Regression")


if __name__ == '__main__':
    main(sys.argv)