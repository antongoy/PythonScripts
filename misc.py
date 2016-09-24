import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss


def encode_cat_feature_using_targets(
    feature_train, targets, feature_test,
    n_folds=None, default_prob=0.5, random_state=None
):

    """

    Encode categorical feature using targets.

    :param feature_train: Using to compute P{ target = 1 | category } for each category in feature_train
    :param targets: Targets consists of 0s and 1s
    :param feature_test: Using to map P{ target = 1 | category } based on feature_train
    :param n_folds: If None then compute probabilities using whole feature_train
                    If int then use cross-validation-like technique to compute probabilities
    :param default_prob: Default probability to fill NaNs
    :param random_state: Seed for random generator
    :return: Encoded feature_test
    """
    assert (n_folds is None) or (n_folds > 1)

    if n_folds:
        folding_maker = KFold(
            len(targets), n_folds=n_folds, random_state=random_state
        )

        encoded_test = pd.Series(index=feature_train.index)

        for train_index, test_index in folding_maker:
            target_means_by_category = targets.iloc[train_index].groupby(
                feature_train.iloc[train_index]
            ).mean()

            encoded_test.iloc[test_index] = feature_train.iloc[test_index].map(
                target_means_by_category
            )

        encoded_test.fillna(default_prob, inplace=True)
    else:
        target_means_by_category = targets.groupby(feature_train).mean()

        encoded_test = feature_test.map(
            target_means_by_category
        ).fillna(default_prob)

    return encoded_test


def make_submission(predictions, path='submission.csv', index=None, name=None):
    pd.Series(
        predictions,
        index=index,
        name=name
    ).to_csv(
        path,
        index=True,
        header=True
    )


def extract_year_month_day(
    df, date_column_name,
    year_column_name='year',
    month_column_name='month',
    day_column_name='day'
):

    df[year_column_name] = df[date_column_name].dt.year
    df[month_column_name] = df[date_column_name].dt.month
    df[day_column_name] = df[date_column_name].dt.day


def test_ssl_algorithm(data, labels, algorithm, n_labeled_objects=10, scoring_function='all', random_state=None):

    n_objects = data.shape[0]

    labeled_inds, unlabeled_inds = train_test_split(np.arange(0, n_objects),
                                                    train_size=n_labeled_objects,
                                                    random_state=random_state)

    new_labels = - np.ones_like(labels)
    new_labels[labeled_inds] = labels[labeled_inds]

    algorithm.fit(data, new_labels)

    distributions = algorithm.label_distributions_[unlabeled_inds, :]
    predictions = algorithm.transduction_[unlabeled_inds]

    switch = {'accuracy': lambda: accuracy_score(labels[unlabeled_inds], predictions),
              'precision': lambda: precision_score(labels[unlabeled_inds], predictions),
              'recall': lambda: recall_score(labels[unlabeled_inds], predictions),
              'f1_score': lambda: f1_score(labels[unlabeled_inds], predictions),
              'log_loss': lambda: log_loss(labels[unlabeled_inds], distributions)}

    scores = {}

    if not isinstance(scoring_function, list):
        assert isinstance(scoring_function, str)

        if scoring_function == 'all':
            scoring_function = switch.keys()
        else:
            scoring_function = [scoring_function]

    for func_name in scoring_function:
        scores[func_name] = switch[func_name]()

    return scores
