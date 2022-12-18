from collections import Counter

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from preprocessing import preprocess


def avg(iterable):
    return sum(iterable) / len(iterable)


def load_from_file(path):
    return pd.read_csv(path, sep=',', header=0)


def load_ds(path):
    ds = load_from_file(path)
    return preprocess(ds)


def load_xy(path):
    return xy_split(load_ds(path))


def xy_split(ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = ds.loc[:, ds.columns != 'is_claim']
    y = ds.loc[:, 'is_claim']

    return X, y


def train_scaler(train_X):
    scaler = StandardScaler()
    scaler.fit(train_X)
    return scaler


def test_classifier(X, y, cons_classifier):
    kf = KFold(n_splits=5)
    scores = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = train_scaler(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        classifier = cons_classifier()
        classifier.fit(X_train, y_train)

        pred = classifier.predict(X_test)
        # counter = Counter(pred)
        # print(pred)
        # print(counter)

        scores.append(f1_score(y_test, pred))

    return avg(scores)


def print_class_name(fn):

    def test_fn(cl, configs, X, y):
        padding = '---------------------'
        print(f'{padding} {cl.__name__} {padding}')
        retval = fn(cl, configs, X, y)
        print(f'{padding}{"-" * (len(cl.__name__) + 2)}{padding}')
        print()
        return retval

    return test_fn


@print_class_name
def test_configs(cl, configs, X, y):
    for conf in configs:
        score = test_classifier(X, y, lambda: cl(**conf))
        print(f'{cl.__name__} {conf}: {score:.3f}')
        # print(f'{cl.__name__} {conf}: {score}')


def test_decision_tree(X, y):
    configs = [{'max_depth': min(x, 73)} for x in range(5, 80, 5)]
    test_configs(DecisionTreeClassifier, configs, X, y)


def test_random_forest(X, y):
    configs = [
            {'max_depth': 45, 'n_estimators': 3},
            {'max_depth': 50, 'n_estimators': 3},
            {'max_depth': 55, 'n_estimators': 3},

            {'max_depth': 45, 'n_estimators': 5},
            {'max_depth': 50, 'n_estimators': 5},
            {'max_depth': 55, 'n_estimators': 5},

            {'max_depth': 45, 'n_estimators': 10},
            {'max_depth': 50, 'n_estimators': 10},
            {'max_depth': 55, 'n_estimators': 10},

            {'max_depth': 45, 'n_estimators': 50},
            {'max_depth': 50, 'n_estimators': 50},
            {'max_depth': 55, 'n_estimators': 50},

            {'max_depth': 45, 'n_estimators': 100},
            {'max_depth': 50, 'n_estimators': 100},
            {'max_depth': 55, 'n_estimators': 100},
            ]
    test_configs(RandomForestClassifier, configs, X, y)


def test_svm(X, y):
    configs = [
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            # {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
            # {'C': 100.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'},
            # {'C': 10.0, 'kernel': 'rbf', 'gamma': 'auto'},
            # {'C': 100.0, 'kernel': 'rbf', 'gamma': 'auto'},

            {'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'},
            # {'C': 10.0, 'kernel': 'linear', 'gamma': 'scale'},
            # {'C': 100.0, 'kernel': 'linear', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'linear', 'gamma': 'auto'},
            # {'C': 10.0, 'kernel': 'linear', 'gamma': 'auto'},
            # {'C': 100.0, 'kernel': 'linear', 'gamma': 'auto'},

            {'C': 1.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
            # {'C': 10.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
            # {'C': 100.0, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},

            {'C': 1.0, 'kernel': 'poly', 'degree': 4, 'gamma': 'scale'},
            # {'C': 10.0, 'kernel': 'poly', 'degree': 4, 'gamma': 'scale'},
            # {'C': 100.0, 'kernel': 'poly', 'degree': 4, 'gamma': 'scale'},

            {'C': 1.0, 'kernel': 'sigmoid', 'gamma': 'scale'},
            # {'C': 10.0, 'kernel': 'sigmoid', 'gamma': 'scale'},
            # {'C': 100.0, 'kernel': 'sigmoid', 'gamma': 'scale'},

            {'C': 1.0, 'kernel': 'sigmoid', 'gamma': 'auto'},
            # {'C': 10.0, 'kernel': 'sigmoid', 'gamma': 'auto'},
            # {'C': 100.0, 'kernel': 'sigmoid', 'gamma': 'auto'},
            ]
    test_configs(SVC, configs, X, y)


def test_xgb(X,  y):
    base = {'objective': 'binary:hinge', 'booster': 'gbtree'}
    configs = [
            # {'max_depth': 10, 'num_parallel_tree': 1},
            # {'max_depth': 45, 'num_parallel_tree': 1},
            # {'max_depth': 50, 'num_parallel_tree': 1},
            # {'max_depth': 55, 'num_parallel_tree': 1},
            # {'max_depth': 73, 'num_parallel_tree': 1},

            # {'max_depth': 10, 'num_parallel_tree': 3},
            # {'max_depth': 45, 'num_parallel_tree': 3},
            # {'max_depth': 50, 'num_parallel_tree': 3},
            # {'max_depth': 55, 'num_parallel_tree': 3},
            # {'max_depth': 73, 'num_parallel_tree': 3},

            # {'max_depth': 10, 'num_parallel_tree': 10},
            # {'max_depth': 45, 'num_parallel_tree': 10},
            # {'max_depth': 50, 'num_parallel_tree': 10},
            # {'max_depth': 55, 'num_parallel_tree': 10},
            # {'max_depth': 73, 'num_parallel_tree': 10},

            # {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.1},
            # {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 1.2},

            {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.6},

            {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.6, 'gamma': 0.5},
            {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.6, 'gamma': 1.0},
            {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.6, 'gamma': 1.5},

            {'max_depth': 55, 'num_parallel_tree': 1, 'eta': 0.6, 'lambda': 1.5, 'alpha': 0.5},
            ]

    for conf in configs:
        conf.update(base)

    test_configs(XGBClassifier, configs, X, y)


def test_all(X, y):
    test_decision_tree(X, y)
    test_random_forest(X, y)
    test_xgb(X, y)
    test_svm(X, y)


def run():
    X, y = load_xy('data/train.csv')

    test_all(X, y)
