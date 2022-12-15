import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

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

        scores.append(f1_score(y_test, pred))

    return avg(scores)


def print_class_name(fn):

    def test_fn(cl, configs, X, y):
        print(f'--------------------- {cl.__name__} ---------------------')
        retval = fn(cl, configs, X, y)
        print('---------------------------------------------------------')
        print()
        return retval

    return test_fn


@print_class_name
def test_configs(cl, configs, X, y):
    for conf in configs:
        score = test_classifier(lambda: cl(**conf), X, y)
        print(f'{cl.__name__} {conf}: {score}')


def test_decision_tree(X, y):
    configs = [
            {'max_depth': 4, },
            {'max_depth': 5, },
            {'max_depth': 6, },
            {'max_depth': 7, },
            {'max_depth': 8, },
            ]
    test_configs(DecisionTreeClassifier, configs, X, y)


def test_random_forest(X, y):
    configs = [
            {'max_depth': 4, 'n_estimators': 10},
            {'max_depth': 6, 'n_estimators': 10},
            {'max_depth': 8, 'n_estimators': 10},

            {'max_depth': 4, 'n_estimators': 100},
            {'max_depth': 6, 'n_estimators': 100},
            {'max_depth': 8, 'n_estimators': 100},
            ]
    test_configs(RandomForestClassifier, configs, X, y)


def test_all(X, y):
    test_decision_tree(X, y)
    test_random_forest(X, y)


def run():
    X, y = load_xy('data/train.csv')

    test_all(X, y)
