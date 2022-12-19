from collections import Counter
import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

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


@dataclass
class ClassifierStats:
    fscore: float
    acc: float
    counter: Counter

    def __str__(self):
        fscore = self.fscore
        accuracy = self.acc
        counter = self.counter
        return f'({fscore=:.3}, {accuracy=:.3}, {counter=})'


def merge_counters(d1: Counter, d2: Counter) -> Counter:
    d = d1.copy()
    for k, v in d2.items():
        d[k] = d.get(k, 0) + v

    return d


def test_classifier(X, y, cons_classifier, preprocess):
    kf = KFold(n_splits=3)
    fscores = []
    accuracies = []
    counter = Counter()

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for fn in preprocess:
            res = fn(X_train, y_train, X_test, y_test)
            X_train, y_train, X_test, y_test = res

        classifier = cons_classifier()
        classifier.fit(X_train, y_train)

        pred = classifier.predict(X_test)
        counter = merge_counters(counter, Counter(pred))

        fscores.append(f1_score(y_test, pred))
        accuracies.append(accuracy_score(y_test, pred))

    fscore = avg(fscores)
    accuracy = avg(accuracies)
    return ClassifierStats(fscore, accuracy, counter)


def print_class_name(fn):

    def test_fn(cl, configs, X, y, preprocess):
        padding = '---------------------'
        print(f'{padding} {cl.__name__} {padding}')
        retval = fn(cl, configs, X, y, preprocess)
        print(f'{padding}{"-" * (len(cl.__name__) + 2)}{padding}')
        print()
        return retval

    return test_fn


def print_row(row):
    row = map(str, row)
    print('| ', ' | '.join(row), ' |', sep='')


def as_table(configs, results):
    keys = set()

    for conf in configs:
        keys.update(conf.keys())

    keys = list(keys)
    header = keys + ['f1 score', 'accuracy', 'counter']
    sep = ['---'] * len(header)

    print_row(header)
    print_row(sep)

    for conf, res in zip(configs, results):
        values = []
        for key in keys:
            values.append(conf.get(key, '-'))
        values += [res.fscore, res.acc, res.counter]
        print_row(values)


@print_class_name
def test_configs(cl, configs, X, y, preprocess):
    results = []
    for conf in configs:
        res = test_classifier(X, y, lambda: cl(**conf), preprocess)
        results.append(res)
        # print(f'{conf}: {res}')
    as_table(configs, results)


def test_decision_tree(X, y, preprocess, max_depth):
    end = max_depth + 5 - (max_depth % 5)
    configs = [{'max_depth': min(x, max_depth)} for x in range(5, end, 5)]
    test_configs(DecisionTreeClassifier, configs, X, y, preprocess)


def test_forest_depths(X, y, preprocess, depths):
    configs = []
    for depth in depths:
        for est in [3, 5, 10, 50, 100]:
            conf = {'max_depth': depth, 'n_estimators': est}
            configs.append(conf)

    test_configs(RandomForestClassifier, configs, X, y, preprocess)


def test_random_forest(X, y, preprocess):
    test_forest_depths(X, y, preprocess, [45, 50, 55])


def test_random_forest_pca(X, y, preprocess):
    test_forest_depths(X, y, preprocess, [10, 15, 20])


def test_svm(X, y, preprocess):
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
    test_configs(SVC, configs, X, y, preprocess)


def test_xgb_confs(X,  y, preprocess, depths, tree_count):
    base = {'objective': 'binary:hinge', 'booster': 'gbtree'}
    configs = []
    for depth in depths:
        for trees in tree_count:
            confs = [
                {'max_depth': depth, 'num_parallel_tree': trees, 'eta': 0.6},

                {'max_depth': depth, 'num_parallel_tree': trees, 'eta': 0.6, 'gamma': 0.5},
                {'max_depth': depth, 'num_parallel_tree': trees, 'eta': 0.6, 'gamma': 1.0},
                {'max_depth': depth, 'num_parallel_tree': trees, 'eta': 0.6, 'gamma': 1.5},

                {'max_depth': depth, 'num_parallel_tree': trees, 'eta': 0.6, 'lambda': 1.5, 'alpha': 0.5},
                ]

            configs += confs

    for conf in configs:
        conf.update(base)

    test_configs(XGBClassifier, configs, X, y, preprocess)


def test_xgb(X, y, preprocess):
    test_xgb_confs(X, y, preprocess, [55], [1])


def test_xgb_pca(X, y, preprocess):
    test_xgb_confs(X, y, preprocess, [10, 15, 20], [1, 3, 10])


def test_mlp(X, y, preprocess, n):
    layers = 'hidden_layer_sizes'
    solver = 'solver'
    activ = 'activation'

    configs = [
            {layers: (n,), solver: 'adam', activ: 'relu'},
            {layers: (n,), solver: 'adam', activ: 'logistic'},
            {layers: (n,), solver: 'lbfgs', activ: 'relu'},
            {layers: (n,), solver: 'lbfgs', activ: 'logistic'},

            {layers: (2*n,), solver: 'adam', activ: 'relu'},
            {layers: (2*n,), solver: 'adam', activ: 'logistic'},
            {layers: (2*n,), solver: 'lbfgs', activ: 'relu'},
            {layers: (2*n,), solver: 'lbfgs', activ: 'logistic'},

            {layers: (2*n, 2*n), solver: 'adam', activ: 'relu'},
            {layers: (2*n, 2*n), solver: 'adam', activ: 'logistic'},
            {layers: (2*n, 2*n), solver: 'lbfgs', activ: 'relu'},
            {layers: (2*n, 2*n), solver: 'lbfgs', activ: 'logistic'},

            {layers: (3*n,), solver: 'adam', activ: 'relu'},
            {layers: (3*n,), solver: 'adam', activ: 'logistic'},
            {layers: (3*n,), solver: 'lbfgs', activ: 'relu'},
            {layers: (3*n,), solver: 'lbfgs', activ: 'logistic'},

            {layers: (4*n,), solver: 'adam', activ: 'relu'},
            {layers: (4*n,), solver: 'adam', activ: 'logistic'},
            {layers: (4*n,), solver: 'lbfgs', activ: 'relu'},
            {layers: (4*n,), solver: 'lbfgs', activ: 'logistic'},
            ]

    test_configs(MLPClassifier, configs, X, y, preprocess)


def test_all(X, y, preprocess=None):
    if preprocess is None:
        preprocess = []

    test_decision_tree(X, y, preprocess, max_depth=73)
    test_random_forest(X, y, preprocess)
    test_xgb(X, y, preprocess)
    test_svm(X, y, preprocess)
    test_mlp(X, y, preprocess, n=73)


def test_all_pca(X, y, preprocess=None):
    if preprocess is None:
        preprocess = []

    test_decision_tree(X, y, preprocess, max_depth=20)
    test_random_forest_pca(X, y, preprocess)
    test_xgb_pca(X, y, preprocess)
    test_svm(X, y, preprocess)
    test_mlp(X, y, preprocess, n=20)


def enhance_ds(train_x, train_y, test_x, test_y):
    smote = SMOTE(random_state=747, k_neighbors=4)
    train_x, train_y = smote.fit_resample(train_x, train_y)
    return train_x, train_y, test_x, test_y


def reduce_dim(train_x, train_y, test_x, test_y):
    pca = PCA(20)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    return train_x, train_y, test_x, test_y


def scale_ds(train_x, train_y, test_x, test_y):
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, train_y, test_x, test_y


def run():
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    X, y = load_xy('data/train.csv')

    print('------------------------------------ Scaled Data -------------------------------------')
    print()
    test_all(X, y, [scale_ds])
    print('---------------------------------- Data Enhancement ----------------------------------')
    print()
    test_all(X, y, [enhance_ds, scale_ds])
    print('---------------------------------------- PCA -----------------------------------------')
    print()
    test_all_pca(X, y, [enhance_ds, scale_ds, reduce_dim])
