class EveryOtherClassifier:
    def __init__(self):
        self.last = 0

    def fit(self, X, y):
        pass

    def get_pred(self):
        pred = 1 - self.last
        self.last = pred
        return pred

    def predict(self, X):
        return [self.get_pred() for _ in X.iterrows()]
