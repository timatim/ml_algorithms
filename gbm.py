import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GBMBase(object):
    def __init__(self, num_stages, step_size=0.1, min_samples_split=2, min_samples_leaf=1, max_depth=3):
        self.num_stages = num_stages
        self.trees = []
        self.step_size = step_size
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def _pseudo_residuals(y, a, loss="square"):
        r_m = np.zeros(len(y))
        if loss=="square":
            r_m = a-y
        elif loss=="logistic":
            exp = np.exp(-y * a)
            r_m = exp / np.power((1 + exp), 2)
        return r_m

    def fit(self, X, y):
        for i in range(self.num_stages):
            f_m = self.predict(X)

            r_m = -(GBMBase._pseudo_residuals(y, f_m))
            dt = DecisionTreeRegressor(min_samples_split=self.min_samples_split,
                                       min_samples_leaf=self.min_samples_leaf,
                                       max_depth=self.max_depth).fit(X, r_m)
            self.trees.append(dt)

    def predict(self, X):
        m, n = X.shape

        predictions = np.zeros(m)

        # predict with each weak learner
        if self.trees:
            for weak in self.trees:
                predictions += self.step_size * weak.predict(X)

        return predictions


class GBMRegressor(GBMBase):
    def __init__(self, num_stages, step_size=0.1, min_samples_split=2, min_samples_leaf=1, max_depth=3):
        super(GBMRegressor, self).__init__(num_stages, step_size=0.1, min_samples_split=2, min_samples_leaf=1, max_depth=3)


if __name__ == "__main__":
    # test regressor
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.datasets import load_boston
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    print("Compare regressor")
    X, y = load_boston(return_X_y=True)
    M = 50
    GBM_reg = GBMRegressor(M)
    GBM_reg.fit(X, y)
    predictions = GBM_reg.predict(X)
    GBMReg_mse = mean_squared_error(y, predictions)

    # set sklearn initial estimator to zero to compare
    class InitEstimator(object):
        def __init__(self):
            pass
        def fit(self, X, y, sample_weight):
            pass
        def predict(self, X):
            m, n = X.shape
            return np.zeros((m,1))

    sklearn_reg = GradientBoostingRegressor(n_estimators=M, criterion="mse", init=InitEstimator())
    sklearn_reg.fit(X, y)
    predictions = sklearn_reg.predict(X)
    sklearn_reg_mse = mean_squared_error(y, predictions)

    print(GBMReg_mse, sklearn_reg_mse)
    assert abs(GBMReg_mse - sklearn_reg_mse) < 1e-4
