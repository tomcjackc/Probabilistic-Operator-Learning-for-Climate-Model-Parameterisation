import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class first_model:
    def __init__(self, low_dim_x, low_dim_y, low_dim_regressor = LinearRegression()):
        self.low_dim_x = low_dim_x
        self.low_dim_y = low_dim_y
        self.PCA_model_x = PCA(n_components=low_dim_x)
        self.PCA_model_y = PCA(n_components=low_dim_y)

        self.low_dim_regressor = low_dim_regressor

    def fit(self, X_train, Y_train, save = False):
        self.n_samples_train = X_train.shape[0]
        self.n_features = X_train.shape[1]
        self.n_targets = Y_train.shape[1]

        self.X_train_low_dim = self.PCA_model_x.fit_transform(X_train)
        self.Y_train_low_dim = self.PCA_model_y.fit_transform(Y_train)

        self.low_dim_regressor = self.low_dim_regressor.fit(self.X_train_low_dim, self.Y_train_low_dim)

        if save:
            # record training results
            self.Y_train_low_dim_pred = self.low_dim_regressor.predict(self.X_train_low_dim)
            self.Y_train_pred = self.PCA_model_y.inverse_transform(self.Y_train_low_dim_pred)
            self.train_rmse = np.sqrt(np.mean((Y_train - self.Y_train_pred) ** 2, axis=1))
            # the following probably only works for sklearn models
            self.R2 = self.low_dim_regressor.score(self.X_train_low_dim, self.Y_train_low_dim)
            #
    
    def predict(self, X):
        if self.n_features != X.shape[1]:
            raise ValueError("Input dimension mismatch")
        X_low_dim = self.PCA_model_x.transform(X)
        Y_low_dim = self.low_dim_regressor.predict(X_low_dim)
        return self.PCA_model_y.inverse_transform(Y_low_dim)
    
    def test(self, X_test, Y_test):
        Y_test_pred = self.predict(X_test)
        return np.sqrt(mean_squared_error(Y_test.T, Y_test_pred.T, multioutput='raw_values'))