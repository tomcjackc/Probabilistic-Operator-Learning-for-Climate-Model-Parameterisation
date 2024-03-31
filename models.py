import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import gpjax as gpx
from jax import jit

class first_model():
    def __init__(self, low_dim_x, low_dim_y = 1, low_dim_regressor = LinearRegression()):
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
        print(self.Y_train_low_dim.shape)

        self.low_dim_regressor.fit(self.X_train_low_dim, self.Y_train_low_dim)

        if save:
            # record training results
            self.Y_train_low_dim_pred = self.low_dim_regressor.predict(self.X_train_low_dim)
            print(self.Y_train_low_dim_pred.shape)
            # self.Y_train_pred = self.PCA_model_y.inverse_transform(self.Y_train_low_dim_pred)
            # self.train_rmse = np.sqrt(np.mean((Y_train - self.Y_train_pred) ** 2, axis=1))
            
    
    def predict(self, X):
        if self.n_features != X.shape[1]:
            raise ValueError("Input dimension mismatch")
        X_low_dim = self.PCA_model_x.transform(X)
        Y_low_dim = self.low_dim_regressor.predict(X_low_dim)
        return self.PCA_model_y.inverse_transform(Y_low_dim)
    
    def test(self, X_test, Y_test):
        Y_test_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test.T, Y_test_pred.T, multioutput='raw_values'))
        R2 = r2_score(Y_test.T, Y_test_pred.T, multioutput='raw_values')
        return rmse, R2
    
class GP_regressor():
    def __init__(self, kernel = gpx.kernels.RBF(), mean_function = gpx.mean_functions.Zero()):
        self.kernel = kernel
        self.mean_function = mean_function
        self.prior = gpx.gps.Prior(mean_function=self.mean_function, kernel=self.kernel)
        return None
    
    def fit(self, X_train, Y_train):
        self.n_samples_train = X_train.shape[0]
        self.n_features = X_train.shape[1]
        self.n_targets = Y_train.shape[1]

        self.D = gpx.Dataset(X=X_train, y=Y_train)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)
        posterior = self.prior * likelihood
        negative_mll = gpx.objectives.ConjugateMLL(negative=True)
        negative_mll = jit(negative_mll)
        self.opt_posterior, self.history = gpx.fit_scipy(model=posterior, objective=negative_mll, train_data=self.D)
        
    def sample_prior(self, X_test, n_samples):
        prior_dist = self.prior.predict(X_test)
        mean = prior_dist.mean()
        cov = prior_dist.covariance()
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        return mean, cov, samples
    
    def predict(self, X_test):
        if self.n_features != X_test.shape[1]:
            raise ValueError("Input dimension mismatch")
        latent_dist = self.opt_posterior.predict(X_test, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        return predictive_mean