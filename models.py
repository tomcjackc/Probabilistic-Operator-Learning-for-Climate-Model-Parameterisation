import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import gpjax as gpx
from jax import jit
import jax.numpy as jnp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

class first_model():
    def __init__(self, low_dim_x, low_dim_y = 1, low_dim_regressor = 'linear', GP_params = None, multiinput = False):
        self.low_dim_x = low_dim_x
        self.low_dim_y = low_dim_y
        self.PCA_model_x = PCA(n_components=low_dim_x)
        if low_dim_y is not None:
            self.PCA_model_y = PCA(n_components=low_dim_y)

        self.low_dim_regressor_name = low_dim_regressor
        self.GP_params = GP_params
        self.multiinput = multiinput
        self.has_train_history = False

    def fit(self, X_train, Y_train, save = False, return_bounds: bool | float | int = False):
        self.n_samples_train = X_train.shape[0]
        self.n_features = X_train.shape[1]
        self.n_targets = Y_train.shape[1]
        self.low_dim_y = self.n_features if self.low_dim_y is None else self.low_dim_y

        self.X_train_low_dim = self.PCA_model_x.fit_transform(X_train)
        if self.low_dim_y != self.n_features:
            self.Y_train_low_dim = self.PCA_model_y.fit_transform(Y_train)
        else:
            self.Y_train_low_dim = Y_train
        print(self.X_train_low_dim.shape)
        print(self.Y_train_low_dim.shape)

        print(self.low_dim_regressor_name)

        if self.low_dim_regressor_name == 'linear':
            self.low_dim_regressor_list = [LinearRegression() for i in range(self.low_dim_y)]
        elif self.low_dim_regressor_name == 'GP':
            self.low_dim_regressor_list = [GP_regressor(self.GP_params) for i in range(self.low_dim_y)]
        elif self.low_dim_regressor_name == 'skl_GP':
            self.low_dim_regressor_list = [GaussianProcessRegressor(kernel = RBF(), alpha = 1e-10,  normalize_y = True, random_state= 1172023) for i in range(self.low_dim_y)]
        else:
            raise ValueError("Invalid regressor type. Must be 'linear', 'GP', or 'skl_GP'")
        
        if self.multiinput:
            for i, regressor in enumerate(self.low_dim_regressor_list):
                print(self.X_train_low_dim.shape, self.Y_train_low_dim[:, i].shape)
                regressor.fit(self.X_train_low_dim, self.Y_train_low_dim[:, i])
        else:
            for i, regressor in enumerate(self.low_dim_regressor_list):
                regressor.fit(self.X_train_low_dim[:,i].reshape(-1,1), self.Y_train_low_dim[:, i])


        if save:
            # record training results
            self.Y_train_low_dim_pred = np.zeros((self.n_samples_train, self.low_dim_y))
            self.Y_train_low_dim_pred_upper = np.zeros((self.n_samples_train, self.low_dim_y))
            self.Y_train_low_dim_pred_lower = np.zeros((self.n_samples_train, self.low_dim_y))
            if self.multiinput:
                for i in range(self.low_dim_y):
                    mean_and_bounds = self.low_dim_regressor_list[i].predict(self.X_train_low_dim)
                    self.Y_train_low_dim_pred[:,i], self.Y_train_low_dim_pred_upper[:,i], self.Y_train_low_dim_pred_lower[:,i] = mean_and_bounds[0], mean_and_bounds[1], mean_and_bounds[2]
            else:
                for i in range(self.low_dim_y):
                    mean_and_bounds = self.low_dim_regressor_list[i].predict(self.X_train_low_dim[:,i].reshape(-1,1))
                    self.Y_train_low_dim_pred[:,i], self.Y_train_low_dim_pred_upper[:,i], self.Y_train_low_dim_pred_lower[:,i] = mean_and_bounds[0], mean_and_bounds[1], mean_and_bounds[2]
            print(self.Y_train_low_dim_pred.shape)
            if self.low_dim_y == self.n_features:
                self.Y_train_pred = self.Y_train_low_dim_pred
            else:
                self.Y_train_pred = self.PCA_model_y.inverse_transform(self.Y_train_low_dim_pred)
                
            self.train_rmse = np.sqrt(np.mean((Y_train - self.Y_train_pred) ** 2, axis=1))
            self.has_train_history = True
            
    
    def predict(self, X, save = False, return_bounds: bool | float | int = False):
        if self.n_features != X.shape[1]:
            raise ValueError("Input dimension mismatch")
        X_low_dim = self.PCA_model_x.transform(X)
        Y_low_dim_pred = np.zeros((X.shape[0], self.low_dim_y))
        Y_low_dim_pred_upper = np.zeros((X.shape[0], self.low_dim_y))
        Y_low_dim_pred_lower = np.zeros((X.shape[0], self.low_dim_y))
        if self.multiinput:
            for i in range(self.low_dim_y):
                Y_low_dim_pred[:,i], Y_low_dim_pred_upper[:,i], Y_low_dim_pred_lower[:,i] = self.low_dim_regressor_list[i].predict(X_low_dim, return_bounds)
        else:
            for i in range(self.low_dim_y):
                Y_low_dim_pred[:,i], Y_low_dim_pred_upper[:,i], Y_low_dim_pred_lower[:,i] = self.low_dim_regressor_list[i].predict(X_low_dim[:,i].reshape(-1,1), return_bounds)
        Y_pred = self.PCA_model_y.inverse_transform(Y_low_dim_pred)
        Y_pred_upper = self.PCA_model_y.inverse_transform(Y_low_dim_pred_upper)
        Y_pred_lower = self.PCA_model_y.inverse_transform(Y_low_dim_pred_lower)

        if save:
            self.X_test_low_dim = X_low_dim
            self.Y_test_low_dim_pred = Y_low_dim_pred
            self.Y_test_low_dim_pred_upper = Y_low_dim_pred_upper
            self.Y_test_low_dim_pred_lower = Y_low_dim_pred_lower

        if return_bounds is False:
            return Y_pred
        else:
            return Y_pred, Y_pred_lower, Y_pred_upper
    
    def test(self, X_test, Y_test):
        Y_test_pred = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test.T, Y_test_pred.T, multioutput='raw_values'))
        R2 = r2_score(Y_test.T, Y_test_pred.T, multioutput='raw_values')
        return rmse, R2
    
class GP_regressor():
    def __init__(self, GP_params = None, tune_hypers = True):
        self.tune_hypers = tune_hypers
        if GP_params is not None:
            self.kernel = GP_params['kernel']
            self.mean_function = GP_params['mean_function']
        else:
            self.kernel = gpx.kernels.RBF()
            self.mean_function = gpx.mean_functions.Zero()
        self.prior = gpx.gps.Prior(mean_function=self.mean_function, kernel=self.kernel)
        return None
    
    def fit(self, X_train, Y_train):
        self.n_samples_train = X_train.shape[0]
        self.n_features = X_train.shape[1]
        Y_train = Y_train.reshape(-1, 1)
        self.n_targets = Y_train.shape[1]

        self.D = gpx.Dataset(X=X_train.astype('double'), y=Y_train.astype('double'))
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)#, obs_stddev=jnp.array(1e-3)) # here i choose the value of obs_stddev
        posterior = self.prior * likelihood
        if self.tune_hypers:
            negative_mll = gpx.objectives.ConjugateMLL(negative=True)
            negative_mll = jit(negative_mll)
            # hyperparam tuning
            self.opt_posterior, self.history = gpx.fit_scipy(model=posterior, objective=negative_mll, train_data=self.D, max_iters=1000)
            # print(dir(self.opt_posterior))
            # print(self.history)
            print(self.opt_posterior.prior)
            print(self.opt_posterior.likelihood)
        else:
            self.opt_posterior = posterior
        
    def sample_prior(self, X_test, n_samples):
        prior_dist = self.prior.predict(X_test)
        mean = prior_dist.mean()
        cov = prior_dist.covariance()
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        return mean, cov, samples
    
    def predict(self, X_test, return_bounds: bool | float | int = False):
        # if self.n_features != X_test.shape[1]:
        #     raise ValueError("Input dimension mismatch")
        latent_dist = self.opt_posterior.predict(X_test, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        print(predictive_std)
        if return_bounds is True:
            lower_bound = predictive_mean - 2 * predictive_std
            upper_bound = predictive_mean + 2 * predictive_std
            return predictive_mean, lower_bound, upper_bound
        elif return_bounds is False:
            return predictive_mean, np.zeros_like(predictive_mean), np.zeros_like(predictive_mean)
        else:
            lower_bound = predictive_mean - return_bounds * predictive_std
            upper_bound = predictive_mean + return_bounds * predictive_std
            return predictive_mean, lower_bound, upper_bound
        
class second_model():
    def __init__(self, x_low_dim):
        self.n_pca = x_low_dim
        return None
    
    # def fit(self, x_train, y_train, model):
    #     self.x_pca = PCA(n_components=self.n_pca)
    #     self.x_train_pca = self.x_pca.fit_transform(x_train)
    #     self.models = []
    #     for i in tqdm(range(y_train.shape[-1])):
    #         model.fit(self.x_train_pca, y_train[:, i])
    #         self.models.append(model)
    #     print(self.models)
    #     return None

    # def pred(self, x_test):
    #     x_test_pca = self.x_pca.transform(x_test)
    #     pred = []
    #     for i in tqdm(range(x_test.shape[-1])):
    #         pred.append(self.models[i].predict(x_test_pca))
    #         # print(self.model.kernel_)
    #     #pred_train = gp.predict(x_train)

    #     return np.stack(pred).T
    
    def train_test(self, x_train, x_test, y_train, model, parallel = False, verbose = False):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.x_pca = PCA(n_components=self.n_pca)
        self.x_train_pca = self.x_pca.fit_transform(x_train)
        self.x_test_pca = self.x_pca.transform(x_test)

        if parallel:
            pool = mp.Pool(mp.cpu_count())
            f = partial(fit_predict, model = model, x_train_pca = self.x_train_pca, y_train = y_train, x_test_pca = self.x_test_pca, verbose = verbose)
            pred = pool.map(f, range(y_train.shape[-1]))
        else:
            pred = []
            for i in tqdm(range(y_train.shape[-1])):
                model.fit(self.x_train_pca, y_train[:, i])
                pred.append(model.predict(self.x_test_pca, return_std = True))


        return np.stack(pred).T
    
def fit_predict(i, model, x_train_pca, y_train, x_test_pca, verbose):
    if verbose:
        print(model.kernel_)
    model.fit(x_train_pca, y_train[:, i])
    return model.predict(x_test_pca, return_std = True)