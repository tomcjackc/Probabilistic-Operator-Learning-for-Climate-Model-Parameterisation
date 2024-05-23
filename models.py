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
import tensorflow_probability.substrates.jax.bijectors as tfb
from scipy.optimize import minimize

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
            self.multiinput = GP_params['multiinput']
        else:
            self.kernel = gpx.kernels.RBF()
            self.mean_function = gpx.mean_functions.Zero()
            self.multiinput = False
        
        
        # constrain lengthscales (exact constraint tbd)
        self.kernel = self.kernel.replace_bijector(lengthscale=tfb.SoftClip(low=jnp.array(1e-3, dtype=jnp.float64)))#, high=jnp.array(3e1, dtype=jnp.float64)))
        self.kernel = self.kernel.replace_bijector(variance=tfb.SoftClip(low=jnp.array(1e-3, dtype=jnp.float64), high=jnp.array(2e1, dtype=jnp.float64)))

        self.prior = gpx.gps.Prior(mean_function=self.mean_function, kernel=self.kernel)
        return None
    
    def fit(self, X_train, Y_train):
        self.n_samples_train = X_train.shape[0]
        self.n_features = X_train.shape[1]
        Y_train = Y_train.reshape(-1, 1)
        self.n_targets = Y_train.shape[1]

        # check statistics
        # print(f'x_train: mean = {np.mean(X_train, axis = 0)}, std = {np.std(X_train, axis = 0)}')
        # print(f'y_train: mean = {np.mean(Y_train)}, std = {np.std(Y_train)}')
        # for i, PC in enumerate(X_train.T):
        #     print(PC.shape)
        #     plt.figure()
        #     plt.hist(PC)
        #     plt.show()

        self.D = gpx.Dataset(X=jnp.array(X_train, dtype=jnp.float64), y=jnp.array(Y_train, dtype=jnp.float64))
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n, obs_stddev=jnp.array([1.0], dtype=jnp.float64)) # here i choose the value of obs_stddev
        likelihood = likelihood.replace_bijector(obs_stddev=tfb.SoftClip(low=jnp.array(1e-3, dtype=jnp.float64)))
        
        posterior = self.prior * likelihood

        if self.tune_hypers:
            negative_mll = gpx.objectives.ConjugateMLL(negative=True)
            negative_mll = jit(negative_mll)
            # hyperparam tuning
            print(likelihood.obs_stddev)
            print(likelihood.obs_stddev.dtype)
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
    
    def sample_posterior(self, X_test, n_samples):
        posterior_dist = self.opt_posterior.predict(X_test, self.D)
        mean = posterior_dist.mean()
        cov = posterior_dist.covariance()
        samples = np.random.multivariate_normal(mean, cov, n_samples)
        return samples.T
    
    def predict(self, X_test, return_bounds: bool | float | int = False):
        # if self.n_features != X_test.shape[1]:
        #     raise ValueError("Input dimension mismatch")
        latent_dist = self.opt_posterior.predict(X_test, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        predictive_mean = predictive_dist.mean()
        predictive_std = predictive_dist.stddev()
        # print(predictive_std)
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


class full_model():
    def __init__(self, n, m, ARD, multiinput, standardise, combine_pca):
        # if combining pca, the dimension of the latent space = n
        self.n = n
        self.m = m
        self.ARD = ARD
        self.multiinput = multiinput
        self.standardise = standardise
        self.combine_pca = combine_pca
        if self.combine_pca != False:
            self.m = self.n

    def fit(self, x_train, y_train, n_samples = None):

        if self.ARD:
            if self.multiinput:
                ls = jnp.full(self.n, 2, dtype=jnp.float64)
                var = jnp.full(self.n, 2, dtype=jnp.float64)
            else:
                ls = jnp.full((1, self.n), 2, dtype=jnp.float64)
                var = jnp.full((1, self.n), 2, dtype=jnp.float64)
        else:
            ls = jnp.full((1), 2, dtype=jnp.float64)
            var = jnp.full((1), 2, dtype=jnp.float64)

        GP_params = {"kernel": gpx.kernels.RBF(lengthscale = ls, variance = jnp.full((1), 1, dtype=jnp.float64)), 'mean_function': gpx.mean_functions.Zero(), 'multiinput': self.multiinput}

        self.x_train = x_train
        self.y_train = y_train

        if self.m is None:
            self.m = self.y_train.shape[-1]

        if self.combine_pca == 'features':
            self.combined_pca = PCA(n_components = self.n)
            self.combined_train = np.concatenate((self.x_train, self.y_train), axis = 1)
            self.combined_train_pca = self.combined_pca.fit_transform(self.combined_train)
            print('does combined pca')
            
            self.x_pca = PCA(n_components = self.n)
            self.y_pca = PCA(n_components = self.n)
            self.x_pca.components_ = self.combined_pca.components_[:, :x_train.shape[1]]
            self.x_pca.mean_ = self.combined_pca.mean_[:x_train.shape[1]]
            self.y_pca.components_ = self.combined_pca.components_[:, x_train.shape[1]:]
            self.y_pca.mean_ = self.combined_pca.mean_[x_train.shape[1]:]

            self.x_train_pca = self.x_pca.transform(self.x_train)
            print('does x pca')
            self.y_train_pca = self.y_pca.transform(self.y_train)
            print('does y pca')
            
        elif self.combine_pca == 'data':
            self.combined_pca = PCA(n_components = self.n)
            combined_train = np.concatenate((x_train, y_train), axis = 0)
            self.combined_pca.fit(combined_train)
            print('does combined pca')

            self.x_pca = self.combined_pca
            self.y_pca = self.combined_pca

            self.x_train_pca = self.x_pca.transform(self.x_train)
            print('does x pca')
            self.y_train_pca = self.y_pca.transform(self.y_train)
            print('does y pca')

        else:
            self.x_pca = PCA(n_components = self.n)
            self.y_pca = PCA(n_components = self.m)
            self.x_train_pca = self.x_pca.fit_transform(self.x_train)
            print('does x pca')
            self.y_train_pca = self.y_pca.fit_transform(self.y_train)
            print('does y pca')

        self.model_list = []

        if n_samples is not None:
            train_samples_pca = np.zeros((x_train.shape[0], self.m, n_samples))
            train_samples = np.zeros((self.y_train.shape[0], self.y_train.shape[1], n_samples))

        if self.standardise:
            self.x_train_pca_stand = (self.x_train_pca - self.x_train_pca.mean(axis = 0))/self.x_train_pca.std(axis = 0)
            self.y_train_pca_stand = (self.y_train_pca - self.y_train_pca.mean(axis = 0))/self.y_train_pca.std(axis = 0)

            if self.multiinput:
                for i in tqdm(range(self.y_train_pca_stand.shape[-1])):
                    local_gp = GP_regressor(GP_params=GP_params)
                    local_gp.fit(self.x_train_pca_stand, self.y_train_pca_stand[:, i])
                    print(local_gp.kernel.lengthscale)
                    print(local_gp.kernel.variance)
                    self.model_list.append(local_gp)
                    if n_samples is not None:
                        train_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_train_pca_stand, n_samples = n_samples)
            
            else:
                for i in tqdm(range(self.y_train_pca_stand.shape[-1])):
                    local_gp = GP_regressor(GP_params=GP_params)
                    local_gp.fit(self.x_train_pca_stand[:, i].reshape(-1, 1), self.y_train_pca_stand[:, i])
                    self.model_list.append(local_gp)
                    if n_samples is not None:
                        train_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_train_pca_stand[:, i].reshape(-1, 1), n_samples = n_samples)

        else:
            if self.multiinput:
                for i in tqdm(range(self.y_train_pca.shape[-1])):
                    local_gp = GP_regressor(GP_params=GP_params)
                    local_gp.fit(self.x_train_pca, self.y_train_pca[:, i])
                    self.model_list.append(local_gp)
                    if n_samples is not None:
                        train_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_train_pca, n_samples = n_samples)
            
            else:
                for i in tqdm(range(self.y_train_pca.shape[-1])):
                    local_gp = GP_regressor(GP_params=GP_params)
                    local_gp.fit(self.x_train_pca[:, i].reshape(-1, 1), self.y_train_pca[:, i])
                    self.model_list.append(local_gp)
                    if n_samples is not None:
                        train_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_train_pca[:, i].reshape(-1, 1), n_samples = n_samples)

        if n_samples is not None:
            for i in range(n_samples):
                train_samples_pca_i = train_samples_pca[:, :, i]
                train_samples_i = self.y_pca.inverse_transform(train_samples_pca_i)
                train_samples[:, :, i] = train_samples_i
            return train_samples
    
    def predict(self, x_test, n_samples = None):

        self.x_test = x_test

        self.x_test_pca = self.x_pca.transform(x_test)

        if n_samples is not None:
            test_samples_pca = np.zeros((x_test.shape[0], self.m, n_samples))
            test_samples = np.zeros((self.x_test.shape[0], self.y_train.shape[1], n_samples))
        
        if self.standardise:
            y_pred_pca_stand = []
            self.x_test_pca_stand = (self.x_test_pca - self.x_train_pca.mean(axis = 0)) / self.x_train_pca.std(axis = 0) # check this
            if self.multiinput:
                for i in tqdm(range(self.m)):
                    local_gp = self.model_list[i]
                    y_pred_pca_stand.append(local_gp.predict(self.x_test_pca_stand, return_bounds = True))
                    if n_samples is not None:
                        test_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_test_pca_stand, n_samples = n_samples)
                        test_samples_pca[:, i, :] = (test_samples_pca[:, i, :] * self.y_train_pca.std(axis = 0)[i]) + self.y_train_pca.mean(axis = 0)[i]
            
            else:
                for i in tqdm(range(self.m)):
                    local_gp = self.model_list[i]
                    y_pred_pca_stand.append(local_gp.predict(self.x_test_pca_stand[:, i].reshape(-1, 1), return_bounds = True))
                    if n_samples is not None:
                        test_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_test_pca_stand[:, i].reshape(-1, 1), n_samples = n_samples)
                        test_samples_pca[:, i, :] = (test_samples_pca[:, i, :] * self.y_train_pca.std(axis = 0)[i]) + self.y_train_pca.mean(axis = 0)[i]
            
            y_pred_pca_stand = np.stack(y_pred_pca_stand).T
            self.y_pred_pca = (y_pred_pca_stand * self.y_train_pca.std(axis = 0)) + self.y_train_pca.mean(axis = 0)
        
        else:
            self.y_pred_pca = []
            if self.multiinput:
                for i in tqdm(range(self.m)):
                    local_gp = self.model_list[i]
                    self.y_pred_pca.append(local_gp.predict(self.x_test_pca, return_bounds = True))
                    if n_samples is not None:
                        test_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_test_pca, n_samples = n_samples)
            
            else:
                for i in tqdm(range(self.m)):
                    local_gp = self.model_list[i]
                    self.y_pred_pca.append(local_gp.predict(self.x_test_pca[:, i].reshape(-1, 1), return_bounds = True))
                    if n_samples is not None:
                        test_samples_pca[:, i, :] = local_gp.sample_posterior(self.x_test_pca[:, i].reshape(-1, 1), n_samples = n_samples)
            
            self.y_pred_pca = np.stack(self.y_pred_pca).T
        
        y_pred = self.y_pca.inverse_transform(self.y_pred_pca)

        if n_samples is not None:
            for i in range(n_samples):
                test_samples_pca_i = test_samples_pca[:, :, i]
                test_samples_i = self.y_pca.inverse_transform(test_samples_pca_i)
                test_samples[:, :, i] = test_samples_i
            return y_pred[:, 0, :], test_samples
        else:
            return y_pred[:, 0, :]
        
def my_joint_PCA(X_1, X_2, rho = 0, n_components = 1):
    result_history = []
    term_history = []
    X_1 = np.array(X_1)
    X_1 = X_1 - np.mean(X_1, axis=0)
    X_2 = np.array(X_2)
    X_2 = X_2 - np.mean(X_2, axis=0)
    
    p_1 = X_1.shape[1] # dimensionality of data_1
    p_2 = X_2.shape[1] # dimensionality of data_2

    w_full = np.zeros((n_components, p_1 + p_2)) # has shape (n_components, p_1 + p_2)

    def joint_objective(w, X_1, X_2, rho):
        w_1 = w[:p_1]
        w_2 = w[p_1:]

        w_1 = w_1 / np.linalg.norm(w_1)
        w_2 = w_2 / np.linalg.norm(w_2)

        result_history[i].append(np.concatenate((w_1, w_2)))
        
        term1 = np.var(np.dot(X_1, w_1))
        term2 = np.var(np.dot(X_2, w_2))
        term3 = 2 * rho * np.cov(np.dot(X_1, w_1), np.dot(X_2, w_2))[0, 1]

        # print(f'term1: {term1}, term2: {term2}, term3: {term3}')

        term_history[i].append([term1, term2, term3])

        # print(f'component {i+1}: iteration {counter}: {-(term1 + term2 + term3)}')

        return -(term1 + term2 + term3)

    for i in tqdm(range(n_components)):
        result_history.append([])
        term_history.append([])

        # Initial guess - random unit vector
        np.random.seed(137)  # For reproducibility
        w0_1 = np.random.randn(p_1)
        w0_1 /= np.linalg.norm(w0_1)

        w0_2 = np.random.randn(p_2)
        w0_2 /= np.linalg.norm(w0_2)

        w0 = np.concatenate((w0_1, w0_2))
        print(f'initial guess: {w0}')

        res = minimize(joint_objective, w0, args=(X_1, X_2, rho))
        w = res.x
        w_1 = w[:p_1]
        w_2 = w[p_1:]
        w_1 = w_1 / np.linalg.norm(w_1)
        w_2 = w_2 / np.linalg.norm(w_2)
        w = np.concatenate((w_1, w_2))
        w_full[i, :] = w

        X_1 = X_1 - np.outer(np.dot(X_1, w_1), w_1)
        X_2 = X_2 - np.outer(np.dot(X_2, w_2), w_2)

    # result_history = np.array(result_history)
    # term_history = np.array(term_history)
    return w_full, result_history, term_history