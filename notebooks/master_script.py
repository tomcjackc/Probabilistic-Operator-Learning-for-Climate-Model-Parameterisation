#%%
import sys

module_dir = '../'

if module_dir not in sys.path:
    sys.path.append(module_dir)

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
from models import *
# import test_systems_1d as systems_1d
import itertools
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import NoNorm
from matplotlib.colors import Normalize
from tqdm import tqdm
import gpjax as gpx
from sklearn.decomposition import PCA
import scipy.io
from scipy.interpolate import griddata
from dataloaders import *
import pandas as pd

from jax import config
config.update("jax_enable_x64", True)


problem = 'darcy'
n_train = 1000
n_test = 1000
r = 15

if problem == 'darcy':
    dataloader_params = [n_train, n_test, r]
    s = int(((421 - 1) / r) + 1)
    extent = [0, 1, 0, 1]
    x_cmap = 'coolwarm'
    y_cmap = 'viridis'
elif problem == 'ns':
    dataloader_params = [n_train, n_test]
    s = 64
    extent = [0, 2*np.pi, 0, 2*np.pi]
    x_cmap = 'viridis'
    y_cmap = 'viridis'

dataloader_dict = {'darcy':get_darcy_data, 'ns':get_navier_stokes_data}
x_train, y_train, x_test, y_test, x_grid, y_grid = dataloader_dict[problem](*dataloader_params)

#%%
min_n = 10
min_m = 10

max_n = 30
max_m = 30

jumps_n, jumps_m = 3, 3

n_list = np.linspace(min_n, max_n, jumps_n, dtype=int)
m_list = np.linspace(min_m, max_m, jumps_m, dtype=int)

ARD_list = [False]
multiinput_list = [True]
standardise_list = [True]
combine_pca_list = [True, False]

row_list = []

for n, m, ARD, multiinput, standardise, combine_pca in tqdm(itertools.product(n_list, m_list, ARD_list, multiinput_list, standardise_list, combine_pca_list), desc = 'outer loop'):
    if n == m:
        print(n, m, ARD, multiinput, standardise, combine_pca)
        model = full_model(n = n, m = m, ARD = ARD, multiinput = multiinput, standardise = standardise, combine_pca = combine_pca)
        model.fit(x_train, y_train)

        y_pred_train = model.predict(x_train)
        relative_L2 = np.linalg.norm(y_pred_train - y_train, axis = -1)/np.linalg.norm(y_train, axis = -1)
        train_relative_L2 = np.mean(relative_L2)
        
        y_pred_test = model.predict(x_test)
        relative_L2 = np.linalg.norm(y_pred_test - y_test, axis = -1)/np.linalg.norm(y_test, axis = -1)
        test_relative_L2 = np.mean(relative_L2)

        row_list.append({'n':n, 'm':m, 'ARD':ARD, 'multiinput':multiinput, 'standardise':standardise, 'combine_pca':combine_pca, 'train_relative_L2':train_relative_L2, 'test_relative_L2':test_relative_L2})

output_df = pd.DataFrame(row_list, columns = ['n', 'm', 'ARD', 'multiinput', 'standardise', 'combine_pca', 'train_relative_L2', 'test_relative_L2'])
output_df.to_csv(f'../output_data/{problem}_ntrain={n_train}_ntest={n_test}_s={s}_{min_n}:{max_n}:{jumps_n}_{min_m}:{max_m}:{jumps_m}_{ARD_list}_{multiinput_list}_{standardise_list}_{combine_pca_list}.csv')

#%%