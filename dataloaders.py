import numpy as np
import scipy
import re
import os
import xarray as xr
from sklearn.model_selection import train_test_split

def get_navier_stokes_data(n_train, n_test):
    '''
    Returns the Navier-Stokes dataset.
    The dataset is loaded from the .npy files in the navier_stokes_data folder.
    The dataset is reshaped to have the shape ([n_train or n_test], 64*64).
    '''
    x = np.load('../navier_stokes_data/NavierStokes_inputs.npy').transpose((2,1,0)).reshape(40000, 64*64)
    y = np.load('../navier_stokes_data/NavierStokes_outputs.npy').transpose((2,1,0)).reshape(40000, 64*64)

    x_train = x[int(len(x)/5):, :]
    y_train = y[int(len(y)/5):, :]

    x_test = x[:int(len(x)/5), :]
    y_test = y[:int(len(y)/5), :]

    print(x.shape, y.shape)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    x_train = x_train[:n_train, :]
    y_train = y_train[:n_train, :]

    x_test = x_test[:n_test, :]
    y_test = y_test[:n_test, :]

    x_grid, y_grid = None, None

    return x_train, y_train, x_test, y_test, x_grid, y_grid

def get_darcy_data(n_train, n_test, r):
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29 - basically r selects the course-grainedness, r=1 means no course-graining
    r = r
    s = int(((421 - 1) / r) + 1)

    # for training data
    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    train_data = scipy.io.loadmat("../darcy_flow_data/rect_cont_PWC/piececonst_r421_N1024_smooth1.mat")
    x_train = train_data["coeff"][:n_train, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y_train = train_data["sol"][:n_train, ::r, ::r].astype(np.float32) * 100

    # The dataset has a mistake that the BC is not 0. this is corrected below
    y_train[:, 0, :] = 0
    y_train[:, -1, :] = 0
    y_train[:, :, 0] = 0
    y_train[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_train = x_train.reshape(n_train, s * s)
    y_train = y_train.reshape(n_train, s * s)

    # same again for test data
    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    test_data = scipy.io.loadmat("../darcy_flow_data/rect_cont_PWC/piececonst_r421_N1024_smooth2.mat")
    x_test = test_data["coeff"][:n_test, ::r, ::r].astype(np.float32) * 0.1 - 0.75
    y_test = test_data["sol"][:n_test, ::r, ::r].astype(np.float32) * 100

    # The dataset has a mistake that the BC is not 0. this is corrected below
    y_test[:, 0, :] = 0
    y_test[:, -1, :] = 0
    y_test[:, :, 0] = 0
    y_test[:, :, -1] = 0

    grids = []
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grids.append(np.linspace(0, 1, s, dtype=np.float32))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    x_test = x_test.reshape(n_test, s * s)
    y_test = y_test.reshape(n_test, s * s)

    x_grid, y_grid = grid, grid
    return x_train, y_train, x_test, y_test, x_grid, y_grid

def get_helmholtz_data(n_train, n_test):
    '''
    Assumes total number of datapoints is greater than n_train + n_test.
    '''
    x = np.load('../helmholtz_data/Helmholtz_inputs.npy')
    print(x.shape)
    x = x.transpose((2,1,0)).reshape(x.shape[2], 101*101)
    y = np.load('../helmholtz_data/Helmholtz_outputs.npy')
    y = y.transpose((2,1,0)).reshape(y.shape[2], 101*101)

    x_train = x[:n_train, :]
    y_train = y[:n_train, :]

    x_test = x[n_train:n_train+n_test, :]
    y_test = y[n_train:n_train+n_test, :]

    x_grid, y_grid = None, None

    return x_train, y_train, x_test, y_test, x_grid, y_grid

def find_files_matching_regex(directory, regex_pattern):
    # Compile the regex pattern for better performance
    pattern = re.compile(regex_pattern)
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                matching_files.append(os.path.join(root, file))
    
    return matching_files

def get_PV_param_data(n_train, n_test, operator, level = 2):
    '''
    Assumes total number of datapoints is greater than n_train + n_test.
    Assumes low res means 64x64 grid size.
    Assumes high res means 256x256 grid size.
    Assumes L=1.0e6.
    Assumes all data in the directory is relevant.
    Don't change level from 2 as the generated data is currently only for level 2.
    '''

    # Load low-res data
    regex = r'^eddy_64_.+years_L=1\.0e6\.nc$'
    low_res_files = find_files_matching_regex('../2d_data/eddy_config/lowres/', regex)[-(n_train+n_test):]

    # low_res_files = [f'{file}' for file in low_res_files]
    low_res_data = xr.open_mfdataset(low_res_files)
    q_low_res = low_res_data.q.sel(lev=level)
    x = q_low_res.values.reshape(-1, 64 * 64)
    print('loaded low-res (x) data')

    # Load coarsened data
    regex = f'^eddy_256_q_operator{operator}_.+years_L=1\.0e6\.csv$'
    coarsened_files = find_files_matching_regex('../2d_data/eddy_config/coarsened/', regex)[-(n_train+n_test):]
    y = []
    for file in coarsened_files:
        y.append(np.genfromtxt(file, delimiter=','))
    y = np.array(y)
    if y.ndim == 4:
        raise ValueError('y has 4 dimensions, this code needs changing so it selects the correct level, as given by the level arguement.')
    y = y.reshape(-1, 64 * 64)
    print('loaded coarsened (y) data')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=n_test, random_state=42)

    x_grid, y_grid = None, None

    return x_train, y_train, x_test, y_test, x_grid, y_grid

