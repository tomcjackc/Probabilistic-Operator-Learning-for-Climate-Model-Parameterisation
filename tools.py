import numpy as np

def extract(ds, key):
    var = ds[key].values
    return var.reshape(-1,*var.shape[2:])

def prepare_PV_data(ds_train, ds_test):
    '''
    Extract Potential vorticity as input ('q')
    and subgrid PV forcing ('q_forcing_advection')
    as output, and normalizes data
    '''
    X_train = extract(ds_train, 'q')
    Y_train = extract(ds_train, 'q_forcing_advection')
    X_test = extract(ds_test, 'q')
    Y_test = extract(ds_test, 'q_forcing_advection')

    print('X_train [0] scale:', np.std(X_train[:, 0, :, :]))
    print('X_train [1] scale:', np.std(X_train[:, 1, :, :]))
    print('Y_train [0] scale:', np.std(Y_train[:, 0, :, :]))
    print('Y_train [1] scale:', np.std(Y_train[:, 1, :, :]))

    x_scale = ChannelwiseScaler(X_train)
    y_scale = ChannelwiseScaler(Y_train)

    X_train = x_scale.normalize(X_train)
    X_test = x_scale.normalize(X_test)
    Y_train = y_scale.normalize(Y_train)
    Y_test = y_scale.normalize(Y_test)

    return X_train, Y_train, X_test, Y_test, x_scale, y_scale

def channelwise_function(X: np.array, fun) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    applies function "fun" for each channel
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''

    N_features = X.shape[1]
    out = np.zeros((1,N_features,1,1))
    for n_f in range(N_features):
        out[0,n_f,0,0] = fun(X[:,n_f,:,:])

    return out.astype('float32')

def channelwise_std(X: np.array) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    Computes standard deviation for each channel
    with double precision
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''
    return channelwise_function(X.astype('float64'), np.std)

def channelwise_mean(X: np.array) -> np.array:
    '''
    For array X of size 
    Nbatch x Nfeatures x Ny x Nx
    Computes mean for each channel
    with double precision
    and returns array of size
    1 x Nfeatures x 1 x 1
    '''
    return channelwise_function(X.astype('float64'), np.mean)

class ChannelwiseScaler:
    '''
    Class containing std and mean
    values for each channel
    '''
    def __init__(self, X=None):
        ''' 
        Stores std and mean values.
        X is numpy array of size
        Nbatch x Nfeatures x Ny x Nx.
        '''
        if X is not None:
            self.mean = channelwise_mean(X)
            self.std  = channelwise_std(X)
    def normalize(self, X):
        '''
        Divide by std
        '''
        return X / self.std
    def denormalize(self, X):
        return X * self.std