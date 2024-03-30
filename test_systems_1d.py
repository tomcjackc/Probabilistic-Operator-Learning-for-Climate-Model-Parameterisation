import numpy as np

def diffusion(x, t, D, V, H_0):
    w_0 = (V**2)/(2*np.pi*H_0**2)
    S = (2*np.pi*H_0**2)/V
    w = np.sqrt(w_0 + 2*D*t)

    return ((S*w_0)/(w*np.sqrt(2*np.pi)))*np.exp(-(x**2)/(2*w**2))