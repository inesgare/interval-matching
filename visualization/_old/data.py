import os
import re
import numpy as np
import matplotlib.pyplot as plt

## Sampling functions

# sample from images or shapes
# sample from 3D volume e.g. tubular data

def rescale(vol) :
    'rescales between 0 and 1'
    vol = vol - vol.min()
    vol = vol/(vol.max())
    return vol

def sample_image_points(u, method = 1, N = 100, noise_scale = .1):
    '''
    - method = 1 supposes u gives the whole spatial proba directly, nb sampling pts is N with possible multiplicity,
    then adding noise to discard multiplicity.
    '''
    u = rescale(u) # between 0 and 1 now
    if method == 1 : # u is the discrete histogram of a probability, with u.sum() = 1
        u /= u.sum()
        indices = np.random.choice(np.prod(u.shape), size=N, p=u.ravel())
        coords = np.unravel_index(indices, shape=u.shape) # xs,ys,zs
        pts = 1.*np.vstack(coords).T
        pts += noise_scale * np.random.randn(*pts.shape)
        
    if method == 2 : # sample in same connected component??
        
        # ...
        
        u /= u.sum()
        indices = np.random.choice(np.prod(u.shape), size=N, p=u.ravel())
        coords = np.unravel_index(indices, shape=u.shape) # xs,ys,zs
        pts = 1.*np.vstack(coords).T
        pts += noise_scale * np.random.randn(*pts.shape)
        
    return pts


def sample_circle(x0 = 0, r = 1, N = 10, dim = 2, noise_scale = .1) :
    '''sample N points uniformly on a circle whose center may be shifted by x0 on X axis '''
    theta = 2 * np.pi * np.random.rand(N)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    if dim == 3 :
        z = np.zeros(len(x))
        return x,y,z
    pts = np.vstack((x0 + x,y)).T # x0 + x, y # for Julia compatibility a tuple of 2 arrays
    pts += noise_scale * np.random.randn(*pts.shape)
    return pts

