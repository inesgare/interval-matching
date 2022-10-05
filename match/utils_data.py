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


# sampling a 3D tubular shape

import nibabel as nib

# "get_fdata is more predictable behavior" but for now PLEASE USE get_data otherwise memory problems on remote server.
def load_nii(nii_file) :
    'Loads a .nii.gz or .nii file into a np.array'
    img = nib.load(nii_file)
    vol = np.asarray(img.get_data())
    return vol


# sample points on a surface (given by 3D field u / niifile)
def sample_points_from_surface_(u, N = 100, noise_scale = .01) :
    print('REMEMBER, MARCHING CUBES WAS BUGGY LAST YEAR, with bad intersecting triangulations.')
    import skimage
    verts, faces, _, _ = skimage.measure.marching_cubes(u, level = 0, allow_degenerate = False) 
    
    # compute cells areas
    As = verts[faces[:,0]]
    Bs = verts[faces[:,1]]
    Cs = verts[faces[:,2]]
    Gs = (As + Bs + Cs) / 3 # barycenters of the mesh cells
    alen = np.linalg.norm(Cs - Bs, axis = 1)
    blen = np.linalg.norm(Cs - As, axis = 1)
    clen = np.linalg.norm(Bs - As, axis = 1)
    midlen = .5 * (alen + blen + clen)
    areas = np.sqrt(midlen * (midlen - alen) * (midlen - blen) * (midlen - clen))
    
    total_area = areas.sum() 
    
    probas = areas / total_area
    cdf = np.cumsum(probas)
    values = np.random.rand(N)
    sampled_indices = np.searchsorted(cdf, values)
    sampled_points = Gs[sampled_indices]
    sampled_points += noise_scale * np.random.randn(*sampled_points.shape)
    return sampled_points
    
def sample_points_from_surface(u = None, nii_file = None, N = 100, noise_scale = .1) :       
    if u is None and nii_file is not None :
        v = load_nii(nii_file)
    if u is not None and nii_file is None :
        v = u
    else :
        raise ValueError("You cannot give both non-trivial arguments u and nii_file.")
    return sample_points_from_surface_(v, N = N, noise_scale = noise_scale)

# sample points inside a volume (given by 3D field u / niifile)
def sample_points_from_vol_(u, N = 100, noise_scale = .1) :
    return sample_image_points(u, method = 1, N = N, noise_scale = noise_scale)
    
def sample_points_from_vol(u = None, nii_file = None, N = 100, noise_scale = .1) :
    if u is None and nii_file is not None :
        v = load_nii(nii_file)
    if u is not None and nii_file is None :
        v = u
    else :
        raise ValueError("You cannot give both non-trivial arguments u and nii_file.")
    return sample_points_from_vol_(v, N = N, noise_scale = noise_scale)

