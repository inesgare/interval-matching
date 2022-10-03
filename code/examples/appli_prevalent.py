import sys
import numpy as np

import pickle

import matplotlib
matplotlib.use('Agg') # prevent from plotting through sbatch

#from itertools import product
from utils_PH import *

from choose_data import DATASET, N_ref, N, N_resamp, noise_scale

####### READ value of SLURM_ARRAY_TASK_ID

script = sys.argv[0]
index = int(sys.argv[1])
print('script', script) # prints python_script.py
print('index', index) # prints var1

##### LOAD DATA

X = pickle.load( open('{}_temp/X_{}.pkl'.format(DATASET, N_ref),'rb'))
list_Y = pickle.load( open('{}_temp/list_Y_samp{}_{}.pkl'.format(DATASET, N_resamp, N),'rb'))
N_resamp = len(list_Y)

# load PH result for ref X
res_X = pickle.load( open('{}_temp/res_X_{}.pkl'.format(DATASET, N_ref),'rb'))
bars_X, reps_X, tight_reps_X, indices_X = res_X


##### SINGLE MATCHING BETWEEN X AND Y_i

Y = list_Y[index]
print('Matching X to Y_{} ...'.format(index))

# Compute the lower distance matrices
ldm_file_X, ldm_file_Y, ldm_file_Z, threshold = \
    create_matrices_image(X, Y, filename_X = '{}_temp/ldm_X_Z_{}'.format(DATASET, index), 
                          filename_Y = '{}_temp/ldm_Y_Z_{}'.format(DATASET, index),
                          filename_Z = '{}_temp/ldm_Z_{}'.format(DATASET, index), return_thr = True)

# Image persistence - apply thresholding (bug fixed)
print('Image PH of X in Z_{}'.format(index))
out_X_Z = compute_image_bars(filename_X = ldm_file_X, filename_Z = ldm_file_Z, threshold = threshold)
bars_X_Z, indices_X_Z = extract_bars_indices(out_X_Z, only_dim_1 = True)

#print('out_X_Z', out_X_Z)

# Image persistence - apply thresholding (bug fixed)
print('Image PH of Y_{} in Z_{}'.format(index,index))
out_Y_Z = compute_image_bars(filename_X = ldm_file_Y, filename_Z = ldm_file_Z, threshold = threshold)
bars_Y_Z, indices_Y_Z = extract_bars_indices(out_Y_Z, only_dim_1 = True)

#print('out_Y_Z', out_Y_Z)

# PH of Yi
print('PH of Y_{}'.format(index))
out_Y = compute_bars_tightreps(Y, filename = '{}_temp/ldm_Y_{}'.format(DATASET, index) )
bars_Y, reps_Y, tight_reps_Y, indices_Y = extract_bars_reps_indices(out_Y, only_dim_1 = True)    
bars_reps_Y = bars_Y, reps_Y, tight_reps_Y

# Discard files - no more used

import os

os.remove('{}_temp/ldm_X_Z_{}.lower_distance_matrix'.format(DATASET, index))
os.remove('{}_temp/ldm_Y_Z_{}.lower_distance_matrix'.format(DATASET, index))
os.remove('{}_temp/ldm_Y_{}.lower_distance_matrix'.format(DATASET, index))
os.remove('{}_temp/ldm_Z_{}.lower_distance_matrix'.format(DATASET, index))

## Match + affinity

affinity_method = 'A'

matched_X_Y, affinity_X_Y, deaths_image_bars = find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, 
                               bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, 
                               affinity_method = affinity_method, check_Morse = False, 
                               check_ambiguous_deaths = False, return_deaths_image_bars = True)

#list_matched_X_Y, list_affinity_X_Y, bars_reps_X, list_bars_reps_Y = \
#    multiple_matching(X, list_Y, dim = 1, verbose_figs = False, affinity_method = 'A')

result_i = matched_X_Y, affinity_X_Y, bars_reps_Y, deaths_image_bars

###### SAVE RESULTS

pickle.dump(result_i, open('{}_temp/res_match_{}.pkl'.format(DATASET, index), 'wb'))

print('Done computing ')
