import sys
import numpy as np

import pickle

import matplotlib
matplotlib.use('Agg') # prevent from plotting through sbatch

#from itertools import product
from match.utils_PH import *

from choose_data import  DATASET, generate_data_matching, N, noise_scale

####### READ value of SLURM_ARRAY_TASK_ID

script = sys.argv[0]
index = int(sys.argv[1])
print('script', script) # prints python_script.py
print('index', index) # prints var1

##### LOAD DATA

list_X = pickle.load( open('{}_temp/list_X.pkl'.format(DATASET),'rb'))


##### SINGLE MATCHING BETWEEN X_i AND X_{i+1}

import time

if DATASET == 'lateral_line_zebrafish':
    # for the zebrafish data recall that there's a step of 3 between slices
    X = list_X[index * 3]
    Y = list_X[(index+1)*3]
    print('Matching X_{} to X_{} ...'.format(index * 3, (index + 1) * 3))
    
elif DATASET == 'heartbeat':
    X = list_X[index]
    Y = list_X[index+1]
    print('Matching X_{} to X_{} ...'.format(index, index + 1))
    
elif DATASET == 'embryogenesis':
    X = list_X[index]
    Y = list_X[index+1]
    print('Matching X_{} to X_{} ...'.format(index, index + 1))
    

t1 = time.time()
print('starting at', time.ctime(t1))
    
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

# PH of X
print('PH of X_{}'.format(index * 3))
out_X = compute_bars_tightreps(None, filename = '{}_temp/ldm_X_Z_{}'.format(DATASET, index) )
bars_X, reps_X, tight_reps_X, indices_X = extract_bars_reps_indices(out_X, only_dim_1 = True)    
bars_reps_X = bars_X, reps_X, tight_reps_X

# PH of X
print('PH of X_{}'.format((index+1) * 3))
out_Y = compute_bars_tightreps(None, filename = '{}_temp/ldm_Y_Z_{}'.format(DATASET, (index) ))
bars_Y, reps_Y, tight_reps_Y, indices_Y = extract_bars_reps_indices(out_Y, only_dim_1 = True)    
bars_reps_Y = bars_Y, reps_Y, tight_reps_Y


## Match + affinity

affinity_method = 'A'

matched_X_Y, affinity_X_Y = find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, 
                               bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, 
                               affinity_method = affinity_method, check_Morse = False, 
                               check_ambiguous_deaths = False)


t2 = time.time()
print('finishing at', time.ctime(t2))
duration_i = t2 - t1 # in SECONDS not minutes

# Discard files - no more used

import os
os.remove('{}_temp/ldm_X_Z_{}.lower_distance_matrix'.format(DATASET, index))
os.remove('{}_temp/ldm_Y_Z_{}.lower_distance_matrix'.format(DATASET, index))
os.remove('{}_temp/ldm_Z_{}.lower_distance_matrix'.format(DATASET, index))


result_i = matched_X_Y, affinity_X_Y, bars_reps_X, bars_reps_Y

###### SAVE RESULTS

pickle.dump(result_i, open('{}_temp/res_match_{}.pkl'.format(DATASET, index), 'wb'))
pickle.dump(duration_i, open('{}_temp/duration_sec_match_{}.pkl'.format(DATASET, index), 'wb'))

print('Done computing ')
