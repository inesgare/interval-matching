import numpy as np
import pandas as pd
import pickle

from match.utils_data import sample_image_points

from skimage.filters import threshold_otsu, threshold_mean, threshold_isodata, threshold_yen, threshold_minimum

from match.utils_PH import *

##### PARAMETERS
# Rmk: the parameters depend on the type of application we have

# 1. choose application
application = 'matching'  #'prevalence'

# 2. choose dataset
DATASET = 'embryogenesis'
# 'cosmic_web', 'actin_cropI', 'actin_cropII', 'actin_cropIII'
# 'lateral_line_zebrafish' ,'heartbeat' , 'noisy_cycles', 

# 3. choose parameters
if application == 'prevalence': 
    N_ref = 1000
    N = 1000
    N_resamp = 20
    noise_scale = .1

if application == 'matching':
    N = 500 # number of points sampled in each image
    # 1000 for lateral_line_zebrafish
    # 500 for heartbeat
    # 200 for embryogenesis
    noise_scale = 0

####### DATA

def generate_data_resamplings(dataset = None):
        
    if 'actin' in dataset :
        if dataset == 'actin_cropI' : # rear
            x1,x2, y1,y2 = 1900,2700, 0, 900 

        elif dataset == 'actin_cropII' : # mixing front and rear
            x1,x2, y1,y2 = 1900,2700, 2000, 2900
        
        elif dataset == 'actin_cropIII' : # front
            x1,x2, y1,y2 = 1900,2700, 3000, 3900
            
        img = plt.imread('data/actin/CIL_24800.jpg', format='jpg')[...,0] # R = G = B, take R
        img = img[x1:x2, y1:y2]
        res = img > threshold_otsu(img)
        img2 = (res == 1) * img
        X = sample_image_points(img2, method = 1, N = N_ref, noise_scale = noise_scale)
        
        list_Y = []
        for i in range(N_resamp) :
            Y = sample_image_points(img2, method = 1, N = N_ref, noise_scale = noise_scale)
            list_Y += [Y]
        full_data = (img, res)
        
    elif dataset == 'cosmic_web' :
        # galaxy data from BOSS CMASS data (DR17 release)

        csv_filepath = 'data/cosmic_web/BOSS_sample_RA_150-210_dec_10-70_z_o4-o8.csv'
        df = pd.read_csv(csv_filepath)
        df = df.rename(columns={"redshift": "z"}) # rename 'redshift' column to 'z'
        # restrict to galaxies, not quasars
        df = df[df['class'] == 'GALAXY']
        df.head()

        # select redshift range

        z_min = .564 #.565
        z_max = .57 #.575
        df_zbin = df[ (df['z'] > z_min) * (df['z'] < z_max) ]

        x = np.array(df_zbin['ra'])
        y = np.array(df_zbin['dec'])

        def create_mask(x,y) :
            return (170 < x) * (x < 190) * (30 < y) * (y < 50)

        mask = create_mask(x,y)
        x = x[mask > 0]
        y = y[mask > 0]

        full_data = np.vstack( (x[None], y[None]) ).T

        #### GENERATE RESAMPLINGS

        # Xref vs X_1,..,X_N
        # x,y already defined above - galaxy RA-dec coords

        nb_galaxy = len(x)

        chosen = np.random.choice(nb_galaxy, size=N_ref)
        X = np.vstack( (x[None,chosen], y[None,chosen]) ).T    
        X += noise_scale * np.random.randn(*X.shape)

        list_Y = []
        for i in range(N_resamp) :
            chosen = np.random.choice(nb_galaxy, size=N)
            Y = np.vstack( (x[None,chosen], y[None,chosen]) ).T
            Y += noise_scale * np.random.randn(*Y.shape)
            list_Y += [Y]

    elif dataset == 'noisy_cycles':
        X = np.random.rand(N_ref,2)
        list_Y = [np.random.rand(N,2) for i in range(N_resamp)]
        full_data = None
        
    else :
        raise ValueError('dataset not recognized')

    return full_data, X, list_Y


def generate_data_matching(dataset = None):
    if dataset == 'lateral_line_zebrafish':
        from skimage import io
        u = io.imread('data/lateral_line_zebrafish/pLLP_data.tif')
        
        #select the relevant frames and crop the image
        u = u[25:70, 300:600, 700:1000]
        
        # apply the otsu threshold
        from skimage.filters import threshold_otsu
        thres_otsu = u > threshold_otsu(u)
        thres_otsu = np.asarray(thres_otsu, dtype=np.float64)
        full_data = (u, thres_otsu)
        
        # create list_X
        list_indices = [i for i in range(0, 45, 3)]
        list_X = {}
        
        for i in list_indices:
            image_X = thres_otsu[i,:,:]
            list_X[i] = sample_image_points(image_X, method = 1, N = N, noise_scale = noise_scale)
    
    elif dataset == 'heartbeat':
        
        # extract the frames from the video 
        
        import cv2
        from skimage.color import rgb2gray
        import matplotlib.pyplot as plt
        vidcap = cv2.VideoCapture('data/heart_zebrafish.mov')
        success,image = vidcap.read()
        u = []
        count = 0
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*50)) 
            success,image = vidcap.read()
            if image is None:
                break
            u.append(rgb2gray(image))
            count += 1
        
        # select the frames we want and thresholding
        u = np.asarray(u, dtype = np.float64)
        u = u[55:65,25:,10:] 
        thres_mean = u > threshold_mean(u)
        thres_mean = np.asarray(thres_mean, dtype=np.float64)
        full_data = (u, thres_mean)
        
        # create list_X
        list_indices = [i for i in range(0, 10, 1)]
        list_X = {}
        
        for i in list_indices:
            image_X = thres_mean[i,:,:]
            list_X[i] = sample_image_points(image_X, method = 1, N = N, noise_scale = noise_scale)

    elif dataset == 'embryogenesis':
        from skimage.filters import sato
        from skimage.filters import threshold_otsu
        
        # Select the frames we want
        u = np.asarray([mpl.image.imread('data/embryogenesis/embryogenesis ({}).jpeg'.format(i)) for i in range(1,50,5)])
        u = u[:, 200:450, 60:320]
        
        # apply some filter to detect tubular shapes (preferred: SATO)
        ridges = np.asarray([sato(u[i,:,:]) for i in range(u.shape[0])], dtype=np.float64)

        # threshold with the otsu method
        thres_otsu = ridges> threshold_otsu(ridges)
        thres_otsu = np.asarray(thres_otsu, dtype = np.float64)
        full_data = (u, thres_otsu)
        
        # create list_X
        list_X = {}
        list_indices = [i for i in range(0, 10, 1)]
        
        for i in list_indices:
            list_X[i] = sample_image_points(thres_otsu[i,:,:], method = 1, N = N, noise_scale = noise_scale)

    else :
        raise ValueError('dataset not recognized')
    
    return full_data, list_X, list_indices