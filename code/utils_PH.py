import os
import re

import numpy as np
import pandas as pd

import subprocess

def send_cmd_windows(cmd) : # cmd is a string
    'dont forget stdout = good output,  stderr = error like outputs e.g. help output for windows'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode('UTF-8').split('\n')
    # choosing first good output
    return out

def send_cmd_linux(cmd) :
    out = os.popen(cmd).read().split('\n')
    return out

#### define your system
send_cmd = send_cmd_linux

############

from utils_plot import *


################

##### PERSISTENCE

def compute_bars_tightreps(inp = None, filename = 'data') :
    '''This function computes the barcode and representatives of a:
    - inp = point cloud in the form of an array
    - filename = file containing the lower diagonal matrix containing the pairwise distances for a finite metric space'''
    if inp is None : # a filename input which gives lower distance matrix
        if filename.endswith('.lower_distance_matrix') :
            ldm_file = filename
        else :
            ldm_file = filename + '.lower_distance_matrix'
    else :
        data = inp # a point cloud in 2D or 3D
        ldm_file = '{}.lower_distance_matrix'.format(filename)
        pairwise = np.sqrt( np.sum( (data[:, None, :] - data[None, :, :])**2, axis=-1) )
        
        f = open(ldm_file, "w") # erase possibly pre-existing
        for i in range(len(pairwise)) :
            f.write(', '.join([ str(x) for x in pairwise[i,:i]]))
            f.write('\n')
        f.close()
        
    software = "./ripser-tight-representatives"
    options = ""
    
    command = "{} {} {}".format(software, options, ldm_file)

    out = send_cmd(command)
    return out

def extract_bars_reps(out, only_dim_1 = False, verbose = False) :
    '''This function converts the output of compute_bars_tight_reps into list of bars and reps, organised by dimension'''
    # find after which line it starts enumerating intervals in dim 0,1,2
    line_PH = {0:len(out), 1:len(out), 2:len(out)}
    for i in range(len(out)) :
        if out[i].startswith('persistent homology intervals in dim ') :
            dim = out[i].rstrip().split(' ')[-1][:-1] # after split '0:'
            dim = int(dim)
            line_PH[dim] = i

    bars = {0:[],1:[],2:[]}
    reps = {0:[],1:[],2:[]}
    tight_reps = {0:[],1:[],2:[]}

    # reps 0 [ [[0],[1]], [[1],[2]], etc ]
    # reps 1 [ [ [0,1], [1,2], [2,3], [3,0] ], same]

    if not only_dim_1 :
        
        # 0-dim PH bars and reps
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
            if not x : raise ValueError("no match found")
            if x.group(2) != ' ' : # finite bars first
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
                y = re.search(r"\{\[(\d+)\], \[(\d+)\]\}", out[i])
                if not y :
                    raise ValueError("finite interval detected but not represented by two vertices")
                reps[dim] += [ [ [int(y.group(1))], [int(y.group(2))] ]  ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]
                y = re.search(r"\{\[(\d+)\].*\}", out[i])
                reps[dim] += [ [ [int(y.group(1))] ]  ]
            i += 1
    
    # 1-dim PH bars and reps
    dim = 1

    i = line_PH[dim]+1
    while i < line_PH[dim + 1] :
        x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
        if i == len(out) - 1 : # trivial string ''
            break

        # all "finite" bars (no missing second endpoint)
        bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]

        i += 1 # next line for tight reps
        y = re.findall(r"\[(\d+),(\d+)\] \(\d*.\d*\)", out[i])
        y = [list(elem) for elem in y]
        y = [[int(e[0]), int(e[1])] for e in y]
        tight_reps[dim] += [ y ]

        i += 1 # again, next line for reps
        y = re.findall(r"\[(\d+),(\d+)\] \(\d*.\d*\)", out[i])
        y = [list(elem) for elem in y]
        y = [[int(e[0]), int(e[1])] for e in y]
        reps[dim] += [ y ]

        i += 1
    
    if verbose :
        print(bars)
        print(reps)
        print(tight_reps)

    return bars, reps, tight_reps

def extract_bars_reps_indices(out, only_dim_1 = False, verbose = False) :
    '''This function converts the output of compute_bars_tight_reps into list of bars, representatives and indices of the persistence pairs,
    organised by dimension. REMARK: you need to use the modified version or ripser_tight_representative_cycles'''
    # find after which line it starts enumerating intervals in dim 0,1,2
    line_PH = {0:len(out), 1:len(out), 2:len(out)}
    for i in range(len(out)) :
        if out[i].startswith('persistent homology intervals in dim ') :
            dim = out[i].rstrip().split(' ')[-1][:-1] # after split '0:'
            dim = int(dim)
            line_PH[dim] = i
    # print(line_PH)
    # allows for repeated line: then takes last one

    bars = {0:[],1:[],2:[]}
    reps = {0:[],1:[],2:[]}
    tight_reps = {0:[],1:[],2:[]}
    indices = {1:[],2:[]}

    # reps 0 [ [[0],[1]], [[1],[2]], etc ]
    # reps 1 [ [ [0,1], [1,2], [2,3], [3,0] ], same]

    if not only_dim_1 :
        
        # 0-dim PH bars and reps
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
            if not x : raise ValueError("no intervals found")
            if x.group(2) != ' ' : # finite bars first
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
                y = re.search(r"\{\[(\d+)\], \[(\d+)\]\}", out[i])
                if not y :
                    raise ValueError("finite interval detected but not represented by two vertices")
                reps[dim] += [ [ [int(y.group(1))], [int(y.group(2))] ]  ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]
                y = re.search(r"\{\[(\d+)\].*\}", out[i])
                reps[dim] += [ [ [int(y.group(1))] ]  ]
            i += 1

    
    # 1-dim PH bars and reps
    dim = 1

    i = line_PH[dim]+1
    while i < line_PH[dim + 1] :
        x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
        if i == len(out) - 1 : # trivial string ''
            break

        # all "finite" bars (no missing second endpoint)
        bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
        

        # indices
        z = re.search(r"indices: (\d*)-(\d*)", out[i])
        if not z : raise ValueError("no iindices found --- are you using the modified version of ripser-tight-representative-cycles?")
        indices[dim] += [ [int(z.group(1)), int(z.group(2))] ]

        i += 1 # next line for tight reps
        y = re.findall(r"\[(\d+),(\d+)\] \(\d*.\d*\)", out[i])
        y = [list(elem) for elem in y]
        y = [[int(e[0]), int(e[1])] for e in y]
        tight_reps[dim] += [ y ]

        i += 1 # again, next line for reps
        y = re.findall(r"\[(\d+),(\d+)\] \(\d*.\d*\)", out[i])
        y = [list(elem) for elem in y]
        y = [[int(e[0]), int(e[1])] for e in y]
        reps[dim] += [ y ]

        i += 1
    
    if verbose :
        print(bars)
        print(reps)
        print(tight_reps)

    return bars, reps, tight_reps, indices

##### IMAGE-PERSISTENCE

def compute_image_bars(filename_X = 'X', filename_Z = 'Z', threshold = None) :
    '''This function computes the barcode of the image-persistence of X inside of Z. 
    The input consists on the two lower distance matrices, using the extension explained in the reference paper, and the treshold up
    to which their VR complexes coincide.'''

    if filename_X.endswith('.lower_distance_matrix') :
        ldm_file_X = filename_X
        ldm_file_Z = filename_Z
    else :
        ldm_file_X = filename_X + '.lower_distance_matrix'
        ldm_file_Z = filename_Z + '.lower_distance_matrix'
        
    software = "./ripser-image"

    if threshold is None :
        options = "--dim 1 --subfiltration {}".format(ldm_file_X)
    else :
        options = "--dim 1 --threshold {} --subfiltration {}".format(threshold, ldm_file_X)
        
    command = "{} {} {}".format(software, options, ldm_file_Z)

    out = send_cmd(command)

    return out

def extract_bars(out, only_dim_1 = False, verbose = False) :
    ''' This function converts the output of compute_image_bars into list of bars organised by dimension 
    (simpler version than extract_bars_reps, no reps for image-persistence)'''
    
    line_PH = {0:len(out), 1:len(out), 2:len(out)}
    for i in range(len(out)) :
        if out[i].startswith('persisten') : # not the same output message depending on ripser-feature version!
            dim = out[i].rstrip().split(' ')[-1][:-1] # after split '0:'
            dim = int(dim)
            line_PH[dim] = i

    bars = {0:[],1:[],2:[]}
    
    if not only_dim_1 :
    
        # 0-dim PH bars
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
            if not x : raise ValueError("no match found")
            if x.group(2) != ' ' : # finite bars first
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]
            i += 1
    
    # 1-dim PH bars
    dim = 1
    i = line_PH[dim]+1
    while i < line_PH[dim + 1] :
        x = re.search(r"\[(\d*.\d*),(\d*.*\d*)\)", out[i])
        if i == len(out) - 1 : # trivial string ''
            break
        if x :
            if x.group(2) != ' ' :
                # "finite" bar (no missing second endpoint)
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]

        i += 1

    if verbose :
        print(bars)
    
    return bars


def extract_bars_indices(out, only_dim_1 = False, verbose = False) :
    ''' This function converts the output of compute_image_bars into list of bars and indices of the persistence pairs organised by dimension 
    (simpler version than extract_bars_reps_indices, no reps for image-persistence). REMARK: need to use the modified version of ripser-image!'''
    
    line_PH = {0:len(out), 1:len(out), 2:len(out)}
    for i in range(len(out)) :
        if out[i].startswith('persisten') : # not the same output message depending on ripser-feature version!
            dim = out[i].rstrip().split(' ')[-1][:-1] # after split '0:'
            dim = int(dim)
            line_PH[dim] = i


    bars = {0:[],1:[],2:[]}
    indices = {1: [], 2: []}
    
    if not only_dim_1 :
    
        # 0-dim PH bars
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            x = re.search(r"\[(\d*.\d*),(\d*.\d*)\)", out[i])
            if not x : raise ValueError("no match found")
            if x.group(2) != ' ' : # finite bars first
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]
            i += 1
    
    # 1-dim PH bars
    dim = 1
    i = line_PH[dim]+1
    while i < line_PH[dim + 1] :
        
        #bars
        x = re.search(r"\[(\d*.\d*),(\d*.*\d*)\)", out[i])
        if i == len(out) - 1 : # trivial string ''
            break
        if x :
            if x.group(2) != ' ' :
                # "finite" bar (no missing second endpoint)
                bars[dim] += [ [float(x.group(1)), float(x.group(2))] ]
            else :
                bars[dim] += [ [float(x.group(1)), np.inf] ]
                
        #indices
        z = re.search(r"indices: (\d*)-(\d*)", out[i])
        indices[dim] += [ [int(z.group(1)), int(z.group(2))] ]
        if not z : raise ValueError("no iindices found --- are you using the modified version of ripser-image?")

        i += 1

    if verbose :
        print(bars)
    
    return bars, indices

##### MATCHING

def Jaccard(a,b,c,d) : 
    # Jaccard index = intersection over union of two intervals [a,b] and [c,d]
    # used to measure affinity of two intervals
    M1 = max(a,c)
    m1 = min(b,d)
    if M1 < m1 :
        Jac = (m1 - M1) / (max(b,d) - min(a,c))
    else :
        Jac = 0
    return Jac

def compute_affinity(birth_X, death_X, death, birth_Y, death_Y, affinity_method = 'A') :
    if affinity_method == 'A' : # Yohai's and Omer's
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity = a_X_Y * a_X_Z * a_Y_Z

    if affinity_method == 'B' :
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity = a_XZ_YZ * a_X_Z * a_Y_Z

    if affinity_method == 'C' :
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity = a_X_Y * a_XZ_YZ * a_X_Z * a_Y_Z

    if affinity_method == 'D' :
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        affinity = a_X_Y * a_XZ_YZ

    return affinity


def argsort(seq, option = 'desc'):
    # what permutation to apply to indices in order to sort seq in ascending / descending order
    if option == 'asc' :
        return sorted(range(len(seq)), reverse = False, key= seq.__getitem__)
    elif option == 'desc' :
        return sorted(range(len(seq)), reverse = True, key= seq.__getitem__)

def show_matches(X, Y, matched_X_Y, affinity_X_Y, tight_reps_X, tight_reps_Y, dim = 1, 
                 zoom_factor = 3, show_together = False) :
    ''' This function displays the matches between two point-clouds X and Y after performing the matching and obtaining
    the lists: matched_X_Y and affinity_X_Y. Also needed the corresponding lists of tight_reps.'''

    Z = np.vstack((X,Y))
    arg = argsort(affinity_X_Y)
    affinity_X_Y = np.array(affinity_X_Y)[arg]
    matched_X_Y = np.array(matched_X_Y)[arg]
    
    if len(matched_X_Y) == 1 or not show_together :
        for match, aff in zip(matched_X_Y, affinity_X_Y) :
            print('new match')
            a,b = match

            if X.shape[1] == 2 :
                fig, axes = plt.subplots(1,2, figsize = (8,5), sharex = True, sharey = True)
                axes[0].scatter(Y[:,0], Y[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = axes[0])
                axes[1].scatter(X[:,0], X[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = axes[1])
                for ax in axes:
                    ax.set_aspect('equal')
                axes[0].set_xlabel('X')
                axes[1].set_xlabel('Y')
                #plt.text(2, 5, 'a match with affinity {}'.format(aff))
                fig.suptitle('a match with affinity {}'.format(aff), y=0.78)
                plt.show()

            if X.shape[1] == 3 :
                fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
                fig.suptitle('a match with affinity {}'.format(aff), y=0.7)
                ax = fig.add_subplot(1,2,1, projection='3d')
                ax.scatter(Y[:,0],Y[:,1],Y[:,2], alpha = .1)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = ax)
                ax.set_title('X')

                ax = fig.add_subplot(1,2,2, projection='3d')
                ax.scatter(X[:,0],X[:,1],X[:,2], alpha = .1) #, c = '#729dcf', s = 50, edgecolors='black')
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = ax)
                ax.set_title('Y')
                plt.show()

    else : # show_together :
        
        if X.shape[1] == 2 :
            fig, axes = plt.subplots(len(matched_X_Y), 2, figsize = (10, 6 * len(matched_X_Y)),
                                                                     sharex = True, sharey = True)
            # will be buggy if len(matched_X_Y) == 1
            i = 0
            for match, aff in zip(matched_X_Y, affinity_X_Y) :
                a,b = match

                axes[i,0].scatter(Y[:,0], Y[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = axes[i,0])
                axes[i,1].scatter(X[:,0], X[:,1], alpha = .2)
                plot_cycreps(Z, [tight_reps_Y[dim][b]],  pts_to_show = Y, ax = axes[i,1])
                
                axes[i,0].set_xlabel('X')
                axes[i,0].set_title('aff =')
                axes[i,1].set_title(aff)
                axes[i,1].set_xlabel('Y')
                #plt.text(2, 5, 'a match with affinity {}'.format(aff))
                #fig.suptitle('a match with affinity {}'.format(aff), y=0.78)
            for ax in axes.ravel():
                ax.set_aspect('equal')
            plt.show()
        
        
        if X.shape[1] == 3 :
            fig, axes = plt.subplots(len(matched_X_Y), 2, subplot_kw={'projection': '3d'},
                                    figsize = (10, 4.5 * len(matched_X_Y)))
            #fig = plt.figure(figsize=plt.figaspect(1.)*zoom_factor) # figaspect(0.5)*1.5
            #fig.suptitle('all matches', y=0.7)
            
            i = 0
            for match, aff in zip(matched_X_Y, affinity_X_Y) :
                a,b = match
                #print('a,b=',a,b)
                ax = axes[i,0]
                #ax = fig.add_subplot(i,2,1, projection='3d')
                ax.scatter(Y[:,0],Y[:,1],Y[:,2], alpha = .1)
                plot_cycreps(Z, [tight_reps_X[dim][a]], pts_to_show = X, ax = ax)
                ax.set_title('X   aff = {}'.format(aff))

                ax = axes[i,1]
                #ax = fig.add_subplot(i,2,2, projection='3d')
                ax.scatter(X[:,0],X[:,1],X[:,2], alpha = .1) #, c = '#729dcf', s = 50, edgecolors='black')
                plot_cycreps(Z, [tight_reps_Y[dim][b]], pts_to_show = Y, ax = ax)
                ax.set_title('Y  a = {} b = {}'.format(a,b))
                
                i += 1
                
            plt.tight_layout()
            plt.show()
                
                
def duplicates_list(a) : # duplicates_list([1,2,3,1,2,2]) = [1, 2, 2]
    seen = set()
    dupes = [x for x in a if x in seen or seen.add(x)]
    return dupes

def find_occurences_list(a, val) :
    indices = [i for i, x in enumerate(a) if x == val]
    return indices


def find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, affinity_method = 'A', 
               check_Morse = False, check_ambiguous_deaths = False) :
    ''' This funtion find the matches between the barcodes of X and Y providing the barcodes of their image-persistence modules in the union.
    Affinity score is automatically set to A but can be changed. Optiona outputs to check if the filtrations provided are Morse and if 
    there are image-bars sharing death times in the barcodes.'''

    matched_X_Y = []
    affinity_X_Y = []
    
    # consider all image-bars
    births_X_Z = [a[0] for a in bars_X_Z[dim]]
    births_Y_Z = [a[0] for a in bars_Y_Z[dim]]
    deaths_X_Z = [a[1] for a in bars_X_Z[dim]]
    deaths_Y_Z = [a[1] for a in bars_Y_Z[dim]]
    
    # consider normal bars
    births_X = [a[0] for a in bars_X[dim]]
    births_Y = [a[0] for a in bars_Y[dim]]
    deaths_X = [a[1] for a in bars_X[dim]]
    deaths_Y = [a[1] for a in bars_Y[dim]]

    if check_Morse :
        # adding noise to your point clouds does not solve the following exceptions.
        # It will create a distance matrix with unique values, so that only adding 1 edge at a time 
        # but possibly many triangles that kill cycles simultaneously in Rips complexes
        if len(duplicates_list(deaths_X_Z)) > 0 :
            print('Found duplicate deaths in X_Z') 
        if len(duplicates_list(deaths_Y_Z)) > 0 :
            print('Found duplicate deaths in Y_Z') 
        if len(duplicates_list(births_X)) > 0 :
            print('Found duplicate births in X') # should never happen for unique distance values
        if len(duplicates_list(births_Y)) > 0 :
            print('Found duplicate births in Y') # should never happen for unique distance values


    considered_deaths_X_Z = set(deaths_X_Z)
    considered_deaths_Y_Z = set(deaths_Y_Z)
    
    # find common (considered) deaths in image
    common_deaths = considered_deaths_X_Z.intersection(considered_deaths_Y_Z)
        
    
    if check_ambiguous_deaths :
        if set(duplicates_list(deaths_X_Z)).intersection(set(duplicates_list(deaths_Y_Z))) != set() :
            print('Found common duplicate deaths in X_Z and Y_Z!!!')
        if set(duplicates_list(deaths_X_Z)).intersection(common_deaths) != set() :
            print('Found duplicate death in X_Z common with Y_Z')
        if set(duplicates_list(deaths_Y_Z)).intersection(common_deaths) != set() :
            print('Found duplicate death in Y_Z common with X_Z')

    # determine ambiguous deaths
    ambiguous_deaths_X_Z = set(duplicates_list(deaths_X_Z)).intersection(common_deaths)
    ambiguous_deaths_Y_Z = set(duplicates_list(deaths_Y_Z)).intersection(common_deaths)
    ambiguous_deaths = ambiguous_deaths_X_Z.union(ambiguous_deaths_Y_Z)
    if ambiguous_deaths != set() :
        print('We will solve ambiguous deaths matching.')

    # now, find common births with the normal bars

    # First case: non-ambiguous matching
    # in this case, deaths in X_Z and Y_Z are unique so matching can be made without ambiguity (even if duplicate deaths in X or in Y)

    for death in common_deaths.difference(ambiguous_deaths) :
        oXZ = deaths_X_Z.index(death)
        oYZ = deaths_Y_Z.index(death)
        birth_X = births_X_Z[oXZ]
        birth_Y = births_Y_Z[oYZ]
        
        # Now we match with the persistence bars of X and Y
        Occ_X = find_occurences_list(births_X, birth_X) 
        Occ_Y = find_occurences_list(births_Y, birth_Y) 

        if len(Occ_X) == 1 and len(Occ_Y) == 1: # if there are no ambiguous births
            a = births_X.index(birth_X) # unique 
            b = births_Y.index(birth_Y) # unique 
          
            matched_X_Y += [[a, b]]
            affinity = compute_affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
            affinity_X_Y += [affinity]
        else:
            # the way we are computing persistent homology, the indices of the 
            # persistent homology of Y can be compared with the indices in the
            # image - persistent homology of Y inside Z
                
            for k, oX in enumerate(Occ_X):
                pos_index_X = indices_X[dim][oX][0]
                pos_index_XZ = indices_X_Z[dim][oXZ][0]
                if pos_index_X == pos_index_XZ:
                    a = oX
            for l, oY in enumerate(Occ_Y):
                pos_index_Y = indices_Y[dim][oY][0]
                pos_index_YZ = indices_Y_Z[dim][oYZ][0] # the bars are presented in the same order, independently on how we arrange X an Y
                if pos_index_Y == pos_index_YZ:
                    b = oY
            matched_X_Y += [[a, b]]
            affinity = compute_affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
            affinity_X_Y += [affinity]
    
    # Second case: ambiguous matching
    
    for death in ambiguous_deaths :
        # detect the indices of the bars with ambiguous death times
        Occ_XZ = find_occurences_list(deaths_X_Z, death)
        Occ_YZ = find_occurences_list(deaths_Y_Z, death)

        for i, oXZ in enumerate(Occ_XZ) :
            # extract negative index of the image persistence bar of X
            neg_index_XZ = indices_X_Z[dim][oXZ][1]
            for j, oYZ in enumerate(Occ_YZ) :
                # extract negative index of the image persistence bar of Y
                neg_index_YZ = indices_Y_Z[dim][oYZ][1]
                if neg_index_XZ == neg_index_YZ:
                    # match the image bars when the indices coincide
                    birth_X = births_X_Z[oXZ] # ! not i
                    birth_Y = births_Y_Z[oYZ] # ! not j

                    # Now we match with the persistence bars of X and Y
                    Occ_X = find_occurences_list(births_X, birth_X) 
                    Occ_Y = find_occurences_list(births_Y, birth_Y) 

                    if len(Occ_X) == 1 and len(Occ_Y) == 1: # if there are no ambiguous births
                        a = births_X.index(birth_X) # unique 
                        b = births_Y.index(birth_Y) # unique 
                        matched_X_Y += [[a, b]]
                        affinity = compute_affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                        affinity_X_Y += [affinity]
                    else:
                        # the way we are computing persistent homology, the indices of the 
                        # persistent homology of Y can be compared with the indices in the
                        # image - persistent homology of Y inside Z
                        for k, oX in enumerate(Occ_X):
                            pos_index_X = indices_X[dim][oX][0]
                            pos_index_XZ = indices_X_Z[dim][oXZ][0]
                            if pos_index_X == pos_index_XZ:
                                a = oX
                        for l, oY in enumerate(Occ_Y):
                            pos_index_Y = indices_Y[dim][oY][0]
                            pos_index_YZ = indices_Y_Z[dim][oYZ][0] # the bars are presented in the same order, independently on how we arrange X an Y
                            if pos_index_Y == pos_index_YZ:
                                b = oY
                        matched_X_Y += [[a, b]]
                        affinity = compute_affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                        affinity_X_Y += [affinity]
        
    return matched_X_Y, affinity_X_Y


def create_matrices_image(X, Y, filename_X = 'X', filename_Y = 'Y', filename_Z = 'Z', return_thr = False):
    '''Function to create the matrices for the computation of image-persistence so that we can compare the indices of the persistence pairs'''
    Z = np.vstack((X,Y))
    nb_X = len(X) 
    ldm_file_X = '{}.lower_distance_matrix'.format(filename_X)
    ldm_file_Y = '{}.lower_distance_matrix'.format(filename_Y)
    ldm_file_Z = '{}.lower_distance_matrix'.format(filename_Z)

    pairwise_Z = np.sqrt( np.sum( (Z[:, None, :] - Z[None, :, :])**2, axis=-1) )
    maxi = np.max(pairwise_Z)
    pairwise_X = pairwise_Z.copy()
    pairwise_X[nb_X:] = 2 * maxi + 1 # add min offset 1 because maxi can be small 
    pairwise_Y = pairwise_Z.copy()
    pairwise_Y[:,:nb_X] = 2 * maxi + 1 #observe here the change wrt the previous line

    # so that later we can apply thresholding in Ripser-image (bug fixed)
    # threshold = 2 * maxi # not 2 * maxi - 1 as maxi could be very small

    f = open(ldm_file_X, "w") # erase possibly pre-existing
    for i in range(len(pairwise_X)) :
        f.write(', '.join([ str(x) for x in pairwise_X[i,:i]]))
        f.write('\n')
    f.close()

    f = open(ldm_file_Y, "w") # erase possibly pre-existing
    for i in range(len(pairwise_Y)) :
        f.write(', '.join([ str(x) for x in pairwise_Y[i,:i]]))
        f.write('\n')
    f.close()

    f = open(ldm_file_Z, "w") # erase possibly pre-existing
    for i in range(len(pairwise_Z)) :
        f.write(', '.join([ str(x) for x in pairwise_Z[i,:i]]))
        f.write('\n')
    f.close()

    if return_thr :
        threshold = 2 * maxi + 0.5 # to make sure we still include people <= maxi (in case maxi = 0... paranoia lol)
        return ldm_file_X, ldm_file_Y, ldm_file_Z, threshold
        
    return ldm_file_X, ldm_file_Y, ldm_file_Z


def matching(X,Y, dim = 1, verbose_figs = False, affinity_method = 'A', check_Morse = False) :

    '''Function that takes as input two pointclouds X and Y and computes the relevant barcodes and the matching.'''
    
    # Compute the lower distance matrices
    ldm_file_X, ldm_file_Y, ldm_file_Z, threshold = \
        create_matrices_image(X, Y, filename_X = 'X', filename_Y = 'Y', return_thr = True)
    
    # Image persistence - apply thresholding (bug fixed)
    out_X_Z = compute_image_bars(filename_X = ldm_file_X, filename_Z = ldm_file_Z, threshold = threshold)
    bars_X_Z, indices_X_Z = extract_bars_indices(out_X_Z, only_dim_1 = True)
    print('bars_X_Z', bars_X_Z)

    out_Y_Z = compute_image_bars(filename_X = ldm_file_Y, filename_Z = ldm_file_Z, threshold = threshold)
    bars_Y_Z, indices_Y_Z = extract_bars_indices(out_Y_Z, only_dim_1 = True)
    
    # Persistent homology
    out_X = compute_bars_tightreps(inp = None, filename = ldm_file_X) 
    # this way we obtain the same bars but the vertices are indexes in the same way as in the image persistence
    bars_X, reps_X, tight_reps_X, indices_X = extract_bars_reps_indices(out_X, only_dim_1 = True)
    
    out_Y = compute_bars_tightreps(inp = None, filename = ldm_file_Y)
    bars_Y, reps_Y, tight_reps_Y, indices_Y = extract_bars_reps_indices(out_Y, only_dim_1 = True)
    
    matched_X_Y, affinity_X_Y = find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, 
                                           bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, 
                                           affinity_method = affinity_method, check_Morse = check_Morse, 
                                           check_ambiguous_deaths = False)
        
    if verbose_figs :
        show_matches(X,Y,matched_X_Y, affinity_X_Y, tight_reps_X, tight_reps_Y, dim = dim, show_together = True)
    
    return matched_X_Y, affinity_X_Y, (bars_X, reps_X, tight_reps_X), (bars_Y, reps_Y, tight_reps_Y)


###### PREVALENCE, CROSS-PREVALENCE

def multiple_matching(X, list_Y, dim = 1, verbose_figs = False, affinity_method = 'A') :    
    
    out_X = compute_bars_tightreps(X)
    bars_X, reps_X, tight_reps_X, indices_X = extract_bars_reps_indices(out_X, only_dim_1 = True)    
    
    list_matched_X_Y = {}
    list_affinity_X_Y = {}
    
    list_bars_reps_Y = []
    
    for y, Y in enumerate(list_Y) :
        print('Matching X to Y_{} ...'.format(y))
        
        # Compute the lower distance matrices
        ldm_file_X, ldm_file_Y, ldm_file_Z, threshold = \
            create_matrices_image(X, Y, filename_X = 'X', filename_Y = 'Y', return_thr = True)

        # Image persistence - apply thresholding 
        out_X_Z = compute_image_bars(filename_X = ldm_file_X, filename_Z = ldm_file_Z, threshold = threshold)
        bars_X_Z, indices_X_Z = extract_bars_indices(out_X_Z, only_dim_1 = True)

        out_Y_Z = compute_image_bars(filename_X = ldm_file_Y, filename_Z = ldm_file_Z, threshold = threshold)
        bars_Y_Z, indices_Y_Z = extract_bars_indices(out_Y_Z, only_dim_1 = True)

        out_Y = compute_bars_tightreps(Y)
        bars_Y, reps_Y, tight_reps_Y, indices_Y = extract_bars_reps_indices(out_Y, only_dim_1 = True)    

        list_bars_reps_Y += [ [bars_Y, reps_Y, tight_reps_Y] ]

        matched_X_Y, affinity_X_Y = find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, 
                                       bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, 
                                       affinity_method = affinity_method, check_Morse = False, 
                                       check_ambiguous_deaths = False)

        list_matched_X_Y[y] = matched_X_Y
        list_affinity_X_Y[y] = affinity_X_Y        
        
        if verbose_figs :

            show_matches(X, list_Y[y], matched_X_Y[y], affinity_X_Y[y], tight_reps_X, tight_reps_Y, dim = dim, show_together = True)
    
    list_matched_X_Y = list(list_matched_X_Y.values())
    list_affinity_X_Y = list(list_affinity_X_Y.values())
    bars_reps_X = [bars_X, reps_X, tight_reps_X]
    
    return list_matched_X_Y, list_affinity_X_Y, bars_reps_X, list_bars_reps_Y
        
def cross_matching(list_X, dim = 1, verbose_figs = False, affinity_method = 'A') :
    
    list_matched_X_Y = {}
    list_affinity_X_Y = {}
    list_bars_reps_indices_X = {}
    list_bars_reps_X = {}
    
    # compute PH and reps and indices of individual spaces
    for i,X in enumerate(list_X) :
        out_X = compute_bars_tightreps(X)
        bars_X, reps_X, tight_reps_X, indices_X = extract_bars_reps_indices(out_X, only_dim_1 = True)
        list_bars_reps_indices_X[i] = [bars_X, reps_X, tight_reps_X, indices_X]
        list_bars_reps_X[i] =  [bars_X, reps_X, tight_reps_X]
    
    # match any X_i to any X_j (j > i)
    for i,X in enumerate(list_X) :
        for j in range(i+1, len(list_X)) :
            print('Matching X_{} to X_{} ...'.format(i,j))
            
            X = list_X[i]
            Y = list_X[j]
            
            # Compute the lower distance matrices
            ldm_file_X, ldm_file_Y, ldm_file_Z, threshold = \
                create_matrices_image(X, Y, filename_X = 'X', filename_Y = 'Y', return_thr = True)
    
            # Image persistence - apply thresholding (bug fixed)
            out_X_Z = compute_image_bars(filename_X = ldm_file_X, filename_Z = ldm_file_Z, threshold = threshold)
            bars_X_Z, indices_X_Z = extract_bars_indices(out_X_Z, only_dim_1 = True)

            out_Y_Z = compute_image_bars(filename_X = ldm_file_Y, filename_Z = ldm_file_Z, threshold = threshold)
            bars_Y_Z, indices_Y_Z = extract_bars_indices(out_Y_Z, only_dim_1 = True)
           
            bars_X, reps_X, tight_reps_X, indices_X = list_bars_reps_indices_X[i]
            bars_Y, reps_Y, tight_reps_Y, indices_Y = list_bars_reps_indices_X[j]
            
                
            matched_X_Y, affinity_X_Y = find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, 
                                           bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, 
                                           affinity_method = affinity_method, check_Morse = False, 
                                           check_ambiguous_deaths = False)
            
            list_matched_X_Y[i,j] = matched_X_Y
            list_affinity_X_Y[i,j] = affinity_X_Y
    
    for i in range(len(list_X)) :
        for j in range(i) :
            aa = list_matched_X_Y[j,i]
            if len(aa) > 0 :
                list_matched_X_Y[i,j] = np.array(aa)[:,::-1].tolist() # reverse column order
            else :
                list_matched_X_Y[i,j] = []
            list_affinity_X_Y[i,j] = list_affinity_X_Y[j,i]
    return list_matched_X_Y, list_affinity_X_Y, list_bars_reps_X

##### SOME FUNCTIONS TO ENABLE TRACKING CYCLES

def track_cycles_from_slice(list_matched_X_Y, list_affinity_X_Y, cycle, list_indices, initial_slice = 0):
    ''' From a list of matched cycles between consecutive slices, obtains a list in which we store the matches 
        that track a particular cycle from a some chosen slice
        output = [[cycle, a],[a, b], [b, c] ...]
        rmk: set of indices does not include the initial slice, counts from the second slice studied'''
    
    tracked_cycle = []
    tracked_affinity = []
    current_match = []
    
    # initialise
    for i, match in enumerate(list_matched_X_Y[initial_slice]):
        if match[0] == cycle:
            tracked_cycle += [match]
            current_match = match
            tracked_affinity.append(list_affinity_X_Y[initial_slice][i])

            
    # track the cycle
    for i in list_indices:
        next_cycle = current_match[1]
        tracked_copy = tracked_cycle.copy()
        for j, match in enumerate(list_matched_X_Y[i]):
            if match[0] == next_cycle:
                #print(match)
                tracked_cycle += [match]
                current_match = match
                tracked_affinity.append(list_affinity_X_Y[i][j])
                continue
        if len(tracked_copy) == len(tracked_cycle): # in case we don't find a next cycle we stop tracking

            break
            
    return tracked_cycle, tracked_affinity
