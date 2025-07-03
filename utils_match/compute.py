from numpy import allclose
from send_cmd import send_cmd_linux as send

#### paths to Ripser
ripser_rep_path = '/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/modified_ripser/ripser-tight-representative-cycles/ripser-representatives'
ripser_img_path = '/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/modified_ripser/ripser-image-persistence-simple/ripser-image'



def bars_tightreps(inp = None, filename = 'data', maxdim=2, verbose = False) :
    '''This function computes the barcode and representatives of a:
    - inp = point cloud in the form of an array
    - filename = file containing the lower diagonal matrix containing the pairwise distances for a finite metric space'''
    if inp is None : # a filename input which gives lower distance matrix
        if filename.endswith('.ldm') :
            ldm_file = filename
        else :
            ldm_file = filename + '.ldm'

    elif len(inp.shape) > 1:
        if inp.shape[0] == inp.shape[1]:
            if allclose(inp, inp.T):
                pairwise = inp;
            if filename.endswith('.ldm'):
                ldm_file = filename
            else:
                ldm_file = filename + '.ldm'
        else:
            data = inp # a point cloud in 2D or 3D
            ldm_file = f"{filename}.ldm"
            pairwise = np.sqrt( np.sum( (data[:, None, :] - data[None, :, :])**2, axis=-1) )

        f = open(ldm_file, "w") # erase possibly pre-existing
        for i in range(len(pairwise)) :
            f.write(', '.join([ str(x) for x in pairwise[i,:i]]))
            f.write('\n')
        f.close()

    software = ripser_rep_path
    options = f"--dim {maxdim}"

    command = f"{software} {options} {ldm_file}"

    out = send(command)
    if verbose:
        ### debug code ###
        print('Command: ' + command)
        print('Received from Ripser (' + str(len(out)) + ' lines of output):')
        # print(*out, sep='\n')
        print('')
        ### debug code ###

    return out


def image_bars(filename_X = 'X', filename_Z = 'Z', threshold = None, maxdim = 2, verbose = False) :
    '''This function computes the barcode of the image-persistence of X inside of Z. 
    The input consists on the two lower distance matrices, using the extension explained in the reference paper, and the threshold up
    to which their VR complexes coincide.'''

    if filename_X.endswith('.ldm') :
        ldm_file_X = filename_X
        ldm_file_Z = filename_Z
    else :
        ldm_file_X = filename_X + '.ldm'
        ldm_file_Z = filename_Z + '.ldm'

    software = ripser_img_path

    if threshold is None :
        options = f"--dim {maxdim} --subfiltration {ldm_file_X}"
    else :
        options = f"--dim {maxdim} --threshold {threshold} --subfiltration {ldm_file_X}"

    command = f"{software} {options} {ldm_file_Z}"

    out = send(command)
    if verbose:
        ### debug code ###
        print('Command: ' + command)
        print('Received from Ripser (' + str(len(out)) + ' lines of output):')
        # print(*out, sep='\n')
        print('')
        ### debug code ###

    return out



##### MATCH AFFINITY

def Jaccard(b1,d1,b2,d2) :
    # Jaccard index = intersection over union of two intervals [b1,d1] and [b2,d2]
    # used to measure affinity of two intervals
    M1 = max(b1,b2)
    m1 = min(d1,d2)
    if M1 < m1 :
        Jac = (m1 - M1) / (max(d1,d2) - min(b1,b2))
    else :
        Jac = 0
    return Jac


def affinity(birth_X, death_X, death, birth_Y, death_Y, affinity_method = 'D') :
    if affinity_method == 'A' : # Yohai's and Omer's
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity_val = a_X_Y * a_X_Z * a_Y_Z

    if affinity_method == 'B' :
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity_val = a_XZ_YZ * a_X_Z * a_Y_Z

    if affinity_method == 'C' :
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        a_X_Z = Jaccard( birth_X, death_X, birth_X, death )
        a_Y_Z = Jaccard( birth_Y, death_Y, birth_Y, death )
        affinity_val = a_X_Y * a_XZ_YZ * a_X_Z * a_Y_Z

    if affinity_method == 'D' :
        a_X_Y = Jaccard( birth_X, death_X, birth_Y, death_Y )
        a_XZ_YZ = Jaccard( birth_X, death, birth_Y, death )
        affinity_val = a_X_Y * a_XZ_YZ

    return affinity_val
