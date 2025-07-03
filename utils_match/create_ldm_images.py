import os
import sys
import argparse
import numpy as np

sys.path.append('/scratch/tyoeasley/brain_representations/src_py')
from generate_subindex import tag_to_subidx

## main function and helpers ##
##########################################################################################################################################
def make_ldm_images(X=[], Y=[], pairwise_Z=[], nb_X=[], return_thr=False,
        write_mode=True, filename_X='X', filename_Y='Y', filename_Z='dZ'):
    '''Function to create the matrices for the computation of image-persistence so that we can compare the indices of the persistence pairs'''
    X, Y, pairwise_Z, nb_X = _std_phom_input(X=X, Y=Y, pairwise_Z=pairwise_Z, nb_X=nb_X)

    maxi = np.max(pairwise_Z)
    pairwise_X = pairwise_Z.copy()
    pairwise_Y = pairwise_Z.copy()
    pairwise_X[nb_X:] = 2 * maxi + 1 # add min offset 1 because maxi can be small
    pairwise_Y[:,:nb_X] = 2 * maxi + 1 #observe here the change wrt the previous line
    threshold = 2 * maxi + 0.5 # to make sure we still include people <= maxi (in case maxi = 0... paranoia lol)

    # so that later we can apply thresholding in Ripser-image (bug fixed)
    # threshold = 2 * maxi # not 2 * maxi - 1 as maxi could be very small

    if write_mode:
        ldm_X = _write_out(filename_X, pairwise_X)
        ldm_Y = _write_out(filename_Y, pairwise_Y)
        ldm_Z = _write_out(filename_Z, pairwise_Z)

    else:
        ldm_X = pairwise_X
        ldm_Y = pairwise_Y
        ldm_Z = pairwise_Z

    if return_thr :
        return ldm_X, ldm_Y, ldm_Z, threshold

    return ldm_X, ldm_Y, ldm_Z


def _subsamp_dZ(dX, subsamp_idx):
    dXY = dX[:, subsamp_idx]
    dY = dXY[subsamp_idx, :]

    dZ = np.block([[dX, dXY], [dXY.T, dY]])
    return dZ, dY


def _write_out(fname, square_pdist):
    if not fname.endswith('.ldm'):
        fname = fname + '.ldm'

    ldm_fname = os.path.abspath(fname)
    with open(ldm_fname, "w") as f: # erase possibly pre-existing
        for i in range(len(square_pdist)):
            f.write(', '.join([ str(x) for x in square_pdist[i,:i]]))
            f.write('\n')

        f.close()

    return ldm_fname


def _std_phom_input(X=[], Y=[], pairwise_Z=[], nb_X=[]):
    if not np.any(pairwise_Z):
        assert np.any(X), "If pairwise_Z is not provided, point clouds X and Y must both be given"
        assert np.any(Y), "If pairwise_Z is not provided, point clouds X and Y must both be given"
        Z = np.vstack((X,Y))
        nb_X = len(X)
        pairwise_Z = np.sqrt( np.sum( (Z[:, None, :] - Z[None, :, :])**2, axis=-1) )    # computes pairwise Euclidean 2-norm on np.vstack((X,Y))
    else:
        assert nb_X, "If pairwise_Z is given and X and Y are not, nb_X must be specified."

    return X, Y, pairwise_Z, nb_X
##########################################################################################################################################



## test code ##
##########################################################################################################################################
# the function below produces fixed results that will behave mildly inconsistent across tags;
# this is ok, because this function only exists to allow code to run smoothly on test data and is not present in any important analyses
def reduce_subidx(subidx, N_new):
    approx_prop = float(len(subidx))/max([float(i) for i in subidx])
    subidx_long = [idx for idx in subidx if idx < N_new]

    new_length = int(N_new*approx_prop)
    subidx_new = subidx_long[:new_length]
    return subidx_new
## test code ##
##########################################################################################################################################


## input parser ##
##########################################################################################################################################
# Computes and saves subsampled distance & image distance matrices given an initial distance matrix and an ASCII-encoded bit array
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Make and save image & subfiltration matrices and subsample matrices given distance matrix and subsample tag"
    )
    parser.add_argument(
        "-x", "--dX_fname", type=str, help="path to distance matrix"
    )
    parser.add_argument(
        "-t", "--tag", type=str, help="bit-encoded subsampling array (as ASCII string)"
    )
    parser.add_argument(
        "-z", "--dZ_fname", type=str, help="outpath to image distance matrix"
    )
    parser.add_argument(
        "-y", "--dY_fname", type=str, help="outpath to subsampled distance matrix"
    )
    parser.add_argument(
        "-i", "--dXZ_fname", type=str, help="outpath to embedded image distance matrix"
    )
    parser.add_argument(
        "-j", "--dYZ_fname", type=str, help="outpath to embedded subsampled image distance matrix"
    )
    args = parser.parse_args()

    dX = np.genfromtxt(args.dX_fname)
    nb_X = dX.shape[0]

    subidx = tag_to_subidx(args.tag)

    if len(subidx) >= nb_X:
        subidx = reduce_subidx(subidx,nb_X)

    dZ, dY = _subsamp_dZ(dX, subidx)

    _write_out(args.dY_fname, dY)
    make_ldm_images(
            pairwise_Z=dZ, 
            nb_X=nb_X,
            filename_X=args.dXZ_fname,
            filename_Y=args.dYZ_fname,
            filename_Z=args.dZ_fname
            )
