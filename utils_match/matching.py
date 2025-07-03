import os
import re
import ast
import argparse
import compute


def find_match_wrapper(
        read=False, phomX_fname='', phomY_fname='', phomXZ_fname='', phomYZ_fname='',
        barsX=None, barsY=None, barsXZ=None, barsYZ=None, 
        idxX=None, idxY=None, idxXZ=None, idxYZ=None,
        dim=1, affinity_method='C', check_Morse=False, check_ambiguous_deaths=False
        ):
    if read:
        assert phomX_fname, 'Must supply filepath (type) of phomX data when in read mode'
        assert phomY_fname, 'Must supply filepath (type) of phomY data when in read mode'
        assert phomXZ_fname,'Must supply filepath (type) of phomXZ data when in read mode'
        assert phomYZ_fname,'Must supply filepath (type) of phomYZ data when in read mode'

        barsX= _read_in(phomX_fname, 'bars')
        barsY= _read_in(phomY_fname, 'bars')
        barsXZ=_read_in(phomXZ_fname,'bars')
        barsYZ=_read_in(phomYZ_fname,'bars')

        idxX= _read_in(phomX_fname,  'indices')
        idxY= _read_in(phomY_fname,  'indices')
        idxXZ=_read_in(phomXZ_fname, 'indices')
        idxYZ=_read_in(phomYZ_fname, 'indices')
    else:
        assert barsX, 'Must supply barsX data when not in read mode'
        assert barsY, 'Must supply barsY data when not in read mode'
        assert barsXZ,'Must supply barsXZ data when not in read mode'
        assert barsYZ,'Must supply barsYZ data when not in read mode'
        assert idxX,  'Must supply idxX data when not in read mode'
        assert idxY,  'Must supply idxY data when not in read mode'
        assert idxXZ, 'Must supply idxXZ data when not in read mode'
        assert idxYZ, 'Must supply idxYZ data when not in read mode'

    matched, affinity, verbose_matches = find_match(
        barsX, barsXZ, idxX, idxXZ,
        barsY, barsYZ, idxY, idxYZ,
        dim=dim, affinity_method=affinity_method, 
        check_Morse=check_Morse, check_ambiguous_deaths=check_ambiguous_deaths
            )

    return matched, affinity, verbose_matches

def _pull_tag(phom_fpath):
    phom_fname = os.path.basename(phom_fpath).split('.')[0]
    tag = re.sub('phom[X-Z]?[X-Z]?_','',phom_fname)
    return tag

def _read_in(phom_fpath, varlabel):
    savedir   = os.path.dirname(phom_fpath)
    phom_fname= os.path.basename(phom_fpath)
    var_fname = phom_fname.replace('phom', varlabel)
    var_fpath = os.path.join(savedir, var_fname)
    with open(var_fpath,'r') as fin:
        var = ast.literal_eval(fin.read())
    return var

def _write_out(var, varlabel, phom_fpath, verbose=False):
    savedir   = os.path.dirname(phom_fpath)
    phom_fname= os.path.basename(phom_fpath)
    var_fname = phom_fname.replace('phom', varlabel)
    var_fpath = os.path.join(savedir, var_fname)
    if verbose:
        print(var_fpath)
    with open(var_fpath,'w') as fout:
        varstring = var.__str__().replace('inf','2e308')     # replace 'inf' with a true literal (may break in future versions of Python)
        fout.write(varstring)

import numpy as np

def get_duplicates_list(values) : # get_duplicates_list([1,2,3,1,2,2]) = [1, 2, 2]
    seen = set()
    dupes_list = [val for val in values if val in seen or seen.add(val)]
    return dupes_list

def find_occurences_list(values, val) :
    indices = [i for i, x in enumerate(values) if x == val]
    return indices


def find_match(bars_X, bars_X_Z, indices_X, indices_X_Z, bars_Y, bars_Y_Z, indices_Y, indices_Y_Z, dim = 1, affinity_method = 'A',
               check_Morse = False, check_ambiguous_deaths = False) :
    ''' This funtion find the matches between the barcodes of X and Y providing the barcodes of their image-persistence modules in the union.
    Affinity score is automatically set to A but can be changed. Optional outputs to check if the filtrations provided are Morse and if 
    there are image-bars sharing death times in the barcodes.'''

    # initialize the match and affinity variables as dictionaries indexed by the indices of bars_XZ
    matched_X_Y = {idx: [] for idx, bar in enumerate(bars_X[dim])}
    affinity_X_Y = {idx: [] for idx, bar in enumerate(bars_X[dim])}
    verbose_matches = [{
        "dim":dim,
        "barX": bar,
        "barY": [],
        "deathZ": [],
        "matchedXY":[],
        "affinity":[]} for bar in bars_X[dim]]

    # consider all image-bars
    births_X_Z = [bar[0] for bar in bars_X_Z[dim]]
    births_Y_Z = [bar[0] for bar in bars_Y_Z[dim]]
    deaths_X_Z = [bar[1] for bar in bars_X_Z[dim]]
    deaths_Y_Z = [bar[1] for bar in bars_Y_Z[dim]]
    ## unzip

    # consider normal bars
    births_X = [bar[0] for bar in bars_X[dim]]
    births_Y = [bar[0] for bar in bars_Y[dim]]
    deaths_X = [bar[1] for bar in bars_X[dim]]
    deaths_Y = [bar[1] for bar in bars_Y[dim]]
    ## unzip

    if check_Morse :
        # adding noise to your point clouds does not solve the following exceptions.
        # It will create a distance matrix with unique values, so that only adding 1 edge at a time 
        # but possibly many triangles that kill cycles simultaneously in Rips complexes
        if len(get_duplicates_list(deaths_X_Z)) > 0 :
            print('Found duplicate deaths in X_Z')
        if len(get_duplicates_list(deaths_Y_Z)) > 0 :
            print('Found duplicate deaths in Y_Z')
        if len(get_duplicates_list(births_X)) > 0 :
            print('Found duplicate births in X') # should never happen for unique distance values
        if len(get_duplicates_list(births_Y)) > 0 :
            print('Found duplicate births in Y') # should never happen for unique distance values


    considered_deaths_X_Z = set(deaths_X_Z)
    considered_deaths_Y_Z = set(deaths_Y_Z)

    # find common (considered) deaths in image
    common_deaths = considered_deaths_X_Z.intersection(considered_deaths_Y_Z)


    if check_ambiguous_deaths :
        if set(get_duplicates_list(deaths_X_Z)).intersection(set(get_duplicates_list(deaths_Y_Z))) != set() :
            print('Found common duplicate deaths in X_Z and Y_Z!!!')
        if set(get_duplicates_list(deaths_X_Z)).intersection(common_deaths) != set() :
            print('Found duplicate death in X_Z common with Y_Z')
        if set(get_duplicates_list(deaths_Y_Z)).intersection(common_deaths) != set() :
            print('Found duplicate death in Y_Z common with X_Z')

    # determine ambiguous deaths
    ambiguous_deaths_X_Z = set(get_duplicates_list(deaths_X_Z)).intersection(common_deaths)
    ambiguous_deaths_Y_Z = set(get_duplicates_list(deaths_Y_Z)).intersection(common_deaths)
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

            matched_X_Y[a].append(b)

            if b:
                affinity_val = compute.affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                verbose_matches = _update_verbose(verbose_matches, birth_X, deaths_X, death, birth_Y, deaths_Y, a, b, affinity_val)
            else:
                affinity_val = 0

            affinity_X_Y[a].append(affinity_val)
        else:
            # the way we are computing persistent homology, the indices of the 
            # persistent homology of Y can be compared with the indices in the
            # image - persistent homology of Y inside Z

            for oX in Occ_X:
                pos_index_X = indices_X[dim][oX][0]
                pos_index_XZ = indices_X_Z[dim][oXZ][0]
                if pos_index_X == pos_index_XZ:
                    a = oX
            if 'a' not in locals():
                print("oXZ = " + str(oXZ))
                print("Occ_X = " + str(Occ_Xz))
                print("pos_index_X = " + str(pos_index_X))
                print("pos_index_XZ = " + str(pos_index_XZ))
                raise Exception("birth index 'a' in X homology is unset -- this should be impossible.")

            for oY in Occ_Y:
                pos_index_Y = indices_Y[dim][oY][0]
                pos_index_YZ = indices_Y_Z[dim][oYZ][0] # the bars are presented in the same order, independently on how we arrange X an Y
                if pos_index_Y == pos_index_YZ:
                    b = oY
            if 'b' not in locals():
                print(f"\nWARNING: match index \'b\' unset in dimension {dim} at oYZ={oYZ}")
                b = ''

            matched_X_Y[a].append(b)

            if b:
                affinity_val = compute.affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                verbose_matches = _update_verbose(verbose_matches, birth_X, deaths_X, death, birth_Y, deaths_Y, a, b, affinity_val)
            else:
                affinity_val = 0

            affinity_X_Y[a].append(affinity_val)

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
                        matched_X_Y[a].append(b)

                        if b:
                            affinity_val = compute.affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                            verbose_matches = _update_verbose(verbose_matches, birth_X, deaths_X, death, birth_Y, deaths_Y, a, b, affinity_val)
                        else:
                            affinity_val = 0

                        affinity_X_Y[a].append(affinity_val)

                    else:
                        # the way we are computing persistent homology, the indices of the 
                        # persistent homology of Y can be compared with the indices in the
                        # image - persistent homology of Y inside Z
                        for oX in Occ_X:
                            pos_index_X = indices_X[dim][oX][0]
                            pos_index_XZ = indices_X_Z[dim][oXZ][0]
                            if pos_index_X == pos_index_XZ:
                                a = oX
                        if 'a' not in locals():
                            print("oXZ = " + str(oXZ))
                            print("Occ_X = " + str(Occ_Xz))
                            print("pos_index_X = " + str(pos_index_X))
                            print("pos_index_XZ = " + str(pos_index_XZ))
                            raise Exception("birth index 'a' in X homology is unset -- this should be impossible.")

                        for oY in Occ_Y:
                            pos_index_Y = indices_Y[dim][oY][0]
                            pos_index_YZ = indices_Y_Z[dim][oYZ][0] # the bars are presented in the same order, independently on how we arrange X an Y
                            if pos_index_Y == pos_index_YZ:
                                b = oY
                        if 'b' not in locals():
                            print(f"\nWARNING: match index \'b\' unset in dimension {dim} at oYZ={oYZ}; setting to empty string (b=\'\')")
                            b = ''

                        matched_X_Y[a].append(b)

                        if b:
                            affinity_val = compute.affinity(birth_X, deaths_X[a], death, birth_Y, deaths_Y[b], affinity_method = affinity_method)
                            verbose_matches = _update_verbose(verbose_matches, birth_X, deaths_X, death, birth_Y, deaths_Y, a, b, affinity_val)
                        else:
                            # used to handle this exceptional case by setting affinity_val = -1 and b=0, updating verbose_matches regardless.
                            # we no longer allow this (failed) "match" type to enter into the (verbose) record.
                            affinity_val = 0

                        affinity_X_Y[a].append(affinity_val)

    # a bar without a match is assigned an affinity score of 0
    affinity_X_Y = np.array(_fix_affinity(affinity_X_Y))
    verbose_matches = _fix_vb_affinity(verbose_matches)

    return matched_X_Y, affinity_X_Y, verbose_matches

def _update_verbose(verbose_match_list, birth_X, deaths_X, deathZ, birth_Y, deaths_Y, a, b, aff_val):
    entry = verbose_match_list[a]

    barX_new = [birth_X, deaths_X[a]]
    assert np.allclose(np.asarray(entry["barX"]), np.asarray(barX_new)), f"Error: conflict between barX entries {entry['barX']} and {barX_new}"

    entry["barY"].append([birth_Y, deaths_Y[b]])
    entry["deathZ"].append(deathZ)
    entry["matchedXY"].append([a,b])
    if aff_val==[]:
        entry["affinity"].append(0)
    else:
        entry["affinity"].append(aff_val)

    return verbose_match_list

def _fix_affinity(aff):
    for i,val in enumerate(aff):
        if val==[]:
            aff[i]=[0]
    return aff

def _fix_vb_affinity(vbmatches):
    for match in vbmatches:
        if match["affinity"]==[]:
            match["affinity"]=[0]
    return vbmatches



if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Given paths to rawtext saveouts of bootstraps to original and subsampled persistent homology, compute and save out cycle matches"
    )
    parser.add_argument(
        "-x", "--phomX_fname", type=str, help="filepath to X (fixed) persistent homology"
    )
    parser.add_argument(
        "-y", "--phomY_fname", type=str, help="filepath to Y (subsampled) persistent homology"
    )
    parser.add_argument(
        "-D", "--dim", default=1, type=int, help="dimension in which to compute affinity"
    )
    parser.add_argument(
        "-o", "--outdir", default="", type=str, help="output directory for matched indices and affinity scores"
    )
    parser.add_argument(
        "-a", "--affinity_method", default='C', type=str, help="selects from various cycle pairings from which to compute the Jaccard index-based cycle affinity measure"
    )
    parser.add_argument(
        "-M", "--Morse", default=False, action='store_true', help="flag to check Morse condition"
    )
    parser.add_argument(
        "-d", "--check_deaths", default=False, action='store_true', help="flag to check for ambiguous death values during cycle-matching"
    )
    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
    else:
        outdir = os.path.join(os.path.dirname(args.phomX_fname), 'matching')

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    
    phomXZ_fname = args.phomY_fname.replace('phomY','phomXZ')
    phomYZ_fname = args.phomY_fname.replace('phomY','phomYZ')

    matched, affinity, verbose_matches = find_match_wrapper(read=True,
            phomX_fname=args.phomX_fname, phomY_fname=args.phomY_fname, 
            phomXZ_fname=phomXZ_fname, phomYZ_fname=phomYZ_fname,
            dim=args.dim, affinity_method=args.affinity_method, 
            check_Morse=args.Morse, check_ambiguous_deaths=args.check_deaths
            )

    print("Saving matching data to: ")
    tag = _pull_tag(args.phomY_fname)
    dummy_filename = os.path.join(outdir,f"phom_dim{args.dim}_{tag}.txt")
    _write_out(matched, 'matched', dummy_filename, verbose=True)
    _write_out(affinity,'affinity',dummy_filename, verbose=True)
    _write_out(verbose_matches, 'verbose_match', dummy_filename, verbose=True)
