import os
import sys
import ast
import json
import argparse
import numpy as np

sys.path.append("/scratch/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD


def all_permset_dists(datapath, permbarsX_listpath,
        persistence_type="diff", homdim=1, q=2, p=2,
        match_perms=True, verbose=True, debug=True):
    
    barsX = dgmD._get_bars(datapath, homdim=homdim)
    with open(permbarsX_listpath, 'r') as fin:
        perm_pathlist = [ _check_and_clean_barspath(ppath, verbose=False) for ppath in fin.read().split() ]
        if debug:
            ### debugging code ###
            print(f"Datapath to bars for permuted dataset (first 11 entries): \n{perm_pathlist[0:10]}")
            ### debugging code ###

    permcount = len(perm_pathlist)
    permdists_out = [None]*permcount

    if match_perms:
        data_labels = _parse_pathname(datapath, perm_pathtype=False)
        matched_perm_pathlist = [ ppath for i,ppath in enumerate(perm_pathlist) if 
                                 (data_labels["modality"], data_labels["feature"], data_labels["metric"])
                                 ==
                                 (perm_labels_list[i]["modality"], perm_labels_list[i]["feature"], perm_labels_list[i]["metric"])
                                 ]
        perm_pathlist = matched_perm_pathlist

    for i, perm_path in enumerate(perm_pathlist):
        perm_labels = _parse_pathname(perm_path, perm_pathtype=True, debug=debug)


        permbarsX = dgmD._get_bars(perm_path, homdim=homdim)
        dist_summ = permset_distance(
                barsX, 
                permbarsX,
                persistence_type=persistence_type,
                q=q, p=p,
                verbose=False
                )
        # do something to collate dist_summ
        permdists_out[i] = perm_labels | dist_summ

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Permutation type(s):", set([ dist["permtype"] for dist in permdists_out ]))
        Wp_X_nullX = np.array( [dist["Wp_XY"] for dist in permdists_out] )
        print(f"Distribution of Wasserstein distance from data persistence diagrams to permuted-null persistence modules: \nmu={np.mean(Wp_X_nullX)}, sigma={np.std(Wp_X_nullX)}")
        PDX_diag = np.mean(np.array( [dist["PDX_diag"] for dist in permdists_out] ))
        print(f"Wasserstein distance from data persistence module to the diagonal diagram: \n{PDX_diag}")
        PDnullX_diag = np.array( [dist["PDY_diag"] for dist in permdists_out] )
        print(f"Wasserstein distance from permuted-null persistence modules to the diagonal diagram: \nmu={np.mean(PDnullX_diag)}, sigma={np.std(PDnullX_diag)}")
        print("\n")



    return permdists_out


def _check_and_clean_barspath(inpath, verbose=True):
    if inpath.endswith("bars_X.txt"):
        barpath = inpath
        if verbose:
            print("Path appears to already point to a file of persistence bars.")
    elif inpath.endswith(".txt") and "dist" in os.path.basename(inpath):
        if verbose:
            print(f"Converting distance path to persistence bar path : \n{inpath}")
        barname = "phom_data_" + os.path.basename(inpath.replace(".txt",""))
        barpath = os.path.join( os.path.dirname(inpath), barname, "bars_X.txt")
    elif os.path.isdir(inpath):
        if verbose:
            print(f"Input path is directory (assumed to contain persistence bar file): \n{inpath}")
        barpath = os.path.join(inpath, "bars_X.txt")
    else:
        if verbose:
            print(f"Input path not of recognized type (passing forward unchanged): \n{ipath}")
        barpath = inpath

    return barpath


def write_out(outdir, null_dists, datapath=None):
    if datapath_name is None:
        outpath = os.path.join(outdir, "all_permdists.json")
    else:
        nameinfo = _parse_pathname(datapath)
        outname = f"permtests_{nameinfo['modality']}_{nameinfo['feature']}_{nameinfo['metric']}.json"
        outpath = os.path.join(outdir, outname)

    with open(outpath, 'w') as fout:
        json.dump(null_dists, fout)
    return outpath


def permset_distance(barsX, permbarsX, 
        persistence_type="diff", q=2, p=2, 
        verbose=True, debug=False):

    # compute standard Wasserstein distance between diagrams from X and Y
    Wp_XY = dgmD.weighted_Wasserstein_dist(barsX, permbarsX, wtfn_type=None, q=q, p=p, verbose=verbose)
    
    PDX_diag = dgmD.weighted_Wasserstein_dist(barsX, None, wtfn_type=None, q=q, p=p, verbose=verbose)
    PDY_diag = dgmD.weighted_Wasserstein_dist(None, permbarsX, wtfn_type=None, q=q, p=p, verbose=verbose)

    results_dict = {
            "Wp_XY": Wp_XY,
            "PDX_diag": PDX_diag, 
            "PDY_diag": PDY_diag, 
            }

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Standard Wasserstein distance between PD(X) and PD(permX):", Wp_XY)
        print("Projection cost of sending PD(X) to the empty diagram:", PDX_diag)
        print("Projection cost of sending PD(permX) to the empty diagram:", PDY_diag)

    return results_dict


def get_registered_distances(vbmatches_filelist, use_affinity=True, 
        persistence_type="diff", q=2, p=2):
    vbmatches = [None]*len(vbmatches_filelist)
    for i,fpath in enumerate(vbmatches_filelist):
        with open(fpath, 'r') as fin:
            vbmatches[i] = ast.literal_eval(fin.read())

    registered_dists = np.array([
        dgmD.module_distance(
        vbmatch,
        use_affinity=use_affinity, 
        persistence_type=persistence_type,
        q=q,
        p=p) for vbmatch in vbmatches])
    return registered_dists

### debugging code ###
def _debug_bars_list(bars_flist, bars_list, name="X-hat_i"):
    Xlen=[len(bars) for bars in bars_list]
    print(f"Pulled {name} bars from {len(bars_flist)} files.")
    print(f"\nNumber of bars in {name} is distributed as:")
    print(np.histogram(Xlen))
    print("")
    if 0 in Xlen:
        print(f"full list of bar numbers: {Xlen}")
        print(f"Empty {name} diagrams come from the following files:")
        for i,num in enumerate(Xlen):
            if num==0:
                print(bars_flist[i])
### debugging code ###


    # sample datapath: /scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within_ICA/ICA50_pNMs/phom_data_ICA50_pNMs_geodesic_dists/bars_X.txt
    # sample for perm_pathtype=True: /scratch/tyoeasley/brain_representations/phom_analysis/null_testing/within_ICA/ICA50_pNMs/permstrapping/phom_data_ICA50_pNMs_geodesic_dists_subjectPerms_perm_set922_n64620/bars_X.txt
def _parse_pathname(datapath, perm_pathtype=False, debug=True):
    labelset={}
    xdirname = os.path.basename(os.path.dirname(datapath))
    xdisttype = xdirname.replace("phom_data_","").split("_dists")[0]
    xdisttype = xdisttype.replace("Psim_ztrans","Psim-ztrans")

    if debug:
        ### debugging code ###
        print(f"xdirname: {xdirname}")
        print(f"xdisttype: {xdisttype}")
        ### debugging code ###

    modality, feature, metric = xdisttype.split("_")
    labelset["modality"] = modality
    labelset["feature"] = feature
    labelset["metric"] = metric
    labelset["datatype"]="Data"

    if perm_pathtype:
        permtype = xdirname.split("_dists")[1]
        permlabel = permtype.split("Perms_")[1]
        if "subject" in permtype:
            permtype = "subject"
        elif "feature" in permtype:
            permtype = "feature"
        else:
            raise ValueError(f"unrecognized permutation type: {permtype}")

        labelset["datatype"] = "Null"
        labelset["permtype"] = permtype
        labelset["permlabel"] = permlabel
    return labelset

# should not be relevant to permtests case; kept in for use as debugging function
def _verify_distinct_spaces(pathX, pathY):
    if pathX==pathY:
        print("Trivial case! The following path were given as seperate identical inputs for distinct spaces:")
        print(pathX)
        print("\n\nExiting.")
        exit()

################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-x",
        "--datapath",
        default=None,
        type=str,
        help="filepath to bars of first persistence diagram"
    )
    parser.add_argument(
        "-y",
        "--nullpath_lists",
        default=None,
        type=str,
        help="filepath to bars of second persistence diagram"
    )
    parser.add_argument(
        "-o", "--outdir",
        default=None, 
        type=str, 
        help="output directory"
    )
    parser.add_argument(
        "-d", "--dim", 
        default=1, 
        type=int, 
        help="homology dimension"
    )
    parser.add_argument(
        "-p", "--p", 
        default=2, 
        type=int, 
        help="norm power of Wasserstein distance (positive integer)"
    )
    parser.add_argument(
        "-q", "--q", 
        default=2, 
        type=int, 
        help="norm power of diagram (i.e., Euclidean) distance (np.inf or positive integer)"
    )
    parser.add_argument(
        "-P", "--persistence_type", 
        default="diff", 
        type=str, 
        help="Either 'difference' or 'quotient' type measuremnt of persistence from birth/death values"
    )
    parser.add_argument(
        "-v", "--verbose", 
        default=False, 
        action="store_true",
        help="toggle verbose output"
    )
    parser.add_argument(
        "-m", "--match_perms", 
        default=False, 
        action="store_true",
        help="toggle: enforce match between modality, feature, and distance metric of data vs. null"
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"List of datasets pulled from: \n{args.datapath}")
        print(f"List of nullsets pulled from: \n{args.nullpath_lists}")

    null_dists = all_permset_dists(
            args.datapath, 
            args.nullpath_lists,
            persistence_type=args.persistence_type, 
            homdim=args.dim, 
            q=args.q, 
            p=args.p,
            match_perms=args.match_perms,
            verbose=args.verbose 
            )
    outpath = write_out(args.outdir, null_dists, datapath=args.datapath)

    print(f"\n\nSaved to: \n{outpath}")
