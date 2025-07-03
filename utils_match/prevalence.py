import ast
import warnings
import argparse
import numpy as np


def summarize_cycle_registration(verbose_match_list):
    # NOTE: in 'matching.py', the 'affinity' dictionary entry should be initialized with empty list (like the rest), NOT [0] 
    # -- will produce unintended behavior in this line, but may do so without throwing an exception.
    affinity_array = np.array([ match["affinity"] for match in verbose_match_list ])
    
    print(f"affinity array has shape {affinity_array.shape}")

    if (affinity_array < 0).any():
        warnings.warn("All affinity scores should be nonnegative!! Affinity array (feature x bootstrap) is shown below:")
        print(affinity_array)
        print("")

    prevalence_scores = np.mean(affinity_array, axis=1)
    matched_B1_dist = np.count_nonzero(affinity_array, axis=0)
    return prevalence_scores, matched_B1_dist

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Given the (collated) verbose matches from a set of bootstraps, compute and write per-cycle prevalence scores"
    )
    parser.add_argument(
        "-i", "--inpath", type=str, help="input filepath to (string literal of) verbose match list"
    )
    parser.add_argument(
        "-p", "--prev_outpath", type=str, default=None, help="output filepath to prevalence scores"
    )
    parser.add_argument(
        "-m", "--B1match_outpath", type=str, default=None, help="output filepath to prevalence scores"
    )
    parser.add_argument(
        "-n", "--num_bootstraps", type=int, default=1000, help="Number of bootstraps of data computed"
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action='store_true', help="Send a string literal of prevalence scores to stdout"
    )
    args = parser.parse_args()
    verbose_match_list_fname = args.inpath

    with open(verbose_match_list_fname,'r') as fin:
        verbose_match_list = ast.literal_eval(fin.read())

    prevalence_scores, B1match_dist = summarize_cycle_registration(verbose_match_list)

    if args.prev_outpath:
        print(f"saving prevalence scores to {args.prev_outpath}...")
        np.savetxt(args.prev_outpath, prevalence_scores)
    else:
        args.verbose=True

    if args.B1match_outpath:
        print(f"saving matched Betti-1 numbers to {args.B1match_outpath}...")
        np.savetxt(args.B1match_outpath, B1match_dist, fmt='%i')
    else:
        args.verbose=True

    if args.verbose:
        print("Prevalence scores dervied from " + args.inpath)
        print(prevalence_scores)
        print('')
