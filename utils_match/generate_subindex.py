import math
import base64
import argparse
import sys
import numpy as np

rng = np.random.default_rng()
def_motherID_fpath="/scratch/tyoeasley/brain_representations/subsampling/RESTRICTED_mother_IDs_eid_order.txt"


################################################################################################################
# unrestricted sub-index generator; requires sampling proportion and number of indices (n_dims) in full sample
def generate_subidx_tag(n_dims=1003, sample_prop=0.8):
    num_choices = math.floor(n_dims * sample_prop)
    subidx = rng.choice(n_dims, num_choices, replace=False)
    return subidx_to_tag(subidx, n_dims)
################################################################################################################


################################################################################################################
# submodule to account for family structure when selecting allowable subsamples
def generate_famidx_tag(fam_struct, fam_unq, fam_occ, sample_prop=0.8, verbose=False):
    # subsample at the family structure level
    fam_subsamp = _get_fam_subsamp(fam_unq, fam_occ, proportion=sample_prop)

    # articulate subject-level subsample determined by family subsample
    subj_subidx = np.array([idx for idx,famID in enumerate(fam_struct)
        if famID in fam_subsamp])

    if verbose:
        print(
                "Requested vs. empricial bootstrap proportion:",
                sample_prop, "vs.",  len(subj_subidx)/len(fam_struct)
                )

    return subidx_to_tag(subj_subidx, len(fam_struct))

# pull family data information
def _get_fam_struct(fam_struct_fpath):
    # get list of family IDs in subject ID order
    fam_struct = np.loadtxt(fam_struct_fpath, dtype=int)

    # get array of unique family IDs
    fam_unq = np.unique(fam_struct)

    # count number of occurrences of each family ID
    fam_occ = np.array([np.count_nonzero([i==fam_val for i in fam_struct]) for fam_val in fam_unq])
    return fam_struct, fam_unq, fam_occ

# subsample family IDs (with distrubtion weighted to give uniform distribution on subject IDs)
def _get_fam_subsamp(fam_unq, fam_occ, proportion=0.8):
    # weight distribution of family ID draws so that probability of single subject's draw is unaffected by family size
    fam_pvals = (1/fam_occ)/(np.sum(1/fam_occ))

    # select weighted random subsample of family IDs
    n_fams = len(fam_unq)
    fam_subsamp = rng.choice(fam_unq, size=math.ceil(n_fams*proportion+1), replace=False, p=fam_pvals)
    return fam_subsamp
################################################################################################################


################################################################################################################
# encodes byte arrays from given subidx
def _subidx_to_bytes(subidx, n_dims):
    byte_arr = bytearray(math.ceil(n_dims / 8))

    def set_bit(bit_idx):
        byte_idx = bit_idx // 8
        bit_offset = bit_idx % 8
        byte_arr[byte_idx] |= 1 << bit_offset

    for chosen_idx in subidx:
        set_bit(chosen_idx)

    return byte_arr


def subidx_to_tag(subidx, n_dims):
    byte_arr = _subidx_to_bytes(subidx, n_dims)
    return base64.urlsafe_b64encode(byte_arr).decode()


def tag_to_subidx(tag):
    byte_arr = base64.urlsafe_b64decode(tag)

    subidx = []
    for byte_idx, b in enumerate(byte_arr):
        for bit_offset in range(8):
            if b >> bit_offset & 0x1:
                subidx.append(byte_idx * 8 + bit_offset)

    return subidx
################################################################################################################


################################################################################################################
# parses input, streams output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate subindex samplings and their unique tags"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=250, help="how many subindexes to generate"
    )
    parser.add_argument(
        "-d", "--dims", type=int, default=1003, help="total number of dimensions"
    )
    parser.add_argument(
        "-p",
        "--proportion",
        type=float,
        default=0.8,
        help="proportion of dimensions to include in subindex",
    )
    parser.add_argument(
        "-f", 
        "--fam_struct_fpath", 
        type=str,
        default=def_motherID_fpath,
        help="file containing subject-order list of family designations"
    )
    parser.add_argument(
        "-N", 
        "--no_fam_struct", 
        default=False,
        action="store_true",
        help="if flag given, then family structure of HCP sample is not taken into account for subsampling"
    )
    parser.add_argument(
        "-o", "--outfile", help="file to write tags to, if absent write to stdout"
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        default=False,
        action="store_true",
        help="if flag given, then the empirical sampling proportion is reported for every subsample"
    )

    args = parser.parse_args()

    if args.fam_struct_fpath:
        fam_struct, fam_unq, fam_occ = _get_fam_struct(args.fam_struct_fpath)

    out_stream = open(args.outfile, "w") if args.outfile else sys.stdout
    with out_stream as f:
        for _ in range(args.count):
            if args.no_fam_struct:
                f.write(generate_subidx_tag(n_dims=args.dims, sample_prop=args.proportion))
                f.write("\n")
            else:
                f.write(
                        generate_famidx_tag(
                            fam_struct, fam_unq, fam_occ, 
                            sample_prop=args.proportion,
                            verbose = args.verbose
                            )
                        )
                f.write("\n")
################################################################################################################
