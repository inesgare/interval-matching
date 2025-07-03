import re
import os
import sys
import warnings
import argparse
import numpy as np

sys.path.append("/scratch/tyoeasley/brain_representations/src_py")
from generate_subindex import tag_to_subidx


def summarize_diagram(phom_fname, img_flag=True, do_hom0=True, verbose=False, write=False):
    with open(phom_fname,'r') as fin:
        phom_str = fin.read().split('\n')

    if img_flag:
        bars, indices = xtr_bars_indices(phom_str, do_hom0 = do_hom0, verbose = verbose)
        if write:
            _write_out(bars,'bars',phom_fname)
            _write_out(indices,'indices',phom_fname)
        else:
            return bars, indices

    else:
        bars, reps, tight_reps, indices = xtr_bars_reps_indices(phom_str, do_hom0 = do_hom0, verbose = verbose)
        if write:
            _write_out(bars,'bars',phom_fname)
            _write_out(reps,'reps',phom_fname)
            _write_out(tight_reps,'tight_reps',phom_fname)
            _write_out(indices,'indices',phom_fname)
        else:
            return bars, reps, tight_reps, indices


def _write_out(var, varlabel, phom_fpath):
    savedir   = os.path.dirname(phom_fpath)
    phom_fname= os.path.basename(phom_fpath)
    var_fname = phom_fname.replace('phom', varlabel)
    var_fpath = os.path.join(savedir, var_fname)
    # print(var_fpath)
    with open(var_fpath,'w') as fout:
        varstring = var.__str__().replace('inf','2e308')     # replace 'inf' with a true literal (may break in future versions of Python)
        fout.write(varstring)



def xtr_bars_reps(out, do_hom0 = False, verbose = True, maxdim=2) :
    '''This function converts the output of compute_bars_tight_reps into list of bars and reps, organised by dimension'''

    line_PH = _get_line_PH(out, maxdim=maxdim)

    bars = _init_dimdict(maxdim)
    reps = _init_dimdict(maxdim)
    tight_reps = _init_dimdict(maxdim)

    # reps 0 [ [[0],[1]], [[1],[2]], etc ]
    # reps 1 [ [ [0,1], [1,2], [2,3], [3,0] ], same]

    if do_hom0 :

        # 0-dim PH bars and reps
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            bar, inf_flag = _get_bar(out[i],i)
            bars[dim] += [ bar ]
            rep = _get_h0rep(out[i], inf_flag)
            reps[dim] += [ rep ]
            i+=1

    # >=1-dim PH bars and reps
    for dim in range(1,maxdim+1):
        i = line_PH[dim]+1

        if verbose:
            ### debug code ###
            print('')
            print('In: extract_bars_reps')
            print('line of output corresponding to PH dim='+str(dim)+': ' + str(i))
            print('line of output corresponding to PH dim='+str(dim+1)+': ' + str(line_PH[dim+1]))
            ### debug code ###

        while i < line_PH[dim + 1] :
            if i == len(out) - 1 : # trivial string ''
                break
            bar,_ = _get_bar(out[i],i)
            # all "finite" bars (no missing second endpoint)
            bars[dim] += [ bar ] 
            i += 1 # next line for tight reps
            tight_reps[dim] += [ _get_genreps(out[i]) ]
            i += 1 # again, next line for reps
            reps[dim] += [ _get_genreps(out[i]) ]
            i += 1

    if verbose :
        print('phom_bars:')
        print(bars)
        print('')
        print('phom_reps:')
        print(reps)
        print('')
        print('phom_tightreps:')
        print(tight_reps)
        print('')

    return bars, reps, tight_reps



def xtr_bars_reps_indices(out, do_hom0 = False, verbose = False, maxdim=2):
    '''This function converts the output of compute_bars_tight_reps into list of bars, representatives and indices of the persistence pairs,
    organised by dimension. REMARK: you need to use the modified version or ripser_tight_representative_cycles'''
    line_PH = _get_line_PH(out, maxdim=maxdim)

    bars = _init_dimdict(maxdim)
    reps = _init_dimdict(maxdim)
    tight_reps = _init_dimdict(maxdim)
    indices = _init_dimdict(maxdim)

    # reps 0 [ [[0],[1]], [[1],[2]], etc ]
    # reps 1 [ [ [0,1], [1,2], [2,3], [3,0] ], same]

    if do_hom0 :

        # 0-dim PH bars and reps
        dim = 0
        i = line_PH[dim]+1

        if verbose:
            ### debug code ###
            print('')
            print('In: extract_bars_reps_indices')
            print('line of output corresponding to PH dim='+str(dim)+': ' + str(i))
            print('line of output corresponding to PH dim='+str(dim+1)+': ' + str(line_PH[dim+1]))
            # print('Output for homology dimension ' + str(dim) + ':')
            # print(*out[i:line_PH[dim+1]], sep='\n')
            ### debug code ###

        while i < line_PH[dim + 1] :
            bar, inf_flag = _get_bar(out[i],i)
            bars[dim] += [ bar ]
            rep = _get_h0rep(out[i], inf_flag)
            reps[dim] += [ rep ]
            i+=1

    # 1-dim PH bars and reps
    for dim in range(1,maxdim):

        i = line_PH[dim]+1

        if verbose:
            ### debug code ###
            print('')
            print('In: extract_bars_reps_indices')
            print('line of output corresponding to PH dim='+str(dim)+': ' + str(i))
            print('line of output corresponding to PH dim='+str(dim+1)+': ' + str(line_PH[dim+1]))
            ### debug code ###

        while i < line_PH[dim + 1] :
            if i == len(out) - 1 : # trivial string ''
                break
            bar,_ = _get_bar(out[i],i)
            # all "finite" bars (no missing second endpoint)
            bars[dim] += [ bar ] 

            # indices
            indices[dim] += [ _get_idx(out[i]) ]

            i += 1 # next line for tight reps
            tight_reps[dim] += [ _get_genreps(out[i]) ]
            i += 1 # again, next line for reps
            reps[dim] += [ _get_genreps(out[i]) ]
            i += 1

    if verbose :
        print(bars)
        print(reps)
        print(tight_reps)

    return bars, reps, tight_reps, indices


def xtr_bars(out, do_hom0 = False, verbose = False, maxdim=2) :
    ''' This function converts the output of compute_image_bars into list of bars organised by dimension
    (simpler version than extract_bars_reps, no reps for image-persistence)'''
    line_PH = _get_line_PH(out, maxdim=maxdim)

    bars = _init_dimdict(maxdim)

    if do_hom0 :

        # 0-dim PH bars
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            bar,_ = _get_bar(out[i],i)
            bars[dim] += [ bar ]
            i+=1

    # 1-dim PH bars
    dim = 1
    i = line_PH[dim]+1

    if verbose:
        ### debug code ###
        print('')
        print('In: extract_bars')
        print('line of output corresponding to PH dim='+str(dim)+': ' + str(i))
        print('line of output corresponding to PH dim='+str(dim+1)+': ' + str(line_PH[dim+1]))
        ### debug code ###

    while i < line_PH[dim + 1] :
        if i == len(out) - 1 : # trivial string ''
            break
        bar,_ = _get_bar(out[i],i)
        bars[dim] += [ bar ]
        i+=1

    if verbose :
        print(bars)

    return bars


def xtr_bars_indices(out, do_hom0 = False, verbose = False, maxdim=2) :
    ''' This function converts the output of compute_image_bars into list of bars and indices of the persistence pairs organised by dimension
    (simpler version than extract_bars_reps_indices, no reps for image-persistence). REMARK: need to use the modified version of ripser-image!'''
    line_PH = _get_line_PH(out, maxdim=maxdim)

    bars = _init_dimdict(maxdim)
    indices = _init_dimdict(maxdim)

    if do_hom0 :

        # 0-dim PH bars
        dim = 0
        i = line_PH[dim]+1
        while i < line_PH[dim + 1] :
            bar,_ = _get_bar(out[i],i)
            bars[dim] += [ bar ]
            i+=1

    # 1-dim PH bars
    dim = 1
    i = line_PH[dim]+1

    if verbose:
        ### debug code ###
        print('')
        print('In: extract_bars_indices')
        print('line of output corresponding to PH dim='+str(dim)+': ' + str(i))
        print('line of output corresponding to PH dim='+str(dim+1)+': ' + str(line_PH[dim+1]))
        ### debug code ###

    while i < line_PH[dim + 1] :
        #bars
        if i == len(out) - 1 : # trivial string ''
            break
        bar,_ = _get_bar(out[i],i)
        bars[dim] += [ bar ]
        #indices
        indices[dim] += [ _get_idx(out[i]) ]

        i += 1

    if verbose :
        print(bars)
        print(indices)

    return bars, indices


def _get_line_PH(out, maxdim=2):
    # find after which line it starts enumerating intervals in dim 0,1,2
    line_PH = {dim: len(out) for dim in range(maxdim+2)}
    dim_prologue1 = 'persistent homology intervals in dim '
    dim_prologue2 = 'persistence intervals in dim '
    for i,line in enumerate(out) :
        if line.endswith(':'): 
            if line.startswith(dim_prologue1) or line.startswith(dim_prologue2):
                dim = int(line.split(':')[0][-1])
                line_PH[dim] = i
    return line_PH

def _init_dimdict(maxdim):
    dimdict = {dim: [] for dim in range(maxdim+1)}
    return dimdict


#  pulls birth-death interval from single line of text output
def _get_bar(line_out, line_num):
    inf_flag=False

    interval_pattern = r"\[(\d*.\d*),(\d*.\d*)\)"
    x = re.search(interval_pattern, line_out)
    if not x:
        sci_note_pattern=r"\b-?[1-9](?:\.\d+)?[Ee][-+]?\d+\b"
        x = re.findall(sci_note_pattern, line_out)     # check for scientific notation
        if not x :
            warnings.warn(f"no intervals found in output line number {line_num}: \'{line_out}\'")
            bar = []
        else:
            if len(x) > 1:
                bar = [float(numstr) for numstr in x]
            else:
                # this isn't quite right -- should also parse line_out for cases where we have
                #   (1) sci-notation birth but not death, or 
                #   (2) nonzero birth with sci-notation death (tho this would be *very* near diagonal so might be ok to overlook)
                bar = [0.] + [float(numstr) for numstr in x]

    else:
        if x.group(2) != ' ': # finite bars first
            bar = [float(x.group(1)), float(x.group(2))]
        else:
            inf_flag = True
            bar = [float(x.group(1)), np.inf]

    return bar, inf_flag


# pulls representative from single line of text output (assuming output comes from 0-dim homology)
def _get_h0rep(line_out, inf_flag):
    if inf_flag:
        rep_pattern = r"\{\[(\d+)\].*\}"
    else:
        rep_pattern = r"\{\[(\d+)\], \[(\d+)\]\}"

    y = re.search(rep_pattern, line_out)

    if inf_flag:
        h0rep = [ [int(y.group(1))] ]
    else:
        h0rep = [ [int(y.group(1)), int(y.group(2))] ]
    return h0rep


# pulls representatives from single line of text output (assuming output does not come from 0-dim homology)
def _get_genreps(line_out):
    y = re.findall(r"\[(\d+),(\d+)\] \(\d*.\d*\)", line_out)
    y = [list(elem) for elem in y]
    genreps  = [[int(e[0]), int(e[1])] for e in y]
    return genreps


# pulls indices from single line of text output
def _get_idx(line_out):
    z = re.search(r"indices: (\d*)-(\d*)", line_out)
    if not z : raise ValueError("no indices found --- are you using the modified version of ripser-tight-representative-cycles?")
    idx = [int(z.group(1)), int(z.group(2))]
    return idx

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Extract persistence bars, filtration(?) indices, and cycle representatives (when possible) from rawtext Ripser output"
    )
    parser.add_argument(
        "-x", "--phom_fname", type=str, help="filepath to distance matrix"
    )
    parser.add_argument(
        "-0", "--do_hom0", default=False, action='store_true', help="True if output is expected to contain H0 data"
    )
    parser.add_argument(
        "-w", "--write", default=False, action='store_true', help="True if derived variables are written to file; if False, derived variables are returned"
    )
    parser.add_argument(
        "-i", "--img_flag", default=False, action='store_true', help="True if parsing image persistence output"
    )
    parser.add_argument(
        "-v", "--verbose", default=False, action='store_true', help="True if verbose output"
    )

    args = parser.parse_args()

    summarize_diagram(
            args.phom_fname,
            img_flag = args.img_flag,
            do_hom0 = args.do_hom0,
            verbose = args.verbose,
            write = args.write
            )
