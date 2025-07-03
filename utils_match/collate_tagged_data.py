import os
import sys
import ast
import argparse
import numpy as np

default_tagfile = "/scratch/tyoeasley/brain_representations/subsampling/taglist100k_90p_famstruct.txt"

# Assumes data to be aggregated has simple list structure; concatenates across all files
def collate_simplist_data(filelist):
    all_data = []

    for filename in filelist:
        try:
            with open(filename,'r') as fin:
                tmp_data = ast.literal_eval(fin.read())

            # assumes that read data evaluates to list type
            all_data += tmp_data
        except FileNotFoundError:
            _handle_FNFerr(filename, tag)
            print('')

    return all_data

# Assumes data to be collated has structure of a list of dictionaries; concatenates over all files within each dict key
def collate_dictlist_data(filelist, no_agg_list=["barX", "dim"]):

    with open(filelist[0],'r') as fin:
        base_dictlist = ast.literal_eval(fin.read())

    agg_namekeys = list(base_dictlist[0].keys())
    for name in no_agg_list:
        agg_namekeys.remove(name)

    for filename in filelist[1:]:
        try:
            with open(filename,'r') as fin:
                new_dictlist = ast.literal_eval(fin.read())

            for idx, entry in enumerate(base_dictlist):
                new_entry = new_dictlist[idx]

                for name in no_agg_list:
                    assert entry[name] == new_entry[name],f"Base dictlist and dictlist {idx} fail to agree on entries in key {name} -- maybe an indexing problem? \nError in {filename}"
                for name in agg_namekeys:
                    if name == "affinity":
                        entry[name].append(new_entry[name][0])
                    else:
                        entry[name].append(new_entry[name])

        except FileNotFoundError:
            _handle_FNFerr(filename, tag)
            print('')

    return base_dictlist


def _get_outpath(outdir, filename_type, note):
    if outdir:
        fname = os.path.basename(filename_type).replace('[tagspot]', note)
        outpath = os.path.join(outdir, fname)
    else:
        outpath = filename_type.replace('[tagspot]', note)

    return outpath


def _handle_FNFerr(filename, tag):
    outdir = os.path.dirname(filename)
    print('')
    print(f"Could not locate {filename}")
    print(f"in directory {outdir}")
    search_results=os.popen('ls ' + os.path.join(outdir, f'*{tag}*')).read()
    print(f"Searching for tag: {tag}")
    print("...")
    if search_results:
        print("Did you mean any of the following?")
        print(search_results)
        print("This error may be the product of an incorrectly parsed tag.")
    else:
        print(f"No data with this tag was found in {outdir}!")
        phomY_fpath=os.path.join(os.path.dirname(outdir),'phom_out', f'phomY_{tag}.txt')
        locate_phom=os.popen('ls ' + phomY_fpath).read()
        if locate_phom:
            print("Corresponding phom data has been verified to exist:")
            print(locate_phom)
            print("Checking sizes of related bars & index files...")
            bar_sizes = [int(i) for i in 
                    os.popen('for i in $(ls ' + os.path.join(os.path.dirname(outdir),'phom_out', f'bars*{tag}*); '
                        +'do echo $(stat --printf=%s $i); done')).read().split('\n')[:-1]]
            idx_sizes = [int(i) for i in 
                    os.popen('for i in $(ls ' + os.path.join(os.path.dirname(outdir),'phom_out', f'indices*{tag}*); '
                        +'do echo $(stat --printf=%s $i); done')).read().split('\n')[:-1]]
            print(f"bars: {bar_sizes}") 
            print(f"indices: {idx_sizes}")
            print("Do bar and index file sizes indicate nontrivial match data should be present?")
            dim = _strip_dim(filename)
            _redo_match(phomY_fpath, dim)
                    
        else:
            print("Corresponding phom data may not exist.")

def _strip_dim(filename):
    dim = filename.split('_dim')[1][0]
    return dim

default_scripter="/scratch/tyoeasley/brain_representations/src_bash/submit_match_sbatch.sh"
def _redo_match(phomY_fpath, dim, scripter=default_scripter):
    print(f"Resubmitting cycle-matching corresponding to {phomY_fpath}")
    print("...")
    datadir = os.path.dirname(os.path.dirname(phomY_fpath))

    data_label = os.path.basename(datadir).replace('phom_data_','')
    sbatch_fpath = os.path.join(os.path.dirname(datadir), f'do_match_{data_label}')
    phomX_fpath = os.path.join(datadir, 'phom_X.txt')

    submit_data_label = os.path.join('','problems', f'{data_label}_%j')

    cmd_string = f"{scripter} -x {phomX_fpath} -y {phomY_fpath} -D {dim} -f {sbatch_fpath} -d {submit_data_label} -s {datadir}"
    print(os.popen(cmd_string).read())




def _get_filelist(filename_type, tagfile=default_tagfile, count=100):
    with open(tagfile, 'r') as fin:
        taglist = [next(fin).split('\n')[0] for N in range(count)]
    
    filelist = [filename_type.replace('[tagspot]', tag) for tag in taglist]

    return filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="collate data referenced by unique tag strings"
    )
    parser.add_argument(
        "-f", "--filename_type", type=str, help="filename patern to splice tags into"
    )
    parser.add_argument(
        "-o", "--outdir", default=None, help="save directory for collated data"
    )
    parser.add_argument(
        "-t", "--tagfile", default=default_tagfile, type=str, help="path to file containing tag labels"
    )
    parser.add_argument(
        "-c", "--count", type=int, default=250, help="how many tags to iterate over"
    )
    parser.add_argument(
        "-v", "--verbose", default=True, action='store_false', help="verbose output flag"
    )
    parser.add_argument(
        "-d", "--datatype", type=str, default="dictlist", help="data structure to be collated"
    )
    args = parser.parse_args()

    filelist = _get_filelist(args.filename_type, tagfile=args.tagfile, count=args.count)

    if args.datatype == "dictlist":
        note = f"n{args.count}_dict"
        all_data = collate_dictlist_data(filelist)
    elif args.datatype == "simplist":
        note = f"n{args.count}_cat"
        all_data = collate_simplist_data(filelist)
    
    outpath = _get_outpath(args.outdir, args.filename_type, note)

    if args.verbose:
        print('Sending collated data to:')
        print(outpath)

    with open(outpath,'w') as fout:
        fout.write(all_data.__str__())
