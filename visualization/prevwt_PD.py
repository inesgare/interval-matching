import os
import ast
import argparse
import numpy as np
import matplotlib.pyplot as plt


def weighted_PD(
        bars, weights, outpath=None, showfig=True, color_weighted=True,
        minsz=5, maxsz=50, alpha=0.66, title=None):
    assert minsz <= maxsz, "Maximum point size of scatter points cannot be smaller than minimum point size"
    
    t = np.linspace(0,1,100)
    plt.plot(t, t, 'k--')

    bars_uz = list(zip(*bars))
    births = bars_uz[0]
    deaths = bars_uz[1]
    nbars = len(bars)
    xlims = [min(births)*0.95, max(births)*1.05]
    ylims = [min(deaths)*0.95, max(deaths)*1.05]

    if color_weighted:
        print(f"num bars: {nbars}")
        print(f"num weights: {weights.size}")
        sizeval = int(100/(1 + np.log(nbars)))
        plt.scatter(births, deaths, alpha=alpha, c=weights, cmap='viridis', s=sizeval, edgecolors='none')
        cb = plt.colorbar()
        cb.set_label("prevalence")
    else:
        size_vec = minsz + weights*(maxsz - minsz)
        plt.scatter(births, deaths, s=size_vec, alpha=alpha, color='m')

        
    if not title:
        title = "Persistence in H1\n(prevalence-weighted)"
    plt.title(title)
    plt.xlabel("Birth (radius)")
    plt.ylabel("Death (radius)")
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])

    if outpath:
        print(f"figure saved to: {outpath}")
        plt.savefig(outpath, dpi=600)
    else:
        showfig=True

    if showfig:
        plt.show()

    plt.close()


def prevalence_hist(prevalence_scores, outpath=None, showfig=True):
    plt.hist(prevalence_scores)
    plt.xlabel("Prevalence Score")
    plt.ylabel("Occurences")
    plt.title("Prevalence over Cycles")

    if outpath:
        plt.savefig(outpath, dpi=600)
    else:
        showfig=True

    if showfig:
        plt.show()

    plt.close()


def_figpath = "weighted_persistence_dgm.png"
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Plots prevalence-weighted persistence diagram given input filepath to bars, input filepath to prevalence scores, and output filepath to figure name"
    )
    parser.add_argument(
        "-b", "--barsX_fname", type=str, help="input filepath to X phom bars"
    )
    parser.add_argument(
        "-p", "--prevscores_fname", type=str, help="input filepath to prevalence scores corresponding to barsX"
    )
    parser.add_argument(
        "-f", "--figure_fpath", default=None, type=str, help="output filepath to persistence diagram image"
    )
    parser.add_argument(
        "-l", "--label", default=None, type=str, help="data label for weighted persistence diagram figure"
    )
    parser.add_argument(
        "-s", "--showfig", default=False, action="store_true", help="Show figure"
    )
    args = parser.parse_args()

    with open(args.barsX_fname, 'r') as fin:
        barsX = ast.literal_eval(fin.read())

    title = "Prevalence-weighted Persistence in H1\n" + args.label
    prevscores = np.loadtxt(args.prevscores_fname)
    weighted_PD(barsX[1], prevscores, outpath = args.figure_fpath, showfig=args.showfig, title=title)

    if args.figure_fpath:
        hist_fpath = args.figure_fpath.replace(os.path.basename(args.figure_fpath),f"prevhist_{args.label}.png")
    else:
        hist_fpath=None

    prevalence_hist(prevscores, outpath=hist_fpath, showfig=args.showfig)
