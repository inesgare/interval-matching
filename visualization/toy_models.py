import os
import scipy
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

rng = np.random.default_rng(42)

def_knownmodels=[
        "S1w",
        "S1",
        "2S1",
        "S2uR",
        "S2v2S1",
        "T2"
        ]

def vary_dim_SNR(model_type="S2uR", N=250, p=2, normalize=True, stochastic=False, add_origin=False, plot=False, outpath=None, verbose=False):
    snr_sweep = np.logspace(-3, -1/2, num=6)
    for snr in snr_sweep:
        vary_embed_dim(
                model_type=model_type,
                N=N,
                noise_prop=snr,
                p=p,
                normalize=normalize,
                stochastic=stochastic,
                add_origin=add_origin,
                plot=plot,
                outpath=outpath,
                verbose=verbose
                )


def vary_SNR(model_type="S2uR", N=250, embedding_dim=3, p=2, normalize=True, stochastic=False, add_origin=False, plot=False, outpath=None, verbose=False):
    dim = embedding_dim
    snr_sweep = np.logspace(-3, -1/2, num=6)
    for snr in snr_sweep:
        outpath_i = get_outpath_i(outpath, p, dim, snr)
        run_model(
                model_type=model_type,
                N=N,
                noise_prop=snr,
                embedding_dim=embedding_dim,
                p=p,
                normalize=normalize,
                stochastic=stochastic,
                add_origin=add_origin,
                outpath=outpath_i,
                plot=plot,
                verbose=verbose
                )


def vary_embed_dim(model_type="S2uR", N=250, noise_prop=1/10, p=2, normalize=True, stochastic=False, add_origin=False, plot=False, outpath=None, verbose=False):
    dim_sweep = [int(i) for i in np.logspace(1/2, 3, num=6)]
    snr = noise_prop
    for dim in dim_sweep:
        outpath_i = get_outpath_i(outpath, p, dim, snr)
        run_model(
                model_type=model_type,
                N=N,
                noise_prop=snr,
                embedding_dim=dim,
                p=p,
                normalize=normalize,
                stochastic=stochastic,
                add_origin=add_origin,
                outpath=outpath_i,
                plot=plot,
                verbose=verbose
                )



def get_outpath_i(outpath, p, dim, snr):
    if snr > 0:
        outpath_i = outpath.replace(f'L{p}', f'd{dim}_snr10e{np.log10(snr)}_L{p}')
    elif snr==0:
        outpath_i = outpath.replace(f'L{p}', f'd{dim}_noiseless_L{p}')
    else:
        raise Exception("signal-to-raise ratio (specified as noise proportion) cannot be a negative number")

    outpath_i, ext = os.path.splitext(outpath_i)
    outpath_i = outpath_i.replace('.0','').replace('.',',')
    outpath_i = outpath_i + ext
    return outpath_i

def run_model(
        model_type="S2uR", 
        N = 250,
        noise_prop=0.1,
        embedding_dim=3,
        p=2,
        normalize=True,
        stochastic=False,
        add_origin=False,
        plot=False,
        outpath=None,
        verbose=True
        ):
    print(f"\nRunning toy model \"{model_type}\"...")
    if verbose:
        print("Model parameters:")
        print(f"N = {N}")
        print(f"noise_proportion = {noise_prop}")
        print(f"embedding_dimension = {embedding_dim}")
        print(f"distance normalization = {normalize}")
        print(f"stochastic = {stochastic}")
        print(f"add_origin = {add_origin}")
        print(f"p = {p}")

    model_fn = get_model_fn(model_type)

    ptcloud = model_fn(
            N=N,
            stochastic=stochastic,
            add_origin=add_origin
            )

    if normalize:
        ptcloud = add_noise(ptcloud, noise_prop, p=p)
        ptcloud = affine_embed_ptcloud(ptcloud, embedding_dim)
    else:
        ptcloud = affine_embed_ptcloud(ptcloud, embedding_dim)
        ptcloud = add_noise(ptcloud, noise_prop, p=p)


    # computes the pairwise Euclidean p-distance from a point cloud list
    dX = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(ptcloud.T, metric='minkowski', p=p))
    if normalize:
        dX = dX / np.max(dX)

    # write out distance matrix if possible; return as output if not
    if outpath:
        print(f"Saving model distance matrix to: \n{outpath}\n")
        np.savetxt(outpath, dX)
        if plot:
            plot_3d(ptcloud[0], ptcloud[1], ptcloud[2], plot=False, savename=outpath.replace('.txt','_plt3d.png'))
    else:
        return dX
#####################################################################################################################################################


#####################################################################################################################################################
def plot_3d(x, y, z, c=None, plot=True, savename=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c)
    if plot:
        plt.show()
    else:
        plt.savefig(savename, dpi=300, format='png')
        plt.close()


def affine_embed_ptcloud(ptcloud, embed_dim):
    source_dim = ptcloud.shape[0]
    mult = 1+int(embed_dim/source_dim)
    T_aff = np.concatenate([np.eye(source_dim)]*mult)[:embed_dim, :]
    ptcloud_embd = 1/mult * T_aff @ ptcloud 
    return ptcloud_embd

def get_model_fn(model_type):
    switch={
        "S1w": warped_circle,
        "S1": circle,
        "2S1": con_circles,
        "S2uR": sphere_w_chord,
        "S2v2S1": sphere_v_2circ,
        "T2": torus
     }
    model_fn = switch.get(model_type, Exception(f"Unrecogized model type: {model_type}"))
    return model_fn

def add_noise(ptcloud, noise_prop, p=2):
    assert ptcloud.ndim == 2, "point cloud must be specified as an array of N d-dimensional points with shape (d,N)"
    base_noise = np.random.standard_normal(ptcloud.shape)
    
    noise_norm = sum(np.power(np.sum(np.power(base_noise, p), axis=0), 1/p))
    sig_norm = sum(np.power(np.sum(np.power(ptcloud, p), axis=0), 1/p))

    noise = base_noise * sig_norm * noise_prop / noise_norm
    return ptcloud + noise
#####################################################################################################################################################


#####################################################################################################################################################
## WARPED CIRCLE
# generate and plot warped circle (unknot) in R3:
def warped_circle(N=250, radius=1, shape_par=2, n_loops=1, amp=1, stochastic=False, add_origin=True):
    if stochastic:
        t = np.random.rand(N)*n_loops*2*np.pi
    else:
        t = np.linspace(0, n_loops*2*np.pi, num=N)
    x = shape_par*np.cos(t)
    y = np.sin(t)
    z = np.cos(n_loops*2*t)*np.sin(n_loops*2*t)+amp*t*np.sin(t)
    if add_origin:
        x,y,z = [_add_origin(coord) for coord in [x,y,z]]
    return radius*np.array([x, y, z])

## CIRCLE
def circle(N=250, radius=1, stochastic=False, add_origin=True):
    if stochastic:
        t = np.random.rand(N)*2*np.pi
    else:
        t = np.linspace(0, 2*np.pi, num=N)
    x = np.cos(t)
    y = np.sin(t)
    if add_origin:
        x,y = [_add_origin(coord) for coord in [x,y]]
    return radius*np.array([x, y])

## CONCECTRIC CIRCLES
def con_circles(N=250, radius=1, ab_ratio=1, warped=False, stochastic=False, add_origin=True):
    Nc = int(N/2)
    if warped:
        circle_fn=warped_circle
    else:
        circle_fn=circle

    xy_in = circle_fn(N=(N-Nc), radius=radius, stochastic=stochastic, add_origin=add_origin)
    xy_out = circle_fn(N=Nc, radius=radius*(1+ab_ratio), stochastic=stochastic, add_origin=False)
    xy = np.concatenate([xy_in, xy_out], axis=1)
    return xy

## SPHERE UNION DIAMETER
def sphere_w_chord(N=250, radius=1, chord_ratio=1/10, stochastic=False, add_origin=True):
    Nc = int(N*chord_ratio)
    Ns = N - Nc
    if stochastic:
        u = np.random.rand(Ns)*2*np.pi
        v = np.arccos(2*np.random.rand(Ns)-1)
        t = 2*np.random.rand(Nc)-1
    else:
        u = np.linspace(0, 2*np.pi, num=Ns)
        rng.shuffle(u)
        v = np.arccos(np.linspace(-1, 1, num=Ns))
        t = np.linspace(-1, 1, num=Nc)
    x = np.concatenate([np.cos(u) * np.sin(v), np.zeros((Nc,))])
    y = np.concatenate([np.sin(u) * np.sin(v), np.zeros((Nc,))])
    z = np.concatenate([np.cos(v), t])
    if add_origin:
        x,y,z = [_add_origin(coord) for coord in [x,y,z]]
    return radius*np.array([x, y, z])

## WEDGE PRODUCT S1vS2vS1
def sphere_v_2circ(N=250, radius=1, circ_ratio=1/10, stochastic=False, add_origin=True):
    Nc = int(N*circ_ratio)
    Ns = N - Nc*2
    if stochastic:
        u = np.random.rand(Ns)*2*np.pi
        v = np.arccos(2*np.random.rand(Ns)-1)
        t = np.random.rand(Nc)*2*np.pi
    else:
        u = np.linspace(0, 2*np.pi, num=Ns)
        rng.shuffle(u)
        v = np.arccos(np.linspace(-1, 1, num=Ns))
        t = np.linspace(0, 2*np.pi, num=Nc)
    x = np.concatenate([np.cos(u) * np.sin(v), np.cos(t), np.cos(t)])
    y = np.concatenate([np.sin(u) * np.sin(v), np.zeros((2*Nc,))])
    z = np.concatenate([np.cos(v), np.sin(t)-2, np.sin(t)+2])
    if add_origin:
        x,y,z = [_add_origin(coord) for coord in [x,y,z]]
    return radius*np.array([x, y, z])

## TORUS
def torus(N=250, radius=1, ab_ratio=2, stochastic=False, add_origin=False):
    if stochastic:
        u = np.random.rand(N)*2*np.pi
        v = np.random.rand(N)*2*np.pi
    else:
        u = np.linspace(0, 2*np.pi, num=N)
        rng.shuffle(u)
        v = np.linspace(0, 2*np.pi, num=N)
    x=(ab_ratio + np.cos(u)) * np.cos(v)
    y=(ab_ratio + np.cos(u)) * np.sin(v)
    z=np.sin(u)
    if add_origin:
        x,y,z = [_add_origin(coord) for coord in [x,y,z]]
    return radius*np.array([x, y, z])

def _add_origin(x):
    x[0]=0
    return x
#####################################################################################################################################################


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="T2",
        help=f"specifies toy model used to generate data -- choose from the following: {def_knownmodels}"
    )
    parser.add_argument(
        "-n",
        "--sample_points",
        type=int,
        default=250,
        help="number of points sampled from toy model space"
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./T2u0_L2dists.txt",
        help="general form of output path"
    )
    parser.add_argument(
        "-p",
        "--p",
        type=int,
        default=2,
        help="power of Euclidean norm"
    )
    parser.add_argument(
        "-s",
        "--stochastic",
        default=False,
        action="store_true",
        help="toggle random (default fixed) sample of toy model space"
    )
    parser.add_argument(
        "-O",
        "--add_origin",
        default=False,
        action="store_true",
        help="union the origin into the toy model space"
    )
    parser.add_argument(
        "-S",
        "--vary_SNR",
        default=False,
        action="store_true",
        help="option to run given model over varied SNR values"
    )
    parser.add_argument(
        "-D",
        "--vary_embed_dim",
        default=False,
        action="store_true",
        help="option to run given model over varied embedding dimensions"
    )
    parser.add_argument(
        "-N",
        "--normalize",
        default=False,
        action="store_true",
        help="option to normalize distance matrices by dividing all distances by maximum value"
    )
    parser.add_argument(
        "-F",
        "--add_noiseless",
        default=False,
        action="store_true",
        help="option to run a set of noiseless embeddings at varying dimensions"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="toggle verbose output"
    )
    args = parser.parse_args()

    if "u0" in args.model_type:
        args.model_type = args.model_type.replace("u0","")
        args.add_origin = True

    if args.vary_SNR and args.vary_embed_dim:
        vary_dim_SNR(
                model_type=args.model_type,
                N=args.sample_points,
                outpath=args.outpath,
                p=args.p,
                normalize=args.normalize,
                stochastic=args.stochastic,
                add_origin=args.add_origin,
                verbose=args.verbose
                )
    elif args.vary_SNR:
        vary_SNR(
                model_type=args.model_type,
                N=args.sample_points,
                outpath=args.outpath,
                p=args.p,
                normalize=args.normalize,
                stochastic=args.stochastic,
                add_origin=args.add_origin,
                verbose=args.verbose
                )
    elif args.vary_embed_dim:
        vary_embed_dim(
                model_type=args.model_type,
                N=args.sample_points,
                outpath=args.outpath,
                p=args.p,
                normalize=args.normalize,
                stochastic=args.stochastic,
                add_origin=args.add_origin,
                verbose=args.verbose
                )
    else:
        run_model(
                model_type=args.model_type,
                N=args.sample_points,
                outpath=args.outpath,
                p=args.p,
                normalize=args.normalize,
                stochastic=args.stochastic,
                add_origin=args.add_origin,
                plot=True,
                verbose=args.verbose
                )

    if args.add_noiseless:
        vary_embed_dim(
                model_type=args.model_type,
                N=args.sample_points,
                outpath=args.outpath,
                noise_prop=0,
                p=args.p,
                normalize=args.normalize,
                stochastic=args.stochastic,
                add_origin=args.add_origin,
                verbose=args.verbose
                )
