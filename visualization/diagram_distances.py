import ast
import numpy as np


# Implements the distance defined by Reani and Bobrowski [1] between matched persistence modules dirsum({barX}) and dirsum({barY}):
# [1] Y. Reani & O. Bobrowski, “Cycle Registration in Persistent Homology with Applications in Topological Bootstrap,” Jan. 03, 2021, arXiv:2101.00698. doi: 10.48550/arXiv.2101.00698.
def module_distance(verbose_match, use_affinity=True, 
        persistence_type="diff", q=2, p=2):
    matched_bars = [[np.array(match["barX"]), np.array(match["barY"])] for match in verbose_match]
    cycle_pdists = np.array([np.nan]*len(matched_bars))
    for i,pair in enumerate(matched_bars):
        if pair[0].size > 0 and pair[1].size > 0:
            cycle_pdists[i] = _pnorm(pair[0] - pair[1], p=q, axis=None)
        elif pair[0].size > 0:
            cycle_pdists[i] = _get_pers(pair[0], ptype=persistence_type)*np.power(2., (1-q)/q)
        elif pair[1].size > 0:
            cycle_pdists[i] = _get_pers(pair[1], ptype=persistence_type)*np.power(2., (1-q)/q)     
        else:
            raise IOError(f"Module distance dIM_qp is not defined for a matched pair of empty cycles: \nbarX={pair[0]}\nbarY={pair[1]}")

    if use_affinity:
        affinities = np.array([match["affinity"] for match in verbose_match]).flatten()
        assert np.all(affinities >= 0) and np.all(affinities <= 1), "By definition, all affinity scores must take values in [0,1]."
        cycle_pdists = np.multiply(cycle_pdists, 1-affinities)

    if p < np.inf:
        module_dist = _pnorm(cycle_pdists, p=p, axis=None)
    else:
        module_dist = max(cycle_pdists)

    return module_dist


def weighted_Wasserstein_dist(
        X1, X2, w1=None, w2=None, 
        wtfn_type='prevalence', q=2, p=2, ot_bknd="POT",
        verbose=False, debug=False):

    X1 = _validate_input(X1)
    X2 = _validate_input(X2)

    if X1.size == 0 and X2.size == 0:
        if verbose:
            print("both diagrams are empty! returning 0.")
        return 0
    elif X1.size == 0:
        if verbose:
            print("Diagram 1 is empty! Projecting Diagram 2 to the diagonal.")
        return _pnorm(_get_proj_cost(X2, w2, p=q), p=p)
    elif X2.size == 0:
        if verbose:
            print("Diagram 2 is empty! Projecting Diagram 1 to the diagonal.")
        return _pnorm(_get_proj_cost(X1, w1, p=q), p=p)


    # default to standard (diagram) Wasserstein when weights are unspecified
    if w1 is None and w2 is None:
        wtfn_type = None
    if not np.any(w1):
        # default to assuming perfectly prevalent generators when weights are unspecified or all 0
        w1 = np.ones(X1.shape[0],)
    if not np.any(w2):
        # default to assuming perfectly prevalent generators when weights are unspecified or all 0
        w2 = np.ones(X2.shape[0],)

    wt_types = [None, "prevalence", "diff", "shrink", "measure", "coord"]
    if wtfn_type in wt_types:
        # compute transport cost matrix (given PD coordinates, p-norm power, and weight application function)
        cost_mtx = weighted_cost_mtx(X1, X2, w1, w2, wtfn_type=wtfn_type, p=p, q=q)
        if verbose:
            print(f"Using weighting function of type {wtfn_type} to compute W_q,p=W_{q,p} distance")
        if debug:
            ### debug code ###
            print(f"Input X1 has shape {X1.shape} and takes values \n{X1}")
            print(f"Input X2 has shape {X2.shape} and takes values \n{X2}")
            ### debug code ###

    else:
        raise Exception("unsupported cost function type")

    signature_1, signature_2 = _get_signatures(w1, w2, wtfn_type=wtfn_type, debug=debug)

    Wp_dist = _get_emd(
        cost_mtx, 
        signature_1, 
        signature_2, 
        p=p,
        ot_bknd=ot_bknd, 
        verbose=verbose
        )

    return Wp_dist


# diagrams have discrite uniform measures D1=\mu' and D2=\nu'.
# diagram transport problem is on measures \mu=\mu'+R\nu' and \nu=\nu'+R\mu',
# where R is the "projection measure" of an arbitrary PD measure to the diagonal.
def _get_signatures(w1, w2, wtfn_type=None, debug=False):

    if not wtfn_type=="measure":
        sig1 = np.ones(w1.shape)
        sig2 = np.ones(w2.shape)
    else:
        sig1 = w1
        sig2 = w2

    renorm = sum(sig1) + sum(sig2)

    if not renorm > 0:
        print(f"weight function type: {wtfn_type}")
        print(f"non-noramlized metric signature 1: {sig1}")
        print(f"non-noramlized metric signature 2: {sig2}")
        raise IOError("Cannot compute \'measure\'-type Wasserstein distance with sets of measure 0!")

    signature_1 = np.append(sig1, sum(sig2))/renorm
    signature_2 = np.append(sig2, sum(sig1))/renorm

    if debug:
        ### debug code ###
        print(f"signature 1 has shape {signature_1.shape}") # and takes the following values:", signature_1)
        print(f"signature 2 has shape {signature_2.shape}") # and takes the following values:", signature_2)
        ### debug code ###

    return signature_1.flatten(), signature_2.flatten()

def _get_emd(
        cost_mtx, signature_1, signature_2,
        ot_bknd="POT", p=2, verbose=True
        ):

    cost_mtx = np.power(cost_mtx, p)

    # checks which optimal transport library user specifies for computation backend
    if ot_bknd=="POT":
        import ot
        # uses the python optimal transport library backend 
        Wp_dist = ot.emd2(
                signature_1, 
                signature_2, 
                cost_mtx
                )
    elif ot_bknd=="OpenCV":
        import cv2
        # uses the python wrapper for OpenCV
        Wp_dist, _, Wp_flow = cv2.EMD(
                np.array(signature_1, dtype=np.float32), 
                np.array(signature_2, dtype=np.float32), 
                distType = cv2.DIST_USER, 
                cost = np.array(cost_mtx, dtype=np.float32)
                )
        Wp_flow = np.power(Wp_flow, 1/p)
    else:
        raise Exception("optimal transport implementation specification not recognized")

    Wp_dist = np.power(Wp_dist, 1/p)

    if verbose:
        print(f"Weighted PD Wasserstein Distance:", Wp_dist)
        if ot_bknd=="OpenCV":
            print("Optimal Flow Matrix:")
            print(Wp_flow)

    return Wp_dist


def weighted_cost_mtx(X1, X2, w1, w2, p=2, q=2, wtfn_type='diff', debug=False):
    proj_cost1 = _get_proj_cost(X1, w1, p=p)
    proj_cost2 = _get_proj_cost(X2, w2, p=p)

    inner_cost = _get_inner_cost(X1, X2, w1, w2, wtfn_type=wtfn_type, p=p, q=q)
    
    cost_mtx = np.block( [[inner_cost, proj_cost1], [proj_cost2.T, 0] ] )

    if debug:
        ### debug code ###
        print(f"Block cost matrix has shape {cost_mtx.shape} and takes the following values:", cost_mtx)
        ### debug code ###

    return cost_mtx

def _get_inner_cost(X1, X2, w1, w2, wtfn_type='diff', p=2, q=2):
    # Assumes that X1.shape=(n1, d) and X2.shape=(n2, d)
    naive_inner_cost = _pnorm(X1[:, None, :] - X2[None, :, :], axis=-1, p=q)

    # Assumes that X1.shape=(n1, d) and X2.shape=(n2, d)
    # assumes w1, w2 are either None or vectors/lists of scalars
    if wtfn_type in [None, "measure", "shrink"]:
        weight_mtx = np.ones((X1.shape[0], X2.shape[0]))
        if wtfn_type=='shrink':
            X1 = np.multiply(X1, w1)
            X2 = np.multiply(X2, w2)
    elif wtfn_type=='prevalence':
        weight_mtx = np.add.outer(w1, w2) 
        conj_proj_1 = _get_proj_cost(X1, 1 - w1, p=q).reshape(w1.shape)
        conj_proj_2 = _get_proj_cost(X2, 1 - w2, p=q).reshape(w2.shape)
        conj_proj = np.abs(np.subtract.outer(conj_proj_1, conj_proj_2))
        ##### OLD DEFINITION: DEPRECATED ####
        # conj_proj = np.power( np.add.outer( np.power(conj_proj_1, p), np.power(conj_proj_2, p) ), 1/p)
        ##### OLD DEFINITION: DEPRECATED ####
    elif wtfn_type=='diff':
        weight_mtx = 1 + np.abs(np.subtract.outer(w1, w2))
    elif wtfn_type=='coord':
        w1 = w1 * max( naive_inner_cost.flatten() ) / 1.
        w2 = w2 * max( naive_inner_cost.flatten() ) / 1.
        X1 = np.concatenate([X1, w1], axis=-1)
        X2 = np.concatenate([X2, w2], axis=-1)
        naive_inner_cost = X1[:, None, :, :] - X2[None, :, :, :]
        inner_cost = _pnorm(naive_inner_cost, axis=(-1, -2), p=q)

    if not wtfn_type=='coord':
        inner_cost = np.multiply(naive_inner_cost, weight_mtx)

    if wtfn_type=='prevalence':
        inner_cost = np.power( np.power(inner_cost, p) + np.power(conj_proj, p), 1/p ) / 2.

    return inner_cost

# length of perpendicular segment connecting X to its nearest point on y=x; note that 2D assumption (X \subset R2) is important here
def _get_proj_cost(X, w, p=2, debug=False):
    if w is None:
        w = np.ones((X.shape[0],))

    assert len(X)==len(w), f"There must be as many weight values as diagram points: X and w have shapes {X.shape} and {w.shape}, respectively."

    # diagonal projection cost for all cycles assuming uniform prevalence scores of 1
    if p < np.inf:
        if debug: 
            try:
                stable_proj_cost = np.abs(np.diff(X, axis=1))*np.power(2., (1-p)/p)      # if a multidiagram with M coordinates, then 2->M
            except np.exceptions.AxisError:
                print("Failed to compute cost of projection to diagonal!")
                print(f"input array X has shape {X.shape}, size {X.size} and takes values: \n{X}")
                exit()
        else:
            stable_proj_cost = np.abs(np.diff(X, axis=1))*np.power(2., (1-p)/p)      # if a multidiagram with M coordinates, then 2->M
    else:
        stable_proj_cost = np.max(np.abs(np.diff(X, axis=1)))/2

    # prevalence scales diagonal projection cost
    proj_cost = np.multiply(w.flatten(), stable_proj_cost.flatten())
    proj_cost = proj_cost.reshape(len(proj_cost),1)
    
    return proj_cost

def _validate_input(X):
    if X is None:
        X = np.array([])
    
    if not isinstance(X, np.ndarray):
        Warning("Input data is expected to have type \"numpy.ndarray\" but has type {type(X)}; attempting to recast.")
        X = np.array(X)

    assert isinstance(X, np.ndarray), "Typecasting failed. Exiting."
    return X

# compute persistence of a given bar (birth, death)
def _get_pers(bar, ptype="diff"):
    if ptype=="diff":
        persistence, = np.abs(np.diff(bar))     # diff returns iteratble (array): 'var, =' syntax verifies single-element list assumption
    if ptype=="quotient":
        persistence, = np.exp(np.diff(np.log(bar)))     # diff returns iteratble (array): 'var, =' syntax verifies single-element list assumption
    return persistence

# read a list of birth-death pairs (of the specified homology dimension) from file
def _get_bars(bars_fname, homdim=1):
    with open(bars_fname, 'r') as fin:
        all_bars = ast.literal_eval(fin.read())
        bars = all_bars[homdim]  # "bars" dictionary is indexed by homology dimension
    return np.array(bars)

def _pnorm(X, p=2, axis=None):
    X = np.abs(X)
    if p < np.inf:
        pnorm = np.power(np.sum(np.power( np.abs(X) , p), axis=axis), 1/p)
    else:
        pnorm = np.max( X, axis=axis )
    return pnorm

