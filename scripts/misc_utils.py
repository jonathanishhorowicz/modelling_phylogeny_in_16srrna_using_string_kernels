import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow as gpf
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from skbio import DistanceMatrix
from skbio.stats.composition import clr, closure
from sklearn.preprocessing import StandardScaler
import itertools

from typing import Dict

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

############################################################################################
# Misc functions
############################################################################################

def closure_df(df) -> pd.DataFrame:
    return pd.DataFrame(
        closure(df),
        columns=df.columns,
        index=df.index
    )


def clr_df(x):
    return pd.DataFrame(
        clr(x),
        columns=x.columns,
        index=x.index
    )

def uniform_zero_replacement(x, rng, dl=1.0):
    # detection limit is smallest positive count
    # dl = np.nanmin(np.where(x > 0, x, np.nan))
    
    # https://www.sciencedirect.com/science/article/pii/S0169743921000162#bib8
    # Table 1: runif(0.1∗DL,DL)
    
    replacements = rng.uniform(low=0.1*dl, high=dl, size=x.shape)
    imputed_x = pd.DataFrame(
        np.where(x>0.0, x, replacements),
        columns=x.columns,
        index=x.index
    )
    
    return imputed_x

def dict_rbind(df, new_names):
    out_df = pd.concat(df).reset_index()
    out_df = out_df.rename(
        columns={f"level_{i}" : x for i,x in enumerate(new_names)}
    )
    columns_to_remove = out_df.columns[out_df.columns.str.match("level_\\d+")]
    if len(columns_to_remove)>0:
        out_df = out_df.drop(columns=columns_to_remove)
    return out_df

def append_sim_args(df, arg_dict):
    return pd.concat([df, pd.DataFrame(arg_dict, index=df.index)], axis=1)

def arrayidx2args(idx, arg_dict) -> Dict:
    
    arg_dict_lengths = {k : len(v) for k,v in arg_dict.items()}
    
    logger.info(f"Parsing index {idx} with args with length {arg_dict_lengths}")
    
    keys, values = zip(*arg_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"this is job {idx} of {len(permutations_dicts)}")
    
    arg_dict = permutations_dicts[idx]
    arg_dict["N_JOBS"] = len(permutations_dicts)
    return arg_dict

def median_heuristic(x):
    return np.median(pdist(x, metric="sqeuclidean"))

def optimise_gpr(model):
    logger.debug("Optimising model")
    gpf.optimizers.Scipy().minimize(model.training_loss, variables=model.trainable_variables)

def cluster_otus(tree_distances, dist_eps):
    flat_tree_distances = DistanceMatrix(tree_distances).condensed_form()
    Z = linkage(flat_tree_distances, method="complete")
    cluster_labels = fcluster(Z, t=np.max(flat_tree_distances)*dist_eps, criterion="distance")
    return pd.DataFrame({'otu' : tree_distances.index, 'cluster' : cluster_labels})

def safe_rescale(train_arr, test_arr=None):
    """Rescale an array so that each column has zero mean and unit variance.

    If a second (testing) array is supplied then the same means/variances from the first
    (training) array is used to rescale that array (preventing data leakage).

    Args:
        train_arr: array to be rescaled (e.g. from training set)
        test_arr: optional second array (e.g. from testing). Default is None.

    Returns:
        2-Tuple containing the two rescaled arrays. The second item is None if
        test_arr was not provided. If train_arr or test_arr are Pandas DataFrames
        then they are returned as Pandas DataFrames (otherwise numpy arrays).
    """
    # if arrays are 1D then convert to (n,1) shaped arrays
    if train_arr.ndim==1:
        train_arr = train_arr[:,np.newaxis]
    
    if test_arr is not None:
        if test_arr.ndim==1:
            test_arr = test_arr[:,np.newaxis]
        assert train_arr.shape[1]==test_arr.shape[1]
    
    # column-wise rescaling
    ss = StandardScaler()
    train_arr_ = ss.fit_transform(train_arr)
    test_arr_ = ss.transform(test_arr) if test_arr is not None else None
    
    # change return type if required
    if isinstance(train_arr, pd.DataFrame):
        train_arr_ = pd.DataFrame(train_arr_, columns=train_arr.columns, index=train_arr.index)
    
    if isinstance(test_arr, pd.DataFrame):
        test_arr_ = pd.DataFrame(test_arr_, columns=test_arr.columns, index=test_arr.index)
    
    return train_arr_, test_arr_
    
############################################################################################
############################################################################################

############################################################################################
# Kernel Linear algebra
############################################################################################

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

############################################################################################
############################################################################################