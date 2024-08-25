import numpy as np
import pandas as pd

from gpflow.kernels import RBF, Matern32

from sklearn.cluster import AgglomerativeClustering

import sys
sys.path.append("../scripts")
from misc_utils import safe_rescale, median_heuristic
from kernel_classes import StringKernel

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def convert_nb_params(mu, size):
    p = size / (size + mu)
    n = p*mu / (1.0 - p)
    return n, p

def rnegbinom(n_samples, mu, size, rng):
    return rng.negative_binomial(*convert_nb_params(mu, size), size=n_samples)

def calc_mmd(K, g1_idxs, g2_idxs):
    m = g1_idxs.shape[0]
    n = g2_idxs.shape[0]
    Kmm = K[np.ix_(g1_idxs,g1_idxs)]
    Knn = K[np.ix_(g2_idxs,g2_idxs)]
    Kmn = K[np.ix_(g1_idxs,g2_idxs)]
    return 1.0/m**2.0 * Kmm.sum() + 1.0/n**2.0 * Knn.sum() - (2.0/(m*n))*Kmn.sum()

def perm_mmd_test(X0, X1, kernel_maker, n_perms, rng):
    logger.debug(f"MMD significance test with {n_perms} permutations")
    
    mmd_perm_vals = np.full(n_perms, np.nan)
    k_fn = kernel_maker(X0, X1)
    K_all = np.array(k_fn(pd.concat([X0, X1]).to_numpy()))
    
    mmd_obs = calc_mmd(K_all, *np.split(np.arange(K_all.shape[0]), [X0.shape[0]]))
        
    # permute samples
    for i in range(n_perms):
        sample_perm_order = rng.permutation(np.arange(K_all.shape[0]))
        sample_perm_order = np.split(sample_perm_order, [X0.shape[0]])
        mmd_perm_vals[i] = calc_mmd(K_all, *sample_perm_order)

    p_value = (1.0 + np.nansum(mmd_perm_vals>=mmd_obs)) / (np.count_nonzero(~np.isnan(mmd_perm_vals)) + 1.0)
        
    return mmd_perm_vals, mmd_obs, p_value

# kernels
def make_rbf_kernel_fn(x0, x1, rescale):
    x0x1 = pd.concat([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)
    return RBF(lengthscales=h_sq**0.5).K

def make_matern32_kernel_fn(x0, x1, rescale):
    x0x1 = pd.concat([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)
    return Matern32(lengthscales=h_sq**0.5).K

def string_kernel_fn_factory(Q, variance, forceQPD):
    
    def _fn(x0=None, x1=None):
        kernel = StringKernel(Q, variance, forceQPD)
        return kernel.K
        
    return _fn

def make_gram_kernel_fn(x0, x1):
    
    def _fn(x, y=None):
        if y is None:
            y = x
        return np.dot(x,y.T)
    
    return _fn

def get_clusters(tree_dist, n_clusters, linkage, cluster_colname="cluster"):
    cluster_labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        linkage=linkage
        ).fit_predict(
            tree_dist
        )
    return pd.DataFrame(
        {'otu' : tree_dist.index, cluster_colname : cluster_labels}
    )
