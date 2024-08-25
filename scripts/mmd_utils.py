"""
This module contains utility functions requried to run the MMD experiments
in run_mmd_simulation_replicates.py.
"""
import numpy as np
import pandas as pd

from gpflow.kernels import RBF, Matern32

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
    """Convert the mu (mean) and size parameters of a negative binomial to the 
    n,p parameters.

    Args:
        mu: mean parameter
        size: size parameter

    Returns:
        n (number of observations) and p (probability)
    """
    p = size / (size + mu)
    n = p*mu / (1.0 - p)
    return n, p

def rnegbinom(n_samples, mu, size, rng):
    """Sample from a negative binomial using the alternative (ecology) parameterisation.
    
    Args:
        n_samples: number of samples
        mu: negative binomial mean parameter
        size: negative binomial size parameter
        rng: numpy pRNG generator
    
    Returns:
        n_samples from NB(mu, size).
    """
    return rng.negative_binomial(*convert_nb_params(mu, size), size=n_samples)

def calc_mmd(K, g1_idxs, g2_idxs):
    """Calculate the MMD from samples using the unbiased estimator of Gretton et al (2012).
    
    Args:
        K: kernel matrix
        g1_idxs: indices of the samples from group 1
        g2_idxs: indices of the samples from group 2

    Returns:
        The unbised MMD estimate.
    """
    m = g1_idxs.shape[0]
    n = g2_idxs.shape[0]
    Kmm = K[np.ix_(g1_idxs,g1_idxs)]
    Knn = K[np.ix_(g2_idxs,g2_idxs)]
    Kmn = K[np.ix_(g1_idxs,g2_idxs)]
    return 1.0/m**2.0 * Kmm.sum() + 1.0/n**2.0 * Knn.sum() - (2.0/(m*n))*Kmn.sum()

def perm_mmd_test(X0, X1, kernel_maker, n_perms, rng):
    """Perform the MMD permutation test.

    Args:
        X0: design matrix for group 0
        X1: design matrix for group 1
        kernel_maker: Callable that returns a function that computes the kernel matrix.
        n_perms: number of permutations.
        rng: numpy pRNG generator

    Returns:
        3-tuple containing (i) the permuted MMD values, (ii) the observed MMD value and
        (iii) the p-value of the test.
    """
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

# kernel factories - these functions return a callable that computes a kernel from
# two arrays
def make_rbf_kernel_fn(x0, x1, rescale):
    """Create a kernel function that computes an RBF kernel with a
    median heuristic lengthscale.

    Args:
        x0: design matrix for group 0
        x1: design matrix for group 1
        rescale (bool): if True, rescale the columns of the design matrices
            before computing the median heurstic lengthscale.

    Returns:
        Callable that can be used to compute the kernel matrix of an RBF kernel
        with a median heuristic lengthscale.
    """
    x0x1 = pd.concat([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)
    return RBF(lengthscales=h_sq**0.5).K

def make_matern32_kernel_fn(x0, x1, rescale):
    """Create a kernel function that computes a Matern32 kernel with a
    median heuristic lengthscale.

    Args:
        x0: design matrix for group 0
        x1: design matrix for group 1
        rescale (bool): if True, rescale the columns of the design matrices
            before computing the median heurstic lengthscale.

    Returns:
        Callable that can be used to compute the kernel matrix of a Matern32 kernel
        with a median heuristic lengthscale.
    """
    x0x1 = pd.concat([x0, x1], axis=0)
    if rescale:
        x0x1 = safe_rescale(x0x1)[0]
    h_sq = median_heuristic(x0x1)
    return Matern32(lengthscales=h_sq**0.5).K

def string_kernel_fn_factory(Q, variance, forceQPD):
    """Create a kernel function that computes a string kernel with a given Q 
    matrix of OTU-wise similiarities.

    Args:
        Q: matrix of OTU-wise similarities. The hyperparameters used to compute this
            matrix (e.g. k-mer length) are the hyperparameters of the resulting
            string kernel.
        variance: kernel signal variance.
        forceQPD (bool): if True and Q is not positive-definite, replace Q with
            its closest positive-definite matrix.

    Returns:
        Callable that can be used to compute the kernel matrix of a string kernel
        with a median heuristic lengthscale.
    """
    def _fn(x0=None, x1=None):
        kernel = StringKernel(Q, variance, forceQPD)
        return kernel.K
        
    return _fn

def make_gram_kernel_fn(x0, x1):
    """Create a kernel function that computes a gram (linear) kernel with a
    median heuristic lengthscale.

    This function has two unused arguments to match the signature of the equivalent
    rbf and matern32 kernel functions.

    Args:
        x0: design matrix for group 0
        x1: design matrix for group 1

    Returns:
        Callable that can be used to compute the kernel matrix of a gram (linear) kernel
        with a median heuristic lengthscale.
    """
    def _fn(x, y=None):
        if y is None:
            y = x
        return np.dot(x,y.T)
    
    return _fn