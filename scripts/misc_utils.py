import numpy as np
import pandas as pd
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

def closure_df(df) -> pd.DataFrame:
    """Apply the closure transformation (i.e. make all samples sum to 1) and return
    the result as a pd.DataFrame.
    
    Args:
        df: dataframe of OTU abundances (counts).

    Returns:
        pd.DataFrame: relative abundances.
    """
    return pd.DataFrame(
        closure(df),
        columns=df.columns,
        index=df.index
    )


def clr_df(df):
    """Apply the CLR transformation (centre log-ratio) and return the result as 
    a pd.DataFrame.
    
    Args:
        df: result of applying closure to OTU counts.

    Returns:
        pd.DataFrame: CLR-transformed counts.
    """
    return pd.DataFrame(
        clr(df),
        columns=df.columns,
        index=df.index
    )

def uniform_zero_replacement(df, rng, dl=1.0):
    """Replace zeroes using the uniform zero placement rule, with the detection 
    limit is the smallest positive count.

    See Table 1 (runif(0.1∗DL,DL)) iin
    https://www.sciencedirect.com/science/article/pii/S0169743921000162#bib8

    Args:
        df: OTU counts.
        rng: numpy RNG.
        dl: lower detection limit.

    Returns:
        pd.DataFrame: counts with zeroes replaced.
    """
    replacements = rng.uniform(low=0.1*dl, high=dl, size=df.shape)
    df_imputed = pd.DataFrame(
        np.where(df>0.0, df, replacements),
        columns=df.columns,
        index=df.index
    )
    
    return df_imputed

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
    """Add a dictionary of scalars to a DatFrame.
    
    Each key is a new column containing the scalar value.

    Args:
        df: dataframe of simulation results.
        arg_dict: dictionary of simulation settings.

    Returns:
        pd.DataFrame: simulation results with new columns with the simulation
        settings.
    """
    return pd.concat([df, pd.DataFrame(arg_dict, index=df.index)], axis=1)

def arrayidx2args(idx, arg_dict: Dict) -> Dict:
    
    arg_dict_lengths = {k : len(v) for k,v in arg_dict.items()}
    
    logger.info(f"Parsing index {idx} with args with length {arg_dict_lengths}")
    
    keys, values = zip(*arg_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"this is job {idx} of {len(permutations_dicts)}")
    
    arg_dict = permutations_dicts[idx]
    arg_dict["N_JOBS"] = len(permutations_dicts)
    return arg_dict

def median_heuristic(x):
    """Calculate the median heuristic estimate of the lengthscale for an RBF kernel.
    
    See Garreau et al. (2018) for discussion of the median heuristic.

    Args:
        X: design matrix.

    Returns:
        float: estimate of the median heuristic.
    """
    return np.median(pdist(x, metric="sqeuclidean"))

def optimise_gpr(model):
    """Optimise a GP model using the L-BFGS."""
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


def remove_rare_otus(X: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Remove OTUs (columns) from an OTU table that are present (i.e. non-zero abundance)
    in fewer than a given number of samples.
    
    Args:
        X: OTU table where samples are on rows and taxa are on columns.
        cutoff: minimum number of samples in which a taxa must be present to be retained.

    Returns:
        OTU table with taxa fewer in `cutoff` samples removed.
    """
    prevalence = np.count_nonzero(X, axis=0)
    common_otus = X.columns[ prevalence >= cutoff ]
    return X[common_otus]
