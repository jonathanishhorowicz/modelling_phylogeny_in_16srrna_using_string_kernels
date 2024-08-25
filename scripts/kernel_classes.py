import numpy as np

import gpflow as gpf
from gpflow.kernels import Kernel
from gpflow.utilities import positive

import tensorflow as tf

from skbio.diversity import beta_diversity

import sys
sys.path.append("../scripts")
from misc_utils import isPD, nearestPD

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class StringKernel(Kernel):
    #
    # TODO: presence/abscence instead of OTU abundances
    # for debugging but might work better
    # Can also simulate y from a GP with the given kernel to 
    # see if it's possible to recover the signal
    def __init__(self, Q, variance=1e-2, variance_lowerlim=0.0, forceQPD=False):
        logger.debug(f"Initialising SpectrumKern with Q shape {Q.shape}")
        super().__init__()
        self._Q = Q.copy() # OTU-wise similarity, p x p
        self.forceQPD = forceQPD
        if not isPD(self._Q):
            # logger.warning("StringKernel Q is not PD")
            if self.forceQPD:
                logger.warning("Forcing String Q to be PD")
                self._Q = nearestPD(self._Q)
            
        self.variance = gpf.Parameter(variance, transform=positive(variance_lowerlim))
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        logger.debug(f"X shape is {X.shape}, X2 shape is {X2.shape}, _Q shape is {self._Q.shape}")
        out = tf.linalg.matmul(tf.linalg.matmul(X, self._Q), tf.transpose(X2))
        logger.debug(f"output has shape {out.shape}")
        return out * self.variance
    
    def K_diag(self, X):
        return np.diag(self.K(X))

class UniFracKernel(Kernel):
    def __init__(self, tree, otu_names, weighted, variance, validate=False):
        super().__init__()
        
        self.tree = tree.copy()
        self.weighted = weighted
        self.otu_names = otu_names.copy() # [t.name for t in tree.tips()]
        self.validate = validate
        self._fn = self._init_k_fn()
        self.variance = gpf.Parameter(variance, transform=positive())
            
    def K(self, X, X2=None):
        
#         if X.shape[1]!=len(self.otu_names):
#             raise ValueError(f"Number of OTUs in X ({X.shape[1]}) does not match tree ({len(self.otu_names)})")
        
        if X2 is not None:
            XX = np.concatenate([X, X2], axis=0)
        else:
            XX = X
        logger.debug(f"XX shape is {XX.shape}")
        
        K = self._fn(XX)
        logger.debug(f"K has shape {K.shape}")
        
        if X2 is not None:
            K = K[np.ix_(np.arange(X.shape[0]), np.arange(X2.shape[0]))]
        
        return self.variance * self._centreK(K)
    
    def K_diag(self, X):
        return np.diag(self.K(X))
    
    def _centreK(self, K):
        n, m = K.shape
        J0 = np.eye(n) - (1.0/n) * np.matmul(np.ones((n,1)), np.transpose(np.ones((n,1))))
        J1 = np.eye(m) - (1.0/m) * np.matmul(np.ones((m,1)), np.transpose(np.ones((m,1))))
        logger.debug(f"J0 has shape {J0.shape}, J1 has shape {J1.shape}, K has shape {K.shape}")
        return -0.5 * np.matmul(J0, np.matmul(K, J1))
    
    def _init_k_fn(self):
        
        def _fn(x):
            return beta_diversity(
                metric="weighted_unifrac" if self.weighted else "unweighted_unifrac",
                counts=x,
                otu_ids=self.otu_names,
                tree=self.tree,
                validate=self.validate
            ).data
        
        return _fn
    
# class UniFracKernel(Kernel):
#     def __init__(self, K, variance, centre):
#         super().__init__()
#         if centre:
#             self._K = tf.convert_to_tensor(self._centreK(K.copy().to_numpy()))
#         else:
#             self._K = tf.convert_to_tensor(K.copy().to_numpy())
# #         if not isPD(self._K):
# #             logger.warning("K is not PD")
#         self.variance = gpf.Parameter(variance, transform=positive())
#         self.centre = centre
            
#     def K(self, X, X2=None):
        
#         if X2 is None:
#             X2 = X
#         logger.debug(f"X shape is {X.shape}, X2 shape is {X2.shape}")
#         out = tf.gather(tf.gather(self._K, tf.cast(X2, tf.int32), axis=1), tf.cast(X, tf.int32), axis=0)
#         logger.debug(f"output has shape {out.shape}")
#         return self.variance * out
    
#     def K_diag(self, X):
#         return np.diag(self.K(X))
    
#     def _centreK(self, K):
#         n = K.shape[0]
#         one_vector = np.ones((n,1))
#         J = np.eye(n) - 1.0/n * np.matmul(one_vector, np.transpose(one_vector))
#         return -0.5 * np.matmul(J, np.matmul(1.0 - K, J))