import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import  KFold
from sklearn.utils.multiclass import type_of_target
from sklearn.base import BaseEstimator

import re

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import gpflow as gpf
from gpflow.kernels import Linear, RBF, Matern32

import sys
sys.path.append("../scripts")
from kernel_classes import StringKernel
from misc_utils import (
    closure_df,
    safe_rescale, median_heuristic, optimise_gpr,
    remove_rare_otus
)
from data_loading_utils import load_string_kernel_Q_matrices

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
logging.getLogger("utils").setLevel('INFO')


# load ASV table and phenotypes
asv_table = pd.read_csv("../data/ravel_asv_table.csv").set_index("sample_id")
phenotypes = pd.read_csv("../data/ravel_phenotypes.csv").set_index("#SampleID")
common_samples = np.intersect1d(asv_table.index, phenotypes.index)

asv_table = asv_table.loc[common_samples,:]
phenotypes = phenotypes.loc[common_samples,:]

asv_table = remove_rare_otus(asv_table, 0.0)

assert np.all(np.isnan(asv_table).sum()==0)
assert np.all(np.isnan(phenotypes).sum()==0)

### load string kernels
kernel_dict = {}
otu_names = asv_table.columns

string_kernels = load_string_kernel_Q_matrices(
    "ravel",
    save_path=Path("../data/string_q_matrices_ravel.zip")
)

string_kernels = pd.concat(
    [string_kernels[x] for x in ["spectrum", "mismatch", "gappy"]]
).reset_index(drop=True)
string_kernels.k = string_kernels.k.astype(int)

string_kernels["kernel_name"] = string_kernels[["type", "k", "m"]].apply(
    lambda x: ",".join([str(xx) for xx in x]), axis=1
)
kernel_dict["string"] = string_kernels

# only keep OTUs in X for string kernels
new_q = []
for i, q_old in enumerate(string_kernels.Q):
    q = q_old.copy()
    q.index = q.index.str.replace("\\[|\\]", "", regex=True)
    q.columns = q.columns.str.replace("\\[|\\]", "", regex=True)
    q = q.loc[otu_names,otu_names]
    new_q.append(q)
string_kernels.Q = new_q


#######################################################################################################################
# Utility functions
#######################################################################################################################


PARAM_LIMITS = {}
OPT = True
NOISE_VAR_STARTING_GUESS = 1.0
SIGNAL_VAR_STARTING_GUESS = 1.0 
SIGNAL_VAR_LOWER_LIM = 0.0
NOISE_VAR_LOWER_LIM = 0.0

def evaluate_model(mod, X_test, y_test):
    lml = mod.log_posterior_density().numpy()
    lpd = np.nansum(mod.predict_log_density((X_test, y_test)).numpy())
    return {'lml' : lml, 'lpd' : lpd}

def get_scatterplot_data(model, data_dict):
    
    out_df = []
    for name, (xx,yy) in data_dict.items():
        
        if not isinstance(model, BaseEstimator):
            pred_mean, pred_var = [x.numpy() for x in model.predict_y(xx)]
        else:
            pred_mean = model.predict(xx)[:,np.newaxis]
            pred_var = np.full(pred_mean.shape[0], np.nan)[:,np.newaxis]

        out_df.append(
                pd.DataFrame({
                'label' : yy.squeeze(axis=1),
                'pred_mean' : pred_mean.squeeze(axis=1),
                'pred_var' : pred_var.squeeze(axis=1),
                'name' : name
            })
        )
    return pd.concat(out_df)

def fit_generic_gpmod(X, y, kernel_maker, noise_variance, opt, 
                      variance_lowerlim=0.0,
                      #param_limits={}
                     ):
    
    target_type = type_of_target(y)
    # logger.debug(f"y has type {target_type}")
    
    if target_type=='continuous':
    
        m = gpf.models.GPR(
            data=(X, y),
            kernel=kernel_maker(),
            noise_variance=gpf.Parameter(
                noise_variance,
                transform=gpf.utilities.positive(NOISE_VAR_LOWER_LIM)
            ),
        )
        
    elif target_type=='binary':
    
        m = gpf.models.VGP(
            data=(X, y),
            likelihood=gpf.likelihoods.Bernoulli(),
            kernel=kernel_maker()
        )
        
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    
    # constrain parameters if required
    if variance_lowerlim > 0.0:
        m.kernel.variance = gpf.Parameter(
            m.kernel.variance.numpy(),
            transform=gpf.utilities.positive(variance_lowerlim)
        )
        
    # ML-II optimisation
    if opt:
        optimise_gpr(m)
        
    return m, None, None

def fit_string_gpmod(
    X, y,
    variance, noise_variance, opt,
    kernel_df, included_otus,
    **kwargs):
    
    logger.debug(f"String kernel model selection with {kernel_df.shape[0]} candidate kernels")
    
    log_marg_liks = np.full(kernel_df.shape[0], np.nan)
    
    def _train_gp(df_row):
        return fit_generic_gpmod(
                X, y,
                kernel_maker=lambda: StringKernel(
                    df_row.Q.loc[included_otus,included_otus].to_numpy(),
                    variance=variance
                ),
                noise_variance=gpf.Parameter(
                    noise_variance,
                    transform=gpf.utilities.positive(NOISE_VAR_LOWER_LIM)
                ),
                opt=opt,
                **kwargs
            )

    # check each string hyperparameter value
    for i, (row_idx, row) in enumerate(kernel_df.iterrows()):
        
        try:
            mm, _, _ = _train_gp(row)

            log_marg_liks[i] = mm.log_posterior_density().numpy()
            
        except Exception as e:
            logger.warning(e)
            log_marg_liks[i] = np.nan # failed fit
        
    best_model_idx = np.nanargmax(log_marg_liks)
    logger.debug(f"{best_model_idx} is the best model")
    best_row = kernel_df.iloc[best_model_idx,:]
    
    # fit model with highest marginal likelihood on all samples
    m, _, _ =  _train_gp(best_row)
    
    lml_df = pd.concat(
        [kernel_df.reset_index(drop=True), pd.Series(log_marg_liks, name="lml").reset_index(drop=True)],
        axis=1
    )
        
    return m, lml_df.drop(columns="Q"), best_row.kernel_name

model_fit_fns = {
    'linear' : lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: Linear(variance=SIGNAL_VAR_STARTING_GUESS),
        NOISE_VAR_STARTING_GUESS,
        OPT,
        variance_lowerlim=SIGNAL_VAR_LOWER_LIM
    ),
    'rbf' : lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: RBF(
            variance=SIGNAL_VAR_STARTING_GUESS,
            lengthscales=median_heuristic(X)
        ),
        NOISE_VAR_STARTING_GUESS,
        OPT,
        variance_lowerlim=SIGNAL_VAR_LOWER_LIM
    ),
    'matern32' : lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: Matern32(
            variance=SIGNAL_VAR_STARTING_GUESS,
            lengthscales=median_heuristic(X)
        ),
        NOISE_VAR_STARTING_GUESS,
        OPT,
        variance_lowerlim=SIGNAL_VAR_LOWER_LIM
    ),
    'string' : lambda X,y: fit_string_gpmod(
        X, y,
        SIGNAL_VAR_STARTING_GUESS,
        NOISE_VAR_STARTING_GUESS,
        OPT,
        string_kernels,
        otu_names,
        variance_lowerlim=SIGNAL_VAR_LOWER_LIM
    )
}

#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# run experiments
#######################################################################################################################

N_CV_FOLDS = 10

cv_splitter = KFold(n_splits=N_CV_FOLDS, shuffle=True)

model_evals = {}

for phenotype in ["Var"]:
    tf.keras.backend.clear_session()
    logger.info(f"{phenotype}")
    
    # transform X (the design matrix seen by models)
    XX = asv_table.copy()
    
    y = phenotypes[phenotype].dropna()
    XX = XX.loc[y.index,:]
    y = y.to_numpy()
        
    XX = closure_df(XX)
        
    # KFold CV:
    for fold_idx, (train_idxs, test_idxs) in enumerate(cv_splitter.split(XX, y)):
        
        logger.info(f"fold_idx={fold_idx}")
    
        # train-test split     
        X_train = XX.to_numpy()[train_idxs,:]
        X_test = XX.to_numpy()[test_idxs,:]
        y_train = y[train_idxs]
        y_test = y[test_idxs]

        # rescale covariates
        X_train_, X_test_ = safe_rescale(X_train, X_test)
        y_train, y_test = safe_rescale(y_train, y_test)

        #
        # FIT GP MODELS
        for kernel_name, kernel_fit_fn in model_fit_fns.items():
            logger.info(f"kernel_name: {kernel_name}")

            # rescale X if using stationary kernel
            if bool(re.match("matern|rbf", kernel_name)):
                xx_train = X_train_
                xx_test = X_test_
            else:
                xx_train = X_train
                xx_test = X_test

            # fit the model
            try: 
                mod, lml_df, best_hparams = kernel_fit_fn(
                    xx_train, y_train 
                )
                
                mod.kernel.variance = gpf.Parameter(
                    mod.kernel.variance.numpy(),
                    transform=gpf.utilities.positive(SIGNAL_VAR_LOWER_LIM)
                )
                
                optimise_gpr(mod)
            except Exception as e:
                logger.warning(f"Failed with exception:\n{e}")
                continue

            # store relevant probability density results
            model_evals[(phenotype, fold_idx, kernel_name)] = evaluate_model(mod, xx_test, y_test)


##############################################################################
##############################################################################

##############################################################################
# format and save LMLs and LPDs
##############################################################################

all_model_evals = pd.concat(
    [pd.DataFrame(dict(zip(["phenotype", "fold_idx", "model"], k), **v), index=[0]) for k,v in model_evals.items()]
).reset_index(drop=True)

all_model_evals.to_csv(
    "../results/ravel_gpr_model_evals.csv",
    index=False
)

logger.info('Script finished successfully')

##############################################################################
##############################################################################