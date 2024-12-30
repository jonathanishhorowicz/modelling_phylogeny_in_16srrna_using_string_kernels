import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
from skbio import TreeNode

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import tensorflow_probability.python.distributions as tfd
import gpflow as gpf
from gpflow.kernels import Linear, RBF, Matern32
from gpflow.utilities import parameter_dict

import sys
sys.path.append("../scripts")
from data_loading_utils import load_otu_table, load_string_kernels, read_feather
from kernel_classes import StringKernel, UniFracKernel
from misc_utils import (
    cluster_otus, clr_df, closure_df, dict_rbind, append_sim_args,
    uniform_zero_replacement, arrayidx2args, median_heuristic,
    safe_rescale, optimise_gpr, rnegbinom
)
from gp_utils import hparam_grid_search

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')
logging.getLogger("utils").setLevel('INFO')

#######################################################################################################################
# setup
#######################################################################################################################

#
# read environment variables
PBS_JOBID = os.environ["PBS_JOBID"]
logger.info(f"PBS_JOBID={PBS_JOBID}")
PBS_ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"])
logger.info(f"PBS_ARRAY_INDEX={PBS_ARRAY_INDEX}")
N_REPLICATES = int(os.environ["N_REPLICATES"])
logger.info(f"N_REPLICATES={N_REPLICATES}")
N_REP_BATCHES = int(os.environ["N_REP_BATCHES"])
logger.info(f"N_REP_BATCHES={N_REP_BATCHES}")
TASK = os.environ['TASK'].lower()
if TASK not in ['classification', 'regression']:
    raise ValueError(f'Invalid TASK: {TASK}')
logger.info(f"TASK={TASK}")

#
# extract args for this job
arg_vals = arrayidx2args(
    PBS_ARRAY_INDEX,
    {
        'RESAMPLE_BATCH_INDEX' : range(N_REP_BATCHES),
        'SIGMA_SQ' :  {'classification': [0.1], 'regression': [0.3, 0.6]}[TASK],
        'N_SAMPLES' : [200, 400],
        'SAMPLE_READ_DISP' : [3, 10, 30]
    }
)

logger.info(arg_vals)

DATASET = 'fame__bacterial'
RESAMPLE_BATCH_INDEX = arg_vals['RESAMPLE_BATCH_INDEX']
SIGMA_SQ = arg_vals['SIGMA_SQ']
N_SAMPLES = arg_vals['N_SAMPLES']
SAMPLE_READ_DISP = arg_vals['SAMPLE_READ_DISP']

# phenotype simulation settings
N_CAUSAL_CLUSTERS = 10
N_CANDIDATE_CLUSTERS = 20
BETA_VAR = 10
EPS = 0.1

# GP optimisation settings
NOISE_VAR_STARTING_GUESS = 1.0
SIGNAL_VAR_STARTING_GUESS = 1.0
OPT = True

SAMPLE_READ_MEAN = int(1e5)
TRANSFORM = 'rel_abund'
TEST_SIZE = 0.2

SEED = 124356

# create save dirs
PBS_ROOT_ID = re.split("\\[|\\.", PBS_JOBID)[0]
logger.info(f"PBS_ROOT_ID: {PBS_ROOT_ID}")
save_path = save_path = os.path.join(
    "../results/gp_simulations",
    PBS_ROOT_ID,
    TASK
)
logger.info(f"Making save directory at {save_path}")
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "metadata"), exist_ok=True)
os.makedirs(os.path.join(save_path, "model_evals"), exist_ok=True)
os.makedirs(os.path.join(save_path, "model_hyperparameters"), exist_ok=True)
os.makedirs(os.path.join(save_path, "model_predictions"), exist_ok=True)
os.makedirs(os.path.join(save_path, "string_lmls"), exist_ok=True)
os.makedirs(os.path.join(save_path, "best_string_hparams"), exist_ok=True)
os.makedirs(os.path.join(save_path, "unifrac_kernel_model_selection"), exist_ok=True)

# save settings
settings = {
    'DATASET': DATASET,
    'TASK': TASK,
    'SAMPLE_READ_MEAN' : SAMPLE_READ_MEAN,
    'N_CAUSAL_CLUSTERS' : N_CAUSAL_CLUSTERS,
    'OPT' : OPT,
    'NOISE_VAR_STARTING_GUESS' : NOISE_VAR_STARTING_GUESS,
    'SIGNAL_VAR_STARTING_GUESS' : SIGNAL_VAR_STARTING_GUESS,
    'N_CAUSAL_CLUSTERS' : N_CAUSAL_CLUSTERS,
    'N_CANDIDATE_CLUSTERS' : N_CANDIDATE_CLUSTERS,
    'BETA_VAR' : BETA_VAR,
    'EPS' : EPS,
    'SEED' : [SEED]
}
settings = dict(**settings, **arg_vals)
pd.DataFrame(
    settings
).to_csv(
    os.path.join(save_path, "metadata", f"{PBS_JOBID}.csv"),
    index=False
)

# setup pRNG(s)
blocked_pre_rng = []
pre_rng = PCG64(SEED)

# advance to sample genes and X
for i in range(0, N_REP_BATCHES):
    blocked_pre_rng.append(pre_rng.jumped(i))

this_pre_rng = blocked_pre_rng[RESAMPLE_BATCH_INDEX]
rng = Generator(this_pre_rng)

tf_seed = rng.integers(low=1, high=1e9)
logger.info(f"tf_seed: {tf_seed}")
tf.random.set_seed(tf_seed)

#######################################################################################################################
# load data
#######################################################################################################################

# load OTU table
data_dict = {}
data_dict['X'] = load_otu_table(dataset=DATASET)

# tree
tree = TreeNode.read(
    os.path.join("../data/clean/formatted/trees", f"{DATASET}.tree"),
    'newick',
    convert_underscores=False
)
data_dict['tree'] = tree
data_dict['tree_dist'] = read_feather(os.path.join("../data/clean/formatted/tree_distances", f"{DATASET}.feather"))

# DMN concentrations
dmn_alphas = pd.read_csv(os.path.join("../data/clean/formatted/dmn_fits", f"alphas_{DATASET}.csv"))
data_dict["alpha_mle"] = dmn_alphas

# standardise OTU names
otu_names = np.intersect1d(data_dict['tree_dist'].index, data_dict['X'].columns)
otu_names = np.intersect1d(otu_names, data_dict["alpha_mle"].OTU)
otu_names = pd.Series(otu_names).str.replace("\\[|\\]", "", regex=True).tolist()
otu_names = np.intersect1d(otu_names, [t.name for t in data_dict['tree'].tips()])
logger.info(f"Using {len(otu_names)} OTUs")

data_dict['tree'] = data_dict['tree'].shear(otu_names)
data_dict['tree_dist'].index = data_dict['tree_dist'].index.str.replace("\\[|\\]", "", regex=True)
data_dict['tree_dist'].columns = data_dict['tree_dist'].columns.str.replace("\\[|\\]", "", regex=True)
data_dict['tree_dist'] = data_dict['tree_dist'].loc[otu_names,otu_names]
data_dict['X'].columns = data_dict['X'].columns.str.replace("\\[|\\]", "", regex=True)
data_dict['X'] = data_dict['X'][otu_names]
data_dict["alpha_mle"].OTU = data_dict["alpha_mle"].OTU.str.replace("\\[|\\]", "", regex=True)
data_dict['alpha_mle'] = data_dict['alpha_mle'].sort_values("OTU").reset_index(drop=True)

### load string kernels
kernel_dict = {}

# string kernels
string_kernels = load_string_kernels(
    DATASET,
    save_path="../data/clean/formatted/Q_matrices_float64"
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
#######################################################################################################################

#######################################################################################################################
# Utility functions
#######################################################################################################################

def evaluate_model(mod, X_test, y_test):
    """Compute the log-marginal likelihood (or ELBO for a GP classifier)
    and the log-predictive density on test data.
    
    The LML is extracted from the fitted model.

    Args:
        mod: GP model
        X_test: test data design matrix
        y_test: test data labels

    Returns:
        Dictionary with keys 'lml' and 'lpd' (both values are floats).
    """
    lml = mod.log_posterior_density().numpy()
    lpd = np.nansum(mod.predict_log_density((X_test, y_test)).numpy())
    return {'lml' : lml, 'lpd' : lpd}

def get_scatterplot_data(model, data_dict):
    
    out_df = []
    for name, (xx,yy) in data_dict.items():
        pred_mean, pred_var = [x.numpy() for x in model.predict_y(xx)]

        out_df.append(
                pd.DataFrame({
                'label' : yy.squeeze(axis=1),
                'pred_mean' : pred_mean.squeeze(axis=1),
                'pred_var' : pred_var.squeeze(axis=1),
                'name' : name
            })
        )
    return pd.concat(out_df)

def fit_generic_gpmod(X, y, kernel_maker, noise_variance, opt):
    """Fit a GP model that doesn't use a string kernel.

    The type of GP model (regressor or classifer) is inferred from
    whether y is continuous or binary.

    Args:
        X: training design matrix
        y: training labels
        kernel_maker: Callable that returns a GPflow kernel object.
        noise_variance: kernel noise variance.
        opt: if True, then optimise the kernel hyperparameters. The
            type of optimisation depends on whether the model is a
            regressor or classifier.

    Returns:
        Fitted model, None, None. The two Nones are to match the return
        signature of fit_string_gpmod, which returns additional information.
    """    
    target_type = type_of_target(y)
    
    if target_type == 'continuous':
    
        m = gpf.models.GPR(
            data=(X, y),
            kernel=kernel_maker(),
            noise_variance=noise_variance
        )
        
    elif target_type == 'binary':
    
        m = gpf.models.VGP(
            data=(X, y),
            likelihood=gpf.likelihoods.Bernoulli(),
            kernel=kernel_maker()
        )
        
    else:
        raise ValueError(f"Unsupported target type: {target_type}")
    
    # ML-II or ELBO optimisation (depends on target type)
    if opt:
        optimise_gpr(m)
        
    return m, None, None

def fit_string_gpmod(
        X,
        y,
        variance,
        noise_variance,
        opt,
        kernel_df,
        included_otus
    ):
    """Fit a GP model using a string kernel. This includes a grid search over the
    string kernel hyperparameters.

    The type of GP model (regressor or classifer) is inferred from
    whether y is continuous or binary.

    Args:
        X: training design matrix
        y: training labels
        variance: kernel signal variance
        noise_variance: kernel noise variance.
        opt: if True, then optimise the kernel hyperparameters. The
            type of optimisation depends on whether the model is a
            regressor or classifier.
        kernel_df: Pandas DataFrame where each row contains the Q matrix for
            a single candidate kernel and its hyperparameters.
        included_otus: the OTUs to include from the kernel Q matrix.
    
    Returns:
        3-tuple containing: (i) the fitted GP model, (ii) the LMLs/ELBOs for each candidate
        kernel and (iii) the hyperparameters of the selected kernel.
    """
    
    logger.debug(f"String kernel model selection with {kernel_df.shape[0]} candidate kernels")
    
    log_marg_liks = np.full(kernel_df.shape[0], np.nan)
    
    # fit a model using each string Q matrix (i.e. each string kernel hyperparameter value)
    for i, (_, row) in enumerate(kernel_df.iterrows()):
        
        try:
    
            mm, _, _ = fit_generic_gpmod(
                X, y,
                kernel_maker=lambda: StringKernel(
                    row.Q.loc[included_otus,included_otus].to_numpy(),
                    variance=variance
                ),
                noise_variance=noise_variance,
                opt=opt
            )

            log_marg_liks[i] = mm.log_posterior_density().numpy()
            
        except Exception as e:
            logger.warning(e)
            log_marg_liks[i] = np.nan # failed fit
        
    best_model_idx = np.nanargmax(log_marg_liks)
    logger.debug(f"{best_model_idx} is the best model")
    best_row = kernel_df.iloc[best_model_idx,:]
    
    # fit model with highest marginal likelihood on all samples
    m, _, _ = fit_generic_gpmod(
        X, y,
        kernel_maker=lambda: StringKernel(
            variance=variance,
            Q=best_row.Q.loc[included_otus,included_otus].to_numpy()
        ),
        noise_variance=noise_variance,
        opt=opt
    )
    
    lml_df = pd.concat(
        [kernel_df.reset_index(drop=True), pd.Series(log_marg_liks, name="lml").reset_index(drop=True)],
        axis=1
    )
        
    return m, lml_df.drop(columns="Q"), best_row.kernel_name

def fit_unifrac_kernel(X, y, weighted, param_grid):

    def unifrac_kernel_maker_factory(signal_variance):
        """Returns a callable that will itself return a UniFrac kernel for a set of OTUs with 
        fixed signal variance."""
        return lambda: UniFracKernel(data_dict['tree'], otu_names, weighted, signal_variance)
    
    def _fit_fn_factory(signal_variance, noise_variance):
        """Returns a function that will fit a UniFrac kernel model with a specified
        signal variance and noise variance."""

        unifrac_kernel_maker = unifrac_kernel_maker_factory(signal_variance)

        def _fit_fn(X, y):
            """Returns a fitted UniFrac model given a signal and noise variance."""
            return fit_generic_gpmod(X, y, unifrac_kernel_maker, noise_variance, opt=False)[0]
        
        return _fit_fn
        
    
    grid_search_res = hparam_grid_search((X, y), _fit_fn_factory, param_grid)
    return grid_search_res[0], grid_search_res[1], None

UNIFRAC_KERNEL_PARAM_GRID = {
    'signal_variance': np.logspace(-1, 0, 3),
    'noise_variance': np.logspace(-1, 0, 3)
}

model_fit_fns = {
    'linear': lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: Linear(variance=SIGNAL_VAR_STARTING_GUESS),
        NOISE_VAR_STARTING_GUESS,
        OPT
    ),
    'rbf': lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: RBF(
            variance=SIGNAL_VAR_STARTING_GUESS,
            lengthscales=median_heuristic(X)
        ),
        NOISE_VAR_STARTING_GUESS,
        OPT
    ),
    'matern32': lambda X,y: fit_generic_gpmod(
        X, y,
        lambda: Matern32(
            variance=SIGNAL_VAR_STARTING_GUESS,
            lengthscales=median_heuristic(X)
        ),
        NOISE_VAR_STARTING_GUESS,
        OPT
    ),
    # 'string': lambda X,y: fit_string_gpmod(
    #     X, y,
    #     SIGNAL_VAR_STARTING_GUESS,
    #     NOISE_VAR_STARTING_GUESS,
    #     OPT,
    #     string_kernels,
    #     otu_names
    # ),
    'unweighted-unifrac': lambda X, y: fit_unifrac_kernel(X, y, weighted=False, param_grid=UNIFRAC_KERNEL_PARAM_GRID),
    'weighted-unifrac': lambda X, y: fit_unifrac_kernel(X, y, weighted=True, param_grid=UNIFRAC_KERNEL_PARAM_GRID)
}

#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# run experiments
#######################################################################################################################

model_evals = {}
preds_and_labels = {}
model_hparams = {}
string_lmls = {}
best_string_hparams = {}
lasso_scores = {} # for sanity check
unifrac_grid_search_results = {}

for i in range(N_REPLICATES):
    tf.keras.backend.clear_session()
    logger.info(f"Replicate {i} of {N_REPLICATES-1}")
    
    # DMN concentrations
    alpha = data_dict["alpha_mle"].copy()
    alpha.alpha = rng.permutation(alpha.alpha)
    total_counts = rnegbinom(N_SAMPLES, SAMPLE_READ_MEAN, SAMPLE_READ_DISP, rng)
    
    # sample OTU abundances (counts)
    count_arr = tfd.DirichletMultinomial(
        total_counts.astype(float),
        concentration=alpha.alpha.to_numpy()
    ).sample().numpy()

    X = pd.DataFrame(
        count_arr,
        index=[f"sample_{j}" for j in range(count_arr.shape[0])],
        columns=alpha.OTU
    )
    
    # transform counts to relative abundance
    Z = closure_df(X)
    
    otu_summary = pd.DataFrame({
        'total_reads' : X.sum(axis=0),
        'prevalence' : 100.0*np.count_nonzero(X, axis=0) / X.shape[0]
    }).reset_index(
    ).rename(
        columns={'OTU' : 'otu'}
    )

    # sample causal OTU clusters
    otu_clusters = cluster_otus(data_dict["tree_dist"], EPS).merge(
        otu_summary,
        on="otu",
        how="inner"
    )

    cluster_total_reads = otu_clusters.groupby(
        "cluster",
        as_index=False
    ).total_reads.sum()

    cluster_sizes = otu_clusters.groupby(
        "cluster",
        as_index=False
    ).size()

    cluster_summary = cluster_total_reads.merge(
        cluster_sizes,
        on="cluster",
        how="inner"
    )

    candidate_clusters = cluster_summary.loc[ cluster_summary["size"]>1 ]
    candidate_clusters = candidate_clusters.sort_values("total_reads", ascending=False)
    candidate_clusters = candidate_clusters.cluster.head(N_CANDIDATE_CLUSTERS).to_numpy()

    causal_clusters = rng.choice(candidate_clusters, size=N_CAUSAL_CLUSTERS, replace=False)
    
    # sample effect sizes (phylogenetic)
    effect_sizes = pd.DataFrame({
        'cluster' : causal_clusters,
        'beta' : rng.normal(scale=BETA_VAR**0.5, size=causal_clusters.shape[0])
    })
    effect_sizes = otu_clusters.merge(
        effect_sizes,
        how="left",
        on="cluster"
    )
    effect_sizes.beta = effect_sizes.beta.fillna(0.0)
    
    # find non-phylogenetic effect sizes by permuting amongst candidate clusters
    non_phylo_effect_sizes = pd.DataFrame({
        'cluster' : candidate_clusters,
    }).merge(
        otu_clusters,
        how="left",
        on="cluster"
    ).merge(
        effect_sizes,
        how="left",
        on=["otu", "cluster", "total_reads", "prevalence"]
    )
    non_phylo_effect_sizes.beta = rng.permutation(non_phylo_effect_sizes.beta)
    non_phylo_effect_sizes = non_phylo_effect_sizes.rename(columns={'beta' : 'beta_nonphylo'})
    non_phylo_effect_sizes = non_phylo_effect_sizes.merge(effect_sizes, how="right")
    non_phylo_effect_sizes = non_phylo_effect_sizes.rename(columns={'beta' : 'beta_phylo'})
    non_phylo_effect_sizes.beta_nonphylo = non_phylo_effect_sizes.beta_nonphylo.fillna(0.0)
    all_effect_sizes = non_phylo_effect_sizes
    
    assert np.array_equal(X.columns, all_effect_sizes.otu)
    assert np.array_equal(all_effect_sizes.otu, otu_names)

    for phylo_spec in ["phylo", "nonphylo"]:
        logger.info(f"phylo_spec: {phylo_spec}")

        # calculate response from relative abundance
        y = np.dot(
            Z,
            all_effect_sizes[f"beta_{phylo_spec}"])
        y = safe_rescale(y)[0]
        if SIGMA_SQ > 0.0:
            y += rng.normal(scale=SIGMA_SQ**0.5, size=y.shape)
        
        # binary response if required
        if TASK == "classification":
            logger.debug("Sampling binary labels")
            y = (y>0).astype(np.float64)
        elif TASK == "regression":
            logger.debug("Keeping continuous labels")
        else:
            raise ValueError(f"Unrecognised TASK: {TASK}")
        
        # transform X (the design matrix seen by models)
        XX = X.copy()
        
        if TRANSFORM == "clr":
            XX = closure_df(XX)
            XX = uniform_zero_replacement(XX, rng)
            XX = clr_df(XX)
        elif TRANSFORM == "rel_abund":
            XX = closure_df(XX)
        else:
            raise ValueError(f"Unrecognised transform {TRANSFORM}")

        # train-test split        
        X_train, X_test, y_train, y_test = train_test_split(
            XX, y,
            test_size=TEST_SIZE,
            stratify=y if TASK == "classification" else None)
        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()

        # rescale covariates (and labels for GP regression)
        X_train_, X_test_ = safe_rescale(X_train, X_test)
        if TASK == "regression":
            y_train, y_test = safe_rescale(y_train, y_test)

        #
        # FIT GP MODELS
        for kernel_name, kernel_fit_fn in model_fit_fns.items():
            logger.info(f"kernel_name: {kernel_name}")
            
            # rescale X if using stationary kernel
            if re.match("matern|rbf", kernel_name) is not None:
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
            except Exception as e:
                logger.warning(f"Failed with exception:\n{e}")
                raise e
            
            # store relevant probability density results
            model_evals[(phylo_spec, kernel_name, i)] = evaluate_model(mod, xx_test, y_test)
            model_hparams[(phylo_spec, kernel_name, i)] = {
                k : v.numpy() for k,v in parameter_dict(mod).items()
            }
            if kernel_name == "string":
                logger.info(f"best_hparams: {best_hparams}")
                best_string_hparams[(phylo_spec, kernel_name, i)] = pd.DataFrame({'kernel_name' : best_hparams}, index=[0])
                string_lmls[(phylo_spec, kernel_name, i)] = lml_df
            elif 'unifrac' in kernel_name:
                unifrac_grid_search_results[(phylo_spec, kernel_name, i)] = lml_df
            
            # save model predictions and labels
            scatterplot_data = get_scatterplot_data(
                mod,
                {'train' : (xx_train, y_train), 'test' : (xx_test, y_test)}
            )
            preds_and_labels[(phylo_spec, kernel_name, i)] = scatterplot_data

#######################################################################################################################
#######################################################################################################################
            
#######################################################################################################################
# save results
#######################################################################################################################

all_model_evals = pd.concat(
    [pd.DataFrame(dict(zip(["phylo_spec", "model", "rep"], k), **v), index=[0]) for k,v in model_evals.items()]
).reset_index(drop=True)
all_model_evals = append_sim_args(all_model_evals, arg_vals)
all_model_evals.to_csv(
    os.path.join(save_path, "model_evals", f"{PBS_JOBID}.csv"),
    index=False
)

red_model_hparams = {
    k : {kk : vv for kk, vv in v.items() if "kernel" in kk}
    for k, v in model_hparams.items()
}
all_model_hparams = pd.concat(
    [
        pd.DataFrame(dict(zip(["phylo_spec", "model", "rep"], k), **v), index=[0])
        for k,v in red_model_hparams.items() if not isinstance(v, np.ndarray)]
).reset_index(drop=True)
all_model_hparams = append_sim_args(all_model_hparams, arg_vals)
all_model_hparams.to_csv(
    os.path.join(save_path, "model_hyperparameters", f"{PBS_JOBID}.csv"),
    index=False
)

all_preds_and_labels = dict_rbind(preds_and_labels, ["phylo_spec", "kernel", "rep"])
all_preds_and_labels = append_sim_args(all_preds_and_labels, arg_vals)
all_preds_and_labels.to_csv(
    os.path.join(save_path, "model_predictions", f"{PBS_JOBID}.csv"),
    index=False
)

# all_string_lmls = dict_rbind(string_lmls, ["phylo_spec", "kernel", "rep"])
# all_string_lmls = append_sim_args(all_string_lmls, arg_vals)
# all_string_lmls.to_csv(
#     os.path.join(save_path, "string_lmls", f"{PBS_JOBID}.csv"),
#     index=False
# )

# all_best_string_hparams = dict_rbind(best_string_hparams, ["phylo_spec", "kernel", "rep"])
# all_best_string_hparams = append_sim_args(all_best_string_hparams, arg_vals)
# all_best_string_hparams.to_csv(
#     os.path.join(save_path, "best_string_hparams", f"{PBS_JOBID}.csv"),
#     index=False
# )

all_unifrac_grid_search_results = dict_rbind(unifrac_grid_search_results, ["phylo_spec", "kernel", "rep"])
all_unifrac_grid_search_results = append_sim_args(all_unifrac_grid_search_results, arg_vals)
all_unifrac_grid_search_results.to_csv(
    os.path.join(save_path, "unifrac_kernel_model_selection", f"{PBS_JOBID}.csv"),
    index=False
)
logger.info("Script finished successfully")

#######################################################################################################################
#######################################################################################################################