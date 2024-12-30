"""
This script runs the replicates for a single data point of Figures 7 and 8.

Each data point is a single value of 
- transformation function (centre log ratio or log(x+1))
- total number of patients
- negative bionomial dispersion parameter 
"""
import numpy as np
import pandas as pd
import re
import os
import sys
from skbio import TreeNode

import tensorflow_probability.python.distributions as tfd

sys.path.append("../scripts")
from data_loading_utils import load_otu_table, read_feather, load_string_kernels
from misc_utils import (
    rnegbinom, cluster_otus, arrayidx2args, dict_rbind,
    append_sim_args, uniform_zero_replacement, closure_df, clr_df
)
from kernel_classes import UniFracKernel
from mmd_utils import *

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logging.getLogger("utils").setLevel('INFO')

#################################################################################################################
# Setup
#################################################################################################################
    
#
# read environment variables
PBS_JOBID = os.environ["PBS_JOBID"] 
logger.info(f"PBS_JOBID={PBS_JOBID}")
PBS_ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) 
logger.info(f"PBS_ARRAY_INDEX={PBS_ARRAY_INDEX}")
N_REPLICATES = int(os.environ["N_REPLICATES"])
logger.info(f"N_REPLICATES={N_REPLICATES}")
N_SEED_CHUNKS = 1
logger.info(f"N_SEED_CHUNKS={N_SEED_CHUNKS}")

#
# extract args for this job
arg_vals = arrayidx2args(
    PBS_ARRAY_INDEX,
    {
        'TRANSFORM' : ["clr", "log1p"],
        'N_TOTAL' : [50, 100, 200, 400],
        'SAMPLE_READ_DISP' : [3.0, 10.0, 30.0],
        'SEED_CHUNK' : range(N_SEED_CHUNKS)
    }
)

logger.info(arg_vals)

DATASET = 'fame__bacterial'
TRANSFORM = arg_vals['TRANSFORM']
STRING_KERNEL_VAR = arg_vals['STRING_KERNEL_VAR']
N_TOTAL = arg_vals['N_TOTAL']
GROUP1_SIZE = arg_vals['GROUP1_SIZE']
SAMPLE_READ_DISP = arg_vals['SAMPLE_READ_DISP']
SEED_CHUNK = arg_vals['SEED_CHUNK']
STRING_KERNEL_VAR = 1e-1
GROUP1_SIZE = 0.5

n = [int(x) for x in [N_TOTAL*GROUP1_SIZE, N_TOTAL*(1.0-GROUP1_SIZE)]]

SAMPLE_READ_MEAN = int(1e5)

EPS_GRID = [-0.1, 1e-2, 1e-1, 1.1]
n_mmd_permutations = 100 # MMD permutation test
n_eps_permutations = 20
n_dmn_resamples = 1

# setup RNGs
SEED = 12345
ss = np.random.SeedSequence(SEED)

child_seeds = ss.spawn(N_SEED_CHUNKS)
streams = [np.random.default_rng(s) for s in child_seeds]
rng = streams[SEED_CHUNK]

# create save dirs
PBS_ROOT_ID = re.split("\\[|\\.", PBS_JOBID)[0]
logger.info(f"PBS_ROOT_ID: {PBS_ROOT_ID}")
save_path = save_path = os.path.join(
    "../results/mmd_simulations",
    PBS_ROOT_ID)
logger.info(f"Making save directory at {save_path}")
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, "metadata"), exist_ok=True)
os.makedirs(os.path.join(save_path, "mmd_and_pvalues"), exist_ok=True)
os.makedirs(os.path.join(save_path, "permuted_mmds"), exist_ok=True)
os.makedirs(os.path.join(save_path, "sampled_alphas"), exist_ok=True)

# save settings
settings = {
    'DATASET': DATASET,
    'SAMPLE_READ_MEAN': SAMPLE_READ_MEAN,
    'n0': n[0],
    'n1': n[1],
    'N_REPLICATES': N_REPLICATES,
    'n_mmd_permutations': n_mmd_permutations,
    'n_eps_permutations': n_eps_permutations,
    'n_dmn_resamples': n_dmn_resamples,
    'SEED': [SEED]
}
pd.DataFrame(
    dict(
        **settings, **arg_vals
    )
).to_csv(
    os.path.join(save_path, "metadata", f"{PBS_JOBID}.csv"),
    index=False
)

#################################################################################################################
#################################################################################################################

#################################################################################################################
# Setup
#################################################################################################################

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
otu_names = np.intersect1d(data_dict['tree_dist'].index, data_dict["alpha_mle"].OTU)
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

# kernels
# - each kernel is a callable that takes two OTU tables (shape (n1,p) and (n2,p)) and returns
# - n1 x n2 kernel matrix of sample-wise similatiries
kernel_dict = {
    'rbf-rescale' : lambda x0,x1: make_rbf_kernel_fn(x0, x1, True),
    'matern32-rescale' : lambda x0,x1: make_matern32_kernel_fn(x0, x1, True),
    'gram' : make_gram_kernel_fn
}

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

new_q = []
for i, q_old in enumerate(string_kernels.Q):
    q = q_old.copy()
    q.index = q.index.str.replace("\\[|\\]", "", regex=True)
    q.columns = q.columns.str.replace("\\[|\\]", "", regex=True)
    q = q.loc[otu_names,otu_names]
    new_q.append(q)
string_kernels.Q = new_q


for i,row in string_kernels.iterrows():
    kernel_dict[row.kernel_name] = string_kernel_fn_factory(row.Q, STRING_KERNEL_VAR, False)

# unifrac kernels
kernel_dict["unweighted-unifrac"] = lambda x0,x1: UniFracKernel(data_dict['tree'], otu_names, False, 1.0).K
kernel_dict["weighted-unifrac"] = lambda x0,x1: UniFracKernel(data_dict['tree'], otu_names, True, 1.0).K

#################################################################################################################
#################################################################################################################

################################################################################################################
# Run replicates
################################################################################################################

all_mmd_results = {}

for alpha_rep_idx in range(N_REPLICATES):
    logger.info(f"alpha resample {alpha_rep_idx} of {N_REPLICATES-1}")

    # alpha_1
    alpha = data_dict["alpha_mle"].copy()
    alpha["alpha_i"] = rng.permutation(alpha.alpha)
    alpha = alpha.drop(columns="alpha").rename(columns={"OTU" : "otu"})

    for eps in EPS_GRID:
        logger.info(f"eps={eps}")

        alpha2 = alpha.merge(
            cluster_otus(data_dict["tree_dist"], eps).rename(
                columns={'cluster' : 'cluster_phylo'}
            ),
            on="otu",
            how="inner"
        )
        alpha2["cluster_random"] = rng.permutation(alpha2.cluster_phylo)

        for spec in ["phylo", "random"]:
            logger.debug(f"spec: {spec}")
            alpha_perm = []
            for cluster_name, cluster_df in alpha2.groupby(f"cluster_{spec}", dropna=False):
                tmpdf = cluster_df.copy()
                tmpdf[f"alpha_j_{spec}"] = rng.permutation(tmpdf.alpha_i)
                alpha_perm.append(tmpdf[["otu", f"alpha_j_{spec}"]])
            alpha_perm = pd.concat(alpha_perm)[["otu", f"alpha_j_{spec}"]]
            alpha2 = alpha2.merge(alpha_perm, on="otu")

            # sample counts
            for dmn_rep in range(n_dmn_resamples):
                logger.debug(f"dmn rep {dmn_rep} of {n_dmn_resamples-1}")
                sampled_counts = []

                for i, (conc, n_samples) in enumerate(zip([alpha2.alpha_i, alpha2[f"alpha_j_{spec}"]], n)):    

                    total_counts = rnegbinom(np.sum(n_samples), SAMPLE_READ_MEAN, SAMPLE_READ_DISP, rng)

                    count_arr = tfd.DirichletMultinomial(
                        total_counts.astype(float),
                        concentration=conc.to_numpy()
                    ).sample().numpy()

                    n_zero_count_otus = np.all(count_arr==0, axis=0).sum()
                    logger.debug(f"{n_zero_count_otus} of {count_arr.shape[1]} OTUs are all zeros")

                    sampled_counts.append(
                        pd.DataFrame(
                            count_arr,
                            index=[f"g{i}_sample{j}" for j in range(count_arr.shape[0])],
                            columns=alpha.otu
                        )
                    )

                # check otu names match across tree, OTU table and string kernel Q matrices
                for x in sampled_counts:
                    assert np.array_equal(x.columns, otu_names)
                    assert x.columns.equals(data_dict["tree_dist"].columns)
                    assert x.columns.equals(data_dict["tree_dist"].index)
                    assert np.array_equal(x.columns, alpha.otu)
                for q in string_kernels.Q:
                    assert np.array_equal(q.index, otu_names)

                # MMD permutation test
                for kernel_name, kernel_maker in kernel_dict.items():
                    logger.debug(f"kernel: {kernel_name}")
                    
                    if TRANSFORM == "log1p":
                        counts = [np.log(x+1.0) for x in sampled_counts]
                    elif TRANSFORM == "clr":
                        counts = [closure_df(x) for x in sampled_counts]
                        counts = [uniform_zero_replacement(x, rng) for x in counts]
                        counts = [clr_df(x) for x in counts]
                    else:
                        raise ValueError(f"Unrecognised transform: {TRANSFORM}")

                    mmd_result = perm_mmd_test(
                        *counts,
                        kernel_maker,
                        n_mmd_permutations,
                        rng
                    )

                    all_mmd_results[(
                        spec, kernel_name, alpha_rep_idx, eps, dmn_rep
                    )] = mmd_result

################################################################################################################
################################################################################################################

################################################################################################################
# save everything to disc
################################################################################################################

#
# format all the results
mmd_df = {}

for k, v in all_mmd_results.items():
    mmd_perm_vals, mmd_obs, p_value = v
    mmd_df[k] = pd.DataFrame(
        {'mmd_emp' : [mmd_obs], 'p_value' : [p_value]}
    )

# Empirical MMD and p-value
all_mmd_df = dict_rbind(mmd_df, ["otu_perm_method", "kernel", "alpha_rep", "eps"])

# save results to disc
append_sim_args(
    all_mmd_df,
    arg_vals
).to_csv(
    os.path.join(save_path, "mmd_and_pvalues", f"{PBS_JOBID}.csv"),
    index=False
)

logger.info("Script finished successfully")

################################################################################################################
################################################################################################################
