############################################################################################
# data loading utility functions
############################################################################################

import os
import pandas as pd
import numpy as np
import datatable as dt
import re

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def read_feather(filepath):
    out = pd.read_feather(filepath)
    if "rownames" in out:
        return out.set_index("rownames")
    else:
        logger.warning("'rownames' not available for use as index")
        return out.set_index("C0")

def load_Xy(dataset=None, phenotype=None, permitted_values=[], load_path=None, n_threads=1):
    
    logger.debug(f"dataset: {dataset}, phenotype: {phenotype}")

    if load_path is None: # real dataset

        logger.debug("Loading real dataset")
        if phenotype is None or dataset is None:
            raise ValueError("Must specify phenotype and dataset for a real dataset")
    
        # load OTU abundances
        x = dt.fread(
            "../data/clean/formatted/X/{}.csv".format(dataset),
            nthreads=n_threads
        ).to_pandas()
        x.C0 = x.C0.astype(str)
        x = x.set_index("C0")
        x.index.rename("sample_id", inplace=True)
        x = x.iloc[:].astype(int) # datatable converts binary columns to boolean for some reason
        logger.debug("Loaded x with shape {}".format(x.shape))
        
        # load phenotype
        if "fame" in dataset:
            study = "fame"
        elif "Busselton" or "CelticFire" in dataset:
            study = "buscf"
        else:
            raise ValueError("Can't get study for dataset {}".format(dataset))
        logger.debug("Study = {}".format(study))
        
        y = pd.read_csv("../data/clean/formatted/y/{}.csv".format(study))
        
        
        # multiple phenotypes
        if isinstance(phenotype, list):
            if not all([pheno in y for pheno in phenotype]):
                raise ValueError("Some phenotypes missing")
            y = y[["sample_id"] + phenotype].set_index("sample_id")
        else: # single phenotype
            if phenotype not in y:
                raise ValueError("{} not in y".format(phenotype))
            y = y[["sample_id", phenotype]].set_index("sample_id")
        logger.debug("Loaded y with shape {}".format(y.shape))
        
        if len(permitted_values)>0:
            logger.debug(f"Subsetting to {permitted_values}")
            y = y.loc[ y[phenotype].isin(permitted_values) ]
        
        otu_info = None

    else: # simulated dataset - also returns otu effect sizes

        logger.debug("Loading simulated dataset")

        x = read_feather(os.path.join(load_path, "Z.feather"))
        y = read_feather(os.path.join(load_path, "y.feather"))
        logger.debug("Loaded x with shape {}".format(x.shape))
        logger.debug("Loaded y with shape {}".format(y.shape))
        otu_info = read_feather(os.path.join(load_path, "otu_summary.feather"))
    
    # merge on samples (those with abundance data and non-NA phenotype)
    y_samples = y.dropna().index
    x_samples = x.index
    common_samples = np.intersect1d(x_samples, y_samples)
    logger.debug("{} samples in both X and y".format(len(common_samples)))
    
    # remove OTUs that are all zero
    x = x.loc[:, (x != 0).any(axis=0)]
        
    y = y.reindex(common_samples)
    x = x.reindex(common_samples)
    logger.debug("x.shape={}, y.shape={}".format(x.shape, y.shape))

    # check samples all match and that there are no NAs
    assert x.index.equals(y.index)
    assert pd.isna(x).sum().sum()==0
    assert pd.isna(y).sum().sum()==0
    
    return {'X' : x, 'y' : y, 'otu_info' : otu_info}

def load_unifrac_kernel(dataset, weighted=True, load_path=None, n_threads=1):
    # Load UniFrac kernel matrix
    try:
        if load_path is None:
            logger.debug("Loading {}weighted unifrac kernel for a real dataset".format(
                "" if weighted else "un"
            ))
            kernel_path = "../data/clean/formatted/pre_computed_K/{}weighted_unifrac_{}.csv".format(
                    "" if weighted else "un", dataset)
            logger.debug(f"Loading kernel from {kernel_path}")
            Kxx = dt.fread(
                kernel_path,
                header=True,
                nthreads=n_threads
            ).to_pandas()
            Kxx.C0 = Kxx.C0.astype(str)
            Kxx = Kxx.set_index("C0")
            Kxx.index.rename("sample_id", inplace=True)
        else:
            logger.debug("Loading unifrac kernel for a simulated dataset")
            Kxx = read_feather(os.path.join(dataset, "{}weighted_unifrac.feather".format("" if weighted else "un")))

        logger.debug("Loaded Kxx with shape {}".format(Kxx.shape))
        assert Kxx.index.equals(Kxx.columns)
        assert pd.isna(Kxx).sum().sum()==0
        
    except ValueError:
        logger.warning(f"No pre-computed unifrac found for {dataset}")
    
    return Kxx

def parse_stringkernel_name(x):
    return dict([xx.split("__", 1) for xx in x.replace(".feather", "").replace("fame__", "fame_").split("____")]) 

def load_string_kernels(base_dataset, substring_lengths=[], save_path="../data/clean/formatted/pre_computed_K"):
    
    logger.debug(f"Loading spectrum kernels for {base_dataset}")
    
    if "CelticFire" in base_dataset or "Busselton" in base_dataset:
        file_key = "buscf"
    elif "fame" in base_dataset:
        file_key = base_dataset
    elif "ravel" in base_dataset:
        file_key = "ravel"
    else:
        logger.warning(f"No pre-computed spectrum kernels for {base_dataset}")
        return None
    
    logger.debug(f"file key: {file_key}")
    
    # load files containing spectrum kernels for these OTUs
    kernel_files = [
        f for f in os.listdir(save_path)
        if os.path.isfile(os.path.join(save_path, f))
    ]
    kernel_files = [f for f in kernel_files if file_key in f and ".feather" in f]
    
    logger.debug(f"Found {len(kernel_files)} string kernel files")

    loaded_kernels = []
    for i, f in enumerate(kernel_files):
        logger.debug(f"Loading kernel {i} of {len(kernel_files)}")
        Q = read_feather(os.path.join(save_path, f))
        if not Q.columns.equals(Q.index):
            logger.warning(f"Columns and index don't match for {kernel_files[k]}")
            continue
        loaded_kernels.append(
            dict({'Q' : Q.astype(float)}, **parse_stringkernel_name(f))
        )
    logger.info(f"Loaded {len(loaded_kernels)} spectrum kernels")
    
    kernel_df = pd.DataFrame([{k : v for k,v in x.items() if k!='Q'} for x in loaded_kernels])
    kernel_df["Q"] = [x['Q'] for x in loaded_kernels]
    kernel_df = kernel_df.drop(columns="dataset")
    
    out_dict = {}

    for g_name, g_df in kernel_df.groupby("type"):
        out_dict[g_name] = g_df
        
    return out_dict