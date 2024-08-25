"""
This module contains utility functions to load the required data (e.g. OTU tables, Q matrices for
string kernels).
"""
import os
import pandas as pd
import datatable as dt
from pathlib import Path

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def read_feather(filepath: Path) -> pd.DataFrame:
    """Read a feather file.

    The feather file must have been saved with a column named 'rownames'
    or 'C0' (from datatable). This column will be set as the row index.
    
    Args:
        filepath: path to feather file.

    Returns:
        pd.DataFrame.
    """
    out = pd.read_feather(filepath)
    if "rownames" in out:
        return out.set_index("rownames")
    else:
        logger.warning("'rownames' not available for use as index")
        return out.set_index("C0")

def load_otu_table(dataset, n_threads=1):
    """Load the OTU table for a given dataset.

    OTUs with zero counts are removed.

    Args:
        dataset: name of dataset.
        n_threads: number of threads to use in datatable.fread.

    Returns:
        pd.DataFrame: the OTU abundances (counts).
    """
    # load OTU abundances
    x = dt.fread(
        f"../data/clean/formatted/X/{dataset}.csv",
        nthreads=n_threads
    ).to_pandas()
    x.C0 = x.C0.astype(str)
    x = x.set_index("C0")
    x.index.rename("sample_id", inplace=True)
    x = x.iloc[:].astype(int) # datatable converts binary columns to boolean for some reason
    logger.debug("Loaded x with shape {}".format(x.shape))
    
    # remove OTUs that are all zero
    x = x.loc[:, (x != 0).any(axis=0)]

    # check there are no NAs
    assert pd.isna(x).sum().sum()==0
    
    return x.sort_index()

def parse_stringkernel_name(x):
    return dict([xx.split("__", 1) for xx in x.replace(".feather", "").replace("fame__", "fame_").split("____")]) 

def load_string_kernels(base_dataset, save_path="../data/clean/formatted/pre_computed_K"):
    
    logger.debug(f"Loading string kernels for {base_dataset}")
    
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