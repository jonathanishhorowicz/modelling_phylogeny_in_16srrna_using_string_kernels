"""
This module contains utility functions to load the required data (e.g. OTU tables, Q matrices for
string kernels).
"""
import os
import pandas as pd
import datatable as dt
from pathlib import Path
import zipfile
import tempfile
from typing import Dict
from contextlib import contextmanager

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

def load_otu_table(path: Path, n_threads=1):
    """Load the OTU table for a given dataset.

    OTUs with zero counts are removed.

    Args:
        dataset: name of dataset.
        n_threads: number of threads to use in datatable.fread.

    Returns:
        pd.DataFrame: the OTU abundances (counts).
    """
    # load OTU abundances
    x = dt.fread(path, nthreads=n_threads).to_pandas()
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

@contextmanager
def _open_possibly_zipped_directory(save_path: Path):
    if save_path.suffix != '.zip':
        yield Path(save_path)

    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(save_path) as archive:
                archive.extractall(tmp_dir)
            # we assume that a directory containing all the Q-matrices has been
            # zipped so we skip the top directory
            yield Path(tmp_dir, os.listdir(tmp_dir)[0])
        

def load_string_kernel_Q_matrices(base_dataset: str, save_path: Path) -> Dict[str, pd.DataFrame]:
    """Load pre-computed string kernel Q matrices. Each string kernel (a single variant and its 
    hyperparameters) are loaded from a feather file in the supplied directory (or zipfile). 
    
    Args:
        base_dataset: a string containing 'fame' (used in the simulation studies) or 'ravel' (used
            in the real host prediction task).
        save_path: path to directory containing the string kernel Q-matrices as .feather files.

    Returns:
        Dictionary where keys are 'spectrum', 'gappy', 'mismatch' and the values are a dataframe where each row
        contains a single set of hyperparameter values and the corresponding Q matrix.
    """
    
    logger.debug(f"Loading string kernels for {base_dataset}")
    
    if "fame" in base_dataset:
        file_key = base_dataset
    elif "ravel" in base_dataset:
        file_key = "ravel"
    else:
        logger.warning(f"No pre-computed spectrum kernels for {base_dataset}")
        return None
    
    logger.debug(f"file key: {file_key}")

    with _open_possibly_zipped_directory(save_path) as kernel_file_path:
        kernel_files = [p for p in kernel_file_path.glob('*.feather')]

        if not kernel_files:
            raise ValueError('Zero kernel Q matrices found!')
        
        logger.info(f"Found {len(kernel_files)} string kernel files")

        loaded_kernels = []
        for i, p in enumerate(kernel_files):
            logger.debug(f"Loading kernel {i} of {len(kernel_files)}")
            Q = read_feather(p)
            if not Q.columns.equals(Q.index):
                logger.warning(f"Columns and index don't match for {p}")
                continue
            loaded_kernels.append(
                dict({'Q' : Q.astype(float)}, **parse_stringkernel_name(p.name))
            )
        logger.info(f"Loaded {len(loaded_kernels)} string kernel Q matrices")
    
    kernel_df = pd.DataFrame([{k : v for k,v in x.items() if k!='Q'} for x in loaded_kernels])
    kernel_df["Q"] = [x['Q'] for x in loaded_kernels]
    kernel_df = kernel_df.drop(columns="dataset")
    
    out_dict = {}

    for g_name, g_df in kernel_df.groupby("type"):
        out_dict[g_name] = g_df
        
    return out_dict