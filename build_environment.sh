#!/usr/bin/env bash
conda create --name stringphylo python=3.8 --yes
conda activate stringphylo
pip install /
    gpflow==2.9.2 /
    tensorflow-probability==0.21.0 /
    pandas==1.5.3 /
    scikit-learn==1.3.2 /
    scikit-bio==0.5.0 /
    datatable==1.1.0 /
    pyarrow==17.0.0

# install R packages - these are requied for the plotting scripts
conda install -y conda-forge::r-base==4.1.3
Rscript scripts/plotting/install_requirements.R