# Modelling phylogeny in 16S rRNA gene sequencing datasets using string kernels

# Repo structure

* `data`: input data files needed to generate the results in the paper, e.g. OTU counts, phylogenetic trees, string kernel otu-wise similarity matrices (called $Q$ in the paper), etc.
* `results`: output files used to create the figures in the paper.
	- `gp_simulations`
	- `mmd_simulations`
	- `host_trait_prediction`
* `run`: runner (shell) scripts for the three sets of results.
* `scripts`: python scripts for the two sets of simulations and the real data results.


# Running the code

### TODO: conda environment setup

The `run` directory is the working directory for running all Python and R scripts (found in `scripts`). Python scripts are used to run simulations (e.g. generating OTU counts, computing kernels, MMD values are fitting GPs) and R scripts are used for making plots.

The code is designed to be run on a high-performance cluster using array jobs. Each job in the array corresponds a single combination of the parameter values. For example, if a script defines a parameter grid  for the number of samples and the $b$ parameter of the negative binomial used to simulate the total OTU counts per individual, then each job in the array runs all the replicates for a single pair (number of samples, $b$). For the parameter grid

```python
{
    'N_TOTAL' : [50, 100, 200, 400], # total number of samples
    'SAMPLE_READ_DISP' : [3.0, 10.0, 30.0], # Negative binomial dispersion
}
```

there will be $4 \times 3 = 12$ jobs in total.

To run an array job of 20 replicates for every value in this parameter grid, use a bash script containing

```sh
export N_REPLICATES=10
export PBS_JOBID=mmd_sims
for PBS_ARRAY_INDEX in {0..11}; do 
	export PBS_JOBID="${PBS_JOBID_ROOT}[${PBS_ARRAY_INDEX}]"
	export PBS_ARRAY_INDEX
	# command to run script, e.g. python {path_to_script}
done
```

If you are working on a cluster with slurm, you can use the script below to run all 12 jobs. This will automatically set the correct `PBS_ARRAY_INDEX` and `PBS_JOBID` environment variables in each job.

```sh
#!/bin/bash

#PBS -N mmd_sims
#PBS -e logs/mmd_sims
#PBS -o logs/mmd_sims
#PBS -j oe
#PBS -l select=1:ncpus=8:mem=25gb:mpiprocs=1:ompthreads=8
#PBS -l walltime=02:00:00
#PBS -J 0-11

export N_REPLICATES=10

# command to run script, e.g. python {path_to_script}
```


## MMD Simulations (Figures 5 and 6)

There are 24 jobs for the MMD simulations.

To run the simulations for a single data point of the MMD simulations (e.g. if you are running locally) run the following commands:

```sh
# Requires environment variables:
# - PBS_JOBID (str)
# - PBS_ARRAY_INDEX (int)
# - N_REPLICATES (int)
# - N_SEED_CHUNKS (int)
cd run
python ../scripts/run_mmd_simulation_replicates.py 
```

If you have all the outputs of all 24 jobs in a directory at `path_to_mmd_sim_results`, you can generate Figure 5a, 5b, 5c, 6a and 6b by running

```sh
Rscript ../scripts/plotting/make_mmd_plots.R path_to_mmd_sim_results
```

## GP simulations (Figures 7 and 8)

To produce Figures 7 and 8, from this directory run

```sh
cd run
Rscript ../scripts/plotting/gp_host_trait_sim_plots.R ../results/gp_simulations/manuscript.zip
```


## Real data results (Figure 9)

To run the analysis, from this directory run 

```sh
cd run
./run_ravel_gp_regression.sh
```

This wlil generate `results/ravel_gpr_model_evals.csv`, which contains the log-marginal likelihoods and log-predictive densities across the ten nseted cross-validation outer folds. To create Figure 10 then run

```sh
Rscript ../scripts/plotting/gp_regression_ravel_plots.R
```