#!/usr/bin/env bash
export PBS_JOBID_ROOT=local_test
export N_REPLICATES=1
export N_REP_BATCHES=1
export TASK='classification'

for PBS_ARRAY_INDEX in {0..11}; do
	export PBS_JOBID="${PBS_JOBID_ROOT}[${PBS_ARRAY_INDEX}]"
	export PBS_ARRAY_INDEX
	echo $PBS_JOBID
	python ../scripts/run_gp_simulation_replicates.py
done

export TASK='classification'


