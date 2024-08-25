#!/usr/bin/env bash
export PBS_JOBID_ROOT=local_test
export N_REPLICATES=20

for PBS_ARRAY_INDEX in {0..18}; do
	export PBS_JOBID="${PBS_JOBID_ROOT}[${PBS_ARRAY_INDEX}]"
	export PBS_ARRAY_INDEX
	echo $PBS_JOBID
	python ../scripts/run_mmd_simulation_replicates.py
done

