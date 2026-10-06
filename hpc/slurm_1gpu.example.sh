#!/usr/bin/env bash
# EXAMPLE ONLY: adapt account/partition/time/memory and environment activation
# to the cluster. This is not a site-validated submission script.
#SBATCH --job-name=scgvb12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=scgvb12-%j.log
set -euo pipefail
# Activate your environment here, e.g. source /path/to/venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
srun bash hpc/run_12pi.sh benchmark
