#!/bin/bash
#SBATCH --job-name=experiment-sparseful
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=davide.domini@unibo.it
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31G
#SBATCH --partition=rtx2080
#SBATCH --output=output_exp_%A_%a.txt
#SBATCH --chdir=/scratch.hpc/davide.domini/experiments-2025-TCD-marl-neighbor-TL
#SBATCH --gres=gpu:1

source ../venv/bin/activate

PARAMS_LIST=(
	"--dataset EMNIST"
	"--dataset CIFAR100"
)

PARAMS="${PARAMS_LIST[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH=/scratch.hpc/davide.domini/experiments-2026-PMC-sparseful

python3 src/simulation.py $PARAMS