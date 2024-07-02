#!/bin/bash

#SBATCH --job-name=ya-wei_job
#SBATCH -p high
#SBATCH --array=0-1
#SBATCH -N 1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-gpu=2
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --output=my_job_%A_%a.out
#SBATCH --error=my_job_%A_%a.err



# Run the task
python3 wasser_classify.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT