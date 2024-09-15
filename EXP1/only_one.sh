#!/bin/bash

#SBATCH --job-name=ya-wei_job
#SBATCH -p high
#SBATCH -N 1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-gpu=256G
#SBATCH --output=my_job_%A_%a.out
#SBATCH --error=my_job_%A_%a.err



# Run the task
python3 validation_naive.py